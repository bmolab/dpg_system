import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from scipy import signal
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import numpy as np
import torch
from dpg_system.torch_base_nodes import TorchDeviceDtypeNode
import scipy

# import torch_kf
# import torch_kf.ckf

def register_torch_butterworth_nodes():
    Node.app.register_node("t.filter_bank", TorchBandPassFilterBankNode.factory)
    Node.app.register_node("t.sav_gol_filter", TorchSavGolFilterNode.factory)
    Node.app.register_node("t.quat_ESEKF", TorchTristateBlendESEKFNode.factory)
    Node.app.register_node("t.ESEKF", TorchTristateBlendEuclideanNode.factory)
    Node.app.register_node("t.smart_clamp_kf", TorchSmartClampKFNode.factory)
    Node.app.register_node("t.smart_clamp_quat_kf", TorchSmartClampQuaternionKFNode.factory)
    Node.app.register_node("t.hybrid_quat_kf", TorchHybridQuaternionKFNode.factory)
    Node.app.register_node("t.persistence_quat_kf", TorchPersistenceKFNode.factory)
    Node.app.register_node("t.jerk_aware_quat_kf", TorchJerkAwareQuatKFNode.factory)


# class TorchKalmanFilterNode(TorchDeviceDtypeNode):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = TorchKalmanFilterNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#
#         self.setup_dtype_device_grad(args)
#         self.dtype = torch.float32
#
#         self.order = self.arg_as_int(default_value=2)
#
#         self.accum = 0.0
#
#         self.kf = torch_kf.ckf.constant_kalman_filter(measurement_std, process_std, dim=1, order=order, expected_model=True)
#
#         self.input = self.add_input('in', triggers_execution=True)
#         self.degree_input = self.add_input('degree', widget_type='drag_float', min=0.0, max=1.0, default_value=self.degree, callback=self.change_degree)
#         self.degree_input.widget.speed = .01
#         self.output = self.add_output('out')
#
#     def change_degree(self, input=None):
#         self.degree = self.degree_input()
#         if self.degree < 0:
#             self.degree = 0
#         elif self.degree > 1:
#             self.degree = 1
#
#     def execute(self):
#         input_value = self.input.get_data()
#         if type(self.accum) != type(input_value):
#             self.accum = any_to_match(self.accum, input_value)
#         t = type(input_value)
#         if t is np.ndarray:
#             if self.accum.size != input_value.size:
#                 self.accum = np.zeros_like(input_value)
#         elif self.app.torch_available and type(input_value) == torch.Tensor:
#             if input_value.device != self.accum.device:
#                 self.accum = any_to_match(self.accum, input_value)
#             if self.accum.size() != input_value.size():
#                 self.accum = torch.zeros_like(input_value)
#
#         self.accum = self.accum * self.degree + input_value * (1.0 - self.degree)
#         self.output.send(self.accum)


class TorchBandPassFilterBankNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchBandPassFilterBankNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.order = 1

        #  these should be algorithmically generated as octaves
        low_cut = 0.01
        high_cut = 12
        num_bands = 8
        # add facility to have band overlap...
        # take multiplier from band to band - sqrt of multiplier is 1/2 step, pow(0.25) is 1/4 step?
        #   take multiplier from band to band, math.log2(multiplier) gives band power
        # expand band up and down by pow(2, overlap * band_power

        # order 3 .75 overlap seems interesting, but not sure about overshoot...


        self.sample_frequency = 60
        self.filter_type = 'bandpass'
        self.filter_design = 'butter'
        self.nyquist = self.sample_frequency * 0.5
        self.overlap = 0

        self.setup_dtype_device_grad(args)
        self.dtype = torch.float32

        self.input = self.add_input("signal", triggers_execution=True)
        self.capture_input = self.add_input('reset', widget_type='button', callback=self.capture)
        self.filter_type_property = self.add_property('filter type', widget_type='combo', default_value=self.filter_type, callback=self.params_changed)
        self.filter_type_property.widget.combo_items = ['bandpass', 'lowpass', 'highpass', 'bandstop']
        self.filter_design_property = self.add_property('filter design', widget_type='combo', default_value=self.filter_design, callback=self.params_changed)
        self.filter_design_property.widget.combo_items = ['butter', 'cheby1', 'cheby2']
        self.order_property = self.add_property('order', widget_type='input_int', default_value=self.order, min=1, max=8, callback=self.params_changed)

        #  these should be algorithmically generated as octaves
        # so we specify low, high, num bands, and it generates band specifications for each band
        self.low_cut_property = self.add_property('low', widget_type='drag_float', default_value=low_cut, callback=self.params_changed)
        self.high_cut_property = self.add_property('high', widget_type='drag_float', default_value=high_cut, callback=self.params_changed)
        self.num_bands = self.add_property('number of bands', widget_type='input_int', default_value=num_bands, callback=self.params_changed)
        self.band_scaling = self.add_property('band scaling', widget_type='combo', default_value='log', callback=self.params_changed)
        self.band_scaling.widget.combo_items = ['log', 'linear']
        self.overlap = self.add_property('overlap fraction', widget_type='drag_float', default_value=0.0, callback=self.params_changed)

        self.sample_frequency_property = self.add_property('sample freq', widget_type='drag_float', default_value=self.sample_frequency, callback=self.params_changed)
        self.bands = []
        self.output = self.add_output('filtered')
        self.filter = None
        self.capture = False
        self.create_dtype_device_grad_properties(option=True)
        self.message_handlers['report_bands'] = self.report_bands

    def custom_create(self, from_file):
        self.params_changed()

    def capture(self):
        self.capture = True

    def report_bands(self, command, data):
        for index, band in enumerate(self.bands):
            band_center = float(band[0] + band[1]) / 2
            print(index, float(band[0]), float(band_center), float(band[1]))

    def calc_bands(self):
        self.bands = []
        if self.num_bands() == 1:
            np_bands = np.linspace(self.high_cut_property(), self.low_cut_property(), 2)
        else:
            if self.band_scaling() == 'log':
                np_bands = np.logspace(np.log10(self.high_cut_property()), np.log10(self.low_cut_property()), self.num_bands())
                factor = np_bands[-2] / np_bands[-1]
                band_power = math.log2(factor)
                overlap_factor = pow(2, band_power * self.overlap())
                if self.num_bands() == 1:
                    self.bands = [[np_bands[1] / overlap_factor, np_bands[0] * overlap_factor]]
                else:
                    for i in range(self.num_bands() - 1):
                        self.bands.append([np_bands[i + 1] / overlap_factor, np_bands[i] * overlap_factor])
            else:
                np_bands = np.linspace(self.high_cut_property(), self.low_cut_property(), self.num_bands())
                factor = np_bands[-2] - np_bands[-1]
                overlap_factor = factor * self.overlap()
                if self.num_bands() == 1:
                    self.bands = [[np_bands[1] - overlap_factor, np_bands[0] + overlap_factor]]
                else:
                    for i in range(self.num_bands() - 1):
                        self.bands.append([np_bands[i + 1] - overlap_factor, np_bands[i] + overlap_factor])


    def params_changed(self):
        self.filter = None
        self.low_cut = self.low_cut_property()
        self.high_cut = self.high_cut_property()
        self.calc_bands()
        self.order = self.order_property()
        self.sample_frequency = self.sample_frequency_property()
        self.nyquist = self.sample_frequency * 0.5
        self.filter_type = self.filter_type_property()
        self.filter_design = self.filter_design_property()
        if self.high_cut > self.nyquist:
            self.high_cut = self.nyquist - 1
        if self.low_cut > self.high_cut:
            self.low_cut = self.high_cut * .5
        if self.filter_type in ['bandpass', 'bandstop']:
            self.filter = TorchIIR2Filter(self.order, self.bands, filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency, device=self.device, dtype=self.dtype)
        elif self.filter_type == 'lowpass':
            low_bands = []
            for band in self.bands:
                low_bands.append([band[1]])
            self.filter = TorchIIR2Filter(self.order, low_bands, filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency, device=self.device, dtype=self.dtype)
        elif self.filter_type == 'highpass':
            high_bands = []
            for band in self.bands:
                high_bands.append([band[1]])
            self.filter = TorchIIR2Filter(self.order, high_bands, filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency, device=self.device, dtype=self.dtype)

    def device_changed(self):
        super().device_changed()
        self.params_changed()

    def execute(self):
        signal = self.input()
        if self.filter is not None:
            if self.capture:
                self.filter.capture(signal)
                self.capture = False
            signal_out = self.filter.filter(signal)
            self.output.send(signal_out)


class TorchIIR2Filter:
    def __init__(self, order, bands, filter_type, design='butter', rp=1, rs=1, fs=0, device='cpu', dtype=torch.float32):
        self.designs = ['butter', 'cheby1', 'cheby2']
        self.filter_types_1 = ['lowpass', 'highpass', 'Lowpass', 'Highpass', 'low', 'high']
        self.filter_types_2 = ['bandstop', 'bandpass', 'Bandstop', 'Bandpass']
        self.error_flag = 0
        self.fir_coefficients = None
        self.iir_coefficients = None
        self.bands = bands
        self.order = order
        self.rp = rp
        self.rs = rs
        self.fs = fs
        self.filter_type = filter_type
        self.design = design
        self.device = device
        self.dtype = dtype
        self.coefficients = self.create_coefficients()
        self.buffers = None
        self.n = 0

    def allocate_buffers(self, width):
        self.buffers = torch.zeros([self.coefficients.shape[0], 3, len(self.bands), width], dtype=self.dtype, device=self.device)

    def update_filter(self, order, filter_type, design='butter', rp=1, rs=1, fs=0):
        self.order = order
        self.rp = rp
        self.rs = rs
        self.fs = fs
        self.filter_type = filter_type
        self.design = design
        self.allocate_buffers(self.buffers.shape[3])
        self.coefficients = self.create_coefficients()

    def capture(self, input_):
        input_ = any_to_tensor(input_, device=self.device).flatten()
        if self.buffers is None or self.buffers.shape[3] != input_.shape[0]:
            self.allocate_buffers(input_.shape[0])
        self.buffers[:, :, :] = input_


    def filter(self, input_):
        shape = input_.shape
        input_ = any_to_tensor(input_, device=self.device).flatten()
        if self.buffers is None or self.buffers.shape[3] != input_.shape[0]:
            self.allocate_buffers(input_.shape[0])

        # input should be shaped [width]
        if len(self.coefficients[0, :]) > 1:
            output = input_.unsqueeze(0)  # shape = [1, width]
            n_base = self.n

            for order in range(self.coefficients.shape[0]):  # [order, 3, bands, 1]
                n0 = n_base
                n1 = (n0 + 1) % 3
                n2 = (n1 + 1) % 3
                self.fir_coefficients = self.coefficients[order][0:3]     # shape = [3, bands, 1]
                self.iir_coefficients = self.coefficients[order][3:6] * -1          # shape = [3, bands, 1]

                # Calculating the accumulated input consisting of the input and the values coming from
                # the feedback loops (delay buffers weighed by the IIR coefficients).

                self.buffers[order, n0] = (
                        output + self.buffers[order, n1] * self.iir_coefficients[1] + self.buffers[order, n2] * self.iir_coefficients[2])

                #  shape[1, 1, num_bands, width] =
                #       shape[1, width]
                #       + shape[num_bands, width] * shape[num_bands, 1]
                #       + shape[num_bands, width] * shape[num_bands, 1]

                output = (
                        self.buffers[order, n0] * self.fir_coefficients[0] + self.buffers[order, n1] * self.fir_coefficients[1] +
                        self.buffers[order, n2] * self.fir_coefficients[2])

                #  shape[1, 1, num_bands, width] =
                #       shape[num_bands, width] * shape[num_bands, 1]
                #       + shape[num_bands, width] * shape[num_bands, 1]
                #       + shape[num_bands, width] * shape[num_bands, 1]

                # Shifting the values on the delay line: acc_input -> buffer1 -> buffer2
            self.n = (self.n - 1) % 3
        return output.transpose(1, 0).view([shape[0], shape[1], -1])   #  shape[num_bands, width]

    def create_coefficients(self):
        # Error handling: other errors can arise too, but those are dealt with in the signal package.

        self.error_flag = 1  # if there was no error then it will be set to 0

        if self.design not in self.designs:
            print('Gave wrong filter design! Remember: butter, cheby1, cheby2.')
        elif self.filter_type not in self.filter_types_1 and self.filter_type not in self.filter_types_2:
            print('Gave wrong filter type! Remember: lowpass, highpass, bandpass, bandstop.')
        elif self.fs < 0:
            print('The sampling frequency has to be positive!')
        else:
            self.error_flag = 0
        coefficient_set = None
        coefficients = None
        #  we want shape of [order, 6, num_bands, 1]
        for index, band in enumerate(self.bands):
            # if fs was given then the given cutoffs need to be normalised to Nyquist
            if self.fs and self.error_flag == 0:
                for i in range(len(band)):
                    band[i] = band[i] / self.fs * 2

            if self.design == 'butter' and self.error_flag == 0:
                coefficients = torch.from_numpy(signal.butter(self.order, band, self.filter_type, output='sos')).to(device=self.device, dtype=self.dtype)
            elif self.design == 'cheby1' and self.error_flag == 0:
                coefficients = torch.from_numpy(signal.cheby1(self.order, self.rp, band, self.filter_type, output='sos')).to(device=self.device, dtype=self.dtype)
            elif self.design == 'cheby2' and self.error_flag == 0:
                coefficients = torch.from_numpy(signal.cheby2(self.order, self.rs, band, self.filter_type, output='sos')).to(device=self.device, dtype=self.dtype)
            #  coefficients shape is [order, 6]
            if coefficients is not None:
                coefficients = coefficients.unsqueeze(-1)        # [order, 6, 1]

                if coefficient_set is None:
                    coefficient_set = coefficients              # [order, 6, 1]

                else:
                    coefficient_set = torch.cat([coefficient_set, coefficients], dim=2)  # [order, 6, n]

        coefficient_set = coefficient_set.unsqueeze(-1)  # [order, 6, num_bands, 1]]
        return coefficient_set


# gemini test

class TorchParallelSavGolFilter:
    """
    Applies a Savitzky-Golay filter to a batch of parallel, independent streams.

    This class is designed for scenarios where an input tensor of shape
    [num_streams, num_components] represents a single time-step for multiple
    independent data streams (e.g., all joint rotations in a skeleton).
    """

    def __init__(self, window_length: int, polyorder: int, num_streams: int, num_components: int,
                 normalize_output: bool = False, device='cpu', dtype=torch.float32):
        self.window_length = window_length
        self.polyorder = polyorder
        self.num_streams = num_streams
        self.num_components = num_components
        self.normalize_output = normalize_output
        self.device = device
        self.dtype = dtype

        self.fir_coefficients = self.create_coefficients()

        # The buffer now holds a history for each parallel stream
        # Shape: [num_streams, window_length, num_components]
        self.buffer = None
        self.ptr = 0
        self.is_warmed_up = False
        self.allocate_buffers(self.num_streams, self.num_components)

    def allocate_buffers(self, num_streams, num_components):
        """Allocates or re-allocates the internal buffer for all streams."""
        self.num_streams = num_streams
        self.num_components = num_components
        self.buffer = torch.zeros(
            (self.num_streams, self.window_length, self.num_components),
            dtype=self.dtype,
            device=self.device
        )
        self.ptr = 0
        self.is_warmed_up = False
        print(f"Allocated Parallel SavGol buffer for {num_streams} streams of {num_components} components.")

    def filter(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Filters a batch of data representing one time-step for all parallel streams.

        Args:
            input_batch (torch.Tensor): A 2D tensor of shape [num_streams, num_components].

        Returns:
            torch.Tensor: The filtered output batch of shape [num_streams, num_components].
        """
        if self.buffer is None or self.buffer.shape[0] != input_batch.shape[0] or self.buffer.shape[2] != \
                input_batch.shape[1]:
            self.allocate_buffers(input_batch.shape[0], input_batch.shape[1])

        # --- Update circular buffer for all streams at once ---
        # input_batch shape: [num_streams, num_components]
        # We need to insert it into the buffer at the current pointer index.
        # buffer shape: [num_streams, window_length, num_components]
        self.buffer[:, self.ptr, :] = input_batch
        self.ptr = (self.ptr + 1) % self.window_length

        # --- Handle warm-up period ---
        if not self.is_warmed_up:
            if self.ptr == 1:
                # Prime the buffer with the first frame
                self.buffer = self.buffer + input_batch.unsqueeze(1)
            if self.ptr == 0:
                self.is_warmed_up = True

        # --- Apply FIR filter in parallel ---
        # Roll the buffer to get a chronologically-ordered view for each stream
        # Shape remains [num_streams, window_length, num_components]
        ordered_buffer = torch.roll(self.buffer, shifts=-self.ptr, dims=1)

        # Batched dot product using einsum. This is the key to parallelism.
        # 'w,swc->sc': For each stream 's', sum over 'w' (window) dimension.
        # coeffs ('w') and buffer ('swc'), result is stream-channel ('sc').
        output = torch.einsum('w,swc->sc', self.fir_coefficients, ordered_buffer)

        # --- Optional: Parallel normalization ---
        if self.normalize_output:
            # Calculate norms for each stream vector. keepdim=True makes it [num_streams, 1]
            norms = torch.linalg.norm(output, dim=1, keepdim=True)
            # Avoid division by zero
            norms[norms < 1e-9] = 1.0
            output = output / norms

        return output

    def create_coefficients(self):
        coeffs = scipy.signal.savgol_coeffs(self.window_length, self.polyorder, deriv=0)
        return torch.tensor(coeffs[::-1].copy(), dtype=self.dtype, device=self.device)

    def reset(self):
        if self.buffer is not None: self.buffer.zero_()
        self.ptr = 0
        self.is_warmed_up = False

# --- The New Node Class ---

class TorchSavGolFilterNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSavGolFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.window_length = 31
        self.polyorder = 3
        self.normalize_output = True  # Defaulting to True as quaternions are the primary use case
        self.setup_dtype_device_grad(args)
        self.dtype = torch.float32
        self.input_port = self.add_input("signal", triggers_execution=True)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)
        self.window_length_property = self.add_property('window length', widget_type='input_int',
                                                        default_value=self.window_length, min=3, max=1001,
                                                        callback=self.params_changed)
        self.polyorder_property = self.add_property('polynomial order', widget_type='input_int',
                                                    default_value=self.polyorder, min=1, max=8,
                                                    callback=self.params_changed)
        self.normalize_property = self.add_property('normalize output', widget_type='checkbox',
                                                    default_value=self.normalize_output, callback=self.params_changed)
        self.output_port = self.add_output('filtered')
        self.filter = None
        self.create_dtype_device_grad_properties(option=True)

    def custom_create(self, from_file):
        self.params_changed()

    def reset_filter(self):
        if self.filter:
            print(f"Node '{self.label}': Resetting Parallel SavGol filter state.")
            self.filter.reset()

    def params_changed(self):
        self.filter = None
        self.window_length = self.window_length_property()
        self.polyorder = self.polyorder_property()
        self.normalize_output = self.normalize_property()
        if self.window_length % 2 == 0: self.window_length += 1
        if self.polyorder >= self.window_length: self.polyorder = self.window_length - 1
        print(f"Node '{self.label}': Parameters changed. Filter will be recreated on next execution.")

    def device_changed(self):
        super().device_changed()
        self.params_changed()

    def execute(self):
        """
        The main execution logic. Handles an input tensor where the first
        dimension is the number of parallel streams to be filtered.
        """
        signal_in = self.input_port()
        if signal_in is None:
            return

        if not isinstance(signal_in, torch.Tensor):
            signal_in = torch.tensor(signal_in, dtype=self.dtype, device=self.device)

        # We now expect a 2D tensor [num_streams, num_components]
        if signal_in.dim() != 2:
            print(
                f"Node '{self.label}': Error: Input must be a 2D tensor of shape [num_streams, num_components]. Got {signal_in.dim()} dimensions.")
            return

        num_streams, num_components = signal_in.shape

        # --- Lazy Instantiation or Resizing of the Filter ---
        if self.filter is None or self.filter.num_streams != num_streams or self.filter.num_components != num_components:
            print(
                f"Node '{self.label}': Instantiating/resizing Parallel SavGol filter for ({num_streams}, {num_components}).")
            try:
                # Use the new parallel filter class
                self.filter = TorchParallelSavGolFilter(
                    window_length=self.window_length,
                    polyorder=self.polyorder,
                    num_streams=num_streams,
                    num_components=num_components,
                    normalize_output=self.normalize_output,
                    device=self.device,
                    dtype=self.dtype
                )
            except ValueError as e:
                print(f"Error creating filter: {e}")
                self.filter = None
                return

        # --- Run the parallel filter ---
        # The filter class now handles the parallel logic internally.
        signal_out = self.filter.filter(signal_in)

        self.output_port.send(signal_out)


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


# --- Helper Functions (JIT Compiled) ---

@torch.jit.script
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)

@torch.jit.script
def q_conjugate(q):
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)

@torch.jit.script
def angular_velocity_to_delta_q(av, dt: float):
    angle = torch.linalg.norm(av, dim=-1) * dt
    # Use a small epsilon to prevent division by zero for zero angular velocity
    axis = F.normalize(av, p=2.0, dim=-1, eps=1e-9)
    angle_half = angle / 2.0
    w = torch.cos(angle_half)
    sin_angle_half = torch.sin(angle_half)
    xyz = axis * sin_angle_half.unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)

@torch.jit.script
def skew_symmetric(v):
    """
    Creates a skew-symmetric matrix from a 3-element vector.
    Used for cross-product operations in matrix form.
    v is a batch of vectors: [num_streams, 3]
    """
    z = torch.zeros_like(v[:, 0])
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    row1 = torch.stack([z, -vz, vy], dim=-1)
    row2 = torch.stack([vz, z, -vx], dim=-1)
    row3 = torch.stack([-vy, vx, z], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)

# --- JIT Kernels for EKF Steps ---

@torch.jit.script
def ekf_predict_kernel(q, av, P, Q, F_base, dt: float):
    # 1. Propagate Nominal State
    delta_q = angular_velocity_to_delta_q(av, dt)
    q_new = F.normalize(q_mult(q, delta_q), p=2.0, dim=-1)

    # 2. Propagate Error Covariance
    # F = [[I - dt*skew(av), dt*I], [0, I]]
    F_mat = F_base.clone()
    av_skew = skew_symmetric(av)
    
    # Top-Left: Rotation of error frame
    # F_mat[:, 0:3, 0:3] -= dt * av_skew # Add this for correctness
    F_mat[:, 0:3, 0:3] = F_base[:, 0:3, 0:3] - dt * av_skew

    # Top-Right: Velocity integration
    # F_mat[:, 0:3, 3:6] = I * dt
    # F_base is Identity. So F_base[0:3, 3:6] is 0.
    # We need to set it to Identity * dt.
    eye_dt = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0) * dt
    F_mat[:, 0:3, 3:6] = eye_dt

    # P = F P F^T + Q
    P_new = torch.bmm(F_mat, torch.bmm(P, F_mat.transpose(1, 2))) + Q
    
    return q_new, P_new

@torch.jit.script
def ekf_update_kernel(q, av, x, P, R, I_x, H, jitter, z_measured_q):
    # 1. Measurement Residual
    residual_q = q_mult(q_conjugate(q), z_measured_q)
    
    # 2. Convert to rotation vector
    angle_half = torch.acos(torch.clamp(residual_q[:, 0], -1.0, 1.0))
    sin_angle_half = torch.sin(angle_half)
    
    # JIT supports torch.where
    # Ensure constants like 2.0 match dtype/device layout if possible, or usually safe in script
    # JIT often handles scalar expansion well.
    scale = torch.where(sin_angle_half.abs() > 1e-9, 
                        (2.0 * angle_half) / sin_angle_half, 
                        torch.full_like(angle_half, 2.0))
    
    residual_v = scale.unsqueeze(-1) * residual_q[:, 1:]

    # 3. Kalman Gain
    P_xz = P[:, :, :3] # P @ H^T
    P_zz = P[:, :3, :3] # H @ P @ H^T
    S = P_zz + R
    
    S_inv = torch.linalg.inv(S + jitter)
    K = P_xz @ S_inv
    
    # 4. Update State
    x_new = torch.einsum('scd,sd->sc', K, residual_v)
    
    # 5. Update Covariance
    I_minus_KH = I_x - K @ H
    
    term1 = I_minus_KH @ P @ I_minus_KH.transpose(1, 2)
    term2 = K @ R @ K.transpose(1, 2)
    P_new = term1 + term2
    P_new = (P_new + P_new.transpose(1, 2)) * 0.5

    # 6. Injection
    error_rot_vec = x_new[:, :3]
    delta_q_correction = angular_velocity_to_delta_q(error_rot_vec, 1.0)
    q_final = F.normalize(q_mult(q, delta_q_correction), p=2.0, dim=-1)
    av_final = av + x_new[:, 3:]
    x_final = torch.zeros_like(x_new) # Reset error

    return q_final, av_final, x_final, P_new

@torch.jit.script
def linear_predict_kernel(pos, vel, P, Q, F_base, dt: float):
    # F = [[I, dt*I], [0, I]]
    # State transition: pos = pos + vel*dt, vel=vel
    pos_new = pos + vel * dt
    vel_new = vel.clone()

    # Propagate Covariance
    F_mat = F_base.clone()
    dim = int(pos.shape[1])
    
    # Update top-right block of F with dt
    # F is [S, 2D, 2D]
    for i in range(dim):
        F_mat[:, i, i + dim] = dt

    # P = F P F^T + Q
    P_new = torch.bmm(F_mat, torch.bmm(P, F_mat.transpose(1, 2))) + Q
    
    return pos_new, vel_new, P_new

@torch.jit.script
def linear_update_kernel(pos, vel, P, R, I_x, H, jitter, z_measured):
    # Measurement Residual: y = z - pos
    residual = z_measured - pos
    
    dim = int(pos.shape[1])
    
    # Kalman Gain K = P H^T (H P H^T + R)^-1
    # H = [I, 0]
    # P H^T = P[:, :, :dim]  (First dim columns)
    # H P H^T = P[:, :dim, :dim] (Top-left block)
    
    P_xz = P[:, :, :dim]
    P_zz = P[:, :dim, :dim]
    S = P_zz + R
    
    S_inv = torch.linalg.inv(S + jitter)
    K = torch.bmm(P_xz, S_inv)
    
    # Update State: x = x + K y
    # residual is [S, D], need [S, D, 1] for matmul, or just einsum
    # K is [S, 2D, D]
    
    dx = torch.einsum('bij,bj->bi', K, residual)
    
    pos_new = pos + dx[:, :dim]
    vel_new = vel + dx[:, dim:]
    
    # Update Covariance: P = (I - KH) P (I - KH)^T + K R K^T (Joseph form)
    I_minus_KH = I_x - torch.bmm(K, H)
    
    term1 = torch.bmm(I_minus_KH, torch.bmm(P, I_minus_KH.transpose(1, 2)))
    term2 = torch.bmm(K, torch.bmm(R, K.transpose(1, 2)))
    
    P_new = term1 + term2
    P_new = (P_new + P_new.transpose(1, 2)) * 0.5
    
    return pos_new, vel_new, P_new




class TorchEuclideanKF:
    """
    Generic Linear Kalman Filter for N streams of D dimensions.
    State is [Position, Velocity] (size 2*D).
    """
    def __init__(self, dt, num_streams, dim, device='cpu', dtype=torch.float32):
        self.dt = dt
        self.num_streams = num_streams
        self.dim = dim
        self.state_dim = 2 * dim
        self.device = device
        self.dtype = dtype
        
        # State
        self.pos = torch.zeros(num_streams, dim, device=device, dtype=dtype)
        self.vel = torch.zeros(num_streams, dim, device=device, dtype=dtype)
        
        # Covariance
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
        
        # Noise Matrices
        self.Q = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.01
        self.R = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
        
        # Constant Matrices matching JIT signature
        self.I_x = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0)
        self.F_base = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)
        self.H = torch.zeros(num_streams, dim, self.state_dim, device=device, dtype=dtype)
        self.H[:, :dim, :dim] = torch.eye(dim, device=device, dtype=dtype)
        
        self.jitter = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0) * 1e-9
        
        self._reset_pending = False

    def update_device_to_match(self, input_tensor):
        if input_tensor.device != self.device:
            self.device = input_tensor.device
            for attr in ['pos', 'vel', 'P', 'Q', 'R', 'I_x', 'F_base', 'H', 'jitter']:
                 setattr(self, attr, getattr(self, attr).to(self.device))
                 
    def set_noise_params(self, meas_noise_vec, vel_change_noise_vec, dt):
        # meas_noise_vec: [S] or [S, D] - std dev of position measurement
        # vel_change_noise_vec: [S] or [S, D] - std dev of velocity change
        self.dt = dt
        
        # Expand scalar noise to all dims if needed
        if meas_noise_vec.dim() == 1:
            meas_noise_vec = meas_noise_vec.unsqueeze(1).expand(-1, self.dim)
        if vel_change_noise_vec.dim() == 1:
            vel_change_noise_vec = vel_change_noise_vec.unsqueeze(1).expand(-1, self.dim)
            
        r_vals = meas_noise_vec ** 2
        
        # Flatten R for diag_embed if D > 1
        # R is [S, D, D] diagonal
        self.R = torch.diag_embed(r_vals)
        
        # Q is [S, 2D, 2D] diagonal
        # Process noise:
        # Pos variance ~ 1/3 * a_var * dt^3 ?? Or just simplified discrete model?
        # Piecewise White Noise Acceleration Model (Discrete Wiener Process Acceleration)
        # Q = [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * sigma_a^2 usually
        # But here we used simplified:
        # Rot part (pos) = (vel_noise * dt)^2
        # Vel part = vel_noise^2
        
        vel_sq = vel_change_noise_vec ** 2
        pos_part = (vel_sq * (self.dt ** 2)) 
        
        # Cat dims?
        # Vector of diagonals [S, 2D]
        Q_diag = torch.cat([pos_part, vel_sq], dim=1)
        self.Q = torch.diag_embed(Q_diag)
        
    def predict(self):
        if self._reset_pending: return
        self.pos, self.vel, self.P = linear_predict_kernel(
            self.pos, self.vel, self.P, self.Q, self.F_base, self.dt
        )
        
    def update(self, z):
        if self._reset_pending:
            self.pos = z.to(self.device, self.dtype)
            self.vel.zero_()
            self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)
            self._reset_pending = False
            return
            
        self.pos, self.vel, self.P = linear_update_kernel(
            self.pos, self.vel, self.P, self.R, self.I_x, self.H, self.jitter, z
        )
        
    def flag_reset(self):
        self._reset_pending = True
        
class TorchTristateBlendEuclideanNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchTristateBlendEuclideanNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.filter = None
        self.filter = None
        self.last_input_vec = None
        self.last_vel = None
        self.last_accel = None

        
        # [damp, resp, err]
        self.blended_alphas = None

        self.motion_min = 1.0
        self.motion_max = 5.0
        self.error_min = 0.5
        self.error_max = 2.0
        
        # Default Params
        self.damp_params = [0.1, 0.01]
        self.resp_params = [0.01, 1.0]
        self.err_params = [0.001, 10.0]
        
        self.damp_tensor = None
        self.resp_tensor = None
        self.err_tensor = None

        self.input = self.add_input("input", triggers_execution=True)
        self.dt_input = self.add_input("dt (sec)", widget_type='drag_float', default_value=1.0 / 60.0, callback=self.update_params)
        self.blending_speed_in = self.add_input('Blending Speed', widget_type='drag_float', default_value=0.1, min=0.01, max=1.0)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

        self.add_property("--- Acceleration Rejection ---", widget_type='label')
        self.accel_filter_in = self.add_input('Accel Rejection Factor', widget_type='drag_float', default_value=0.0, min=0.0, max=100.0)


        self.add_property("--- Damping Mode ---", widget_type='label')
        self.damp_meas_in = self.add_input('Damp Meas Noise', widget_type='drag_float', default_value=self.damp_params[0], callback=self.update_params)
        self.damp_vel_in = self.add_input('Damp Vel Noise', widget_type='drag_float', default_value=self.damp_params[1], callback=self.update_params)

        self.add_property("--- Responsive Mode ---", widget_type='label')
        self.resp_meas_in = self.add_input('Resp Meas Noise', widget_type='drag_float', default_value=self.resp_params[0], callback=self.update_params)
        self.resp_vel_in = self.add_input('Resp Vel Noise', widget_type='drag_float', default_value=self.resp_params[1], callback=self.update_params)

        self.add_property("--- Error Correction Mode ---", widget_type='label')
        self.err_meas_in = self.add_input('Err Correct Meas Noise', widget_type='drag_float', default_value=self.err_params[0], callback=self.update_params)
        self.err_vel_in = self.add_input('Err Correct Vel Noise', widget_type='drag_float', default_value=self.err_params[1], callback=self.update_params)

        self.add_property("--- Transition Ranges ---", widget_type='label')
        self.motion_min_in = self.add_input('Min Motion (units/s)', widget_type='drag_float', default_value=self.motion_min, callback=self.update_params)
        self.motion_max_in = self.add_input('Max Motion (units/s)', widget_type='drag_float', default_value=self.motion_max, callback=self.update_params)
        self.error_min_in = self.add_input('Min Error (units)', widget_type='drag_float', default_value=self.error_min, callback=self.update_params)
        self.error_max_in = self.add_input('Max Error (units)', widget_type='drag_float', default_value=self.error_max, callback=self.update_params)

        self.output_port = self.add_output('filtered')
        self.alphas_output = self.add_output('alphas')

    def custom_create(self, from_file):
        self.update_params()

    def reset_filter(self):
        if self.filter: self.filter.flag_reset()
        self.last_input_vec = None
        self.last_vel = None
        self.last_accel = None

        if self.blended_alphas is not None:
            self.blended_alphas.zero_()
            self.blended_alphas[:, 0] = 1.0

    def device_changed(self):
        super().device_changed()
        self.update_params()
        
    def update_params(self):
        self.damp_params = [self.damp_meas_in(), self.damp_vel_in()]
        self.resp_params = [self.resp_meas_in(), self.resp_vel_in()]
        self.err_params = [self.err_meas_in(), self.err_vel_in()]
        self.motion_min = self.motion_min_in()
        self.motion_max = self.motion_max_in()
        self.error_min = self.error_min_in()
        self.error_max = self.error_max_in()
        
        self.damp_tensor = torch.tensor(self.damp_params, device=self.device, dtype=self.dtype)
        self.resp_tensor = torch.tensor(self.resp_params, device=self.device, dtype=self.dtype)
        self.err_tensor = torch.tensor(self.err_params, device=self.device, dtype=self.dtype)

    def execute(self):
        signal_in = any_to_tensor(self.input())
        if signal_in is None: return

        # Flatten input to [S, D] if needed, or handle [B, ..., D]
        # For simplicity, treat dim=0 as streams/batch, and the rest as flat dimension D
        input_shape = signal_in.shape
        if len(input_shape) == 1:
            # Single stream, vector
            signal_in = signal_in.unsqueeze(0)
            
        # Treat last dim as D, but flatten all prior dims to S? 
        # Or just S = dim 0
        if len(input_shape) > 2:
             # Flatten [B, T, D] -> [B*T, D] for filtering? 
             # Usually node logic assumes [Batch, Features]
             # Let's enforce 2D for now: [Stream, Dim]
             signal_in = signal_in.view(input_shape[0], -1)
             
        num_streams, dim = signal_in.shape
        dt = self.dt_input()
        if dt is None or dt <= 0: return
        
        if signal_in.device != self.device:
            self.device = signal_in.device
            self.update_params()

        if self.filter is None or self.filter.num_streams != num_streams or self.filter.dim != dim:
            self.filter = TorchEuclideanKF(dt, num_streams, dim, self.device, self.dtype)
            self.reset_filter()
            self.blended_alphas = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype).unsqueeze(0).repeat(num_streams, 1)

        self.filter.update_device_to_match(signal_in)
        if self.last_input_vec is not None and self.last_input_vec.device != self.device:
            self.last_input_vec = self.last_input_vec.to(self.device)
        self.blended_alphas = self.blended_alphas.to(self.device)

        self.blended_alphas = self.blended_alphas.to(self.device)


        # 1. Motion Calc
        motion_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_vec is not None:
            dist = torch.linalg.norm(signal_in - self.last_input_vec, dim=-1)
            motion_per_sec = dist / dt

        # 2. Error Calc
        # We need estimated position for error calc, but we haven't run predict yet (to allow Q update).
        # So we manually predict position for the error metric: pos + vel * dt
        predicted_pos_temp = self.filter.pos + self.filter.vel * dt
        error_dist = torch.linalg.norm(signal_in - predicted_pos_temp, dim=-1)

        
        # 3. Blending
        strength_resp = torch.clamp((motion_per_sec - self.motion_min) / (self.motion_max - self.motion_min + 1e-9), 0.0, 1.0)
        strength_err = torch.clamp((error_dist - self.error_min) / (self.error_max - self.error_min + 1e-9), 0.0, 1.0)
        
        alpha_active = torch.maximum(strength_resp, strength_err)
        total_strength = strength_resp + strength_err + 1e-9
        
        target_resp = alpha_active * (strength_resp / total_strength)
        target_err = alpha_active * (strength_err / total_strength)
        target_damp = 1.0 - target_resp - target_err
        
        target_alphas = torch.stack([target_damp, target_resp, target_err], dim=1)
        
        blending_speed = self.blending_speed_in()
        self.blended_alphas.lerp_(target_alphas, blending_speed)
        
        # 4. Set Params
        # self.damp_tensor is [2], blended_alphas [S, 3]
        param_stack = torch.stack([self.damp_tensor, self.resp_tensor, self.err_tensor], dim=0) # [3, 2]
        blended_params = self.blended_alphas @ param_stack # [S, 2]

        # --- Acceleration Rejection Logic ---
        accel_rejection_factor = self.accel_filter_in()
        
        if accel_rejection_factor > 0 and self.filter is not None:
            current_vel = self.filter.vel
            if self.last_vel is not None:
                # Calculate current acceleration (frame k-1)
                current_accel = (current_vel - self.last_vel) / dt
                
                if self.last_accel is not None:
                     # Dot product to detect reversal [S]
                     accel_dot = (current_accel * self.last_accel).sum(dim=1) # [S]
                     
                     reversal_mask = accel_dot < 0
                     if reversal_mask.any():
                         damping = torch.ones_like(accel_dot)
                         damping[reversal_mask] = 1.0 + accel_rejection_factor
                         
                         # Apply: Increase Meas Noise (0), Decrease Vel Noise (1)
                         blended_params[:, 0] = blended_params[:, 0] * damping
                         blended_params[:, 1] = blended_params[:, 1] / damping

                
                self.last_accel = current_accel
            
            self.last_vel = current_vel.clone()
        
        self.filter.set_noise_params(
            meas_noise_vec=blended_params[:, 0],
            vel_change_noise_vec=blended_params[:, 1],
            dt=dt
        )
        
        # NOW we predict, using the updated Q (noise) from the blending/rejection logic
        self.filter.predict()
        
        self.filter.update(signal_in)

        
        # Output reshape
        output = self.filter.pos
        if len(input_shape) > 2:
            output = output.view(input_shape)
            
        self.output_port.send(output)
        self.alphas_output.send(self.blended_alphas)
        self.last_input_vec = signal_in.clone()

# --- REFACTORED AND RENAMED FILTER CLASS ---

class TorchQuaternionESEKF:  # Renamed from UKF to EKF
    """
    A highly optimized, fully vectorized Error-State Extended Kalman Filter (ES-EKF)
    for parallel quaternion streams. This architecture is rotationally invariant.
    """

    def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
        self.dim_x = 6  # State is a 6D error vector: [err_rot_3d, err_av_3d]
        self.dim_z = 3  # Measurement is a 3D rotation error vector
        self.dt = dt
        self.num_streams = num_streams
        self.device = device
        self.dtype = dtype

        # --- Nominal State (our best guess, kept outside the filter) ---
        self.q = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q[:, 0] = 1.0  # w=1, x=y=z=0 identity quaternion
        self.av = torch.zeros(num_streams, 3, device=device, dtype=dtype)  # angular velocity

        # --- Error State (what the EKF actually filters) ---
        self.x = torch.zeros(num_streams, self.dim_x, device=device, dtype=dtype)
        self.P = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1

        # --- Noise matrices ---
        self.Q = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.01
        self.R = torch.eye(self.dim_z, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1

        # --- Pre-allocated Constant Matrices for Optimization ---
        self.I_x = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0) # [1, 6, 6]
        self.H = torch.zeros(num_streams, self.dim_z, self.dim_x, device=device, dtype=dtype) # NOTE: swapped dims for broadcasting? No, H is usually z by x.
        # Original code had H as [num_streams, x, z] but did K @ H.mT which means H was [x, z].
        # Let's verify: K is [x, z]. P_update = ... K @ H^T.
        # Standard EKF: y = z - Hx. H maps state space to measurement space. H is [z, x].
        # The previous code had: H = zeros(..., x, z); H[:, :3, :3] = Eye.
        # I_minus_KH = I - K @ H.transpose(-1, -2) => K [x, z] @ H.T [z, x] -> [x, x]. Correct.
        # So H in previous code was effectively [x, z] layout?
        # WAIT. In previous code:
        # H = torch.zeros(self.num_streams, self.dim_x, self.dim_z, ...) => [S, 6, 3]
        # H[:, :3, :3] = Eye
        # I_minus_KH = I - K @ H.transpose(-1, -2) => K[6,3] @ H^T[3,6] = [6,6]. Yes.
        # So "H" stored was actually H^T (or similar).
        # Standard notation H is [z_dim, x_dim] = [3, 6].
        # Let's store standard H [S, 3, 6] so we don't need to transpose it.
        # H = [I3, 0]
        self.H = torch.zeros(num_streams, self.dim_z, self.dim_x, device=device, dtype=dtype)
        self.H[:, :3, :3] = torch.eye(self.dim_z, device=device, dtype=dtype)
        
        # F matrix base: Identity
        self.F_base = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)
        
        # Jitter for robustness
        self.jitter = torch.eye(self.dim_z, device=device, dtype=dtype).unsqueeze(0) * 1e-9

        self._reset_pending = False

    def update_device_to_match(self, input_tensor):
        if input_tensor.device != self.device:
            self.device = input_tensor.device
            # Move all tensor attributes
            for attr in ['q', 'av', 'x', 'P', 'Q', 'R', 'I_x', 'H', 'F_base', 'jitter']:
                setattr(self, attr, getattr(self, attr).to(self.device))

    def set_noise_params(self, meas_noise_vec, vel_change_noise_vec, dt):
        """
        Sets the noise covariance matrices R and Q based on per-stream noise parameters.
        optimized to avoid full matrix reconstruction.
        """
        self.dt = dt

        # R is diagonal [S, 3, 3]
        # meas_noise_vec is [S]
        # We can just update the diagonal.
        r_vals = meas_noise_vec ** 2
        
        # Optimized: exploiting the fact that we know R is diagonal
        # This is strictly faster than creating diag_embed every frame
        # However, advanced indexing can be just as slow.
        # But diag_embed is creating a new tensor of zeros every time.
        # Let's try to update in place if strict structure allows, else just optimized creation.
        # Constructing R from diagonals is fine, but let's avoid intermediate expands if possible.
        
        # R diagonals [S, 3] = r_vals [S, 1] expand
        # self.R.diagonal(dim1=-2, dim2=-1).copy_(...) <- this might be the fastest if supported in batch
        # Batch diagonal view is tricky in older torch, but we can assume modern torch.
        
        # Fallback to diag_embed but faster inputs
        R_diag = r_vals.unsqueeze(1).expand(-1, 3) 
        self.R = torch.diag_embed(R_diag)

        # Q is diagonal [S, 6, 6]
        vel_sq = vel_change_noise_vec ** 2
        rot_part = (vel_sq * (self.dt ** 2)) # Approx (vel * dt)^2
        vel_part = vel_sq
        
        Q_diag = torch.cat([rot_part.unsqueeze(1).expand(-1, 3), vel_part.unsqueeze(1).expand(-1, 3)], dim=1)
        self.Q = torch.diag_embed(Q_diag)

    def predict(self):
        if self._reset_pending:
            return
            
        # Call JIT kernel
        # We need to ensure types match for JIT
        self.q, self.P = ekf_predict_kernel(
            self.q, self.av, self.P, self.Q, self.F_base, self.dt
        )

    def update(self, z_measured_q):
        if self._reset_pending:
            self.q = z_measured_q.to(self.device, self.dtype)
            self.av.zero_()
            self.x.zero_()
            # Reset P to Identity * 0.1
            self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)
            self._reset_pending = False
            return
            
        # Call JIT kernel
        self.q, self.av, self.x, self.P = ekf_update_kernel(
            self.q, self.av, self.x, self.P, self.R, 
            self.I_x, self.H, self.jitter, z_measured_q
        )

    def reset(self):
        self.q.zero_()
        self.q[:, 0] = 1.0
        self.av.zero_()
        self.x.zero_()
        self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)

    def flag_reset(self):
        self._reset_pending = True

    def reset_to_defaults(self):
        self.reset()
        self._reset_pending = False


# --- CORRECTED NODE CLASS ---

class TorchTristateBlendESEKFNode(TorchDeviceDtypeNode):  # Renamed
    # ... (Assuming TorchDeviceDtypeNode and other setup methods are defined elsewhere)
    @staticmethod
    def factory(name, data, args=None):
        return TorchTristateBlendESEKFNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.filter = None
        self.last_input_quat = None
        self.last_av = None
        self.last_accel = None

        # --- NEW STATE FOR SMOOTH BLENDING ---
        # Stored as [damp, resp, err] for each stream
        self.blended_alphas = None

        self.motion_min = 15.0
        self.motion_max = 90.0
        self.error_min = 5.0
        self.error_max = 45.0

        # Parameters are [Measurement Noise StdDev (radians), AngVel Change Noise StdDev (rad/s)]
        # We store them as standard python lists for the UI access, 
        # BUT we also maintain pre-allocated tensors for the fast path.
        self.damp_params = [0.5, 0.1]
        self.resp_params = [0.05, 1.0]
        self.err_params = [0.005, 10.0]
        
        self.damp_tensor = None
        self.resp_tensor = None
        self.err_tensor = None

        # --- Inputs ---
        self.quat_input = self.add_input("quaternions", triggers_execution=True)
        self.dt_input = self.add_input("dt (sec)", widget_type='drag_float', default_value=1.0 / 60.0,
                                       callback=self.update_params)
        # --- NEW BLENDING SPEED INPUT ---
        self.blending_speed_in = self.add_input('Blending Speed', widget_type='drag_float', default_value=0.1,
                                                min=0.01, max=1.0)
        self.blending_speed_in = self.add_input('Blending Speed', widget_type='drag_float', default_value=0.1,
                                                min=0.01, max=1.0)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

        self.add_property("--- Acceleration Rejection ---", widget_type='label')
        self.accel_filter_in = self.add_input('Accel Rejection Factor', widget_type='drag_float', default_value=0.0, min=0.0, max=100.0)


        self.add_property("--- Damping Mode ---", widget_type='label')
        self.damp_meas_in = self.add_input('Damp Meas Noise', widget_type='drag_float',
                                           default_value=self.damp_params[0], callback=self.update_params)
        self.damp_vel_in = self.add_input('Damp AngVel Noise', widget_type='drag_float',
                                          default_value=self.damp_params[1], callback=self.update_params)

        self.add_property("--- Responsive Mode ---", widget_type='label')
        self.resp_meas_in = self.add_input('Resp Meas Noise', widget_type='drag_float',
                                           default_value=self.resp_params[0], callback=self.update_params)
        self.resp_vel_in = self.add_input('Resp AngVel Noise', widget_type='drag_float',
                                          default_value=self.resp_params[1], callback=self.update_params)

        self.add_property("--- Error Correction Mode ---", widget_type='label')
        self.err_meas_in = self.add_input('Err Correct Meas Noise', widget_type='drag_float',
                                          default_value=self.err_params[0], callback=self.update_params)
        self.err_vel_in = self.add_input('Err Correct AngVel Noise', widget_type='drag_float',
                                         default_value=self.err_params[1], callback=self.update_params)

        self.add_property("--- Transition Ranges ---", widget_type='label')
        self.motion_min_in = self.add_input('Min Motion (deg/sec)', widget_type='drag_float',
                                            default_value=self.motion_min, callback=self.update_params)
        self.motion_max_in = self.add_input('Max Motion (deg/sec)', widget_type='drag_float',
                                            default_value=self.motion_max, callback=self.update_params)
        self.error_min_in = self.add_input('Min Error (deg)', widget_type='drag_float', default_value=self.error_min,
                                           callback=self.update_params)
        self.error_max_in = self.add_input('Max Error (deg)', widget_type='drag_float', default_value=self.error_max,
                                           callback=self.update_params)

        # Outputs
        self.output_port = self.add_output('filtered')
        self.alphas_output = self.add_output('alphas (damp, resp, err)')
        
    def custom_create(self, from_file):
        # Initialize tensors after widgets are ready
        self.update_params()

    def reset_filter(self):
        # Safety check if the filter hasn't been created yet
        if self.filter is None:
            return

        # 1. Command the filter to perform a reset on its next update.
        self.filter.flag_reset()

        # 2. Reset the node's own state variables to their initial conditions.
        # This prevents calculating a huge motion spike on the next frame.
        self.last_input_quat = None
        self.last_av = None
        self.last_accel = None


        # 3. Reset the blended alphas to the default "Damping" state.
        # This is now the ONLY place this reset happens, ensuring consistency.
        if self.blended_alphas is not None:
            self.blended_alphas.zero_()
            self.blended_alphas[:, 0] = 1.0
            
    def device_changed(self):
        super().device_changed()
        # Re-create tensors on new device
        self.update_params()

    def update_params(self):
        self.damp_params = [self.damp_meas_in(), self.damp_vel_in()]
        self.resp_params = [self.resp_meas_in(), self.resp_vel_in()]
        self.err_params = [self.err_meas_in(), self.err_vel_in()]
        self.motion_min = self.motion_min_in()
        self.motion_max = self.motion_max_in()
        self.error_min = self.error_min_in()
        self.error_max = self.error_max_in()
        
        # Pre-allocate tensors for execute loop
        self.damp_tensor = torch.tensor(self.damp_params, device=self.device, dtype=self.dtype)
        self.resp_tensor = torch.tensor(self.resp_params, device=self.device, dtype=self.dtype)
        self.err_tensor = torch.tensor(self.err_params, device=self.device, dtype=self.dtype)

    def _calculate_angular_difference_deg(self, q1, q2):
        # Optimized: assume q1, q2 normalized.
        # dot product
        dot = (q1 * q2).sum(dim=-1).abs_()
        angle_rad = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
        return torch.rad2deg(angle_rad)

    def execute(self):
        signal_in = any_to_tensor(self.quat_input())
        if signal_in is None or signal_in.dim() != 2 or signal_in.shape[1] != 4:
            return

        dt = self.dt_input()
        if dt is None or dt <= 0: return

        # Optimize: don't check shape[0] vs num_streams every single time if filter exists? 
        # Actually filter check is fast.
        num_streams = signal_in.shape[0]
        
        # Device check is handled by base class usually calling device_changed, 
        # but if input comes from elsewhere we must check.
        if signal_in.device != self.device:
            self.device = signal_in.device
            self.update_params() # Move param tensors to new device

        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchQuaternionESEKF(dt, num_streams, self.device, self.dtype)
            self.reset_filter()
            self.blended_alphas = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype).unsqueeze(0).repeat(num_streams, 1)

        self.filter.update_device_to_match(signal_in)
        if self.last_input_quat is not None and self.last_input_quat.device != self.device:
             self.last_input_quat = self.last_input_quat.to(self.device)
        if self.blended_alphas.device != self.device:
             self.blended_alphas = self.blended_alphas.to(self.device)

        # Predict
        if self.blended_alphas.device != self.device:
             self.blended_alphas = self.blended_alphas.to(self.device)

        # Predict MOVED DOWN


        # 1. Calculate the raw strength for each active mode.
        motion_deg_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_quat is not None:
            motion_deg_per_frame = self._calculate_angular_difference_deg(signal_in, self.last_input_quat)
            motion_deg_per_sec = motion_deg_per_frame / dt
            
        strength_resp_vec = torch.clamp(
            (motion_deg_per_sec - self.motion_min) / (self.motion_max - self.motion_min + 1e-9), 0.0, 1.0)

            
        # We need estimated quaternion for error calc.
        # manually predict q: q + delta_q(av)
        delta_q = angular_velocity_to_delta_q(self.filter.av, dt)
        predicted_q_temp = q_mult(self.filter.q, delta_q)
        # Normalize? Not strictly needed for difference calc but safe
        # predicted_q_temp = F.normalize(predicted_q_temp, dim=-1)

        error_deg = self._calculate_angular_difference_deg(signal_in, predicted_q_temp)

        strength_err_vec = torch.clamp((error_deg - self.error_min) / (self.error_max - self.error_min + 1e-9), 0.0,
                                       1.0)


        # 2. Calculate target alphas
        alpha_active = torch.maximum(strength_resp_vec, strength_err_vec)
        total_strength = strength_resp_vec + strength_err_vec + 1e-9
        
        # Optimized: logical masking or direct math
        target_alpha_resp = alpha_active * (strength_resp_vec / total_strength)
        target_alpha_err = alpha_active * (strength_err_vec / total_strength)
        target_alpha_damp = 1.0 - target_alpha_resp - target_alpha_err
        
        target_alphas = torch.stack([target_alpha_damp, target_alpha_resp, target_alpha_err], dim=1)

        # 3. Smooth the alphas
        blending_speed = self.blending_speed_in()
        self.blended_alphas.lerp_(target_alphas, blending_speed) # In-place lerp

        # 4. Use smoothed alphas to set filter parameters
        # Optimized: Use pre-allocated parameter tensors (damp_tensor, etc).
        # We need to mix them.
        # blended_params = alphas * params
        # [S, 3] * [3, 2] -> [S, 2]? No
        # alphas is [S, 3] (damp, resp, err)
        # params is 3 sets of [2] (meas, vel)
        # We want result [S, 2]
        
        # Stack params: [3, 2]
        # self.damp_tensor: [2]
        param_stack = torch.stack([self.damp_tensor, self.resp_tensor, self.err_tensor], dim=0) # [3, 2]
        
        # blended = alphas @ param_stack
        # [S, 3] @ [3, 2] -> [S, 2]
        blended_params = self.blended_alphas @ param_stack
        
        # --- Acceleration Rejection Logic ---
        # Detect acceleration reversals and reduce velocity process noise (blended_params[:, 1])
        accel_rejection_factor = self.accel_filter_in()
        
        if accel_rejection_factor > 0 and self.filter is not None:
            current_av = self.filter.av
            if self.last_av is not None:
                # Calculate current acceleration (frame k-1)
                # av is a tensor [S, 3]
                current_accel = (current_av - self.last_av) / dt
                
                if self.last_accel is not None:
                     # Dot product to detect reversal [S]
                     # normalized dot product might be better, but let's stick to raw sign first?
                     # actually, we want to know if they oppose.
                     accel_dot = (current_accel * self.last_accel).sum(dim=1) # [S]
                     
                     # Reversal if dot < 0.
                     # We want a damping factor. 
                     # If dot < 0, we want to reduce noise.
                     # Let's say we define a penalty based on how strong the reversal is.
                     
                     # Simple logic: if dot < 0, huge penalty.
                     # Damping = 1 / (1 + factor * |dot_norm|) ?
                     
                     reversal_mask = accel_dot < 0
                     if reversal_mask.any():
                         # Calculate a scalar intensity of reversal?
                         # Or just apply constant damping?
                         # Plan said: "scale down the vel_change_noise_vec"
                         
                         # Let's use the rejection factor as a multiplier for the reduction
                         # modification: new_noise = noise / (1 + factor)
                         
                         # But checking magnitude of reversal is good.
                         # Let's use the magnitude of the acceleration as well?
                         
                         damping = torch.ones_like(accel_dot)
                         damping[reversal_mask] = 1.0 + accel_rejection_factor
                         
                         # Apply: Increase Meas Noise (0), Decrease Vel Noise (1)
                         blended_params[:, 0] = blended_params[:, 0] * damping
                         blended_params[:, 1] = blended_params[:, 1] / damping

                
                self.last_accel = current_accel
            
            self.last_av = current_av.clone()

        self.filter.set_noise_params(
            meas_noise_vec=blended_params[:, 0],
            vel_change_noise_vec=blended_params[:, 1],
            dt=dt
        )

        # Predict NOW (using new Q)
        self.filter.predict()

        # 5. Run Update
        self.filter.update(signal_in)

        
        self.output_port.send(self.filter.q)
        self.alphas_output.send(self.blended_alphas)
        self.last_input_quat = signal_in.clone()


class TorchSmartClampKF:
    """
    A Linear Kalman Filter with 'Smart Innovation Clamping' to reject 
    physically impossible acceleration spikes while tracking plausible fast motion.
    
    This filter limits the MAGNITUDE of the 'correction' (Innovation) that can be applied
    in a single frame. This effectively sets a 'maximum position jump' or 'maximum velocity change'
    limit, preventing 1-frame glitches from teleporting the filter state.
    """
    def __init__(self, dt, num_streams, dim, device='cpu', dtype=torch.float32):
        self.dt = dt
        self.num_streams = num_streams
        self.dim = dim
        self.state_dim = 2 * dim
        self.device = device
        self.dtype = dtype

        # State [Position, Velocity]
        self.x = torch.zeros(num_streams, self.state_dim, device=device, dtype=dtype)
        
        # P: Error Covariance
        # Initialize with reasonable uncertainty
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1

        # Matrices
        # F: State Transition
        # [ I  dt*I ]
        # [ 0  I    ]
        self.F_base = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)
        for i in range(dim):
             self.F_base[:, i, i + dim] = dt

        # H: Measurement Matrix (Measure Position only)
        # [ I  0 ]
        self.H = torch.zeros(num_streams, dim, self.state_dim, device=device, dtype=dtype)
        self.H[:, :dim, :dim] = torch.eye(dim, device=device, dtype=dtype)

        # Q and R will be set dynamically or have defaults
        self.Q = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.01
        self.R = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
        
        self.I_x = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0)
        self.jitter = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0) * 1e-9

        self._reset_pending = False
        self.clamp_radius = 0.1 # Default
        self.accel_rejection_factor = 0.0
        self.last_velocity = None

    def update_params(self, process_noise_q, measurement_noise_r, clamp_radius, accel_rejection_factor, max_accel, dt):
        self.dt = dt
        self.clamp_radius = clamp_radius
        self.accel_rejection_factor = accel_rejection_factor
        
        # --- Accel Rejection Logic ---
        # If enabled, checking for acceleration reversal
        current_vel = self.x[:, self.dim:] # Estimated velocity
        
        scaling_factor = 1.0
        if self.accel_rejection_factor > 0 and self.last_velocity is not None:
            # Calc Accel (Change in Velocity)
            # We don't have last_accel stored explicitly, but we can compare direction of Velocity Change
            # Wait, noise manifests as high frequency velocity changes. 
            # Rejection logic: If acceleration vector reverses (dot product < 0), suppress Process Noise.
            # But we need PREVIOUS acceleration to compare against CURRENT acceleration.
            # Accel_t = Vel_t - Vel_{t-1}
            # Accel_{t-1} = Vel_{t-1} - Vel_{t-2}
            
            # Since we don't store Accel_{t-1}, let's just stick to a simpler heuristic or add state?
            # actually we need to store last_velocity to compute current_accel, AND last_accel to compare.
            pass 
        
        # To keep it efficient inside update_params:
        # We can't do per-frame state logic easily here unless we store it.
        # But update_params is called every frame before predict/update.
        
        # Let's modify process_noise_q based on the state BEFORE rebuilding Q.
        # But we need the previous frame's acceleration.
        
        if self.accel_rejection_factor > 0 and hasattr(self, 'last_accel') and self.last_accel is not None:
             # Calculate current acceleration (estimated)
             # Note: self.x is the updated state from LAS frame (since we haven't predicted yet? No, we are about to predict)
             # self.x is posterior from t-1.
             # self.last_velocity is posterior velocity from t-2?
             
             # Actually, let's just use the logic from TristateBlend:
             # It calculates accel *outside* and passes it in.
             # But here we want the filter to be self-contained.
             
             # Let's perform the check here.
             current_v = self.x[:, self.dim:]
             if self.last_velocity is not None:
                 curr_a = (current_v - self.last_velocity) / dt
                 # Dot with last accel
                 dot = (curr_a * self.last_accel).sum(dim=1, keepdim=True) # (S, 1)
                 
                 # Mask where dot < 0 (Reversal)
                 reversal_mask = (dot < 0).float()
                 
                 # Apply damping
                 # If reversal, divide process noise by (1 + factor)
                 # factor = rejection * 10? Scaled 0-10 input.
                 
                 # New Q multiplier: 1 / (1 + factor * reversal)
                 multiplier = 1.0 / (1.0 + self.accel_rejection_factor * reversal_mask * 100.0) # Boosted strength
                 scaling_factor = multiplier
                 
                 self.last_accel = curr_a # Store for NEXT frame (actually we need to store it after update?)
                 # Wait, if we update last_accel here, we are using (v_{t-1} - v_{t-2}). Correct.
                 
             else:
                 if self.last_velocity is not None:
                    self.last_accel = (current_v - self.last_velocity) / dt

             self.last_velocity = current_v.clone()
        else:
             # First run init
             if not hasattr(self, 'last_accel'):
                 self.last_accel = None
                 self.last_velocity = None
             if self.last_velocity is None:
                 self.last_velocity = self.x[:, self.dim:].clone()

        # Apply Scaling to Q
        # Q is process noise. If we think it's noise/chatter, we want to REDUCE Q (trust model/prediction more, trust innovation less? No.)
        # If Q is small, we trust the model physics (Constant Velocity). 
        # If Q is large, we allow changes (Responsive).
        # So for chatter suppression (which is high freq deviation from constant velocity), we want LOW Q.
        # So reducing Q is correct.
        
        q_eff = process_noise_q * scaling_factor
        
        # Rebuild F with new dt
        self.F_base[:, :self.dim, self.dim:] = torch.eye(self.dim, device=self.device, dtype=self.dtype) * dt
        
        # Rebuild Q
        pos_var = (q_eff * dt)**2
        vel_var = q_eff**2
        
        q_diag_pos = torch.full((self.num_streams, self.dim), 1.0, device=self.device, dtype=self.dtype) * pos_var
        q_diag_vel = torch.full((self.num_streams, self.dim), 1.0, device=self.device, dtype=self.dtype) * vel_var
        
        # If scaling factor is per-stream (tensor), we need to broadcast properly
        # q_eff is (S, 1) or scalar.
        if isinstance(scaling_factor, torch.Tensor):
             q_diag_pos = q_diag_pos * 1.0 # Ensure tensor
             q_diag_vel = q_diag_vel * 1.0
             
        Q_diag = torch.cat([q_diag_pos, q_diag_vel], dim=1)
        self.Q = torch.diag_embed(Q_diag)


        # Update R (Measurement Noise)
        self.R = torch.eye(self.dim, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * measurement_noise_r

    def predict(self):
        if self._reset_pending: return
        
        # x = F x
        # P = F P F^T + Q
        
        # F is [S, 2D, 2D]
        # x is [S, 2D]
        
        # self.x = (self.F_base @ self.x.unsqueeze(-1)).squeeze(-1)
        self.x = torch.bmm(self.F_base, self.x.unsqueeze(-1)).squeeze(-1)
        
        # P = FPF' + Q
        self.P = torch.bmm(self.F_base, torch.bmm(self.P, self.F_base.transpose(1, 2))) + self.Q

    def update(self, z):
        if self._reset_pending:
            self.x.zero_()
            self.x[:, :self.dim] = z
            self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)
            self._reset_pending = False
            return self.x[:, :self.dim]

        # 1. Innovation
        # y = z - Hx
        # H selects position part
        x_pos = self.x[:, :self.dim]
        y = z - x_pos

        # --- SMART CLAMPING ---
        # Limit the magnitude of y
        if self.clamp_radius > 0:
            # Calculate norm per stream
            innovation_mag = torch.norm(y, dim=1, keepdim=True) + 1e-9
            
            # Scale factor: min(1.0, limit / mag)
            # If mag < limit, scale = 1.0 (pass through)
            # If mag > limit, scale = limit / mag < 1.0 (shrink)
            scale = torch.clamp(self.clamp_radius / innovation_mag, max=1.0)
            
            y_clamped = y * scale
            # Note: We use Clamped Innovation for the STATE update,
            # but theoretically the Covariance update shouldn't 'know' about the clamp 
            # if we want to trust the model more? 
            # Actually, standard EKF gating rejects it. 
            # Here we are effectively modifying the measurement to be 'closer'.
            y = y_clamped
            
        # 2. Kalman Gain
        # S = H P H^T + R
        # P is [S, 2D, 2D]
        # H is [S, D, 2D]
        
        # H P H^T  -> H @ P @ H.T
        # H is just selection of top-left block of P
        P_pos = self.P[:, :self.dim, :self.dim]
        S = P_pos + self.R
        
        # K = P H^T S^-1
        # P H^T is P[:, :, :dim]
        PHt = self.P[:, :, :self.dim]
        
        # S_inv
        S_inv = torch.linalg.inv(S + self.jitter)
        K = torch.bmm(PHt, S_inv)
        
        # 3. Update State
        # x = x + K y
        # K [S, 2D, D] @ y [S, D, 1] -> [S, 2D, 1]
        correction = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
        self.x = self.x + correction
        
        # 4. Update Covariance
        # P = (I - KH) P
        # I - KH
        I = self.I_x
        KH = torch.bmm(K, self.H)
        I_KH = I - KH
        
        # Joseph form or simple form?
        # Simple: P = (I-KH)P
        self.P = torch.bmm(I_KH, self.P)
        
        return self.x[:, :self.dim]

    def reset(self):
        self._reset_pending = True

    def flag_reset(self):
        self._reset_pending = True
        
    def update_device_to_match(self, tensor):
        if tensor.device != self.device:
            self.device = tensor.device
            for attr in ['x', 'P', 'F_base', 'H', 'Q', 'R', 'I_x', 'jitter']:
                setattr(self, attr, getattr(self, attr).to(self.device))


class TorchSmartClampKFNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchSmartClampKFNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
        self.filter = None
        self.last_input_shape = None

        # Inputs
        self.input_port = self.add_input("input", triggers_execution=True)
        self.dt_input = self.add_input("dt", widget_type='drag_float', default_value=1.0/60.0)
        
        self.add_property("--- Filter Parameters ---", widget_type='label')
        
        # Responsiveness (Process Noise)
        self.process_noise_prop = self.add_property('responsiveness', widget_type='drag_float', default_value=1.0, min=0.0001, max=1000.0)
        
        # Smoothness (Measurement Noise)
        self.meas_noise_prop = self.add_property('smoothness', widget_type='drag_float', default_value=0.1, min=0.0001, max=10.0)
        
        # Clamp Radius (Max Jump)
        self.clamp_prop = self.add_property('max jump (clamp)', widget_type='drag_float', default_value=0.1, min=0.0, max=100.0)

        # Accel Rejection (Noise Reduction on chatter)
        self.accel_reject_prop = self.add_property('accel_rejection', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0)

        # Max Acceleration Clamp (deg/s^2)
        self.max_accel_prop = self.add_property('max_accel (deg/s^2)', widget_type='drag_float', default_value=50000.0, min=1.0, max=100000.0)
        
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

        self.output_port = self.add_output('filtered')

    def reset_filter(self):
        if self.filter:
            self.filter.flag_reset()

    def execute(self):
        input_tensor = any_to_tensor(self.input_port())
        dt = self.dt_input()
        
        if input_tensor is None:
            return

        # Handle formatting
        # Expect [Streams, Dims]
        original_shape = input_tensor.shape
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Treat [B, T, D] as flat [BT, D] ??
        # Or just enforce 2D
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.flatten(0, -2)
            
        num_streams, dim = input_tensor.shape
        
        # Initialize or Resize
        if self.filter is None or self.filter.num_streams != num_streams or self.filter.dim != dim:
            self.filter = TorchSmartClampKF(dt, num_streams, dim, device=self.device, dtype=self.dtype)
            
        self.filter.update_device_to_match(input_tensor)
        
        # Update Parameters
        p_noise = self.process_noise_prop()
        m_noise = self.meas_noise_prop()
        clamp = self.clamp_prop()
        accel_rej = self.accel_reject_prop()
        max_accel = self.max_accel_prop()
        
        self.filter.update_params(p_noise, m_noise, clamp, accel_rej, max_accel, dt)
        
        # Execution
        self.filter.predict()
        filtered_pos = self.filter.update(input_tensor)
        
        # Reshape output
        if len(original_shape) > 2:
            filtered_pos = filtered_pos.view(original_shape)
        elif len(original_shape) == 1:
            filtered_pos = filtered_pos.squeeze(0)
            
        self.output_port.send(filtered_pos)

@torch.jit.script
def smart_clamp_quat_update_kernel(q, av, x, P, R, I_x, H, jitter, z_measured_q, dt: float, clamp_rad: float, max_accel_rad: float):
    # 1. Measurement Residual
    # residual_q = q.conj * z
    residual_q = q_mult(q_conjugate(q), z_measured_q)

    # 3. Shortest Path Check
    # If w < 0, negate the quaternion to take the shorter path
    mask = (residual_q[:, 0] < 0).float().unsqueeze(-1)
    residual_q = (1.0 - mask) * residual_q + mask * (-residual_q)

    # 2. Convert to rotation vector (Tangent Space Innovation)
    # angle = 2 * acos(w)
    angle_half = torch.acos(torch.clamp(residual_q[:, 0], -1.0, 1.0))
    sin_angle_half = torch.sin(angle_half)

    # Avoid div by zero
    scale_to_rotvec = torch.where(sin_angle_half.abs() > 1e-9,
                        (2.0 * angle_half) / sin_angle_half,
                        torch.full_like(angle_half, 2.0))

    residual_v = scale_to_rotvec.unsqueeze(-1) * residual_q[:, 1:]

    # --- SMART CLAMPING (Angular) ---
    if clamp_rad > 0.0:
        # Residual_v magnitude is the angle in radians
        mag = torch.norm(residual_v, dim=1, keepdim=True) + 1e-9
        # clamp_rad is in radians
        scale = torch.clamp(clamp_rad / mag, max=1.0)
        residual_v = residual_v * scale
    # --------------------------------

    # 3. Kalman Gain
    P_xz = P[:, :, :3] # P @ H^T (H is identity [I 0])
    P_zz = P[:, :3, :3] # H @ P @ H^T
    S = P_zz + R

    # Invert S
    S_inv = torch.linalg.inv(S + jitter)
    K = P_xz @ S_inv

    # 4. Update State (Error State)
    # x_new = K @ residual_v
    x_new = torch.einsum('scd,sd->sc', K, residual_v)

    # --- MAX ACCELERATION CLAMP ---
    # Clamp both Position jump (which implies infinite accel) and Velocity jump
    if max_accel_rad < 1000000.0: # Only engage if reasonable limit
        pos_impulse = x_new[:, :3]
        vel_impulse = x_new[:, 3:]
        
        # Accel from Pos = d / dt^2
        # Accel from Vel = v / dt
        accel_from_pos = torch.norm(pos_impulse, dim=1, keepdim=True) / (dt * dt)
        accel_from_vel = torch.norm(vel_impulse, dim=1, keepdim=True) / dt
        
        # Context Velocity Check (Discriminator)
        # Frame 75 (Artifact) happens at Rest (Vel < 0.8 rad/s)
        # Frame 1903 (Real) happens at Speed (Vel > 10 rad/s)
        
        # av is the predicted angular velocity [S, 3]
        curr_vel_mag = torch.norm(av, dim=1, keepdim=True)
        
        # Scale limit: 1x at Rest, up to 50x at Speed
        # Use a Smooth Ramp to prevent on/off inconsistency
        # Low Threshold: 1.0 rad/s (~60 deg/s) -> Strict
        # High Threshold: 1.5 rad/s (~90 deg/s) -> Relaxed (50x)
        # Frame 75 (~0.6) will be Strict. Frame 1903 (~1.8) will be Relaxed.
        
        t_val = torch.clamp((curr_vel_mag - 1.0) / (1.5 - 1.0), 0.0, 1.0)
        
        # Linearly interpolate
        # scale = 1.0 * (1-t) + 50.0 * t  => 1 + 49*t
        limit_scale = 1.0 + 49.0 * t_val
        
        effective_limit = max_accel_rad * limit_scale

        # Total implied acceleration load
        implied_accel = accel_from_pos + accel_from_vel
        
        # Scale down if exceeds effective_limit
        accel_scale = torch.clamp(effective_limit / (implied_accel + 1e-9), max=1.0)
        
        # Apply to entire state correction
        x_new = x_new * accel_scale
    # ------------------------------

    # 5. Update Covariance
    I_minus_KH = I_x - K @ H

    # Joseph Form: P = (I-KH)P(I-KH)' + KRK'
    term1 = I_minus_KH @ P @ I_minus_KH.transpose(1, 2)
    term2 = K @ R @ K.transpose(1, 2)
    P_new = term1 + term2
    P_new = (P_new + P_new.transpose(1, 2)) * 0.5

    # 6. Injection (Error State -> Nominal State)
    # Split position (rot var) and velocity parts
    error_rot_vec = x_new[:, :3]
    error_vel_vec = x_new[:, 3:]

    # Apply rotation correction to Quaternion state
    delta_q_correction = angular_velocity_to_delta_q(error_rot_vec, 1.0)
    q_final = F.normalize(q_mult(q, delta_q_correction), p=2.0, dim=-1)

    # Apply velocity correction
    av_final = av + error_vel_vec

    # Reset error state
    x_final = torch.zeros_like(x_new)

    return q_final, av_final, x_final, P_new


class TorchSmartClampQuaternionKF:
    def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
        self.dt = dt
        self.num_streams = num_streams
        self.device = device
        self.dtype = dtype

        self.state_dim = 6 # 3 for Rot (Error), 3 for AngVel
        self.dim = 3       # Measurement dimension (Rot Vector)

        # State:
        # q: Nominal Quaternion [S, 4]
        # av: Angular Velocity [S, 3]
        # x: Error State [S, 6] (Usually zero, kept for consistency)
        # P: Covariance [S, 6, 6]

        self.q = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q[:, 0] = 1.0 # Identity quaternions
        self.av = torch.zeros(num_streams, 3, device=device, dtype=dtype)
        self.x = torch.zeros(num_streams, self.state_dim, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)

        # F_base: Transition Matrix Base
        # [[I, dt*I], [0, I]] but for Error State kinematic relationship
        # F = [[I, dt*I], [0, I]] IS NOT SUFFICIENT for Rotations implies specific F_base?
        # Typically F depends on AV. ekf_predict_kernel builds F dynamically.
        # But we pass a base F to it.
        # The base is Identity.
        self.F_base = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)

        # H: Measurement Matrix (Measure Rotation Error only)
        # H = [I, 0]
        self.H = torch.zeros(num_streams, self.dim, self.state_dim, device=device, dtype=dtype)
        self.H[:, :self.dim, :self.dim] = torch.eye(self.dim, device=device, dtype=dtype)

        # Q and R
        self.Q = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)
        self.R = torch.eye(self.dim, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1)

        self.I_x = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0)
        self.jitter = torch.eye(self.dim, device=device, dtype=dtype).unsqueeze(0) * 1e-9

        self._reset_pending = False
        self.clamp_angle_rad = 0.1 # Default clamp (radians)
        self.accel_rejection_factor = 0.0
        self.last_av = None
        self.last_accel = None
        self.max_accel_rad = 1000.0 # Default high

    def update_device_to_match(self, tensor):
        if tensor.device != self.device:
            self.device = tensor.device
            for attr in ['q', 'av', 'x', 'P', 'F_base', 'H', 'Q', 'R', 'I_x', 'jitter']:
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.to(self.device))
                else:
                    # Handle list of tensors if any, but above are all tensors
                    pass

    def update_params(self, process_noise_q, measurement_noise_r, clamp_angle_deg, accel_rejection_factor, max_accel_deg, dt):
        self.dt = dt
        self.clamp_angle_rad = math.radians(clamp_angle_deg)
        self.accel_rejection_factor = accel_rejection_factor
        self.max_accel_rad = math.radians(max_accel_deg)

        # --- Accel Rejection Logic (Quaternion) ---
        # State av is angular velocity [S, 3]
        current_av = self.av
        
        if isinstance(process_noise_q, torch.Tensor) and process_noise_q.dim() == 1:
            process_noise_q = process_noise_q.unsqueeze(1)
            
        scaling_factor = 1.0
        if self.accel_rejection_factor > 0:
            if self.last_av is not None:
                curr_accel = (current_av - self.last_av) / dt
                if self.last_accel is not None:
                     # Dot product (S, 1)
                     dot = (curr_accel * self.last_accel).sum(dim=1, keepdim=True)
                     
                     reversal_mask = (dot < 0).float()
                     
                     multiplier = 1.0 / (1.0 + self.accel_rejection_factor * reversal_mask * 100.0)
                     scaling_factor = multiplier # [S, 1]
                
                self.last_accel = curr_accel
            self.last_av = current_av.clone()
        else:
            self.last_av = self.av.clone()
            self.last_accel = None
            
        q_eff = process_noise_q * scaling_factor

        # Update Q (Process Noise)
        # Standard white noise acceleration model for rotation
        # Q_angle = (dt^3 / 3) * q_mag
        # Q_vel = dt * q_mag
        if isinstance(q_eff, torch.Tensor) and q_eff.dim() == 1:
            q_eff = q_eff.unsqueeze(1)
        
        pos_var = (q_eff * dt)**2
        vel_var = q_eff**2
        
        q_diag_pos = torch.ones((self.num_streams, self.dim), device=self.device, dtype=self.dtype) * pos_var
        q_diag_vel = torch.ones((self.num_streams, self.dim), device=self.device, dtype=self.dtype) * vel_var
        Q_diag = torch.cat([q_diag_pos, q_diag_vel], dim=1)
        self.Q = torch.diag_embed(Q_diag)

        # Update R
        if isinstance(measurement_noise_r, torch.Tensor):
             if measurement_noise_r.dim() == 1:
                 measurement_noise_r = measurement_noise_r.view(-1, 1, 1)
             elif measurement_noise_r.dim() == 2:
                 # In case it is [S, 1]
                 measurement_noise_r = measurement_noise_r.unsqueeze(-1)
                 
        self.R = torch.eye(self.dim, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * measurement_noise_r

    def predict(self):
        if self._reset_pending: return
        self.q, self.P = ekf_predict_kernel(self.q, self.av, self.P, self.Q, self.F_base, self.dt)

    def update(self, z_quat):
        if self._reset_pending:
            self.q = z_quat.clone()
            self.av.zero_()
            self.x.zero_()
            self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)
            self._reset_pending = False
            return self.q

        # Ensure Z is normalized
        z = F.normalize(z_quat, p=2.0, dim=-1)

        # Use Smart Clamp Update Kernel
        self.q, self.av, self.x, self.P = smart_clamp_quat_update_kernel(
            self.q, self.av, self.x, self.P, self.R, self.I_x, self.H, self.jitter, z, self.dt, self.clamp_angle_rad, self.max_accel_rad
        )
        return self.q

    def reset(self):
        self._reset_pending = True


class TorchSmartClampQuaternionKFNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSmartClampQuaternionKFNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
        self.input_port = self.add_input('input', triggers_execution=True)

        self.add_property("--- Filter Parameters ---", widget_type='label')

        # Properties
        # Process Noise (Responsiveness)
        self.process_noise_prop = self.add_property('responsiveness', widget_type='drag_float', default_value=1.0, min=0.0001, max=1000.0)
        
        # Measurement Noise (Smoothness)
        self.meas_noise_prop = self.add_property('smoothness', widget_type='drag_float', default_value=0.1, min=0.0001, max=10.0)
        
        # Max Jump In Degrees
        self.clamp_prop = self.add_property('max jump (deg)', widget_type='drag_float', default_value=5.0, min=0.0, max=180.0)

        # Accel Rejection
        self.accel_reject_prop = self.add_property('accel_rejection', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0)

        # Max Acceleration Clamp (deg/s^2)
        self.max_accel_prop = self.add_property('max_accel (deg/s^2)', widget_type='drag_float', default_value=50000.0, min=1.0, max=100000.0)

        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)
        
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)
        
        # self.input_port is already defined above
        self.dt_input = self.add_input("dt", widget_type='drag_float', default_value=1.0/60.0)
        self.output = self.add_output("filtered")

        self.filter = None
    
    def reset_filter(self):
        if self.filter:
            self.filter.reset()

    def execute(self):
        input_tensor = self.input_port()
        if input_tensor is None:
            return
            
        input_tensor = any_to_tensor(input_tensor, device=self.device, dtype=self.dtype)
        
        # Check Shape (Expect Nx4)
        if input_tensor.dim() == 1 and input_tensor.shape[0] == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        if input_tensor.dim() != 2 or input_tensor.shape[-1] != 4:
            return # Invalid input
            
        num_streams = input_tensor.shape[0]
        dt = self.dt_input()
        if dt <= 0: dt = 1.0/60.0

        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchSmartClampQuaternionKF(dt, num_streams, device=self.device, dtype=self.dtype)
            self.filter.reset()

        if self.filter.device != self.device:
            # Re-init on device change (simplest for now)
            self.filter = TorchSmartClampQuaternionKF(dt, num_streams, device=self.device, dtype=self.dtype)
            self.filter.reset()

        # Update Params
        self.filter.update_params(
            process_noise_q=self.process_noise_prop(),
            measurement_noise_r=self.meas_noise_prop(),
            clamp_angle_deg=self.clamp_prop(),
            accel_rejection_factor=self.accel_reject_prop(),
            max_accel_deg=self.max_accel_prop(),
            dt=dt
        )

        # Predict & Update
        self.filter.predict()
        filtered_q = self.filter.update(input_tensor)
        
        self.output.send(filtered_q)

class TorchHybridQuaternionKFNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchHybridQuaternionKFNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
        self.filter = None
        self.last_input_quat = None
        self.blended_alphas = None # [S, 3]

        self.input_port = self.add_input('input', triggers_execution=True)
        
        # --- Tristate Parameters ---
        self.add_property("--- Tristate Noise Params ---", widget_type='label')
        # Damping (Base State)
        self.damp_meas_in = self.add_input('Damp Smoothness', widget_type='drag_float', default_value=0.1, callback=self.update_params)
        self.damp_vel_in = self.add_input('Damp Responsiveness', widget_type='drag_float', default_value=1.0, callback=self.update_params)
        
        # Responsive (Motion State)
        self.resp_meas_in = self.add_input('Resp Smoothness', widget_type='drag_float', default_value=0.01, callback=self.update_params)
        self.resp_vel_in = self.add_input('Resp Responsiveness', widget_type='drag_float', default_value=100.0, callback=self.update_params)
        
        # Error Correction (Divergence State)
        self.err_meas_in = self.add_input('Err Correct Smoothness', widget_type='drag_float', default_value=0.5, callback=self.update_params)
        self.err_vel_in = self.add_input('Err Correct Responsiveness', widget_type='drag_float', default_value=50.0, callback=self.update_params)
        
        self.add_property("--- Transition Ranges ---", widget_type='label')
        self.motion_min_in = self.add_input('Min Motion (deg/sec)', widget_type='drag_float', default_value=10.0, callback=self.update_params)
        self.motion_max_in = self.add_input('Max Motion (deg/sec)', widget_type='drag_float', default_value=90.0, callback=self.update_params)
        self.error_min_in = self.add_input('Min Error (deg)', widget_type='drag_float', default_value=5.0, callback=self.update_params)
        self.error_max_in = self.add_input('Max Error (deg)', widget_type='drag_float', default_value=45.0, callback=self.update_params)
        
        self.blending_speed_in = self.add_input('Blending Speed', widget_type='drag_float', default_value=0.1)

        # --- Smart Clamp Parameters ---
        self.add_property("--- Smart Clamp Params ---", widget_type='label')
        self.clamp_prop = self.add_property('max jump (clamp)', widget_type='drag_float', default_value=5.0, min=0.0, max=100.0)
        self.accel_reject_prop = self.add_property('accel_rejection', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0)
        self.max_accel_prop = self.add_property('max_accel (deg/s^2)', widget_type='drag_float', default_value=500.0, min=1.0, max=100000.0)

        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

        # Outputs
        self.output_port = self.add_output('filtered')
        self.alphas_output = self.add_output('alphas (damp, resp, err)')
        
    def custom_create(self, from_file):
        self.update_params()

    def reset_filter(self):
        if self.filter:
            self.filter.reset()
        self.last_input_quat = None
        if self.blended_alphas is not None:
            self.blended_alphas.zero_()
            self.blended_alphas[:, 0] = 1.0

    def update_params(self):
        # Cache params for speed
        self.damp_params = [self.damp_meas_in(), self.damp_vel_in()]
        self.resp_params = [self.resp_meas_in(), self.resp_vel_in()]
        self.err_params = [self.err_meas_in(), self.err_vel_in()]
        
        self.motion_min = self.motion_min_in()
        self.motion_max = self.motion_max_in()
        self.error_min = self.error_min_in()
        self.error_max = self.error_max_in()
        
        if self.device is not None:
             self.damp_tensor = torch.tensor(self.damp_params, device=self.device, dtype=self.dtype)
             self.resp_tensor = torch.tensor(self.resp_params, device=self.device, dtype=self.dtype)
             self.err_tensor = torch.tensor(self.err_params, device=self.device, dtype=self.dtype)

    def _calculate_angular_difference_deg(self, q1, q2):
        # q1, q2 are [S, 4]
        diff_q = q_mult(q_conjugate(q1), q2)
        w = torch.clamp(diff_q[:, 0], -1.0, 1.0)
        angle_rad = 2 * torch.acos(w)
        return torch.rad2deg(angle_rad)

    def execute(self):
        input_tensor = any_to_tensor(self.input_port())
        dt = 1.0/60.0 # Standardize or get from input? Tristate usually infers or fixed?
        # Check parent node for dt input?
        # Actually Tristate node had dt_input separate? No, looking at my view_file output it wasn't obvious.
        # Let's add dt input to be safe.
        # Wait, TristateBlendESEKFNode uses self.dt_input() in execute usually.
        # I didn't verify if it has one. Let's add it.
        
        if input_tensor is None: return
        
        # Ensure 2D [S, 4]
        if input_tensor.dim() == 1: input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 2 or input_tensor.shape[-1] != 4: return
        
        num_streams = input_tensor.shape[0]
        
        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchSmartClampQuaternionKF(dt, num_streams, device=self.device, dtype=self.dtype)
            self.reset_filter()
            self.blended_alphas = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype).unsqueeze(0).repeat(num_streams, 1)

        self.filter.update_device_to_match(input_tensor)
        if self.blended_alphas.device != self.device: self.blended_alphas = self.blended_alphas.to(self.device)
        
        # --- 1. Calculate Tristate Alphas ---
        motion_deg_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_quat is not None:
            motion_deg_per_frame = self._calculate_angular_difference_deg(input_tensor, self.last_input_quat)
            motion_deg_per_sec = motion_deg_per_frame / dt
            
        strength_resp_vec = torch.clamp((motion_deg_per_sec - self.motion_min) / (self.motion_max - self.motion_min + 1e-9), 0.0, 1.0)
        
        # Error prediction
        # We need to use filter's AV to predict where we SHOULD be
        # Then compare to input.
        # Note: SmartClampKF stores state in self.filter.q, self.filter.av
        
        delta_q = angular_velocity_to_delta_q(self.filter.av, dt)
        predicted_q_temp = q_mult(self.filter.q, delta_q)
        error_deg = self._calculate_angular_difference_deg(input_tensor, predicted_q_temp)
        strength_err_vec = torch.clamp((error_deg - self.error_min) / (self.error_max - self.error_min + 1e-9), 0.0, 1.0)
        
        # Blending
        alpha_active = torch.maximum(strength_resp_vec, strength_err_vec)
        total_strength = strength_resp_vec + strength_err_vec + 1e-9
        
        target_alpha_resp = alpha_active * (strength_resp_vec / total_strength)
        target_alpha_err = alpha_active * (strength_err_vec / total_strength)
        target_alpha_damp = 1.0 - target_alpha_resp - target_alpha_err
        
        target_alphas = torch.stack([target_alpha_damp, target_alpha_resp, target_alpha_err], dim=1)
        blending_speed = self.blending_speed_in()
        self.blended_alphas.lerp_(target_alphas, blending_speed)
        
        # Mix Params
        param_stack = torch.stack([self.damp_tensor, self.resp_tensor, self.err_tensor], dim=0) # [3, 2]
        blended_params = self.blended_alphas @ param_stack # [S, 2]
        
        # --- 2. Update Filter Params (Smart Clamp Integrated) ---
        process_noise_q = blended_params[:, 1] # vel responsiveness
        measurement_noise_r = blended_params[:, 0] # smoothness
        
        # We pass these to update_params.
        # Note: SmartClampKF.update_params expects SCALARS or TENSORS?
        # Looking at implementation: 
        # q_diag_pos = torch.full(...) * pos_var. 
        # If pos_var is tensor, it broadcasts.
        # So passing tensors [S] is fine.
        
        self.filter.update_params(
            process_noise_q=process_noise_q,
            measurement_noise_r=measurement_noise_r,
            clamp_angle_deg=self.clamp_prop(),
            accel_rejection_factor=self.accel_reject_prop(),
            max_accel_deg=self.max_accel_prop(),
            dt=dt
        )
        
        # --- 3. Run Filter ---
        self.filter.predict()
        filtered_q = self.filter.update(input_tensor)
        
        self.output_port.send(filtered_q)
        self.alphas_output.send(self.blended_alphas)
        self.last_input_quat = input_tensor.clone()

@torch.jit.script
def persistence_quat_update_kernel(q, av, x, P, R, I_x, H, jitter, z_measured_q, dt: float, 
                                   clamp_rad: float, max_accel_rad: float, 
                                   persistence_counts, persistence_threshold: int):
    # 1. Measurement Residual
    residual_q = q_mult(q_conjugate(q), z_measured_q)

    # Shortest Path Check
    mask = (residual_q[:, 0] < 0).float().unsqueeze(-1)
    residual_q = (1.0 - mask) * residual_q + mask * (-residual_q)

    # 2. Tangent Space
    angle_half = torch.acos(torch.clamp(residual_q[:, 0], -1.0, 1.0))
    sin_angle_half = torch.sin(angle_half)
    
    scale_to_rotvec = torch.where(sin_angle_half.abs() > 1e-9,
                        (2.0 * angle_half) / sin_angle_half,
                        torch.full_like(angle_half, 2.0))

    residual_v = scale_to_rotvec.unsqueeze(-1) * residual_q[:, 1:]

    # --- SMART CLAMPING (Innovation) ---
    if clamp_rad > 0.0:
        mag = torch.norm(residual_v, dim=1, keepdim=True) + 1e-9
        scale = torch.clamp(clamp_rad / mag, max=1.0)
        residual_v = residual_v * scale
    # -----------------------------------

    # 3. Kalman Gain
    P_xz = P[:, :, :3]
    P_zz = P[:, :3, :3]
    S = P_zz + R
    S_inv = torch.linalg.inv(S + jitter)
    K = P_xz @ S_inv

    # 4. Update State
    x_new = torch.einsum('scd,sd->sc', K, residual_v)

    # --- PERSISTENCE LOGIC ---
    
    # Defaults
    new_counts = persistence_counts
    final_x_new = x_new

    if max_accel_rad < 1000000.0:
        pos_impulse = x_new[:, :3]
        vel_impulse = x_new[:, 3:]
        
        accel_from_pos = torch.norm(pos_impulse, dim=1, keepdim=True) / (dt * dt)
        accel_from_vel = torch.norm(vel_impulse, dim=1, keepdim=True) / dt
        implied_accel = accel_from_pos + accel_from_vel
        
        # Check Violation
        # implied [S, 1], max_accel_rad scalar
        is_violation = (implied_accel > max_accel_rad).squeeze(-1) # [S] bool
        
        # DEBUG
        if max_accel_rad < 1000.0:
           print("DEBUG:", implied_accel.mean(), max_accel_rad, is_violation.float().mean())
        
        # Update Counts
        # If violation, increment. If safe, reset to 0.
        new_counts = torch.where(is_violation, persistence_counts + 1, torch.zeros_like(persistence_counts))
        
        # Determine if we should ACCEPT despite violation (Persistence met)
        # We accept if NOT violation OR (violation AND count >= threshold)
        # So we CLAMP if (violation AND count < threshold)
        
        should_clamp = is_violation & (new_counts < persistence_threshold)
        should_clamp_float = should_clamp.float().unsqueeze(-1) # [S, 1]
        
        # Calculate Clamped Update
        accel_scale = torch.clamp(max_accel_rad / (implied_accel + 1e-9), max=1.0)
        x_clamped = x_new * accel_scale
        
        # Mix: If clamp, use clamped. vary hard.
        final_x_new = (1.0 - should_clamp_float) * x_new + should_clamp_float * x_clamped
        
    # -------------------------

    # 5. Update Covariance
    I_minus_KH = I_x - K @ H
    term1 = I_minus_KH @ P @ I_minus_KH.transpose(1, 2)
    term2 = K @ R @ K.transpose(1, 2)
    P_new = term1 + term2
    P_new = (P_new + P_new.transpose(1, 2)) * 0.5

    # 6. Injection
    error_rot_vec = final_x_new[:, :3]
    delta_q_correction = angular_velocity_to_delta_q(error_rot_vec, 1.0)
    q_final = F.normalize(q_mult(q, delta_q_correction), p=2.0, dim=-1)
    av_final = av + final_x_new[:, 3:]
    x_final = torch.zeros_like(final_x_new)

    return q_final, av_final, x_final, P_new, new_counts


class TorchPersistenceQuaternionKF(TorchSmartClampQuaternionKF):
    def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
        super().__init__(dt, num_streams, device, dtype)
        self.persistence_counts = torch.zeros(num_streams, device=device, dtype=torch.long)
        self.persistence_threshold = 2
        
    def update_device_to_match(self, tensor):
        super().update_device_to_match(tensor)
        if self.persistence_counts.device != self.device:
            self.persistence_counts = self.persistence_counts.to(self.device)

    def update_params(self, process_noise_q, measurement_noise_r, clamp_angle_deg, accel_rejection_factor, max_accel_deg, dt, persistence_threshold):
        super().update_params(process_noise_q, measurement_noise_r, clamp_angle_deg, accel_rejection_factor, max_accel_deg, dt)
        self.persistence_threshold = int(persistence_threshold)
        
    def update(self, z_quat):
        if self._reset_pending:
            self.q = z_quat.clone()
            self.av.zero_()
            self.x.zero_()
            self.P.zero_().diagonal(dim1=-2, dim2=-1).fill_(0.1)
            self.persistence_counts.zero_()
            self._reset_pending = False
            return self.q

        # Ensure Z is normalized
        z = F.normalize(z_quat, p=2.0, dim=-1)

        # Use Persistence Update Kernel
        self.q, self.av, self.x, self.P, self.persistence_counts = persistence_quat_update_kernel(
            self.q, self.av, self.x, self.P, self.R, self.I_x, self.H, self.jitter, z, self.dt, 
            self.clamp_angle_rad, self.max_accel_rad,
            self.persistence_counts, self.persistence_threshold
        )
        return self.q
        
    def reset(self):
        super().reset()
        self.persistence_counts.zero_()


class TorchPersistenceKFNode(TorchSmartClampQuaternionKFNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPersistenceKFNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_property("--- Persistence ---", widget_type='label')
        self.persistence_prop = self.add_property('threshold_frames', widget_type='drag_int', default_value=2, min=1, max=10)

    def execute(self):
        input_tensor = self.input_port() # Use correct port
        if input_tensor is None: return
        
        input_tensor = any_to_tensor(input_tensor, device=self.device, dtype=self.dtype)
        if input_tensor.dim() == 1 and input_tensor.shape[0] == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        num_streams = input_tensor.shape[0]
        dt = self.dt_input()
        if dt <= 0: dt = 1.0/60.0

        if self.filter is None or self.filter.num_streams != num_streams or not isinstance(self.filter, TorchPersistenceQuaternionKF):
            self.filter = TorchPersistenceQuaternionKF(dt, num_streams, device=self.device, dtype=self.dtype)
            self.filter.reset()

        if self.filter.device != self.device:
            self.filter.update_device_to_match(input_tensor)

        p_noise = self.process_noise_prop()
        m_noise = self.meas_noise_prop()
        clamp = self.clamp_prop()
        accel_rej = self.accel_reject_prop()
        max_accel = self.max_accel_prop()
        persist = self.persistence_prop()
        
        self.filter.update_params(p_noise, m_noise, clamp, accel_rej, max_accel, dt, persist)
        
        self.filter.predict()
        filtered_pos = self.filter.update(input_tensor)
        
        self.output.send(filtered_pos)


# =============================================================================
# Jerk-Aware Quaternion Filter
# =============================================================================

class MotionStateEnum:
    """Motion classification states for jerk-aware filtering."""
    NORMAL = 0
    RAPID_MOTION = 1
    NOISE_SPIKE = 2


@torch.jit.script
def jerk_quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (scalar-first convention)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


@torch.jit.script
def jerk_quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate."""
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


@torch.jit.script
def jerk_quat_to_angular_velocity(q0: torch.Tensor, q1: torch.Tensor, dt: float) -> torch.Tensor:
    """Extract angular velocity from consecutive quaternions."""
    dq = jerk_quat_multiply(q1, jerk_quat_conjugate(q0))
    
    # Ensure shortest path
    mask = dq[..., 0:1] < 0
    dq = torch.where(mask, -dq, dq)
    
    w = dq[..., 0]
    xyz = dq[..., 1:4]
    
    w_clamped = torch.clamp(w, -1.0, 1.0)
    near_identity = torch.abs(w_clamped) > 0.9999
    
    # Small angle approximation
    omega_small = 2.0 * xyz / dt
    
    # Large angle extraction
    half_angle = torch.acos(w_clamped)
    sin_half = torch.sin(half_angle)
    axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=1e-10)
    omega_large = (2.0 * half_angle.unsqueeze(-1) / dt) * axis
    
    return torch.where(near_identity.unsqueeze(-1), omega_small, omega_large)


@torch.jit.script
def jerk_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation with per-stream blend factors."""
    q0 = F.normalize(q0, p=2.0, dim=-1)
    q1 = F.normalize(q1, p=2.0, dim=-1)
    
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    
    # Ensure shortest path
    mask = dot < 0
    q1 = torch.where(mask, -q1, q1)
    dot = torch.where(mask, -dot, dot)
    
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Near-identical: linear interpolation
    near_identical = dot > 0.9995
    
    # SLERP computation
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.clamp(torch.sin(theta_0), min=1e-10)
    
    theta = theta_0 * t.unsqueeze(-1)
    
    s0 = torch.cos(theta) - dot * torch.sin(theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    
    result_slerp = s0 * q0 + s1 * q1
    
    # Linear fallback
    result_linear = q0 + t.unsqueeze(-1) * (q1 - q0)
    
    result = torch.where(near_identical, result_linear, result_slerp)
    return F.normalize(result, p=2.0, dim=-1)


@torch.jit.script
def jerk_classify_motion(
    jerk_mag: torch.Tensor,
    accel_mag: torch.Tensor,
    omega_mag: torch.Tensor,
    alpha_new: torch.Tensor,
    alpha_prev: torch.Tensor,
    prev_state: torch.Tensor,
    spike_counter: torch.Tensor,
    jerk_threshold: torch.Tensor,
    accel_limit: float,
    velocity_threshold: float,
    spike_recovery_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classify motion states with velocity hysteresis.
    
    NOISE_SPIKE: high jerk + LOW velocity (momentary noise flicker)
    DYNAMIC_MOTION: in active motion (velocity above entry threshold OR 
                    velocity above exit threshold after recent high velocity)
    NORMAL: everything else
    
    Hysteresis: Once in DYNAMIC_MOTION, stay there until velocity drops
    below EXIT threshold (30% of entry) for several consecutive frames.
    While in DYNAMIC_MOTION, never classify as spike.
    """
    NORMAL = 0
    DYNAMIC_MOTION = 1  # Replaces both RAPID_MOTION and MOTION_COOLDOWN
    NOISE_SPIKE = 2
    
    num_streams = jerk_mag.shape[0]
    device = jerk_mag.device
    
    new_state = torch.zeros(num_streams, dtype=torch.long, device=device)
    new_counter = spike_counter.clone()
    
    # Hysteresis thresholds
    velocity_entry = velocity_threshold  # Enter dynamic mode above this
    velocity_exit = velocity_threshold * 0.3  # Exit dynamic mode below this
    frames_to_exit = spike_recovery_frames  # Must stay below exit for this many frames
    
    # Check previous state
    was_dynamic = (prev_state == DYNAMIC_MOTION)
    was_spike = (prev_state == NOISE_SPIKE)
    
    # Velocity conditions
    is_high_velocity = omega_mag >= velocity_entry
    is_above_exit = omega_mag >= velocity_exit
    is_definitely_stopped = omega_mag < velocity_exit
    
    # DYNAMIC MOTION: Enter when velocity is high
    entering_dynamic = is_high_velocity & (~was_dynamic)
    
    # Stay in dynamic mode if:
    # 1. Velocity is still above exit threshold, OR
    # 2. Velocity dropped below exit but counter hasn't reached threshold
    staying_dynamic = was_dynamic & (is_above_exit | (new_counter < frames_to_exit))
    
    # Update counter for exiting dynamic mode
    # Increment when below exit threshold, reset when above
    new_counter = torch.where(was_dynamic & is_definitely_stopped, 
                              spike_counter + 1, new_counter)
    new_counter = torch.where(was_dynamic & is_above_exit, 
                              torch.zeros_like(new_counter), new_counter)
    new_counter = torch.where(entering_dynamic, 
                              torch.zeros_like(new_counter), new_counter)
    
    # Apply DYNAMIC_MOTION state
    is_dynamic = entering_dynamic | staying_dynamic
    new_state = torch.where(is_dynamic, torch.full_like(new_state, DYNAMIC_MOTION), new_state)
    
    # NOISE SPIKE: high jerk AND definitely low velocity AND NOT in dynamic mode
    is_jerk_spike = jerk_mag > jerk_threshold
    is_low_velocity = omega_mag < velocity_threshold
    is_spike = is_jerk_spike & is_low_velocity & (~is_dynamic)
    
    # Also check for acceleration reversal (but not in dynamic mode)
    dot = torch.sum(alpha_new * alpha_prev, dim=-1)
    alpha_prev_mag = torch.norm(alpha_prev, dim=-1)
    reversal_spike = (
        (accel_mag > accel_limit * 0.5) & 
        (dot < 0) & 
        (alpha_prev_mag > accel_limit * 0.3) &
        is_low_velocity &
        (~is_dynamic)
    )
    is_spike = is_spike | reversal_spike
    
    # Apply spike state (overrides NORMAL but not DYNAMIC_MOTION)
    new_state = torch.where(is_spike & (~is_dynamic), 
                            torch.full_like(new_state, NOISE_SPIKE), new_state)
    new_counter = torch.where(is_spike & (~is_dynamic), 
                              torch.zeros_like(new_counter), new_counter)
    
    # Spike recovery: stay in spike mode for a few frames after spike ends
    still_recovering = was_spike & ~is_spike & ~is_dynamic & (spike_counter < spike_recovery_frames)
    new_state = torch.where(still_recovering, torch.full_like(new_state, NOISE_SPIKE), new_state)
    new_counter = torch.where(still_recovering, spike_counter + 1, new_counter)
    
    return new_state, new_counter


@torch.jit.script
def jerk_compute_blend(
    motion_states: torch.Tensor,
    jerk_mag: torch.Tensor,
    jerk_threshold: float,
    min_responsiveness: float,
    max_responsiveness: float
) -> torch.Tensor:
    """Compute blend factors based on motion state."""
    NORMAL = 0
    RAPID_MOTION = 1
    NOISE_SPIKE = 2
    
    blend = torch.full_like(jerk_mag, max_responsiveness)
    
    # Moderate smoothing for rapid motion
    moderate = 0.5 * (min_responsiveness + max_responsiveness)
    blend = torch.where(motion_states == RAPID_MOTION, torch.full_like(blend, moderate), blend)
    
    # Strong smoothing for spikes
    severity = torch.clamp(jerk_mag / (jerk_threshold * 2), max=1.0)
    spike_blend = min_responsiveness * (1 - 0.5 * severity)
    blend = torch.where(motion_states == NOISE_SPIKE, spike_blend, blend)
    
    return blend


class TorchJerkAwareQuatKF:
    """
    Jerk-aware adaptive quaternion filter.
    
    Distinguishes between noise spikes and genuine rapid movements
    by analyzing temporal coherence of angular jerk from RAW INPUT.
    
    During spikes, extrapolates from the LAST GOOD POSE using
    SMOOTHED VELOCITY (not single-frame velocity which may be noisy).
    """
    
    def __init__(self, dt: float, num_streams: int, device='cpu', dtype=torch.float32):
        self.dt = dt
        self.num_streams = num_streams
        self.device = device
        self.dtype = dtype
        
        # Filtered output state
        self.q_filtered = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q_filtered[:, 0] = 1.0
        self.q_filtered_prev = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q_filtered_prev[:, 0] = 1.0
        
        # Smoothed velocity for robust prediction (EMA)
        self.omega_smooth = torch.zeros(num_streams, 3, device=device, dtype=dtype)
        self.velocity_smooth_factor = 0.3  # Lower = more smoothing
        
        # Last good pose (before spike started)
        self.q_last_good = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q_last_good[:, 0] = 1.0
        self.frames_in_spike = torch.zeros(num_streams, dtype=torch.long, device=device)
        self.frames_since_spike = torch.zeros(num_streams, dtype=torch.long, device=device)  # For gradual recovery
        
        # Raw input history (for motion classification)
        self.q_raw_prev = torch.zeros(num_streams, 4, device=device, dtype=dtype)
        self.q_raw_prev[:, 0] = 1.0
        self.omega_raw_prev = torch.zeros(num_streams, 3, device=device, dtype=dtype)
        self.alpha_raw_prev = torch.zeros(num_streams, 3, device=device, dtype=dtype)
        
        # Motion classification state
        self.motion_states = torch.zeros(num_streams, dtype=torch.long, device=device)
        self.spike_counters = torch.full((num_streams,), 3, dtype=torch.long, device=device)
        
        # Global mocap glitch detection (multi-joint synchronized spikes)
        self.global_glitch_mode = False  # True when in global glitch
        self.global_glitch_frames = 0    # Frames since glitch started
        self.glitch_joint_threshold = 0.3  # Fraction of joints to trigger global glitch
        self.global_glitch_transition_frames = 5  # Frames to smoothly transition to new position
        
        # Parameters
        self.jerk_threshold = 2000.0
        
        # Per-joint jerk threshold scale factors based on SMPL hierarchy
        # Proximal joints (high inertia) get LOWER threshold (stricter)
        # Distal joints (low inertia) get HIGHER threshold (more permissive)
        # Scale factors multiply the base jerk_threshold
        self.joint_threshold_scales = self._create_smpl_joint_scales(num_streams, device, dtype)
        self.accel_limit = 50.0
        self.velocity_threshold = 5.0
        self.min_responsiveness = 0.1
        self.max_responsiveness = 0.95
        self.spike_recovery_frames = 3
        
        self._warmup_frames = 0
        self._initialized = False
    
    def _create_smpl_joint_scales(self, num_streams: int, device, dtype) -> torch.Tensor:
        """
        Create per-joint jerk threshold scale factors based on SMPL hierarchy.
        
        Proximal joints (high inertia): Lower scale = stricter threshold
        Distal joints (low inertia): Higher scale = more permissive
        
        Returns tensor of shape (num_streams,) with scale factors.
        """
        # Default: all joints same scale
        scales = torch.ones(num_streams, device=device, dtype=dtype)
        
        # SMPL 22-joint hierarchy scale factors
        # Based on physical inertia - lower = stricter filtering
        if num_streams == 22 or num_streams == 24:
            smpl_scales = {
                0: 0.25,   # pelvis - very strict
                1: 0.4,    # left_hip
                2: 0.4,    # right_hip
                3: 0.4,    # spine1
                4: 0.6,    # left_knee
                5: 0.6,    # right_knee
                6: 0.5,    # spine2
                7: 1.2,    # left_ankle
                8: 1.2,    # right_ankle
                9: 0.5,    # spine3
                10: 1.5,   # left_foot
                11: 1.5,   # right_foot
                12: 0.4,   # neck
                13: 0.6,   # left_collar
                14: 0.6,   # right_collar
                15: 0.5,   # head
                16: 0.7,   # left_shoulder
                17: 0.7,   # right_shoulder
                18: 1.0,   # left_elbow
                19: 1.0,   # right_elbow
                20: 2.0,   # left_wrist
                21: 2.0,   # right_wrist
            }
            # Additional joints for 24-joint model
            if num_streams == 24:
                smpl_scales[22] = 2.5  # left_hand
                smpl_scales[23] = 2.5  # right_hand
            
            for j, scale in smpl_scales.items():
                if j < num_streams:
                    scales[j] = scale
        
        # SMPL-X 52-joint (includes fingers) - expand hand scales
        elif num_streams == 52 or num_streams == 55:
            # First 22 joints same as above
            base_scales = [
                0.25, 0.4, 0.4, 0.4, 0.6, 0.6, 0.5, 1.2, 1.2, 0.5,
                1.5, 1.5, 0.4, 0.6, 0.6, 0.5, 0.7, 0.7, 1.0, 1.0,
                2.0, 2.0
            ]
            # Finger joints get very high scale (can move very fast)
            finger_scale = 4.0
            while len(base_scales) < num_streams:
                base_scales.append(finger_scale)
            
            scales = torch.tensor(base_scales[:num_streams], device=device, dtype=dtype)
        
        return scales
    
    def update_device_to_match(self, tensor: torch.Tensor):
        if tensor.device != self.device:
            self.device = tensor.device
            for attr in ['q_filtered', 'q_filtered_prev', 'omega_smooth', 
                         'q_last_good', 'frames_in_spike', 'frames_since_spike',
                         'q_raw_prev', 'omega_raw_prev', 'alpha_raw_prev', 
                         'motion_states', 'spike_counters', 'joint_threshold_scales']:
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.to(self.device))
    
    def update_params(
        self, 
        jerk_threshold: float,
        accel_limit: float,
        velocity_threshold: float,
        min_responsiveness: float,
        max_responsiveness: float,
        spike_recovery_frames: int,
        glitch_joint_threshold: float,
        global_glitch_transition_frames: int,
        dt: float
    ):
        self.dt = dt
        self.jerk_threshold = jerk_threshold
        self.accel_limit = accel_limit
        self.velocity_threshold = velocity_threshold
        self.min_responsiveness = min_responsiveness
        self.max_responsiveness = max_responsiveness
        self.spike_recovery_frames = spike_recovery_frames
        self.glitch_joint_threshold = glitch_joint_threshold
        self.global_glitch_transition_frames = global_glitch_transition_frames
    
    def reset(self):
        self.q_filtered[:, 0] = 1.0
        self.q_filtered[:, 1:] = 0.0
        self.q_filtered_prev[:, 0] = 1.0
        self.q_filtered_prev[:, 1:] = 0.0
        self.omega_smooth.zero_()
        self.q_last_good[:, 0] = 1.0
        self.q_last_good[:, 1:] = 0.0
        self.frames_in_spike.zero_()
        self.frames_since_spike.zero_()
        self.q_raw_prev[:, 0] = 1.0
        self.q_raw_prev[:, 1:] = 0.0
        self.omega_raw_prev.zero_()
        self.alpha_raw_prev.zero_()
        self.motion_states.zero_()
        self.spike_counters.fill_(self.spike_recovery_frames)
        self.global_glitch_mode = False
        self.global_glitch_frames = 0
        self._warmup_frames = 0
        self._initialized = False
    
    def filter(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Filter a frame of quaternions.
        
        During spikes: Extrapolate from LAST GOOD POSE using SMOOTHED VELOCITY
        During normal: Track input with high responsiveness
        """
        quaternions = F.normalize(quaternions, p=2.0, dim=-1)
        
        # Initialize on first frame
        if not self._initialized:
            self.q_filtered = quaternions.clone()
            self.q_filtered_prev = quaternions.clone()
            self.q_last_good = quaternions.clone()
            self.q_raw_prev = quaternions.clone()
            self._warmup_frames = 0
            self._initialized = True
            return quaternions
        
        dt = self.dt
        
        # Compute RAW input derivatives (for motion classification)
        omega_raw = jerk_quat_to_angular_velocity(self.q_raw_prev, quaternions, dt)
        
        self._warmup_frames += 1
        if self._warmup_frames < 3:
            # During warmup, track with max responsiveness while building history
            q_out = jerk_slerp(self.q_filtered, quaternions, 
                              torch.full((quaternions.shape[0],), self.max_responsiveness, 
                                         device=quaternions.device, dtype=quaternions.dtype))
            
            alpha_raw = (omega_raw - self.omega_raw_prev) / dt
            
            # Update smoothed velocity
            omega_inst = jerk_quat_to_angular_velocity(self.q_filtered_prev, q_out, dt)
            self.omega_smooth = self.velocity_smooth_factor * omega_inst + (1 - self.velocity_smooth_factor) * self.omega_smooth
            
            # Update state
            self.q_filtered_prev = self.q_filtered.clone()
            self.q_filtered = q_out
            self.q_last_good = q_out.clone()
            
            self.q_raw_prev = quaternions.clone()
            self.omega_raw_prev = omega_raw
            self.alpha_raw_prev = alpha_raw
            
            return q_out
        
        # Compute acceleration and jerk from RAW input sequence
        alpha_raw = (omega_raw - self.omega_raw_prev) / dt
        jerk_raw = (alpha_raw - self.alpha_raw_prev) / dt
        
        jerk_mag = torch.norm(jerk_raw, dim=-1)
        accel_mag = torch.norm(alpha_raw, dim=-1)
        
        # IMPORTANT: Use SMOOTHED velocity (trend from before this frame) for classification
        # This prevents a glitch from "voting for itself" as dynamic motion
        # The glitch will see the pre-glitch velocity trend, not its own corrupted velocity
        omega_mag_for_classification = torch.norm(self.omega_smooth, dim=-1)
        
        # Scale thresholds based on framerate
        # Reference: 60fps (dt = 1/60 ≈ 0.0167s)
        # Jerk scales with 1/dt³, accel scales with 1/dt²
        reference_dt = 1.0 / 60.0
        dt_ratio = reference_dt / dt
        # Apply per-joint scale factors (from SMPL hierarchy)
        # Proximal joints get lower threshold (stricter), distal get higher (more permissive)
        jerk_threshold_scaled = self.jerk_threshold * (dt_ratio ** 3) * self.joint_threshold_scales
        accel_limit_scaled = self.accel_limit * (dt_ratio ** 2)
        
        # Classify motion using SMOOTHED velocity for hysteresis
        motion_states, spike_counters = jerk_classify_motion(
            jerk_mag, accel_mag, omega_mag_for_classification, alpha_raw, self.alpha_raw_prev,
            self.motion_states, self.spike_counters,
            jerk_threshold_scaled, accel_limit_scaled, self.velocity_threshold,
            self.spike_recovery_frames
        )
        
        # Determine spike status
        NOISE_SPIKE = 2
        is_spike = (motion_states == NOISE_SPIKE)
        was_spike = (self.motion_states == NOISE_SPIKE)
        
        # === GLOBAL MOCAP GLITCH DETECTION ===
        # Detect when many joints spike simultaneously at low velocity
        # This indicates a mocap recalibration event, not noise to reject
        spike_fraction = is_spike.float().mean().item()
        max_velocity = omega_mag_for_classification.max().item()
        
        # Enter global glitch mode if:
        # 1. >30% of joints are spiking
        # 2. Maximum velocity is low (not during rapid motion)
        # 3. We're not already in global glitch mode
        entering_global_glitch = (
            spike_fraction >= self.glitch_joint_threshold and 
            max_velocity < self.velocity_threshold and
            not self.global_glitch_mode
        )
        
        # Also check RAW input velocity for detecting real motion during glitch
        raw_omega_mag = torch.norm(omega_raw, dim=-1)
        max_raw_velocity = raw_omega_mag.max().item()
        
        if entering_global_glitch:
            self.global_glitch_mode = True
            self.global_glitch_frames = 1
        elif self.global_glitch_mode:
            self.global_glitch_frames += 1
            
            # ADAPTIVE EXIT: If real motion detected, exit glitch mode early
            # Check RAW input velocity - if it exceeds threshold, real motion is happening
            # The velocity-limited blend will still prevent jerk during this transition
            real_motion_detected = max_raw_velocity > self.velocity_threshold * 0.5
            
            # Exit global glitch mode when:
            # 1. Transition is complete, OR
            # 2. Real motion is detected (velocity increased)
            if self.global_glitch_frames > self.global_glitch_transition_frames or real_motion_detected:
                self.global_glitch_mode = False
                self.global_glitch_frames = 0
        
        # In global glitch mode, override individual spike detection
        # Use smooth velocity-limited transition to input (the new ground truth)
        # rather than prediction from pre-glitch pose
        if self.global_glitch_mode:
            # All joints follow smooth transition path, not spike/predict path
            is_spike = torch.zeros_like(is_spike)
            # Mark all joints as "in transition" for velocity limiting
        
        # Track frames in spike and last good pose (only for individual spikes, not global)
        entering_spike = is_spike & ~was_spike
        in_spike = is_spike
        exiting_spike = ~is_spike & was_spike
        
        # When entering spike, save current filtered pose as last good
        self.q_last_good = torch.where(entering_spike.unsqueeze(-1), 
                                        self.q_filtered, self.q_last_good)
        
        # Increment frames_in_spike during spike, reset otherwise
        self.frames_in_spike = torch.where(in_spike, self.frames_in_spike + 1, 
                                           torch.zeros_like(self.frames_in_spike))
        
        # === SPIKE PATH: Predict from last good pose using smoothed velocity ===
        # Extrapolate: q_predicted = q_last_good * delta_q(omega_smooth * dt * frames)
        total_rotation = self.omega_smooth * dt * self.frames_in_spike.unsqueeze(-1).float()
        angle = torch.norm(total_rotation, dim=-1, keepdim=True)
        axis = F.normalize(total_rotation, p=2.0, dim=-1, eps=1e-9)
        half_angle = angle * 0.5
        w_pred = torch.cos(half_angle)
        xyz_pred = axis * torch.sin(half_angle)
        delta_q = torch.cat([w_pred, xyz_pred], dim=-1)
        q_predicted = F.normalize(jerk_quat_multiply(self.q_last_good, delta_q), p=2.0, dim=-1)
        
        # === NORMAL PATH: Track input ===
        blend = torch.full((quaternions.shape[0],), self.max_responsiveness,
                          device=quaternions.device, dtype=quaternions.dtype)
        
        # In dynamic motion mode, track with high responsiveness
        DYNAMIC_MOTION = 1
        is_dynamic = (motion_states == DYNAMIC_MOTION)
        # Dynamic motion uses same high responsiveness as normal
        # (no need for moderate blend since we're tracking real motion)
        
        q_tracked = jerk_slerp(self.q_filtered, quaternions, blend)
        
        # === GRADUAL RECOVERY PATH ===
        # Track frames since spike ended for gradual blend-back
        just_exited_spike = exiting_spike
        in_recovery = (self.frames_since_spike > 0) & (self.frames_since_spike <= self.spike_recovery_frames)
        
        # Update recovery counter
        post_recovery_frames = 3  # Must match value used later
        total_recovery_frames = self.spike_recovery_frames + post_recovery_frames
        
        self.frames_since_spike = torch.where(just_exited_spike, 
                                              torch.ones_like(self.frames_since_spike),
                                              self.frames_since_spike)
        in_any_recovery = (self.frames_since_spike > 0) & (self.frames_since_spike <= total_recovery_frames)
        self.frames_since_spike = torch.where(in_any_recovery & ~just_exited_spike,
                                              self.frames_since_spike + 1,
                                              self.frames_since_spike)
        self.frames_since_spike = torch.where(self.frames_since_spike > total_recovery_frames,
                                              torch.zeros_like(self.frames_since_spike),
                                              self.frames_since_spike)
        # Reset if entering new spike
        self.frames_since_spike = torch.where(entering_spike,
                                              torch.zeros_like(self.frames_since_spike),
                                              self.frames_since_spike)
        
        # Gradual recovery blend: starts at 20%, increases to max over recovery_frames
        # This prevents sudden jerk when transitioning from prediction back to tracking
        recovery_progress = self.frames_since_spike.float() / max(self.spike_recovery_frames, 1)
        base_recovery_blend = 0.2 + (self.max_responsiveness - 0.2) * recovery_progress
        
        # ACCELERATED RECOVERY: If filtered pose is close to raw input, we can recover faster
        # Compute angle difference between current filtered pose and raw input
        dot = torch.abs(torch.sum(self.q_filtered * quaternions, dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)
        angle_diff = 2.0 * torch.acos(dot)  # radians
        angle_diff_degrees = angle_diff * (180.0 / 3.14159265)
        
        # If angle diff < 5 degrees, accelerate recovery
        close_threshold = 5.0  # degrees
        closeness = torch.clamp(1.0 - angle_diff_degrees / close_threshold, 0.0, 1.0)
        boost_factor = 1.0 + 2.0 * closeness  # 1.0 to 3.0
        
        # Apply boost to recovery blend (capped at max_responsiveness)
        recovery_blend = torch.clamp(base_recovery_blend * boost_factor, 0.2, self.max_responsiveness)
        
        # VELOCITY-LIMITED RECOVERY: Limit how fast we can move during recovery
        # This prevents jerk spikes in the filtered output
        # Max angular velocity during recovery = 1.2x the smoothed velocity trend
        omega_smooth_mag = torch.norm(self.omega_smooth, dim=-1)
        max_recovery_velocity = torch.clamp(omega_smooth_mag * 1.2, min=1.5)  # rad/s, min 1.5
        max_angle_per_frame = max_recovery_velocity * dt  # radians
        max_angle_degrees = max_angle_per_frame * (180.0 / 3.14159265)
        
        # Limit the blend so we don't exceed max velocity
        velocity_limit_blend = torch.where(
            angle_diff_degrees > 0.01,
            torch.clamp(max_angle_degrees / angle_diff_degrees, 0.0, 1.0),
            torch.ones_like(angle_diff_degrees)
        )
        
        # Apply velocity limit during recovery
        in_recovery_now = (self.frames_since_spike > 0) & (self.frames_since_spike <= self.spike_recovery_frames)
        
        # EXTENDED VELOCITY LIMIT: Also apply a decaying velocity limit for a few frames after recovery
        # This prevents jerk when transitioning from recovery back to normal
        frames_since_recovery_ended = torch.clamp(
            self.frames_since_spike - self.spike_recovery_frames, 
            min=0
        )
        post_recovery_frames = 3  # Additional frames of gradual transition
        in_post_recovery = (self.frames_since_spike > self.spike_recovery_frames) & \
                           (self.frames_since_spike <= self.spike_recovery_frames + post_recovery_frames)
        
        # Blend between velocity-limited and normal as we exit post-recovery
        post_recovery_progress = frames_since_recovery_ended.float() / post_recovery_frames
        post_recovery_limit = velocity_limit_blend + (1.0 - velocity_limit_blend) * post_recovery_progress
        
        # Combine limits
        effective_blend = torch.where(
            in_recovery_now, 
            torch.minimum(recovery_blend, velocity_limit_blend),
            blend
        )
        effective_blend = torch.where(
            in_post_recovery,
            torch.minimum(blend, post_recovery_limit),
            effective_blend
        )
        
        # For recovery/post-recovery, blend from CURRENT filtered toward input with limited velocity
        q_recovery = jerk_slerp(self.q_filtered, quaternions, effective_blend)
        
        # === GLOBAL GLITCH MODE: Smooth velocity-limited transition to new position ===
        # During global glitch, all joints transition to input with velocity limiting
        # This treats the new position as ground truth, not noise
        if self.global_glitch_mode:
            # Progress through global glitch transition
            glitch_progress = self.global_glitch_frames / max(self.global_glitch_transition_frames, 1)
            # Start slow (0.1), increase to max over transition frames
            glitch_blend = 0.1 + (self.max_responsiveness - 0.1) * glitch_progress
            # Apply velocity limiting to global glitch transition
            glitch_effective_blend = torch.minimum(
                torch.full_like(velocity_limit_blend, glitch_blend),
                velocity_limit_blend
            )
            q_glitch_transition = jerk_slerp(self.q_filtered, quaternions, glitch_effective_blend)
            # In global glitch mode, use glitch transition for ALL joints
            in_global_glitch = torch.ones_like(in_spike)
        else:
            in_global_glitch = torch.zeros_like(in_spike)
            q_glitch_transition = q_tracked  # Placeholder, won't be used
        
        # Choose output based on motion state
        in_transition = in_recovery_now | in_post_recovery
        q_out = torch.where(in_spike.unsqueeze(-1), q_predicted, q_tracked)
        q_out = torch.where(in_transition.unsqueeze(-1), q_recovery, q_out)
        # Global glitch overrides everything
        q_out = torch.where(in_global_glitch.unsqueeze(-1), q_glitch_transition, q_out)
        q_out = F.normalize(q_out, p=2.0, dim=-1)
        
        # Update smoothed velocity (only during normal tracking)
        omega_inst = jerk_quat_to_angular_velocity(self.q_filtered_prev, q_out, dt)
        update_smooth = ~in_spike
        self.omega_smooth = torch.where(
            update_smooth.unsqueeze(-1),
            self.velocity_smooth_factor * omega_inst + (1 - self.velocity_smooth_factor) * self.omega_smooth,
            self.omega_smooth
        )
        
        # Update filtered trajectory
        self.q_filtered_prev = self.q_filtered.clone()
        self.q_filtered = q_out
        
        # Update last good pose when not in spike
        self.q_last_good = torch.where((~in_spike).unsqueeze(-1), q_out, self.q_last_good)
        
        # Update raw input history
        self.q_raw_prev = quaternions.clone()
        self.omega_raw_prev = omega_raw
        self.alpha_raw_prev = alpha_raw
        
        # Update motion state
        self.motion_states = motion_states
        self.spike_counters = spike_counters
        
        return q_out


class TorchJerkAwareQuatKFNode(TorchDeviceDtypeNode):
    """
    Node for jerk-aware adaptive quaternion filtering.
    
    Distinguishes between noise spikes and genuine rapid movements
    by analyzing temporal coherence of angular jerk.
    """
    
    @staticmethod
    def factory(name, data, args=None):
        node = TorchJerkAwareQuatKFNode(name, data, args)
        return node
    
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
        self.input_port = self.add_input('input', triggers_execution=True)
        self.dt_input = self.add_input('dt', widget_type='drag_float', default_value=1.0/60.0)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)
        
        self.add_property("--- Jerk Detection ---", widget_type='label')
        self.jerk_threshold_prop = self.add_property(
            'jerk_threshold (rad/s³)', 
            widget_type='drag_float', 
            default_value=2000.0, 
            min=10.0, 
            max=10000.0
        )
        self.jerk_threshold_prop.widget.speed = 10.0
        
        self.accel_limit_prop = self.add_property(
            'accel_limit (rad/s²)', 
            widget_type='drag_float', 
            default_value=50.0, 
            min=1.0, 
            max=500.0
        )
        
        self.velocity_threshold_prop = self.add_property(
            'velocity_threshold (rad/s)', 
            widget_type='drag_float', 
            default_value=5.0, 
            min=0.5, 
            max=50.0
        )
        self.velocity_threshold_prop.widget.speed = 0.1
        
        self.add_property("--- Responsiveness ---", widget_type='label')
        self.min_resp_prop = self.add_property(
            'min_responsiveness', 
            widget_type='drag_float', 
            default_value=0.1, 
            min=0.01, 
            max=0.5
        )
        self.min_resp_prop.widget.speed = 0.01
        
        self.max_resp_prop = self.add_property(
            'max_responsiveness', 
            widget_type='drag_float', 
            default_value=0.95, 
            min=0.5, 
            max=1.0
        )
        self.max_resp_prop.widget.speed = 0.01
        
        self.spike_recovery_prop = self.add_property(
            'spike_recovery_frames', 
            widget_type='input_int', 
            default_value=3, 
            min=1, 
            max=10
        )
        
        self.add_property("--- Global Glitch Detection ---", widget_type='label')
        self.glitch_threshold_prop = self.add_property(
            'glitch_joint_threshold', 
            widget_type='drag_float', 
            default_value=0.3,  # 30% of joints
            min=0.1, 
            max=0.8
        )
        self.glitch_threshold_prop.widget.speed = 0.01
        
        self.glitch_transition_prop = self.add_property(
            'glitch_transition_frames', 
            widget_type='input_int', 
            default_value=5, 
            min=2, 
            max=20
        )
        
        self.output = self.add_output('filtered')
        self.filter = None
    
    def reset_filter(self):
        if self.filter:
            self.filter.reset()
    
    def execute(self):
        input_tensor = self.input_port()
        if input_tensor is None:
            return
        
        input_tensor = any_to_tensor(input_tensor, device=self.device, dtype=self.dtype)
        
        # Handle 1D input (single quaternion)
        if input_tensor.dim() == 1 and input_tensor.shape[0] == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Validate shape
        if input_tensor.dim() != 2 or input_tensor.shape[-1] != 4:
            return
        
        num_streams = input_tensor.shape[0]
        dt = self.dt_input()
        if dt <= 0:
            dt = 1.0 / 60.0
        
        # Create or resize filter
        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchJerkAwareQuatKF(dt, num_streams, device=self.device, dtype=self.dtype)
            self.filter.reset()
        
        # Update device if needed
        if self.filter.device != self.device:
            self.filter.update_device_to_match(input_tensor)
        
        # Update parameters
        self.filter.update_params(
            jerk_threshold=self.jerk_threshold_prop(),
            accel_limit=self.accel_limit_prop(),
            velocity_threshold=self.velocity_threshold_prop(),
            min_responsiveness=self.min_resp_prop(),
            max_responsiveness=self.max_resp_prop(),
            spike_recovery_frames=self.spike_recovery_prop(),
            glitch_joint_threshold=self.glitch_threshold_prop(),
            global_glitch_transition_frames=self.glitch_transition_prop(),
            dt=dt
        )
        
        # Filter
        filtered = self.filter.filter(input_tensor)
        
        self.output.send(filtered)
