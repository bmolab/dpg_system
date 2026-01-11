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
    # F = [[I, -dt*skew(av)], [0, I]]
    F_mat = F_base.clone()
    av_skew = skew_symmetric(av)
    
    # We can't use complex slicing in JIT easily with inplace mod sometimes
    # F_mat[:, :3, 3:] = -dt * av_skew
    # Let's try explicit indexing which is usually safe in JIT
    F_mat[:, 0:3, 3:6] = -dt * av_skew

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
        self.last_input_vec = None
        
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
        signal_in = self.input()
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

        self.filter.predict()

        # 1. Motion Calc
        motion_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_vec is not None:
            dist = torch.linalg.norm(signal_in - self.last_input_vec, dim=-1)
            motion_per_sec = dist / dt

        # 2. Error Calc
        error_dist = torch.linalg.norm(signal_in - self.filter.pos, dim=-1)
        
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
        
        self.filter.set_noise_params(
            meas_noise_vec=blended_params[:, 0],
            vel_change_noise_vec=blended_params[:, 1],
            dt=dt
        )
        
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
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

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
        signal_in = self.quat_input()
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
        self.filter.predict()

        # 1. Calculate the raw strength for each active mode.
        motion_deg_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_quat is not None:
            motion_deg_per_frame = self._calculate_angular_difference_deg(signal_in, self.last_input_quat)
            motion_deg_per_sec = motion_deg_per_frame / dt

        # Vectorized strength calc
        # clamp((val - min) / (max - min), 0, 1)
        # Pre-calculation of denominators could be done in update_params, but float div is fast enough.
        
        strength_resp_vec = torch.clamp(
            (motion_deg_per_sec - self.motion_min) / (self.motion_max - self.motion_min + 1e-9), 0.0, 1.0)

        error_deg = self._calculate_angular_difference_deg(signal_in, self.filter.q)
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
        
        self.filter.set_noise_params(
            meas_noise_vec=blended_params[:, 0],
            vel_change_noise_vec=blended_params[:, 1],
            dt=dt
        )

        # 5. Run Update
        self.filter.update(signal_in)
        
        self.output_port.send(self.filter.q)
        self.alphas_output.send(self.blended_alphas)
        self.last_input_quat = signal_in.clone()