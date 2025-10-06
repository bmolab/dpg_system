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
        self.ptr = 0;
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


# --- Helper Functions (assuming these exist and are correct) ---
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)


def q_conjugate(q):
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def angular_velocity_to_delta_q(av, dt):
    angle = torch.linalg.norm(av, dim=-1) * dt
    # Use a small epsilon to prevent division by zero for zero angular velocity
    axis = F.normalize(av, p=2, dim=-1, eps=1e-9)
    angle_half = angle / 2.0
    w = torch.cos(angle_half)
    sin_angle_half = torch.sin(angle_half)
    xyz = axis * sin_angle_half.unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


# Assuming cholesky_mps_safe is defined elsewhere if needed, but it's not used by the EKF
# from your_utils import cholesky_mps_safe

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
        # Unused UKF parameters have been removed.

        self._reset_pending = False

    def update_device_to_match(self, input_tensor):
        if input_tensor.device != self.device:
            self.device = input_tensor.device
            for attr in ['q', 'av', 'x', 'P', 'Q', 'R']:  # Removed unused UKF attrs
                setattr(self, attr, getattr(self, attr).to(self.device))

    def set_noise_params(self, meas_noise_vec, vel_change_noise_vec, dt):
        """
        Sets the noise covariance matrices R and Q based on per-stream noise parameters.
        """
        self.dt = dt

        # R is [num_streams, 3, 3], representing the measurement uncertainty.
        # We start with a [num_streams] vector and expand it to [num_streams, 3].
        R_diagonals_3d = meas_noise_vec.unsqueeze(1).repeat(1, 3)
        self.R = torch.diag_embed(R_diagonals_3d ** 2)

        # Q is [num_streams, 6, 6], representing the process noise.
        # We need 3 diagonal elements for rotation error and 3 for velocity error.

        # Expand the [num_streams] velocity noise vector to [num_streams, 3]
        vel_noise_3d = vel_change_noise_vec.unsqueeze(1).repeat(1, 3)

        # The noise for the angular velocity part of the state
        vel_part = vel_noise_3d ** 2

        # The noise for the rotation part of the state, derived from velocity noise.
        # This models that larger velocity uncertainty also leads to larger position uncertainty.
        rot_part = (vel_noise_3d * self.dt) ** 2

        # Concatenate the two [num_streams, 3] tensors to get a [num_streams, 6] tensor.
        Q_diagonals = torch.cat([rot_part, vel_part], dim=1)
        self.Q = torch.diag_embed(Q_diagonals)

    def predict(self):
        if self._reset_pending:
            return
        # 1. Propagate the nominal state using the current best guess of angular velocity
        delta_q = angular_velocity_to_delta_q(self.av, self.dt)
        self.q = F.normalize(q_mult(self.q, delta_q), p=2, dim=-1)

        # 2. Propagate the error covariance using the linearized state transition matrix F
        # F = [[I, dt*I], [0, I]] -- NOTE: The original F was for a different state representation.
        # For state [err_rot, err_av], the correct linearization is:
        # err_rot_k = err_rot_{k-1} + dt * err_av_{k-1}
        # err_av_k = err_av_{k-1}
        # Let's stick with the user's original F, as it's also a valid model:
        # err_rot_k = err_rot_{k-1} - dt*skew(av)*err_av_{k-1}
        # This one is actually more correct as it handles the rotating frame.
        F_mat = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1)
        av_skew = skew_symmetric(self.av)
        F_mat[:, :3, 3:] = -self.dt * av_skew  # Correct for rotating frame

        # Propagate the covariance: P = F * P * F^T + Q
        self.P = F_mat @ self.P @ F_mat.transpose(-1, -2) + self.Q

    def update(self, z_measured_q):
        # Check if a reset is pending and use the current measurement for initialization
        if self._reset_pending:

            self.q = z_measured_q.to(self.device, self.dtype)
            self.av.zero_() # Reset angular velocity to zero
            self.x.zero_()   # Reset error state
            self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * 0.1 # Reset covariance

            self._reset_pending = False # Clear the flag
            # After initialization, we can optionally skip the rest of the update for this frame
            # to prevent a large initial correction if P is high and the "measurement" is already
            # the new nominal state. However, performing the update is generally safer to
            # immediately use the covariance matrix to refine the first measurement.
            # Let's proceed with the update as usual, as `q` is now `z_measured_q`, making residual zero (or very small).

        # 1. Calculate the measurement residual in quaternion form
        residual_q = q_mult(q_conjugate(self.q), z_measured_q)

        # 2. Convert the residual quaternion to a 3D rotation vector (the measurement residual z)
        angle_half = torch.acos(torch.clamp(residual_q[:, 0], -1.0, 1.0))
        sin_angle_half = torch.sin(angle_half)
        scale = torch.where(sin_angle_half.abs() > 1e-9, 2.0 * angle_half / sin_angle_half, 2.0)
        residual_v = scale.unsqueeze(-1) * residual_q[:, 1:]

        # 3. EKF Update using the linearized measurement model H = [I_3x3, 0_3x3]
        P_zz = self.P[:, :3, :3]
        S = P_zz + self.R

        # The cross-covariance P_xz = P * H^T is just the first 3 columns of P.
        P_xz = self.P[:, :, :3]

        # 4. Calculate Kalman Gain K = P_xz * S^-1
        # --- ROBUSTNESS IMPROVEMENT ---
        # Add a small 'jitter' to S to prevent numerical instability during inversion.
        jitter = torch.eye(self.dim_z, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-9
        S_inv = torch.inverse(S + jitter)
        K = P_xz @ S_inv

        # 5. Update the 6D error state `x` using the 3D measurement residual
        self.x = torch.einsum('scd,sd->sc', K, residual_v)

        # 6. Update the error covariance P using the numerically stable Joseph form
        I = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0)
        H = torch.zeros(self.num_streams, self.dim_x, self.dim_z, device=self.device, dtype=self.dtype)
        H[:, :3, :3] = torch.eye(self.dim_z, device=self.device, dtype=self.dtype)

        I_minus_KH = I - K @ H.transpose(-1, -2)
        P_update = I_minus_KH @ self.P @ I_minus_KH.transpose(-1, -2) + K @ self.R @ K.transpose(-1, -2)
        self.P = (P_update + P_update.transpose(-1, -2)) * 0.5  # Ensure symmetry

        # 7. Inject the estimated error into the nominal state
        error_rot_vec = self.x[:, :3]
        delta_q_correction = angular_velocity_to_delta_q(error_rot_vec, dt=1.0)
        self.q = F.normalize(q_mult(self.q, delta_q_correction), p=2, dim=-1)
        self.av += self.x[:, 3:]

        # 8. Reset the error state for the next cycle
        self.x.zero_()

    def reset(self):
        self.q.zero_()
        self.q[:, 0] = 1.0
        self.av.zero_()
        self.x.zero_()
        self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1,
                                                                                                 1) * 0.1
    def flag_reset(self):
        """
        Flags that a reset is needed. The next call to `update` will use its
        measurement to initialize the filter state.
        """
        print("Reset flagged. Next 'update' will re-initialize the filter.")
        self._reset_pending = True

    def reset_to_defaults(self):
        print("Performing immediate reset to default values.")
        self.q.zero_()
        self.q[:, 0] = 1.0
        self.av.zero_()
        self.x.zero_()
        self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1,
                                                                                                 1) * 0.1
        self._reset_pending = False # Ensure flag is off if doing a hard reset


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
        self.damp_params = [0.5, 0.1]
        self.resp_params = [0.05, 1.0]
        self.err_params = [0.005, 10.0]

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

    def reset_filter(self):
        if self.filter: self.filter.flag_reset()
        self.last_input_quat = None

        # Reset the blended alphas to the default "Damping" state
        if self.filter:
             self.blended_alphas = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.filter.num_streams, 1)

    def update_params(self):
        self.damp_params = [self.damp_meas_in(), self.damp_vel_in()]
        self.resp_params = [self.resp_meas_in(), self.resp_vel_in()]
        self.err_params = [self.err_meas_in(), self.err_vel_in()]
        self.motion_min = self.motion_min_in()
        self.motion_max = self.motion_max_in()
        self.error_min = self.error_min_in()
        self.error_max = self.error_max_in()

    def _calculate_angular_difference_deg(self, q1, q2):
        dot = torch.abs(torch.einsum('sc,sc->s', q1, q2))
        angle_rad = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
        return torch.rad2deg(angle_rad)

    def execute(self):
        signal_in = self.quat_input()
        if signal_in is None or signal_in.dim() != 2 or signal_in.shape[1] != 4:
            return

        dt = self.dt_input()
        if dt is None or dt <= 0: return

        num_streams = signal_in.shape[0]
        self.device = signal_in.device

        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchQuaternionESEKF(dt, num_streams, self.device, self.dtype)
            self.reset_filter()

        self.filter.update_device_to_match(signal_in)
        if self.last_input_quat is not None:
            self.last_input_quat = self.last_input_quat.to(self.device)
        self.blended_alphas = self.blended_alphas.to(self.device)

        self.filter.predict()

        # 1. Calculate the raw strength for each active mode (Responsive and Error Correct)
        motion_deg_per_sec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_quat is not None:
            motion_deg_per_frame = self._calculate_angular_difference_deg(signal_in, self.last_input_quat)
            motion_deg_per_sec = motion_deg_per_frame / dt

        strength_resp_vec = torch.clamp(
            (motion_deg_per_sec - self.motion_min) / (self.motion_max - self.motion_min + 1e-9), 0.0, 1.0)

        error_deg = self._calculate_angular_difference_deg(signal_in, self.filter.q)
        strength_err_vec = torch.clamp((error_deg - self.error_min) / (self.error_max - self.error_min + 1e-9), 0.0,
                                       1.0)

        # 2. Determine the overall "active" level by taking the max of the two strengths.
        # This decides how much we move away from the default Damping state.
        alpha_active = torch.maximum(strength_resp_vec, strength_err_vec)

        # 3. Distribute the `alpha_active` amount between Resp and Err based on their relative strengths.
        total_strength = strength_resp_vec + strength_err_vec + 1e-9  # Add epsilon to avoid div by zero

        target_alpha_resp = alpha_active * (strength_resp_vec / total_strength)
        target_alpha_err = alpha_active * (strength_err_vec / total_strength)

        # 4. Damping alpha is whatever is left over.
        target_alpha_damp = 1.0 - target_alpha_resp - target_alpha_err

        target_alphas = torch.stack([target_alpha_damp, target_alpha_resp, target_alpha_err], dim=1)

        # Smooth the alphas using LERP
        blending_speed = self.blending_speed_in()
        self.blended_alphas = torch.lerp(self.blended_alphas, target_alphas, blending_speed)

        # Use the smoothed alphas to set filter parameters
        alpha_damp_vec, alpha_resp_vec, alpha_err_vec = self.blended_alphas.unbind(dim=-1)

        damp = torch.tensor(self.damp_params, device=self.device, dtype=self.dtype)
        resp = torch.tensor(self.resp_params, device=self.device, dtype=self.dtype)
        err = torch.tensor(self.err_params, device=self.device, dtype=self.dtype)

        blended_params = (alpha_damp_vec.unsqueeze(1) * damp +
                          alpha_resp_vec.unsqueeze(1) * resp +
                          alpha_err_vec.unsqueeze(1) * err)

        self.filter.set_noise_params(
            meas_noise_vec=blended_params[:, 0],
            vel_change_noise_vec=blended_params[:, 1],
            dt=dt
        )

        # Run Update and send outputs
        self.filter.update(signal_in)
        self.output_port.send(self.filter.q)
        self.alphas_output.send(self.blended_alphas)
        self.last_input_quat = signal_in.clone()