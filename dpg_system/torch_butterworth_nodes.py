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
    Node.app.register_node("t.quat_UKF", TorchTristateBlendUKFNode.factory)

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


# --- Quaternion helper functions (these are correct and unchanged) ---
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1);
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)


def angular_velocity_to_delta_q(av, dt):
    angle = torch.linalg.norm(av, dim=-1)
    axis = torch.where(angle.unsqueeze(-1) > 1e-9, av / angle.unsqueeze(-1),
                       torch.tensor([0.0, 0.0, 1.0], device=av.device))
    angle_dt_half = angle * dt * 0.5
    w = torch.cos(angle_dt_half);
    sin_part = torch.sin(angle_dt_half)
    x = axis[..., 0] * sin_part;
    y = axis[..., 1] * sin_part;
    z = axis[..., 2] * sin_part
    return torch.stack((w, x, y, z), -1)


def cholesky_mps_safe(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    A wrapper for torch.linalg.cholesky that works with MPS tensors
    by temporarily moving the computation to the CPU.
    """
    # Check if the tensor is on an MPS device and if the operation is not implemented.
    # This check makes the function robust. If a future PyTorch version
    # implements cholesky for MPS, this `if` block will be skipped.
    if A.device.type == 'mps':
        # Move the tensor to CPU
        A_cpu = A.to('cpu')

        # Perform Cholesky decomposition on CPU
        L_cpu = torch.linalg.cholesky(A_cpu, upper=upper)

        # Move the result back to the original MPS device
        return L_cpu.to(A.device)
    else:
        # If not on MPS, or if the operation is implemented, use the standard function
        return torch.linalg.cholesky(A, upper=upper)


class TorchQuaternionUKF:
    """
    A highly optimized, fully vectorized UKF for parallel quaternion streams.
    All Python loops in the core algorithm have been removed for performance.
    """

    def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
        self.dim_x = 7
        self.dim_z = 4
        self.dt = dt
        self.num_streams = num_streams
        self.device = device
        self.dtype = dtype
        self.num_sigmas = 2 * self.dim_x + 1

        alpha, beta, kappa = 0.1, 2.0, 0.0
        self.lambda_ = alpha ** 2 * (self.dim_x + kappa) - self.dim_x
        self.W_m = torch.full((self.num_sigmas,), 0.5 / (self.dim_x + self.lambda_), device=device, dtype=dtype)
        self.W_c = self.W_m.clone()
        self.W_m[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.W_c[0] = self.W_m[0] + (1 - alpha ** 2 + beta)

        self.x = torch.zeros(num_streams, self.dim_x, device=device, dtype=dtype);
        self.x[:, 0] = 1.0
        self.P = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
        self.Q = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.01
        self.R = torch.eye(self.dim_z, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1

    def update_device_to_match(self, input_tensor):
        if input_tensor.device != self.device:
            print('changing device to', input_tensor.device)
            self.device = input_tensor.device
            self.W_m = self.W_m.to(self.device)
            self.W_c = self.W_m.to(self.device)
            self.x = self.x.to(self.device)
            self.P = self.P.to(self.device)
            self.Q = self.Q.to(self.device)
            self.R = self.R.to(self.device)

    def set_noise_params(self, meas_noise_vec, drift_noise_vec, vel_change_noise_vec, dt):
        self.dt = dt
        R_diagonals = meas_noise_vec.unsqueeze(1).repeat(1, self.dim_z) ** 2
        self.R = torch.diag_embed(R_diagonals)
        Q_diagonals = torch.cat(
            [drift_noise_vec.unsqueeze(1).repeat(1, 4), vel_change_noise_vec.unsqueeze(1).repeat(1, 3)], dim=1) ** 2
        self.Q = torch.diag_embed(Q_diagonals)

    def _generate_sigma_points(self):
        n = self.dim_x
        P_stable = self.P + torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-6
        try:
            P_sqrt = cholesky_mps_safe(P_stable * (n + self.lambda_))
        except torch._C._LinAlgError:
            P_sqrt = torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 0.1

        # --- OPTIMIZATION: Vectorized sigma point generation ---
        # Before: Python for loop. After: Batched tensor addition/subtraction.
        sigmas = torch.zeros(self.num_streams, self.num_sigmas, n, device=self.device, dtype=self.dtype)
        sigmas[:, 0, :] = self.x
        x_unsqueeze = self.x.unsqueeze(1)
        sigmas[:, 1:n + 1, :] = x_unsqueeze + P_sqrt
        sigmas[:, n + 1:, :] = x_unsqueeze - P_sqrt
        return sigmas

    def f_process_model(self, x_batch):
        quat, ang_vel = x_batch[:, :4], x_batch[:, 4:7]
        delta_q = angular_velocity_to_delta_q(ang_vel, self.dt)
        quat_new = F.normalize(q_mult(quat, delta_q), p=2, dim=-1)
        return torch.cat([quat_new, ang_vel], dim=-1)

    def h_measurement_model(self, x_batch):
        return x_batch[:, :4]

    def predict(self):
        sigma_points = self._generate_sigma_points()

        # --- OPTIMIZATION: Vectorized propagation ---
        # Before: List comprehension and torch.stack. After: Reshape and one function call.
        # sigma_points shape: [num_streams, 15, 7]
        s_flat = sigma_points.view(-1, self.dim_x)  # -> [num_streams * 15, 7]
        sigma_points_pred_flat = self.f_process_model(s_flat)
        sigma_points_pred = sigma_points_pred_flat.view(self.num_streams, self.num_sigmas, self.dim_x)

        self.x = torch.einsum('i,sic->sc', self.W_m, sigma_points_pred)
        self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)
        y = sigma_points_pred - self.x.unsqueeze(1)
        self.P = torch.einsum('i,sic,sid->scd', self.W_c, y, y) + self.Q

    def update(self, z):
        sigma_points = self._generate_sigma_points()

        # --- OPTIMIZATION: Vectorized propagation ---
        s_flat = sigma_points.view(-1, self.dim_x)
        sigma_points_z_flat = self.h_measurement_model(s_flat)
        sigma_points_z = sigma_points_z_flat.view(self.num_streams, self.num_sigmas, self.dim_z)

        z_pred = torch.einsum('i,sic->sc', self.W_m, sigma_points_z)
        z_pred = F.normalize(z_pred, p=2, dim=-1)
        y = sigma_points_z - z_pred.unsqueeze(1)

        # Add jitter to S for stability before inverse
        S = torch.einsum('i,sic,sid->scd', self.W_c, y, y)
        S += self.R
        S_inv = torch.inverse(S + torch.eye(self.dim_z, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-9)

        T = torch.einsum('i,sic,sid->scd', self.W_c, sigma_points - self.x.unsqueeze(1), y)
        K = torch.einsum('scd,sdj->scj', T, S_inv)

        residual = z - z_pred
        self.x += torch.einsum('scd,sd->sc', K, residual)

        P_update = K @ S @ K.transpose(-1, -2)
        self.P -= P_update
        self.P = (self.P + self.P.transpose(-1, -2)) * 0.5
        self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)

    def reset(self):
        self.x.zero_();
        self.x[:, 0] = 1.0
        self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * 0.1

# --- The Complete and Correct Filter Class ---
# class TorchQuaternionUKF:
#     """
#     The full UKF implementation with all methods restored and correct dt handling.
#     """
#
#     def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
#         self.dim_x = 7
#         self.dim_z = 4
#         self.dt = dt
#         self.num_streams = num_streams
#         self.device = device
#         self.dtype = dtype
#         alpha, beta, kappa = 0.1, 2.0, 0.0
#         self.lambda_ = alpha ** 2 * (self.dim_x + kappa) - self.dim_x
#         self.W_m = torch.full((2 * self.dim_x + 1,), 0.5 / (self.dim_x + self.lambda_), device=device, dtype=dtype)
#         self.W_c = self.W_m.clone();
#         self.W_m[0] = self.lambda_ / (self.dim_x + self.lambda_)
#         self.W_c[0] = self.W_m[0] + (1 - alpha ** 2 + beta)
#         self.x = torch.zeros(num_streams, self.dim_x, device=device, dtype=dtype)
#         self.x[:, 0] = 1.0
#         self.P = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
#         # Q and R are now initialized as batched matrices
#         self.Q = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.01
#         self.R = torch.eye(self.dim_z, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
#
#     def set_noise_params(self, meas_noise_vec, drift_noise_vec, vel_change_noise_vec, dt):
#         """
#         UPDATED to accept per-stream noise vectors and build batched Q and R matrices.
#         """
#         self.dt = dt
#
#         # Build a batched R matrix. Shape: [num_streams, 4, 4]
#         # meas_noise_vec has shape [num_streams]
#         R_diagonals = meas_noise_vec.unsqueeze(1).repeat(1, self.dim_z) ** 2
#         self.R = torch.diag_embed(R_diagonals)
#
#         # Build a batched Q matrix. Shape: [num_streams, 7, 7]
#         Q_diagonals = torch.cat([
#             drift_noise_vec.unsqueeze(1).repeat(1, 4),
#             vel_change_noise_vec.unsqueeze(1).repeat(1, 3)
#         ], dim=1) ** 2
#         self.Q = torch.diag_embed(Q_diagonals)
#
#     def _generate_sigma_points(self):
#         n = self.dim_x
#         P_stable = self.P + torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-6
#         try:
#             P_sqrt = torch.linalg.cholesky(P_stable * (n + self.lambda_))
#         except torch._C._LinAlgError:
#             P_sqrt = torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 0.1
#
#         sigmas = torch.zeros(self.num_streams, 2 * n + 1, n, device=self.device, dtype=self.dtype)
#         sigmas[:, 0, :] = self.x
#         for i in range(n):
#             sigmas[:, i + 1, :] = self.x + P_sqrt[:, :, i]
#             sigmas[:, n + i + 1, :] = self.x - P_sqrt[:, :, i]
#         return sigmas
#
#     def f_process_model(self, x):
#         """ The kinematic model. This is where dt is correctly used. """
#         quat, ang_vel = x[:, :4], x[:, 4:7]
#         delta_q = angular_velocity_to_delta_q(ang_vel, self.dt)
#         quat_new = F.normalize(q_mult(quat, delta_q), p=2, dim=-1)
#         return torch.cat([quat_new, ang_vel], dim=-1)
#
#     def h_measurement_model(self, x):
#         """ Measurement model: we only measure the quaternion. """
#         return x[:, :4]
#
#     def predict(self):
#         """ Predicts the next state and covariance. """
#         sigma_points = self._generate_sigma_points()
#         sigma_points_pred = torch.stack(
#             [self.f_process_model(sigma_points[:, i, :]) for i in range(2 * self.dim_x + 1)], dim=1)
#
#         self.x = torch.einsum('i,sic->sc', self.W_m, sigma_points_pred)
#         self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)
#
#         y = sigma_points_pred - self.x.unsqueeze(1)
#         self.P = torch.einsum('i,sic,sid->scd', self.W_c, y, y) + self.Q
#
#     def update(self, z):
#         """ Updates the state and covariance based on a new measurement. """
#         sigma_points = self._generate_sigma_points()
#         sigma_points_z = torch.stack(
#             [self.h_measurement_model(sigma_points[:, i, :]) for i in range(2 * self.dim_x + 1)], dim=1)
#
#         z_pred = torch.einsum('i,sic->sc', self.W_m, sigma_points_z)
#         z_pred = F.normalize(z_pred, p=2, dim=-1)
#
#         y = sigma_points_z - z_pred.unsqueeze(1)
#         S = torch.einsum('i,sic,sid->scd', self.W_c, y, y) + self.R
#         T = torch.einsum('i,sic,sid->scd', self.W_c, sigma_points - self.x.unsqueeze(1), y)
#
#         S_inv = torch.inverse(S + torch.eye(self.dim_z, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-9)
#         K = torch.einsum('scd,sdj->scj', T, S_inv)
#
#         residual = z - z_pred
#
#         self.x += torch.einsum('scd,sd->sc', K, residual)
#
#         P_update = K @ S @ K.transpose(-1, -2)
#         self.P -= P_update
#
#         self.P = (self.P + self.P.transpose(-1, -2)) * 0.5
#         self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)
#
#     def reset(self):
#         """ Resets the filter's internal state. """
#         self.x.zero_()
#         self.x[:, 0] = 1.0
#         self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * 0.1
# class TorchQuaternionUKF:
#     """
#     A UKF specifically designed to filter a batch of parallel quaternion streams.
#     This version includes the fix for the P_update shape mismatch.
#     """
#
#     def __init__(self, dt, num_streams, device='cpu', dtype=torch.float32):
#         self.dim_x = 7
#         self.dim_z = 4
#         self.dt = dt
#         self.num_streams = num_streams
#         self.device = device
#         self.dtype = dtype
#
#         # --- UKF parameters (unchanged) ---
#         alpha, beta, kappa = 0.1, 2.0, 0.0
#         self.lambda_ = alpha ** 2 * (self.dim_x + kappa) - self.dim_x
#         self.W_m = torch.full((2 * self.dim_x + 1,), 0.5 / (self.dim_x + self.lambda_), device=device, dtype=dtype)
#         self.W_c = self.W_m.clone()
#         self.W_m[0] = self.lambda_ / (self.dim_x + self.lambda_)
#         self.W_c[0] = self.W_m[0] + (1 - alpha ** 2 + beta)
#
#         # --- State and Covariance (unchanged) ---
#         self.x = torch.zeros(num_streams, self.dim_x, device=device, dtype=dtype)
#         self.x[:, 0] = 1.0
#         self.P = torch.eye(self.dim_x, device=device, dtype=dtype).unsqueeze(0).repeat(num_streams, 1, 1) * 0.1
#         self.Q = torch.eye(self.dim_x, device=device, dtype=dtype) * 0.01
#         self.R = torch.eye(self.dim_z, device=device, dtype=dtype) * 0.1
#
#     def _generate_sigma_points(self):
#         n = self.dim_x
#         P_stable = self.P + torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-6
#         try:
#             P_sqrt = torch.linalg.cholesky(P_stable * (n + self.lambda_))
#         except torch._C._LinAlgError:
#             P_sqrt = torch.eye(n, device=self.device, dtype=self.dtype).unsqueeze(0) * 0.1
#
#         sigmas = torch.zeros(self.num_streams, 2 * n + 1, n, device=self.device, dtype=self.dtype)
#         sigmas[:, 0, :] = self.x
#         for i in range(n):
#             sigmas[:, i + 1, :] = self.x + P_sqrt[:, :, i]
#             sigmas[:, n + i + 1, :] = self.x - P_sqrt[:, :, i]
#         return sigmas
#
#     # --- f_process_model and h_measurement_model are unchanged ---
#     def f_process_model(self, x):
#         quat, ang_vel = x[:, :4], x[:, 4:7]
#         delta_q = angular_velocity_to_delta_q(ang_vel, self.dt)
#         quat_new = F.normalize(q_mult(quat, delta_q), p=2, dim=-1)
#         return torch.cat([quat_new, ang_vel], dim=-1)
#
#     def h_measurement_model(self, x):
#         return x[:, :4]
#
#     def predict(self):
#         sigma_points = self._generate_sigma_points()
#         sigma_points_pred = torch.stack(
#             [self.f_process_model(sigma_points[:, i, :]) for i in range(2 * self.dim_x + 1)], dim=1)
#         self.x = torch.einsum('i,sic->sc', self.W_m, sigma_points_pred)
#         self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)
#         y = sigma_points_pred - self.x.unsqueeze(1)
#         self.P = torch.einsum('i,sic,sid->scd', self.W_c, y, y) + self.Q
#
#     def update(self, z):
#         # --- THIS METHOD CONTAINS THE FIX ---
#         sigma_points = self._generate_sigma_points()
#         sigma_points_z = torch.stack(
#             [self.h_measurement_model(sigma_points[:, i, :]) for i in range(2 * self.dim_x + 1)], dim=1)
#
#         z_pred = torch.einsum('i,sic->sc', self.W_m, sigma_points_z)
#         z_pred = F.normalize(z_pred, p=2, dim=-1)
#
#         y = sigma_points_z - z_pred.unsqueeze(1)
#         S = torch.einsum('i,sic,sid->scd', self.W_c, y, y) + self.R
#         T = torch.einsum('i,sic,sid->scd', self.W_c, sigma_points - self.x.unsqueeze(1), y)
#
#         # S_inv can have stability issues, add jitter.
#         S_inv = torch.inverse(S + torch.eye(self.dim_z, device=self.device, dtype=self.dtype).unsqueeze(0) * 1e-9)
#         K = torch.einsum('scd,sdj->scj', T, S_inv)
#
#         residual = z - z_pred
#
#         self.x += torch.einsum('scd,sd->sc', K, residual)
#
#         # --- THE FIX IS HERE ---
#         # Replace the faulty einsum with clear, correct batch matrix multiplication.
#         P_update = K @ S @ K.transpose(-1, -2)
#         self.P -= P_update
#         # --- END OF FIX ---
#
#         self.P = (self.P + self.P.transpose(-1, -2)) * 0.5
#         self.x[:, :4] = F.normalize(self.x[:, :4], p=2, dim=-1)
#
#     def reset(self):
#         self.x.zero_();
#         self.x[:, 0] = 1.0
#         self.P = torch.eye(self.dim_x, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_streams, 1, 1) * 0.1
#
#         # In the TorchQuaternionUKF class:
#         # ... (all other methods are the same)
#
#     def set_noise_params(self, meas_noise, drift_noise, vel_change_noise, dt):
#         """
#         Updates the filter's noise matrices and time step.
#         Process noise (Q) is scaled by dt for frame rate independence.
#         """
#         self.dt = dt
#
#         # R is measurement noise, independent of dt.
#         self.R = torch.eye(4, device=self.device, dtype=self.dtype) * meas_noise ** 2
#
#         # Q is process noise, representing variance accumulated over time.
#         # We scale it by dt to make the filter's behavior consistent across frame rates.
#         Q_diag = torch.cat([
#             torch.full((4,), drift_noise ** 2, device=self.device, dtype=self.dtype),
#             torch.full((3,), vel_change_noise ** 2, device=self.device, dtype=self.dtype)
#         ])
#         self.Q = torch.diag(Q_diag) * self.dt  # Scale by dt

# --- Placeholder classes and helper functions from previous examples ---
# ... (TorchDeviceDtypeNode, MockWidget, etc.)


class TorchTristateBlendUKFNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchTristateBlendUKFNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.setup_dtype_device_grad(args)
        self.dtype = torch.float32
        self.filter = None
        self.last_input_quat = None

        self.motion_min = 3.0
        self.motion_max = 10.0
        self.error_min = 1.5
        self.error_max = 8
        self.damp_params = [0.2, 0.001, 0.005]
        self.resp_params = [0.05, 0.01, 0.1]
        self.err_params =  [0.01, 0.01, 0.02]

        # --- Inputs for the three 'vertices' of our parameter triangle ---
        self.quat_input = self.add_input("quaternions", triggers_execution=True)
        self.dt_input = self.add_input("dt (sec)", widget_type='drag_float', default_value=1.0 / 60.0, callback=self.update_params)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)

        self.add_property("--- Damping Mode ---", widget_type='label')
        self.damp_meas_in = self.add_input('Damp Meas Noise', widget_type='drag_float', default_value=self.damp_params[0], callback=self.update_params)
        self.damp_drift_in = self.add_input('Damp Drift Noise', widget_type='drag_float', default_value=self.damp_params[1], callback=self.update_params)
        self.damp_vel_in = self.add_input('Damp AngVel Noise', widget_type='drag_float', default_value=self.damp_params[2], callback=self.update_params)

        self.add_property("--- Responsive Mode ---", widget_type='label')
        self.resp_meas_in = self.add_input('Resp Meas Noise', widget_type='drag_float', default_value=self.resp_params[0], callback=self.update_params)
        self.resp_drift_in = self.add_input('Resp Drift Noise', widget_type='drag_float', default_value=self.resp_params[1], callback=self.update_params)
        self.resp_vel_in = self.add_input('Resp AngVel Noise', widget_type='drag_float', default_value=self.resp_params[2], callback=self.update_params)

        self.add_property("--- Error Correction Mode ---", widget_type='label')
        self.err_meas_in = self.add_input('Err Correct Meas Noise', widget_type='drag_float', default_value=self.err_params[0], callback=self.update_params)
        self.err_drift_in = self.add_input('Err Correct Drift Noise', widget_type='drag_float', default_value=self.err_params[1], callback=self.update_params)
        self.err_vel_in = self.add_input('Err Correct AngVel Noise', widget_type='drag_float', default_value=self.err_params[2], callback=self.update_params)

        self.add_property("--- Transition Ranges ---", widget_type='label')
        self.motion_min_in = self.add_input('Min Motion (deg/sec)', widget_type='drag_float', default_value=self.motion_min, callback=self.update_params)
        self.motion_max_in = self.add_input('Max Motion (deg/sec)', widget_type='drag_float', default_value=self.motion_max, callback=self.update_params)
        self.error_min_in = self.add_input('Min Error (deg)', widget_type='drag_float', default_value=self.error_min, callback=self.update_params)
        self.error_max_in = self.add_input('Max Error (deg)', widget_type='drag_float', default_value=self.error_max, callback=self.update_params)

        # Outputs
        self.output_port = self.add_output('filtered')
        self.alphas_output = self.add_output('alphas (damp, resp, err)')

    def reset_filter(self):
        if self.filter: self.filter.reset()
        self.last_input_quat = None

    def update_params(self):
        self.motion_min = self.motion_min_in()
        self.motion_max = self.motion_max_in()
        self.error_min = self.error_min_in()
        self.error_max = self.error_max_in()
        self.damp_params = [self.damp_meas_in(), self.damp_drift_in(), self.damp_vel_in()]
        self.resp_params = [self.resp_meas_in(), self.resp_drift_in(), self.resp_vel_in()]
        self.err_params = [self.err_meas_in(), self.err_drift_in(), self.err_vel_in()]

    def execute(self):
        signal_in = self.quat_input()
        if signal_in is None or signal_in.dim() != 2 or signal_in.shape[1] != 4:
            return

        dt = self.dt_input()
        if dt is None or dt <= 0:
            return
        num_streams = signal_in.shape[0]

        self.device = signal_in.device

        # Instantiate filter
        if self.filter is None or self.filter.num_streams != num_streams:
            self.filter = TorchQuaternionUKF(dt, num_streams, self.device, self.dtype)
            self.reset_filter()

        self.filter.update_device_to_match(signal_in)

        if self.last_input_quat is not None:
            self.last_input_quat = self.last_input_quat.to(device=self.device)

        # --- OPTIMIZATION: Fully vectorized per-stream alpha calculation ---

        # 1. Calculate per-stream strength_resp vector
        strength_resp_vec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.last_input_quat is not None:
            dot = torch.abs(torch.einsum('sc,sc->s', signal_in, self.last_input_quat))
            motion_rad_per_frame = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
            motion_deg_per_sec = torch.rad2deg(motion_rad_per_frame) / dt
            if self.motion_max > self.motion_min:
                strength_resp_vec = (motion_deg_per_sec - self.motion_min) / (self.motion_max - self.motion_min)

        # 2. Calculate per-stream strength_err vector
        if self.filter.x.device is not signal_in.device:
            self.filter.x = self.filter.x.to(signal_in.device)
        dot = torch.abs(torch.einsum('sc,sc->s', signal_in, self.filter.x[:, :4]))
        error_rad = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
        error_deg = torch.rad2deg(error_rad)

        strength_err_vec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
        if self.error_max > self.error_min:
            strength_err_vec = (error_deg - self.error_min) / (self.error_max - self.error_min)

        # Clamp strengths to [0, 1] range
        strength_resp_vec = torch.clamp(strength_resp_vec, 0.0, 1.0)
        strength_err_vec = torch.clamp(strength_err_vec, 0.0, 1.0)

        # 3. Calculate per-stream strength_damp vector
        strength_damp_vec = (1.0 - strength_resp_vec) * (1.0 - strength_err_vec)

        # 4. Normalize the strength vectors to get final alpha weight vectors
        total_strength = strength_damp_vec + strength_resp_vec + strength_err_vec + 1e-9
        alpha_damp_vec = strength_damp_vec / total_strength
        alpha_resp_vec = strength_resp_vec / total_strength
        alpha_err_vec = strength_err_vec / total_strength

        # 5. Blend parameters using the alpha vectors
        current_meas_vec = self.damp_params[0] * alpha_damp_vec + self.resp_params[0] * alpha_resp_vec + self.err_params[0] * alpha_err_vec
        current_drift_vec = self.damp_params[1] * alpha_damp_vec + self.resp_params[1] * alpha_resp_vec + self.err_params[1] * alpha_err_vec
        current_vel_vec = self.damp_params[2] * alpha_damp_vec + self.resp_params[2] * alpha_resp_vec + self.err_params[2] * alpha_err_vec

        # 6. Set filter parameters
        self.filter.set_noise_params(
            meas_noise_vec=current_meas_vec,
            drift_noise_vec=current_drift_vec,
            vel_change_noise_vec=current_vel_vec,
            dt=dt
        )

        # 7. Run UKF cycle and output
        self.filter.predict()
        self.filter.update(signal_in)

        self.output_port.send(self.filter.x[:, :4])
        self.alphas_output.send(torch.stack([alpha_damp_vec, alpha_resp_vec, alpha_err_vec], dim=1))
        self.last_input_quat = signal_in.clone()

# class TorchPerStreamAdaptiveUKFNode(TorchDeviceDtypeNode):
#     @staticmethod
#     def factory(name, data, args=None):
#         return TorchPerStreamAdaptiveUKFNode(name, data, args)
#
#     def __init__(self, label: str, data, args):
#         # The __init__ method is identical to the last version.
#         # It defines the scalar UI controls for the min/max of the blend.
#         super().__init__(label, data, args)
#         self.setup_dtype_device_grad(args)
#         self.dtype = torch.float32
#         self.filter = None
#         self.last_input_quat = None
#         self.quat_input = self.add_input("quaternions", triggers_execution=True)
#         self.dt_input = self.add_input("dt (sec)", widget_type='drag_float', default_value=1.0 / 60.0, min=0.001,
#                                        max=0.1)
#         self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_filter)
#         self.damp_meas_noise_in = self.add_input('Damping Meas Noise', widget_type='drag_float', default_value=0.2)
#         self.damp_drift_noise_in = self.add_input('Damping Drift Noise', widget_type='drag_float', default_value=0.001)
#         self.damp_vel_noise_in = self.add_input('Damping AngVel Noise', widget_type='drag_float', default_value=0.005)
#         self.resp_meas_noise_in = self.add_input('Responsive Meas Noise', widget_type='drag_float', default_value=0.05)
#         self.resp_drift_noise_in = self.add_input('Responsive Drift Noise', widget_type='drag_float',
#                                                   default_value=0.01)
#         self.resp_vel_noise_in = self.add_input('Responsive AngVel Noise', widget_type='drag_float', default_value=0.1)
#         self.motion_min_thresh_in = self.add_input('Min Motion (deg/sec)', widget_type='drag_float', default_value=3.0)
#         self.motion_max_thresh_in = self.add_input('Max Motion (deg/sec)', widget_type='drag_float', default_value=10.0)
#         self.error_min_thresh_in = self.add_input('Min Error (deg)', widget_type='drag_float', default_value=1.0)
#         self.error_max_thresh_in = self.add_input('Max Error (deg)', widget_type='drag_float', default_value=5.0)
#         self.output_port = self.add_output('filtered')
#         self.alpha_output = self.add_output('alpha per stream')
#
#     def reset_filter(self):
#         if self.filter: self.filter.reset()
#         self.last_input_quat = None
#
#     def execute(self):
#         # --- THIS METHOD CONTAINS THE NEW VECTORIZED LOGIC ---
#         signal_in = self.quat_input()
#         if signal_in is None or signal_in.dim() != 2 or signal_in.shape[1] != 4: return
#
#         num_streams = signal_in.shape[0]
#         dt = self.dt_input();
#         if dt is None or dt <= 0: return
#
#         # Get scalar parameter values
#         damp_meas = self.damp_meas_noise_in()
#         damp_drift = self.damp_drift_noise_in()
#         damp_vel = self.damp_vel_noise_in()
#         resp_meas = self.resp_meas_noise_in()
#         resp_drift = self.resp_drift_noise_in()
#         resp_vel = self.resp_vel_noise_in()
#         motion_min = self.motion_min_thresh_in()
#         motion_max = self.motion_max_thresh_in()
#         error_min = self.error_min_thresh_in()
#         error_max = self.error_max_thresh_in()
#
#         # Instantiate filter
#         if self.filter is None or self.filter.num_streams != num_streams:
#             self.filter = TorchQuaternionUKF(dt, num_streams, self.device, self.dtype)
#             self.reset_filter()
#
#         # 1. Calculate alpha_motion_vec
#         alpha_motion_vec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
#         if self.last_input_quat is not None:
#             dot = torch.abs(torch.einsum('sc,sc->s', signal_in, self.last_input_quat))
#             motion_rad_per_frame = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
#             motion_deg_per_sec = torch.rad2deg(motion_rad_per_frame) / dt
#             if motion_max > motion_min:
#                 alpha_motion_vec = (motion_deg_per_sec - motion_min) / (motion_max - motion_min)
#
#         # 2. Run predict()
#         self.filter.predict()
#
#         # 3. Calculate alpha_error_vec
#         predicted_quat = self.filter.x[:, :4]
#         dot = torch.abs(torch.einsum('sc,sc->s', signal_in, predicted_quat))
#         error_rad = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))
#         error_deg = torch.rad2deg(error_rad)
#
#         alpha_error_vec = torch.zeros(num_streams, device=self.device, dtype=self.dtype)
#         if error_max > error_min:
#             alpha_error_vec = (error_deg - error_min) / (error_max - error_min)
#
#         # 4. Combine and clamp to get the final alpha vector
#         final_alpha_vec = torch.max(alpha_motion_vec, alpha_error_vec)
#         final_alpha_vec = torch.clamp(final_alpha_vec, 0.0, 1.0)
#
#         # 5. Interpolate to get per-stream parameter vectors
#         current_meas_noise = damp_meas * (1 - final_alpha_vec) + resp_meas * final_alpha_vec
#         current_drift_noise = damp_drift * (1 - final_alpha_vec) + resp_drift * final_alpha_vec
#         current_vel_noise = damp_vel * (1 - final_alpha_vec) + resp_vel * final_alpha_vec
#
#         # 6. Set filter parameters with the new per-stream vectors
#         self.filter.set_noise_params(
#             meas_noise_vec=current_meas_noise,
#             drift_noise_vec=current_drift_noise,
#             vel_change_noise_vec=current_vel_noise,
#             dt=dt
#         )
#
#         # 7. Run update() and output
#         self.filter.update(signal_in)
#         self.output_port.send(self.filter.x[:, :4])
#         self.alpha_output.send(final_alpha_vec)
#         self.last_input_quat = signal_in.clone()