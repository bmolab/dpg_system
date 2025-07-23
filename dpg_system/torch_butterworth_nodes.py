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

    def capture(self, input):
        input = any_to_tensor(input, device=self.device).flatten()
        if self.buffers is None or self.buffers.shape[3] != input.shape[0]:
            self.allocate_buffers(input.shape[0])
        self.buffers[:, :, :] = input


    def filter(self, input):
        input = any_to_tensor(input, device=self.device).flatten()
        if self.buffers is None or self.buffers.shape[3] != input.shape[0]:
            self.allocate_buffers(input.shape[0])

        # input should be shaped [width]
        if len(self.coefficients[0, :]) > 1:
            output = input.unsqueeze(0)  # shape = [1, width]
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
        return output.transpose(1, 0)   #  shape[num_bands, width]

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
