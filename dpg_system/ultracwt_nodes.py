from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import stft
import scipy.signal as signal
# from fast_cwt import fast_cwt
#from time import time
import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.matrix_nodes import RollingBuffer
from dpg_system.conversion_utils import *


def register_ultracwt_nodes():
    Node.app.register_node('ultracwt', NumpyUltraCWTNode.factory)


class NumpyUltraCWTNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUltraCWTNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.subframe_size = 1000
        self.widths = list(range(1, 100, 10))
        if len(args) > 0:
            self.subframe_size = any_to_int(args[0])
        if len(args) > 1:
            widths, widths_type = decode_arg(args, 1)
            if widths_type == list:
                self.widths = widths

        self.cwt = MyCWT(signal.morlet2, self.subframe_size, self.widths)
        self.input = self.add_input('in 1', triggers_execution=True)

        self.frame_size_widget = self.add_input('scales', widget_type='drag_int', default_value=self.subframe_size)
        self.frame_size_widget.add_callback(self.frame_size_changed)

        self.cwt_mode = self.add_property('mode', widget_type='combo', default_value='normal')
        self.cwt_mode.widget.combo_items = ['normal', 'half', 'unskewed']

        widths_string = any_to_string(self.widths)
        self.widths_property = self.add_input('scales', widget_type='text_input', default_value=widths_string)
        self.widths_property.add_callback(self.widths_changed)
        self.output = self.add_output('cwt out')

    def frame_size_changed(self, val=0):
        self.cwt = None
        self.subframe_size = self.frame_size_widget()
        self.cwt = MyCWT(signal.morlet2, self.subframe_size, self.widths)

    def widths_changed(self, val=0):
        self.cwt = None
        widths = self.widths_property()
        widths_list = re.findall(r'[-+]?\d+', widths)
        widths = []
        for dim_text in widths_list:
            widths.append(any_to_int(dim_text))
        self.widths = widths
        self.cwt = MyCWT(signal.morlet2, self.subframe_size, self.widths)

    def execute(self):
        input_value = any_to_array(self.input())
        if self.cwt != None:
            mode = self.cwt_mode()
            self.cwt.set_mode(mode)
            cwt_out = self.cwt.receive_new_data(input_value)
            self.output.send(cwt_out)


class MyCWT:
    def __init__(self, wavelet: Callable, subframe_size: int, widths: list, dtype=None):
        """
        :param data: The source data, like the quaternion array for one dimension
        :param wavelet: The wavelet function for cwt, like signal.morlet2
        :param subframe_size: The length of data to calculate the cwt for each frame, for frame i, it's data[i: i + subframe_size]
        :param widths: An array of int, each element is the width for the wavelet used to calculate cwt, resulting in one row of the output
        :param dtype: Refer to scipy.signal.cwt, could just leave as None
        """
        self.wavelet = wavelet
        self.subframe_size = subframe_size
        self.widths = widths
        self.input_buffer = RollingBuffer([self.subframe_size, 1])
        self.input_buffer.set_update_style_code(2)
        self.mode = 'normal'

        if dtype is None:
            if np.asarray(wavelet(1, widths[0])).dtype.char in 'FDG':
                self.dtype = np.complex128
            else:
                self.dtype = np.float64
        # The cash data for each algorithm, potentially we can have a better design for caching
        # self.fast_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame = np.zeros([1, len(self.widths), self.subframe_size], dtype=self.dtype)
        # self.scipy_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame_2 = np.zeros([1, len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_skewed_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.wavelet_data = []
        for ind, width in enumerate(self.widths):
            # Potentially we can further speed up by changing the 10 below to a smaller number sacrificing accuracy
            wavelet_length = np.min([10 * width, self.subframe_size])
            self.wavelet_data.append(np.conj(self.wavelet(wavelet_length, width)[::-1]))

    def set_mode(self, mode):
        self.mode = mode

    def receive_new_data(self, data_frame):
        if len(data_frame) != self.ultra_last_frame.shape[0]:
            new_batch = len(data_frame)
            self.ultra_last_frame = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)
            self.ultra_last_frame_2 = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)
            self.ultra_skewed_last_frame = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)

        self.input_buffer.update(data_frame)
        if self.mode == 'normal':
            results = self.ultra_cwt()
        elif self.mode == 'half':
            results = self.ultra_cwt_2()
        elif self.mode == 'unskewed':
            results = self.ultra_skewed()
        if results is None:
            return [0] * len(data_frame)
        return np.abs(results)
    #     insert this data into the buffer
    #     adjust pointers as needed

    def get_N(self, W):
        """
        Get the N value for the right triangle
        :param W: the wavelet length, which is 10 * width parameter
        :return: N length
        """
        return (W - 1) // 2

    def clear(self):
        """
        reset the calculation to the beginning
        """
        self.current_subframe = 0
        # self.fast_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        # self.scipy_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame_2 = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_skewed_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)

    # def next_frame(self):
    #     """
    #     set the current_subframe pointer to the next, should be called after each calculation
    #     """
    #     self.current_subframe += 1
    #     self.current_subframe %= self.num_subframe

    # def fast_cwt(self):
    #     """
    #     The fast cwt implementation, where we calculate both boundary of the triangle (stable part, which could be cached)
    #     and the inside of triangle (where we need to recalculate for each frame)
    #
    #     :return: The cwt result for the current frame
    #     """
    #     self.fast_last_frame[:, :-1] = self.fast_last_frame[:, 1:]
    #     for ind, width in enumerate(self.widths):
    #         wavelet_length = np.min([10 * width, self.subframe_size])
    #         wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
    #         convolve_data = self.data[self.current_subframe: self.current_subframe + self.subframe_size]
    #         convolve_data = convolve_data[-wavelet_length:]
    #         convolve_out = signal.convolve(convolve_data, wavelet_data, mode='same')
    #         N = self.get_N(wavelet_length)
    #         self.fast_last_frame[ind, -N - 1:] = convolve_out[-N - 1:]
    #     return self.fast_last_frame

    def ultra_cwt(self):
        """
        The ultra fast cwt implementation, where we only calculate the boundary of the triangle and thus save time

        :return: The cwt result for the current frame
        """
        self.ultra_last_frame[:, :-1] = self.ultra_last_frame[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        convolve_data = np.swapaxes(convolve_data, 0, 1)

        for ind, width in enumerate(self.widths):
            # Potentially we can further speed up by changing the 10 below to a smaller number sacrificing accuracy
            wavelet_data = self.wavelet_data[ind]
            wavelet_length = np.min([10 * width, self.subframe_size])
            # wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[:, -wavelet_length:]
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data)
            self.ultra_last_frame[:, ind, -1] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, :, -1]

    def ultra_cwt_2(self):
        """
        The ultra fast2 cwt implementation, where we only calculate the boundary of the triangle and thus save time
        Compared to ultra_cwt, we now calculate the middle line between the triangle boundary and the end of the data

        :return: The cwt result for the current frame
        """
        self.ultra_last_frame_2[:, :-1] = self.ultra_last_frame_2[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        for ind, width in enumerate(self.widths):
            wavelet_length = np.min([10 * width, self.subframe_size])
            wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[-wavelet_length:]
            convolve_data_trimmed = convolve_data_trimmed.flatten()
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data[:wavelet_length//2])
            self.ultra_last_frame_2[ind, -1] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, -1]

    def ultra_skewed(self):
        """
        The ultra implementation, where we restore the normal look for the output
        :return:
        """
        self.ultra_skewed_last_frame[:, :-1] = self.ultra_skewed_last_frame[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        for ind, width in enumerate(self.widths):
            wavelet_length = np.min([10 * width, self.subframe_size])
            wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[-wavelet_length:]
            convolve_data_trimmed = convolve_data_trimmed.flatten()
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data)
            N = self.get_N(wavelet_length)
            # The only change compared to ultra, where we store the value at the boundary of the triangle
            self.ultra_skewed_last_frame[ind, -1 - N] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, -1]


class MyTorchCWT:
    def __init__(self, wavelet: Callable, subframe_size: int, widths: list, dtype=None):
        """
        :param data: The source data, like the quaternion array for one dimension
        :param wavelet: The wavelet function for cwt, like signal.morlet2
        :param subframe_size: The length of data to calculate the cwt for each frame, for frame i, it's data[i: i + subframe_size]
        :param widths: An array of int, each element is the width for the wavelet used to calculate cwt, resulting in one row of the output
        :param dtype: Refer to scipy.signal.cwt, could just leave as None
        """
        self.wavelet = wavelet
        self.subframe_size = subframe_size
        self.widths = widths
        self.input_buffer = RollingBuffer([self.subframe_size, 1])
        self.input_buffer.set_update_style_code(2)
        self.mode = 'normal'

        if dtype is None:
            if np.asarray(wavelet(1, widths[0])).dtype.char in 'FDG':
                self.dtype = np.complex128
            else:
                self.dtype = np.float64
        # The cash data for each algorithm, potentially we can have a better design for caching
        # self.fast_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame = np.zeros([1, len(self.widths), self.subframe_size], dtype=self.dtype)
        # self.scipy_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame_2 = np.zeros([1, len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_skewed_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.wavelet_data = []
        for ind, width in enumerate(self.widths):
            # Potentially we can further speed up by changing the 10 below to a smaller number sacrificing accuracy
            wavelet_length = np.min([10 * width, self.subframe_size])
            self.wavelet_data.append(np.conj(self.wavelet(wavelet_length, width)[::-1]))

    def set_mode(self, mode):
        self.mode = mode

    def receive_new_data(self, data_frame):
        if len(data_frame) != self.ultra_last_frame.shape[0]:
            new_batch = len(data_frame)
            self.ultra_last_frame = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)
            self.ultra_last_frame_2 = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)
            self.ultra_skewed_last_frame = np.zeros([new_batch, len(self.widths), self.subframe_size], dtype=self.dtype)

        self.input_buffer.update(data_frame)
        if self.mode == 'normal':
            results = self.ultra_cwt()
        elif self.mode == 'half':
            results = self.ultra_cwt_2()
        elif self.mode == 'unskewed':
            results = self.ultra_skewed()
        if results is None:
            return [0] * len(data_frame)
        return np.abs(results)

    #     insert this data into the buffer
    #     adjust pointers as needed

    def get_N(self, W):
        """
        Get the N value for the right triangle
        :param W: the wavelet length, which is 10 * width parameter
        :return: N length
        """
        return (W - 1) // 2

    def clear(self):
        """
        reset the calculation to the beginning
        """
        self.current_subframe = 0
        # self.fast_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        # self.scipy_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame_2 = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_skewed_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)

    # def next_frame(self):
    #     """
    #     set the current_subframe pointer to the next, should be called after each calculation
    #     """
    #     self.current_subframe += 1
    #     self.current_subframe %= self.num_subframe

    # def fast_cwt(self):
    #     """
    #     The fast cwt implementation, where we calculate both boundary of the triangle (stable part, which could be cached)
    #     and the inside of triangle (where we need to recalculate for each frame)
    #
    #     :return: The cwt result for the current frame
    #     """
    #     self.fast_last_frame[:, :-1] = self.fast_last_frame[:, 1:]
    #     for ind, width in enumerate(self.widths):
    #         wavelet_length = np.min([10 * width, self.subframe_size])
    #         wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
    #         convolve_data = self.data[self.current_subframe: self.current_subframe + self.subframe_size]
    #         convolve_data = convolve_data[-wavelet_length:]
    #         convolve_out = signal.convolve(convolve_data, wavelet_data, mode='same')
    #         N = self.get_N(wavelet_length)
    #         self.fast_last_frame[ind, -N - 1:] = convolve_out[-N - 1:]
    #     return self.fast_last_frame

    def ultra_cwt(self):
        """
        The ultra fast cwt implementation, where we only calculate the boundary of the triangle and thus save time

        :return: The cwt result for the current frame
        """
        self.ultra_last_frame[:, :-1] = self.ultra_last_frame[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        convolve_data = np.swapaxes(convolve_data, 0, 1)

        for ind, width in enumerate(self.widths):
            # Potentially we can further speed up by changing the 10 below to a smaller number sacrificing accuracy
            wavelet_data = self.wavelet_data[ind]
            wavelet_length = np.min([10 * width, self.subframe_size])
            # wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[:, -wavelet_length:]
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data)
            self.ultra_last_frame[:, ind, -1] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, :, -1]

    def ultra_cwt_2(self):
        """
        The ultra fast2 cwt implementation, where we only calculate the boundary of the triangle and thus save time
        Compared to ultra_cwt, we now calculate the middle line between the triangle boundary and the end of the data

        :return: The cwt result for the current frame
        """
        self.ultra_last_frame_2[:, :-1] = self.ultra_last_frame_2[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        for ind, width in enumerate(self.widths):
            wavelet_length = np.min([10 * width, self.subframe_size])
            wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[-wavelet_length:]
            convolve_data_trimmed = convolve_data_trimmed.flatten()
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data[:wavelet_length // 2])
            self.ultra_last_frame_2[ind, -1] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, -1]

    def ultra_skewed(self):
        """
        The ultra implementation, where we restore the normal look for the output
        :return:
        """
        self.ultra_skewed_last_frame[:, :-1] = self.ultra_skewed_last_frame[:, 1:]
        convolve_data = self.input_buffer.get_buffer()
        if convolve_data is None:
            return None
        for ind, width in enumerate(self.widths):
            wavelet_length = np.min([10 * width, self.subframe_size])
            wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[-wavelet_length:]
            convolve_data_trimmed = convolve_data_trimmed.flatten()
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data)
            N = self.get_N(wavelet_length)
            # The only change compared to ultra, where we store the value at the boundary of the triangle
            self.ultra_skewed_last_frame[ind, -1 - N] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, -1]
