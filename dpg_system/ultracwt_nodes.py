from typing import Callable

import numpy as np
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
    print('reg ultra')
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

        print(self.subframe_size, self.widths)
        self.cwt = MyCWT(signal.morlet2, self.subframe_size, self.widths)
        self.input = self.add_input('in 1', triggers_execution=True)

        self.frame_size_widget = self.add_input('scales', widget_type='drag_int', default_value=self.subframe_size)
        self.frame_size_widget.add_callback(self.frame_size_changed)

        self.cwt_mode = self.add_property('mode', widget_type='combo', default_value='normal')
        self.cwt_mode.widget.combo_items = ['normal', 'half', 'unskewed']

        widths_string = any_to_string(self.widths)
        print(widths_string)
        self.widths_property = self.add_input('scales', widget_type='text_input', default_value=widths_string)
        self.widths_property.add_callback(self.widths_changed)
        self.output = self.add_output('cwt out')

    def frame_size_changed(self, val=0):
        self.cwt = None
        self.subframe_size = self.frame_size_widget()
        self.cwt = MyCWT(signal.morlet2, self.subframe_size, self.widths)

    def widths_changed(self, val=0):
        self.cwt = None
        print('widths changed')
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
        self.mode = 'normal'

        if dtype is None:
            if np.asarray(wavelet(1, widths[0])).dtype.char in 'FDG':
                self.dtype = np.complex128
            else:
                self.dtype = np.float64
        # The cash data for each algorithm, potentially we can have a better design for caching
        # self.fast_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        # self.scipy_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_last_frame_2 = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)
        self.ultra_skewed_last_frame = np.zeros([len(self.widths), self.subframe_size], dtype=self.dtype)

    def set_mode(self, mode):
        self.mode = mode

    def receive_new_data(self, data_frame):
        self.input_buffer.update(data_frame)
        if self.mode == 'normal':
            results = self.ultra_cwt()
        elif self.mode == 'half':
            results = self.ultra_cwt_2()
        elif self.mode == 'unskewed':
            results = self.ultra_skewed()
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
        for ind, width in enumerate(self.widths):
            # Potentially we can further speed up by changing the 10 below to a smaller number sacrificing accuracy
            wavelet_length = np.min([10 * width, self.subframe_size])
            wavelet_data = np.conj(self.wavelet(wavelet_length, width)[::-1])
            convolve_data_trimmed = convolve_data[-wavelet_length:]
            convolve_data_trimmed = convolve_data_trimmed.flatten()
            convolve_out = np.dot(convolve_data_trimmed, wavelet_data)
            self.ultra_last_frame[ind, -1] = convolve_out
        self.input_buffer.release_buffer()
        return self.ultra_last_frame[:, -1]

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

    # def experiment(self, num_iter: int):
    #     """
    #     This is the experiment used for notebook. Basically we run all calculation for num_iter times and return the time
    #     for each algorithm
    #
    #     :param num_iter: the number of iteration of calculation we run for the experiment
    #     :return: The three algorithm runtime in tuple
    #     """
    #     fast_start = time()
    #     for i in range(num_iter):
    #         self.fast_cwt()
    #         self.next_frame()
    #     fast_end = time()
    #     fast_time = fast_end - fast_start
    #     self.clear()
    #
    #     ultra_start = time()
    #     for i in range(num_iter):
    #         self.ultra_cwt()
    #     ultra_end = time()
    #     ultra_time = ultra_end - ultra_start
    #     self.clear()
    #
    #     scipy_start = time()
    #     for i in range(num_iter):
    #         self.scipy_cwt()
    #     scipy_end = time()
    #     scipy_time = scipy_end - scipy_start
    #     self.clear()
    #
    #     return scipy_time, fast_time, ultra_time
    #
    #
# def verify_my_cwt():
#     wave_data = np.load("take.npz")['quats'][:, 9, 0]
#     widths = list(range(1, 5))
#     subframe_size = 50
#     wavelet = signal.morlet2
#     my_cwt = MyCWT(wave_data, wavelet, subframe_size, widths)
#     fast_out = None
#     for i in range(1000):
#         assert i == my_cwt.current_subframe
#         cur_signal = wave_data[i:i + subframe_size]
#         fast_out = fast_cwt(cur_signal, signal.morlet2, widths, last_frame=fast_out)
#         my_out = my_cwt.fast_cwt()
#         assert np.allclose(fast_out, my_out, 0), f"i={i}, {np.abs(fast_out - my_out)}"


# if __name__ == "__main__":
#     wave_data = np.load("take.npz")['quats'][:, 9, 0]
#     widths = list(range(1, 100, 10))
#     subframe_size = 1000
#     wavelet = signal.morlet2
#     my_cwt = MyCWT(wave_data, wavelet, subframe_size, widths)
#
#     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, figsize=(10 ,10), constrained_layout=True)
#     cur_signal = wave_data[:subframe_size]
#
#     ax1.set_xlim(-5, 0)
#     ax1.set_ylim(-2, 2)
#     ax1.autoscale(False)
#     h1, = ax1.plot(np.linspace(-5, 0, subframe_size), cur_signal)
#
#     cwt_out = signal.cwt(cur_signal, signal.morlet2, widths)
#     h2 = ax2.imshow(np.abs(cwt_out), vmin=0, vmax=2, extent=[-5, 0, 1, 10], aspect='auto', cmap='jet')
#
#     fast_out = my_cwt.fast_cwt()
#     h3 = ax3.imshow(np.abs(fast_out),vmin=0, vmax=2, extent=[-5, 0, 1, 10], aspect='auto', cmap='jet')
#
#     ultra_out = my_cwt.ultra_cwt()
#     h4 = ax4.imshow(np.abs(ultra_out), vmin=0, vmax=2,extent=[-5, 0, 1, 10], aspect='auto', cmap='jet')
#
#     ultra_skewed_out = my_cwt.ultra_skewed()
#     h5 = ax5.imshow(np.abs(ultra_skewed_out), vmin=0, vmax=2,extent=[-5, 0, 1, 10], aspect='auto', cmap='jet')
#
#     my_cwt.next_frame()
#
#     ax1.set_title("Signal")
#     ax2.set_title("Scipy CWT")
#     ax3.set_title("My Fast CWT")
#     ax4.set_title("My Ultra CWT")
#     ax5.set_title("My Ultra CWT Skewed")
#     ax5.set_xlabel("Time (s)")
#
#     def animate(i):
#         cur_frame = my_cwt.current_subframe
#         cur_signal = wave_data[cur_frame: cur_frame + subframe_size]
#         h1.set_data(np.linspace(-5, 0, subframe_size), cur_signal)
#         global cwt_out
#         cwt_out[:, :-1] = cwt_out[:, 1:]
#         tmp = signal.cwt(cur_signal, signal.morlet2, widths)
#         cwt_out[:, -subframe_size//2:] = tmp[:, -subframe_size//2:]
#         h2.set_data(np.abs(cwt_out))
#         fast_out = my_cwt.fast_cwt()
#         h3.set_data(np.abs(fast_out))
#         ultra_out = my_cwt.ultra_cwt()
#         h4.set_data(np.abs(ultra_out))
#         ultra_skewed_out = my_cwt.ultra_skewed()
#         h5.set_data(np.abs(ultra_skewed_out))
#         my_cwt.next_frame()
#         return h1, h2, h3, h4, h5
#
#     ani = animation.FuncAnimation(fig, animate, frames=1000, interval=1, blit=True)
#     plt.show()
#     # video_path = "ultra_fast_cwt_comparison.mp4"
#     # ani.save(video_path, writer='ffmpeg', fps=30)