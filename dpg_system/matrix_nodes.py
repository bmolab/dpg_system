import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import numpy as np
import threading
import ssqueezepy


t_BufferFill = 0
t_BufferCircularHorizontal = 1
t_BufferCircularVertical = 2


def register_matrix_nodes():
    Node.app.register_node('buffer', BufferNode.factory)
    Node.app.register_node('rolling_buffer', RollingBufferNode.factory)
    Node.app.register_node('cwt', WaveletNode.factory)
    Node.app.register_node('confusion', ConfusionMatrixNode.factory)


class BufferNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = BufferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_count = self.arg_as_int(256)

        self.update_style = 1
        self.output_style = 1
        self.input = self.add_input("input", triggers_execution=True)
        self.index_input = self.add_input("sample to output", triggers_execution=True)
        self.output = self.add_output("output")
        self.sample_count_option = self.add_option('sample count', widget_type='drag_int', default_value=self.sample_count)
        self.update_style_option = self.add_option('update style', widget_type='combo', default_value='input is stream of samples', width=250, callback=self.update_style_changed)
        self.update_style_option.widget.combo_items = ['buffer holds one sample of input', 'input is stream of samples', 'input is multi-channel sample']
        self.output_style_option = self.add_option('output style', widget_type='combo', default_value='output samples on demand by index', width=250, callback=self.output_style_changed)
        self.output_style_option.widget.combo_items = ['output buffer on every input', 'output samples on demand by index']
        self.buffer = np.zeros((self.sample_count))
        self.write_pos = 0

    def output_style_changed(self):
        output_style = self.output_style_option()
        if output_style == 'output buffer on every input':
            self.output_style = 0
        elif output_style == 'output samples on demand by index':
            self.output_style = 1

    def update_style_changed(self):
        update_style = self.update_style_option()
        if update_style == 'buffer holds one sample of input':
            self.update_style = 0
        elif update_style == 'input is stream of samples':
            self.update_style = 1
        elif update_style == 'input is multi-channel sample':
            self.update_style = 2

    def execute(self):
        self.sample_count = self.sample_count_option()

        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)

            if self.update_style == t_BufferFill:
                self.buffer = data.copy()
            elif self.update_style == t_BufferCircularHorizontal:
                if self.sample_count != self.buffer.shape[0] or len(self.buffer.shape) > 1:
                    self.buffer.resize((self.sample_count))
                    self.write_pos = 0
                if self.write_pos > self.sample_count:
                    self.write_pos = 0
                front_size = data.shape[0]
                back_size = 0
                if front_size + self.write_pos >= self.sample_count:
                    front_size = self.sample_count - self.write_pos
                    back_size = data.shape[0] - front_size
                if back_size > self.sample_count:
                    self.buffer[:] = data[-back_size:]
                    self.write_pos = 0
                else:
                    self.buffer[self.write_pos:self.write_pos + front_size] = data[:front_size]
                    if back_size > 0:
                        self.buffer[:back_size] = data[front_size:front_size + back_size]
                        self.write_pos = back_size
                    else:
                        self.write_pos += front_size
                if self.write_pos >= self.sample_count:
                    self.write_pos = 0
            elif self.update_style == t_BufferCircularVertical:
                if len(self.buffer.shape) == 1 or self.buffer.shape[1] != data.shape[0]:
                    self.buffer.resize((self.sample_count, data.shape[0]), refcheck=False)
                    self.write_pos = 0
                self.buffer[self.write_pos, :] = data
                self.write_pos += 1
                if self.write_pos >= self.sample_count:
                    self.write_pos = 0
            if self.output_style == 0:
                if self.write_pos != 0:
                    output_buffer = np.concatenate((self.buffer[self.write_pos:], self.buffer[:self.write_pos]), axis=0)
                    self.output.send(output_buffer)
                    del output_buffer
                else:
                    self.output.send(self.buffer)

        if self.index_input.fresh_input:
            index = any_to_int(self.index_input())
            if 0 <= index < self.sample_count:
                output_sample = self.buffer[index]
                self.output.send(output_sample)


# double the size of capacity, save each incoming value twice,
# buffer = np.zeros((capacity * 2, ))
# buffer[write_ptr] = buffer[write_ptr + capacity] = input
# write_ptr = (write_ptr + 1) % capacity
# output = buffer[write_ptr:write_ptr + capacity]

class RollingBuffer:
    def __init__(self, shape, roll_along_x=True):
        self.breadth = 1
        self.buffer_changed_callback = None
        self.owner = None
        if type(shape) == tuple:
            shape = list(shape)
        if type(shape) == list:
            self.sample_count = shape[0]
            self.breadth = shape[1]
            if len(shape) > 1 and shape[1] > 1:
                self.update_style = t_BufferCircularVertical
            else:
                self.update_style = t_BufferCircularHorizontal
        else:
            length = any_to_int(shape)
            self.sample_count = length
            self.update_style = t_BufferCircularHorizontal
            self.breadth = 1

        self.roll_along_x = roll_along_x
        self.order = 'C'
        if not roll_along_x:
            self.order = 'F'

        self.in_get_buffer = False
        self.lock = threading.Lock()
        self.buffer = None
        self.allocate((self.sample_count, self.breadth), roll_along_x)
        self.elapsed = 0

    def set_update_style(self, update_style):
        if update_style == 'buffer holds one sample of input':
            self.update_style = t_BufferFill
        elif update_style == 'input is stream of samples':
            self.update_style = t_BufferCircularHorizontal
        elif update_style == 'input is multi-channel sample':
            self.update_style = t_BufferCircularVertical
        self.allocate((self.sample_count, self.breadth), self.roll_along_x)

    def set_value(self, x, value):
        if not self.lock.locked():
            if self.lock.acquire(blocking=False):
                if x < self.buffer.shape[1] and x >= 0:
                    if self.roll_along_x:
                        self.buffer[x, 0] = value
                    else:
                        self.buffer[0, x] = value
                self.lock.release()

    def set_write_pos(self, pos):
        if not self.lock.locked():
            if self.lock.acquire(blocking=False):
                self.write_pos = pos
            self.lock.release()

    def update(self, incoming):
        if not self.lock.locked():
            if self.update_style == t_BufferFill:
                if self.buffer.shape != incoming.shape:
                    if len(incoming.shape) == 1:
                        if self.roll_along_x:
                            if self.buffer.shape != (incoming.shape[0], 1):
                                self.allocate((incoming.shape[0], 1), self.roll_along_x)
                        else:
                            if self.buffer.shape != (1, incoming.shape[0]):
                                self.allocate((1, incoming.shape[0]), self.roll_along_x)
                    else:
                        self.allocate(incoming.shape, self.roll_along_x)
                if self.lock.acquire(blocking=False):
                    if len(incoming.shape) == 1:
                        if self.roll_along_x:
                            self.buffer[:, 0] = incoming[:]
                        else:
                            self.buffer[0, :] = incoming[:]
                    else:
                        self.buffer[:, :] = incoming[:, :]
                    self.lock.release()
                self.write_pos = 0

            elif self.update_style == t_BufferCircularHorizontal:
                if self.roll_along_x:
                    if self.sample_count * 2 != self.buffer.shape[0] or self.buffer.shape[1] != 1:
                        shape = (self.sample_count, 1)
                        self.allocate(shape, self.roll_along_x)
                else:
                    if self.sample_count * 2 != self.buffer.shape[1] or self.buffer.shape[0] != 1:
                        shape = (self.sample_count, 1)
                        self.allocate(shape, self.roll_along_x)

                front_size = incoming.shape[0]
                back_size = 0

                if self.lock.acquire(blocking=False):
                    if self.write_pos > self.sample_count:
                        self.write_pos = 0

                    if front_size + self.write_pos >= self.sample_count:
                        front_size = self.sample_count - self.write_pos
                        back_size = incoming.shape[0] - front_size
                    if back_size > self.sample_count:
                        if self.roll_along_x:
                            self.buffer[:self.sample_count, 0] = self.buffer[self.sample_count:, 0] = incoming[-back_size:]
                        else:
                            self.buffer[0, :self.sample_count] = self.buffer[0, self.sample_count:] = incoming[-back_size:]
                        self.write_pos = 0
                        self.lock.release()
                    else:
                        start = self.write_pos
                        end = self.write_pos + front_size
                        if self.roll_along_x:
                            self.buffer[start:end, 0] = incoming[:front_size]
                            self.buffer[start + self.sample_count:end + self.sample_count, 0] = incoming[:front_size]
                            if back_size > 0:
                                self.buffer[:back_size, 0] = self.buffer[self.sample_count:back_size + self.sample_count, 0] = incoming[front_size:front_size + back_size]
                                self.write_pos = back_size
                            else:
                                self.write_pos += front_size
                        else:
                            self.buffer[0, start:end] = incoming[:front_size]
                            self.buffer[0, start + self.sample_count:end + self.sample_count] = incoming[:front_size]
                            if back_size > 0:
                                self.buffer[0, :back_size] = self.buffer[0,
                                                          self.sample_count:back_size + self.sample_count] = incoming[
                                                                                                             front_size:front_size + back_size]
                                self.write_pos = back_size
                            else:
                                self.write_pos += front_size
                        if self.write_pos >= self.sample_count:
                            self.write_pos = 0
                    self.lock.release()

            elif self.update_style == t_BufferCircularVertical:
                if self.roll_along_x:
                    if len(self.buffer.shape) == 1 or self.buffer.shape[1] != incoming.shape[0]:
                        shape = (self.sample_count, incoming.shape[0])
                        self.allocate(shape, self.roll_along_x)
                    if self.lock.acquire(blocking=False):
                        if len(incoming.shape) == 1:
                            self.buffer[self.write_pos, :] = self.buffer[self.write_pos + self.sample_count, :] = incoming[:]
                        else:
                            self.buffer[self.write_pos, :] = self.buffer[self.write_pos + self.sample_count, :] = incoming[0, :]

                        self.write_pos += 1
                        if self.write_pos >= self.sample_count:
                            self.write_pos = 0
                        self.lock.release()

                else:
                    if len(self.buffer.shape) == 1 or self.buffer.shape[0] != incoming.shape[0]:
                        self.allocate((self.sample_count, incoming.shape[0]), self.roll_along_x)
                    if self.lock.acquire(blocking=False):
                        if len(incoming.shape) == 1:
                            self.buffer[:, self.write_pos] = self.buffer[:, self.write_pos + self.sample_count] = incoming[:]
                        else:
                            self.buffer[:, self.write_pos] = self.buffer[:, self.write_pos + self.sample_count] = incoming[:, 0]
                        self.write_pos += 1
                        if self.write_pos >= self.sample_count:
                            self.write_pos = 0
                        self.lock.release()

            return True
        return False

    def get_buffer(self, block=False):
        if self.lock.acquire(blocking=block):
            self.in_get_buffer = True
            if self.roll_along_x:
                b = self.buffer[self.write_pos:self.write_pos + self.sample_count]
            else:
                if len(self.buffer.shape) > 1:
                    b = self.buffer[:, self.write_pos:self.write_pos + self.sample_count]
                else:
                    b = self.buffer[self.write_pos:self.write_pos + self.sample_count]
            return b
        return None

    def release_buffer(self):
        if self.lock.locked and self.in_get_buffer:
            self.lock.release()
            self.in_get_buffer = False

    def allocate(self, shape, roll_along_x):
        self.lock.acquire(blocking=True)
        if self.update_style == t_BufferFill:
            if self.roll_along_x:
                self.breadth = shape[1]
                self.sample_count = shape[0]
            else:
                self.breadth = shape[0]
                self.sample_count = shape[1]
            self.buffer = np.zeros(shape, order=self.order)
        elif self.update_style == t_BufferCircularHorizontal:
            self.sample_count = shape[0]
            self.breadth = 1
            if len(shape) > 1:
                if shape[0] == 1:
                    self.sample_count = shape[1]
            if roll_along_x:
                self.order = 'C'
                self.buffer = np.zeros((self.sample_count * 2, 1), order=self.order)
            else:
                self.order = 'F'
                self.buffer = np.zeros((1, self.sample_count * 2), order=self.order)
        elif self.update_style == t_BufferCircularVertical:
                if roll_along_x:
                    self.order = 'C'
                    self.sample_count = shape[0]
                    self.breadth = shape[1]
                    self.buffer = np.zeros((self.sample_count * 2, self.breadth), order=self.order)
                else:
                    self.order = 'F'
                    self.sample_count = shape[0]
                    self.breadth = shape[1]
                    self.buffer = np.zeros((self.breadth, self.sample_count * 2), order=self.order)
        self.roll_along_x = roll_along_x
        self.write_pos = 0
        if self.buffer_changed_callback is not None:
            self.buffer_changed_callback(self)
        self.lock.release()

    def set_roll_axis(self, roll_along_x):
        if roll_along_x != self.roll_along_x:
            if self.update_style == t_BufferCircularHorizontal:
                self.allocate((self.sample_count, 1), roll_along_x)

            elif self.update_style == t_BufferCircularVertical:
                self.allocate((self.sample_count, self.breadth), roll_along_x)


class RollingBufferNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RollingBufferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_count = 256
        self.scroll_direction = 'horizontal'
        if self.ordered_args is not None and len(self.ordered_args) > 0:
            count, t = decode_arg(self.ordered_args, 0)
            if t == int:
                self.sample_count = count
        self.rolling_buffer = RollingBuffer(self.sample_count, roll_along_x=False)
        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")
        self.sample_count_option = self.add_option('sample count', widget_type='drag_int', default_value=self.sample_count)
        self.update_style_option = self.add_option('update style', widget_type='combo', default_value='input is stream of samples', width=250, callback=self.update_style_changed)
        self.update_style_option.widget.combo_items = ['buffer holds one sample of input', 'input is stream of samples', 'input is multi-channel sample']
        self.scroll_direction_option = self.add_option('scroll direction', widget_type='combo', default_value=self.scroll_direction, callback=self.scroll_direction_changed)
        self.scroll_direction_option.widget.combo_items = ['horizontal', 'vertical']

    def scroll_direction_changed(self):
        self.scroll_direction = self.scroll_direction_option()
        if self.scroll_direction == 'horizontal':
            self.rolling_buffer.set_roll_axis(roll_along_x=True)
        else:
            self.rolling_buffer.set_roll_axis(roll_along_x=False)

    def update_style_changed(self):
        update_style = self.update_style_option()
        self.rolling_buffer.set_update_style(update_style)

    def execute(self):
        self.rolling_buffer.sample_count = self.sample_count_option()

        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)

            self.rolling_buffer.update(data)
            output_buffer = self.rolling_buffer.get_buffer()
            if output_buffer is not None:
                self.output.send(output_buffer)
                del output_buffer
                self.rolling_buffer.release_buffer()


# reshape?
# slice
# transpose
# invert

class ConfusionMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ConfusionMatrixNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("input", triggers_execution=True)
        self.input2 = self.add_input("input2", triggers_execution=True)
        self.output = self.add_output("output")
        self.confusion_matrix = np.zeros((1, 1))
        self.data2 = None

    def execute(self):
        if self.input2.fresh_input:
            self.data2 = self.input2()
        if self.data2 is not None and len(self.data2) > 0:
            data1 = self.input()
            self.confusion_matrix = np.ndarray((len(self.data2), len(data1)))
            for index, word in enumerate(data1):
                for index2, word2 in enumerate(self.data2):
                    if word == word2:
                        self.confusion_matrix[index2, index] = 1.0
                    else:
                        self.confusion_matrix[index2, index] = 0.0
            self.output.send(self.confusion_matrix)


class WaveletNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WaveletNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")
        self.octaves = 2
        self.octaves_property = self.add_property('octaves', widget_type='drag_int', default_value=self.octaves)
        self.wavelets = 'gmw'
        self.wavelets_property = self.add_property('wavelet', widget_type='combo', default_value=self.wavelets)
        self.wavelets_property.widget.combo_items = ['cmhat', 'gmw', 'bump', 'hhhat', 'morlet']

    def execute(self):
        self.octaves = self.octaves_property()
        self.wavelets = self.wavelets_property()
        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)
            wavelets, _ = ssqueezepy.cwt(data.ravel(), nv=self.octaves, wavelet=self.wavelets, scales='log-piecewise')
            self.output.send(np.abs(wavelets))

