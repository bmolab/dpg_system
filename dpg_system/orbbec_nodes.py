import threading
from multiprocessing import Process, shared_memory
import glfw
from multiprocessing.connection import Client, Listener
import subprocess
# import dearpygui.dearpygui as dpg
# import math
# import numpy as np
# import time
# from scipy import signal
from dpg_system.node import Node
from dpg_system.conversion_utils import *

PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


def register_orbbec_nodes():
    Node.app.register_node("femto", FemtoNode.factory)

class SharedMemoryClientNode(Node):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # launch server prccess
        self.server_name = 'depth_server.py'
        self.comm_ports = [6000, 6001]
        self.existing_shm = None
        self.shape = [640, 576]
        self.dtype = np.uint16
        self.setup()
        self.start_server()

    def setup(self):
        self.server_name = 'depth_server.py'
        self.comm_ports = [6000, 6001]
        self.shared_memory_name = 'my_shared_memory'
        self.shape = [640, 576]
        self.dtype = np.uint16

    def start_server(self):
        self.server = subprocess.Popen(["python", self.server_name])
        waiting = True
        address = ('localhost', self.comm_ports[0])
        while waiting:
            try:
                self.send_conn = Client(address)
                waiting = False
            except Exception as e:
                pass

        address = ('localhost', self.comm_ports[1])  # family is deduced to be 'AF_INET'
        self.listener = Listener(address)
        self.receive_conn = self.listener.accept()
        msg = self.receive_conn.recv()
        if msg == 'ready':
            print("Connection established")
        self.send_conn.send('ready_ack')

        # Connect to the existing shared memory block
        self.existing_shm = shared_memory.SharedMemory(name=self.shared_memory_name)
        self.shared_array = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.existing_shm.buf)


class FemtoNode(SharedMemoryClientNode):
    @staticmethod
    def factory(name, data, args=None):
        node = FemtoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.depth_profile = None
        self.running = False
        self.frame_rate = 30
        self.bin_depth = False
        self.wide = False
        self.enable_input = self.add_input('enable', widget_type='checkbox', default_value=False, triggers_execution=True)

        self.bin_depth_input = self.add_input('bin_depth', widget_type='checkbox', default_value=False, callback=self.bin_depth_changed)
        self.wide_angle_input = self.add_input('wide_angle', widget_type='checkbox', default_value=False, callback=self.wide_angle_changed)
        self.frame_rate_input = self.add_input('frame rate', widget_type='combo', default_value='30 fps', callback= self.frame_rate_changed)
        self.depth_out = self.add_output('depth')
        self.acquire = False
        self.new_data = False
        self.depth_data = None
        self.keep_thread_running = True
        self.read_thread = threading.Thread(target=self.receive_data)
        self.add_frame_task()
        self.read_thread.start()

    def receive_data(self):
        while self.keep_thread_running:
            msg = self.receive_conn.recv()
            if msg == 'frame':
                self.depth_data = self.shared_array.copy()
                self.new_data = True

    def setup(self):
        self.server_name = 'depth_server.py'
        self.comm_ports = [6000, 6001]
        self.shared_memory_name = 'depth_buffer'
        self.shape = [640, 576]
        self.dtype = np.uint16

    def wide_angle_changed(self):
        wide = self.wide_angle_input()
        if wide != self.wide:
            self.wide = wide
            self.update_param('wide', self.wide)

    def bin_depth_changed(self):
        bin_depth = self.bin_depth_input()
        if bin_depth != self.bin_depth:
            self.bin_depth = bin_depth
            self.update_param('bin_depth', self.bin_depth)

    def frame_rate_changed(self):
        frame_rate = any_to_int(self.frame_rate_input())
        if frame_rate != self.frame_rate:
            self.frame_rate = frame_rate
            self.update_param('frame_rate', self.frame_rate)

    def update_param(self, name, value):
        pass
        # ultimately parameter controls will send messages via connection to server
        #

    def frame_task(self):
        if self.new_data:
            self.depth_out.send(self.depth_data)


