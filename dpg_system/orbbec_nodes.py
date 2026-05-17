import threading
import time
from multiprocessing import Process, shared_memory
import glfw
from multiprocessing.connection import Client, Listener
import subprocess
import queue
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
        self.server_name = 'dpg_system/depth_server.py'
        self.comm_ports = [6000, 6001]
        self.send_conn = None
        self.receive_conn = None
        self.listener = None
        self.existing_shm = None
        self.shared_memory = None
        self.shared_memory_name = []
        self.shape = [576, 640]
        self.dtype = np.uint16
        self.server = None
        self.message_queue = queue.Queue(16)
        self.setup()
        self.start_server()

    def setup(self):
        self.server_name = 'dpg_system/depth_server.py'
        self.comm_ports = [6000, 6001]
        self.shared_memory_name = ['my_shared_memory']
        self.shape = [576, 640]
        self.dtype = np.uint16

    def start_server(self):
        try:
            self.server = subprocess.Popen(["python", self.server_name])
        except OSError as e:
            print(f'SharedMemoryClientNode: failed to launch {self.server_name}: {e}')
            return False

        # Connect to the subprocess with a hard timeout so node creation
        # cannot hang forever if the server script never comes up.
        deadline = time.monotonic() + 10.0
        address = ('localhost', self.comm_ports[0])
        while True:
            try:
                self.send_conn = Client(address)
                break
            except Exception:
                if time.monotonic() > deadline:
                    print(f'SharedMemoryClientNode: timed out connecting to {self.server_name} on port {self.comm_ports[0]}')
                    return False
                time.sleep(0.05)

        address = ('localhost', self.comm_ports[1])  # family is deduced to be 'AF_INET'
        try:
            self.listener = Listener(address)
            self.receive_conn = self.listener.accept()
        except OSError as e:
            print(f'SharedMemoryClientNode: listener/accept failed on port {self.comm_ports[1]}: {e}')
            return False

        try:
            msg = self.receive_conn.recv()
        except (EOFError, OSError) as e:
            print(f'SharedMemoryClientNode: handshake recv failed: {e}')
            return False
        if msg == 'ready':
            print("Connection established")
        try:
            self.send_conn.send('ready_ack')
        except (BrokenPipeError, OSError) as e:
            print(f'SharedMemoryClientNode: ready_ack send failed: {e}')
            return False
        try:
            self.setup_shared_buffers()
        except FileNotFoundError as e:
            print(f'SharedMemoryClientNode: shared memory not available: {e}')
            return False
        return True

        # Connect to the existing shared memory block
    def setup_shared_buffers(self):
        self.existing_shm = shared_memory.SharedMemory(name=self.shared_memory_name[0])
        self.shared_array = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.existing_shm.buf)

    def send_parameter_update(self, param_name, value):
        self.send_conn.send([param_name, value])

    def receive_message(self, message):
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            # process_message can't keep up; drop the oldest and retry once
            try:
                self.message_queue.get_nowait()
                self.message_queue.put_nowait(message)
            except (queue.Empty, queue.Full):
                pass

    def process_message(self, message):
        pass


class FemtoNode(SharedMemoryClientNode):
    @staticmethod
    def factory(name, data, args=None):
        node = FemtoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.existing_point_cloud_shm = None
        self.shared_point_cloud_array = None
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
        self.point_cloud_out = self.add_output('point_cloud')
        self.acquire = False
        self.new_data = False
        self.depth_data = None
        self.point_cloud = None
        self.keep_thread_running = True
        self.read_thread = threading.Thread(target=self.receive_data, daemon=True)

        self.add_frame_task()
        self.read_thread.start()

    def custom_cleanup(self):
        # Stop the read thread, close IPC, terminate the subprocess, and
        # release shared memory so a destroyed node doesn't leak any of
        # those resources.
        self.keep_thread_running = False
        for conn in (self.send_conn, self.receive_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
            if self.read_thread.is_alive():
                print('FemtoNode: read thread did not exit within 1s')
        if self.server is not None:
            try:
                self.server.terminate()
                self.server.wait(timeout=2.0)
            except Exception as e:
                print(f'FemtoNode: server shutdown error: {e}')
        for shm in (self.existing_shm, self.existing_point_cloud_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass

    def receive_data(self):
        while self.keep_thread_running:
            try:
                msg = self.receive_conn.recv()
            except (EOFError, ConnectionResetError, OSError) as e:
                # The server process closed or the socket died — exit cleanly
                # rather than spinning on a dead connection.
                if self.keep_thread_running:
                    print(f'FemtoNode receive_data: connection closed ({e})')
                return
            except Exception as e:
                print(f'FemtoNode receive_data error: {e}')
                continue
            if type(msg) is str and msg == 'frame':
                if self.shared_array is not None:
                    self.depth_data = self.shared_array.astype(dtype=np.float32)
                if self.shared_point_cloud_array is not None:
                    self.point_cloud = self.shared_point_cloud_array.copy()
                self.new_data = True
            else:
                self.receive_message(msg)

    def setup(self):
        self.server_name = 'dpg_system/depth_server.py'
        self.comm_ports = [7000, 7001]
        self.shared_memory_name = ['femto_depth', 'femto_point_cloud']
        self.shape = [576, 640]
        self.dtype = np.uint16

    def setup_shared_buffers(self):
        self.existing_shm = shared_memory.SharedMemory(name=self.shared_memory_name[0])
        self.shared_array = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.existing_shm.buf)
        self.existing_point_cloud_shm = shared_memory.SharedMemory(name=self.shared_memory_name[1])
        self.shared_point_cloud_array = np.ndarray(shape=(self.shape[0] * self.shape[1], 3), dtype=np.float32, buffer=self.existing_point_cloud_shm.buf)

    def wide_angle_changed(self):
        wide = self.wide_angle_input()
        if wide != self.wide:
            self.wide = wide
            self.send_parameter_update('wide', self.wide)

    def bin_depth_changed(self):
        bin_depth = self.bin_depth_input()
        if bin_depth != self.bin_depth:
            self.bin_depth = bin_depth
            self.send_parameter_update('bin_depth', self.bin_depth)

    def frame_rate_changed(self):
        # The combo widget yields strings like '30 fps' — pull out the
        # leading integer rather than handing the whole string to
        # any_to_int (which can't parse the ' fps' suffix).
        raw = str(self.frame_rate_input()).split()
        if not raw:
            return
        try:
            frame_rate = int(raw[0])
        except (ValueError, TypeError):
            return
        if frame_rate != self.frame_rate:
            self.frame_rate = frame_rate
            self.send_parameter_update('frame_rate', self.frame_rate)

    def frame_task(self):
        if self.new_data:
            self.depth_out.send(self.depth_data)
            self.point_cloud_out.send(self.point_cloud)
            self.new_data = False
        if not self.message_queue.empty():
            message = self.message_queue.get()
            self.process_message(message)

    def process_message(self, message):
        if type(message) is list:
            code = message[0]
            if code == 'depth_width':
                pass
            elif code == 'depth_height':
                pass
            elif code == 'camera_intrinsics':
                pass





