import dearpygui.dearpygui as dpg
from dpg_system.conversion_utils import *
import asyncio
from dpg_system.node import Node
import threading
import socket
import struct
import errno
import torch
import torch.distributed as dist
import os
import select
from numpysocket import NumpySocket

def register_socket_nodes():
    Node.app.register_node("udp_numpy_send", UDPNumpySendNode.factory)
    Node.app.register_node("udp_numpy_receive", UDPNumpyReceiveNode.factory)
    Node.app.register_node('process_group', ProcessGroupNode.factory)
    Node.app.register_node('tcp_numpy_send', TCPNumpySendNode.factory)
    Node.app.register_node('tcp_numpy_receive', TCPNumpyReceiveNode.factory)



class UDPSendSocket:
    def __init__(self, ip, port):
        self.ip = '127.0.0.1'
        self.port = 3500

        if string_is_valid_ip(ip):
            self.ip = ip
        port = any_to_int(port, validate=True)
        if port is not None:
            self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def cleanup(self):
        if self.sock is not None:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()

    def send(self, data):
        self.sock.sendto(data, (self.ip, self.port))


class UDPReceiveSocket:
    def __init__(self, port, max_size=2048):
        self.ip = '127.0.0.1'
        self.port = 3500

        port = any_to_int(port, validate=True)
        if port is not None:
            self.port = port
        self.max_size = max_size

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((self.ip, self.port))

    def cleanup(self):
        if self.sock is not None:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()

    def receive(self):
        try:
            data, addr = self.sock.recvfrom(self.max_size)
        except socket.error as e:
            err = e.args[0]
            if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                return None, ''
            else:
                # a "real" error occurred
                print('UDPReceiveSocket receive:', e)
                return None, ''
        return data, addr


class UDPNumpySendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = UDPNumpySendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if len(args) > 0:
            ip = args[0]
            if string_is_valid_ip(ip):
                self.ip = ip

        if len(args) > 1:
            port = any_to_int(args[1], validate=True)
            if port is not None:
                self.port = port

        self.socket = UDPSendSocket(ip=self.ip, port=self.port)
        self.data_input = self.add_input('data', triggers_execution=True)
        self.ip_in = self.add_input('ip', widget_type='text_input', default_value= self.ip, triggers_execution=True)
        self.port_in = self.add_input('port', widget_type='drag_int', default_value=self.port, triggers_execution=True)

    def pack_data(self, data):
        header_format = 'ciic'
        if data.dtype in [float, np.float32]:
            dtype_code = b'f'
        if data.dtype == np.double:
            dtype_code = b'd'
        elif data.dtype == np.int64:
            dtype_code = b'l'
        elif data.dtype == np.int32:
            dtype_code = b'i'
        elif data.dtype == bool:
            dtype_code = b'b'
        shape_length = len(data.shape)
        shape_size = int(np.prod(data.shape))
        order_code = b'C'
        header = struct.pack(header_format, dtype_code, shape_length, shape_size, order_code)
        data_bytes = data.tobytes()
        shape_bytes = struct.pack('i' * shape_length, *data.shape)
        message_data = header + shape_bytes + data_bytes
        return message_data

    def execute(self):
        if self.active_input == self.ip_in:
            ip = str(self.ip_in())
            if string_is_valid_ip(ip):
                self.ip = ip
                if self.socket is not None:
                    self.socket.cleanup()
                self.socket = None
        elif self.active_input == self.port_in:
            port = any_to_int(self.port_in(), validate=True)
            if port is not None:
                self.port = port
                if self.socket is not None:
                    self.socket.cleanup()
                self.socket = None
        if self.socket is None:
            self.socket = UDPSendSocket(ip=self.ip, port=self.port)
        if self.socket is not None:
            if self.active_input == self.data_input:
                data = self.data_input()
                data = any_to_array(data, validate=True)
                if data is not None:
                    packed_data = self.pack_data(data)
                    self.socket.send(packed_data)


class UDPNumpyReceiveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = UDPNumpyReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if len(args) > 0:
            port = any_to_int(args[0], validate=True)
            if port is not None:
                self.port = port

        self.socket = UDPReceiveSocket(port=self.port)
        self.port_in = self.add_input('port', widget_type='drag_int', triggers_execution=True)
        self.data_output = self.add_output('received data')
        self.buffer = None
        self.buffer_ptr = 0
        self.header_size = struct.calcsize('ciic')
        self.header_parsed = False
        self.incoming_dtype = 'd'
        self.incoming_shape_length = 0
        self.incoming_shape_size = 0
        self.incoming_order_code = 'C'
        self.incoming_element_size = 8
        self.add_frame_task()

    def frame_task(self):
        if self.socket:
            data, address = self.socket.receive()
            if data is not None:
                if self.buffer_ptr == 0:
                    self.buffer = data
                else:
                    self.buffer += data
                self.buffer_ptr += len(data)
                if len(self.buffer) > self.header_size and not self.header_parsed:
                    self.unpack_header(self.buffer)
                    self.header_parsed = True
                    self.expected_bytes = self.header_size + self.incoming_shape_length * 4 + self.incoming_shape_size * self.incoming_element_size
                    print('expected', self.incoming_shape_size, self.incoming_element_size, self.incoming_element_size * self.incoming_shape_size)
                print('received', len(self.buffer))
                if self.header_parsed and len(self.buffer) >= self.expected_bytes:
                    array = self.unpack_data(self.buffer)
                    self.buffer = None
                    self.buffer_ptr = 0
                    self.header_parsed = False
                    self.data_output.send(array)

    def unpack_header(self, data):
        header_size = struct.calcsize('ciic')
        dtype_code, shape_length, shape_size, order_code = struct.unpack('ciic', data[:header_size])
        self.incoming_dtype = dtype_code
        self.incoming_shape_length = shape_length
        self.incoming_shape_size = shape_size
        self.incoming_order_code = order_code
        if self.incoming_dtype in [b'f', b'i']:
            self.incoming_element_size = 4
        elif self.incoming_dtype in [b'd', b'l']:
            self.incoming_element_size = 8
        elif self.incoming_dtype == b'b':
            self.incoming_element_size = 1

    def unpack_data(self, data):
        header_size = struct.calcsize('ciic')
        dtype_code, shape_length, shape_size, order_code = struct.unpack('ciic', data[:header_size])
        print(dtype_code, shape_length, shape_size, order_code)
        shape_holder = [1] * shape_length
        for i in range(shape_length):
            shape_holder[i] = struct.unpack('i', data[header_size + i * 4:header_size + i * 4 + 4])[0]
        dtype = np.float64
        code = b'd'
        if dtype_code == b'f':
            code = 'f'
            dtype = np.float32
        elif dtype_code == b'd':
            code = 'd'
            dtype = np.float64
        elif dtype_code == b'i':
            code = 'i'
            dtype = np.int32
        elif dtype_code == b'l':
            code = 'q'
            dtype = np.int64
        elif dtype_code == b'b':
            code = 'B'
            dtype = bool
        offset = header_size + shape_length * 4
        available = len(data) - offset
        print(available, shape_size, code)
        data = struct.unpack(code * shape_size, data[offset:])
        array = np.array(data, dtype=dtype).reshape(shape_holder)
        return array


def string_is_valid_ip(ip_string):
    try:
        socket.inet_aton(ip_string)  # try to convert the string to an IPv4 address
        return True  # the string is a valid IPv4 address
    except socket.error:
        try:
            socket.inet_pton(socket.AF_INET6, ip_string)  # try to convert the string to an IPv6 address
            return True  # the string is a valid IPv6 address
        except socket.error:
            return False


class TCPNumpySendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TCPNumpySendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.numpysocket = None
        self.ip = '127.0.0.1'
        self.port = 4501
        self.connected = False
        self.server_has_changed = False
        if len(args) > 0:
            ip = args[0]
            if string_is_valid_ip(ip):
                self.ip = ip

        if len(args) > 1:
            port = any_to_int(args[1], validate=True)
            if port is not None:
                self.port = port
        self.data_input = self.add_input('data', triggers_execution=True)

        self.ip_property = self.add_property('ip', widget_type='text_input', default_value=self.ip)
        self.port_property = self.add_property('port', widget_type='input_int', default_value=self.port, max=32767)

    def server_changed(self):
        if self.app.verbose:
            print('server changed')
        self.ip = self.ip_property()
        self.port = self.port_property()
        if self.connected:
            if self.app.verbose:
                print('was connected')
            self.connected = False
            if self.numpysocket is not None:
                if self.app.verbose:
                    print('shutting down send socket')
                try:
                    self.numpysocket.shutdown(socket.SHUT_RDWR)
                    self.numpysocket.close()
                    if self.app.verbose:
                        print('send socket closed')
                except Exception as e:
                    print('send socket trying to shut down')
        self.numpysocket = None
        if self.app.verbose:
            print('send destroyed self.numpysocket')

    def socket_release(self):
        if self.connected:
            if self.app.verbose:
                print('socket_release was connected')
            self.connected = False
        if self.numpysocket is not None:
            if self.app.verbose:
                print('socket_release shutting down send socket')
            try:
                self.numpysocket.shutdown(socket.SHUT_RDWR)
                self.numpysocket.close()
                if self.app.verbose:
                    print('send socket_release socket closed')
            except Exception as e:
                print('send socket_release socket trying to shut down')
        self.numpysocket = None


    def execute(self):
        port = self.port_property()
        ip = self.ip_property()
        if ip != self.ip or port != self.port:
            if string_is_valid_ip(ip):
                self.server_changed()

        if self.numpysocket is None:
            if self.app.verbose:
                print('send created socket')
            self.connected = False
            self.numpysocket = NumpySocket()

        if self.numpysocket is not None:
            if not self.connected:
                try:
                    if self.app.verbose:
                        print('send connecting to', self.ip, self.port)
                    self.numpysocket.connect((self.ip, self.port))
                    if self.app.verbose:
                        print('send socket connected')
                    self.connected = True
                except Exception as e:
                    print('send trying to connect', e)
                    self.socket_release()
            if self.connected:
                if self.active_input == self.data_input:
                    data = self.data_input()
                    data = any_to_array(data, validate=True)
                    if data is not None:
                        if self.app.verbose:
                            print('send')
                        try:
                            self.numpysocket.sendall(data)
                        except Exception as e:
                            if e == errno.ECONNRESET:
                                print('send failed due to connection reset')
                                self.connected = False
                            else:
                                print('trying to send other error', e)

    def custom_cleanup(self):
        if self.numpysocket is not None:
            if self.connected:
                self.numpysocket.shutdown(socket.SHUT_RDWR)
                self.numpysocket.close()
            self.numpysocket = None


class TCPNumpyReceiveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TCPNumpyReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.port = 4500
        self.numpysocket = None
        if len(args) > 0:
            port = any_to_int(args[0], validate=True)
            if port is not None:
                self.port = port
        self.listening = False
        self.running = True
        self.listening = False
        self.connection = None
        self.reconnect = False
        self.port_changed = False
        self.client_Address = ''
        self.connection = None
        self.socket_released = False
        self.received = None
        self.ready_to_listen = True
        self.port_property = self.add_property('port', widget_type='input_int', default_value=self.port, max=32767)
        self.data_out = self.add_output('data')

    def post_creation_callback(self):
        if self.app.verbose:
            print('receive post_create')
        self.port = self.port_property()
        self.socket_init()
        self.port_changed = False
        self.receive_thread = threading.Thread(target=self.receive_thread)
        self.receive_thread.start()
        self.add_frame_task()

# determine how to have a call after any port changes have loaded, to create the socket them
#     def port_has_changed(self):
#         if self.numpysocket is not None:
#             print('receive port changed callback')
#             self.port = self.port_property()
#             self.port_changed = True

    def frame_task(self):
        if self.received is not None:
            if self.app.verbose:
                print('output')
            self.data_out.send(self.received)
            self.received = None
        if self.reconnect:
            if self.app.verbose:
                print('receive reconnecting')
            self.reconnect = False
            self.socket_release()
            if self.app.verbose:
                print('creating new socket receive')
            self.numpysocket = NumpySocket()
            if self.app.verbose:
                print('binding to', self.port)
            self.numpysocket.setblocking(False)
            self.numpysocket.bind(('', self.port))
            self.ready_to_listen = True

    def socket_release(self):
        if self.app.verbose:
            print('receive releasing socket')
        if self.numpysocket is not None:
            if self.connection is not None:
                if self.app.verbose:
                    print('receive closing connection')
                try:
                    self.connection.close()
                except Exception as e:
                    print('receive connection close error', e)
                self.connection = None
            if self.listening:
                try:
                    if self.app.verbose:
                        print('receive shutting down')
                    self.numpysocket.shutdown(socket.SHUT_RDWR)
                except Exception as e:
                    print('receive shutting down error', e)
                self.listening = False
            try:
                self.numpysocket.close()
            except Exception as e:
                print('receive socket close error', e)
            self.numpysocket = None
            self.socket_released = True

    def socket_init(self):
        self.numpysocket = NumpySocket()
        try:
            self.numpysocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.numpysocket.setblocking(False)
            if self.app.verbose:
                print('bind to', self.port)
            self.numpysocket.bind(('', self.port))
            self.numpysocket.listen()
            self.ready_to_listen = True
            self.listening = True
        except Exception as e:
            print('receive socket_init_exception', e)

    def receive_thread(self):
        print('enter receive_thread')
        while self.running:
            if self.port != self.port_property() or self.port_changed:
                if self.app.verbose:
                    print('receive port changed')
                self.port = self.port_property()
                self.socket_release()
            if self.numpysocket is None:
                self.socket_init()

            if self.ready_to_listen:
                if not self.listening:
                    if self.app.verbose:
                        print('listen')
                    try:
                        self.numpysocket.listen()
                        self.listening = True
                    except Exception as e:
                        print('receive thread listen exception', e)
                try:
                    self.connection, addr = self.numpysocket.accept()
                    if self.connection is not None:
                        if self.app.verbose:
                            print('connection is made')
                        while self.connection is not None and self.running:
                            if self.port != self.port_property() or self.port_changed:
                                if self.app.verbose:
                                    print('receive closing due to port change')
                                self.port = self.port_property()
                                self.socket_release()
                            else:
                                try:
                                    frame = self.connection.recv(1024)
                                    if len(frame) > 0:
                                        self.received = frame
                                    else:
                                        self.socket_release()
                                except Exception as e:
                                    if e.errno != 35:
                                        print('connection.recv error', e)
                                    # else:
                                    #     print('socket', self.port, e)
                        if self.app.verbose:
                            print('connected or running == False')
                        if self.connection is not None:
                            self.connection.close()
                            self.connection = None

                except Exception as e:
                    if e.errno == 35:
                        if self.socket_released:
                            print('receive lost connection', e)
                            self.socket_released = False

        if self.app.verbose:
            print('running == False... cleaning up receive socket')
        self.numpysocket.shutdown(socket.SHUT_RDWR)
        self.numpysocket.close()

    def custom_cleanup(self):
        self.running = False
        if self.app.verbose:
            print('join')
        self.receive_thread.join()
        if self.app.verbose:
            print('joined')


class ProcessGroup:
    def __init__(self, ip='127.0.0.1', port='29500', backend='gloo', rank=0, world_size=2):
        os.environ['MASTER_ADDR'] = ip
        os.environ['MASTER_PORT'] = port
        self.req = None
        self.backend = backend
        self.rank = rank
        self.world_size = world_size
        self.setup_thread = threading.Thread(target=self.setup_process_group)
        self.ready = False
        self.setup_thread.start()
        print('after starting thread')
        # dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    def setup_process_group(self):
        print('preparing process group', self.rank)
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)
        print('process group ready', self.rank)
        self.ready = True

    def send(self, tensor, dest=1):
        self.req = dist.isend(tensor=tensor, dst=dest)

    def receive(self, tensor, source=0):
        self.req = dist.irecv(tensor=tensor, src=source)

    def wait(self):
        self.req.wait()

    def is_completed(self):
        return self.req.is_completed()

class ProcessGroupNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ProcessGroupNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ip = '127.0.0.1'
        self.port = 29500
        self.rank = 0
        self.world_size = 2
        self.backend = 'gloo'
        self.sending = False
        self.receiving = False
        self.destination = 1
        if self.rank > 0:
            self.destination = 0

        if len(args) > 0:
            ip = args[0]
            if string_is_valid_ip(ip):
                self.ip = ip

        if len(args) > 1:
            port = any_to_int(args[1], validate=True)
            if port is not None:
                self.port = port

        if len(args) > 2:
            rank = any_to_int(args[2], validate=True)
            if rank is not None:
                self.rank = rank

        if len(args) > 2:
            self.backend = args[2]

        if len(args) > 3:
            world_size = any_to_int(args[3], validate=True)
            if world_size is not None:
                self.world_size = world_size

        print('ip', self.ip, 'port', str(self.port), 'backend', self.backend, 'rank', self.rank, 'world_size', self.world_size)
        self.process_group = ProcessGroup(ip=self.ip, port=str(self.port), backend=self.backend, rank=self.rank, world_size=self.world_size)

        self.ip_widget = self.add_property('ip', widget_type='text_input', default_value=self.ip)
        self.ip_port_widget = self.add_property('port', widget_type='input_int', default_value=self.port)
        self.ip_rank_widget = self.add_property('rank', widget_type='input_int', default_value=self.rank)
        self.backend_widget = self.add_property('backend', widget_type='text_input', default_value=self.backend)
        self.world_size = self.add_property('world_size', widget_type='input_int', default_value=self.world_size)

        self.data_in = self.add_input('data_to_send', triggers_execution=True)
        self.destination_in = self.add_input('destination_rank', widget_type='input_int', default_value=self.destination)
        self.sending_complete_out = self.add_output('sending_complete')
        self.expected_tensor_example_in = self.add_input('expected_tensor_example', callback=self.expected)
        self.received_data_out = self.add_output('received_data')
        self.tensor = None

    def expected(self):
        example = self.expected_tensor_example_in()
        if type(example) == torch.tensor:
            self.tensor = torch.zeros_like(self.expected_tensor_example_in())

    def send(self, tensor, dest=1):
        if self.sending:
            print('already sending tensor')
        elif self.process_group is not None:
            self.process_group.send(tensor, dest=dest)
            self.sending = True

    def receive(self, tensor, source=1):
        if self.receiving:
            print('already receiving tensor')
        elif self.process_group is not None and self.tensor is not None:
            self.process_group.receive(self.tensor, source=source)
            self.receiving = True

    def is_completed(self):
        if self.process_group is not None:
            return self.process_group.is_completed()

    def wait(self):
        if self.process_group is not None:
            return self.process_group.wait()

    def frame_task(self):
        if self.is_completed():
            if self.sending:
                self.sending_complete_out.send('bang')
                self.sending = False
            elif self.receiving:
                self.received_data_out.send(self.tensor)
                self.receiving = False

    def execute(self):
        tensor = self.data_in()
        tensor = any_to_tensor(tensor)
        self.send(tensor)
