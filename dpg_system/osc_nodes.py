import dearpygui.dearpygui as dpg
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from dpg_system.conversion_utils import *
import asyncio
from dpg_system.node import Node
import threading
from dpg_system.interface_nodes import ValueNode, ButtonNode, ToggleNode, MenuNode, RadioButtonsNode

# NOTE changing target name changed, changing target port crashed


def register_osc_nodes():
    Node.app.register_node('osc_device', OSCDeviceNode.factory)
    Node.app.register_node('osc_source', OSCSourceNode.factory)
    Node.app.register_node('osc_source_async', OSCAsyncIOSourceNode.factory)
    Node.app.register_node('osc_receive', OSCReceiveNode.factory)
    Node.app.register_node('osc_target', OSCTargetNode.factory)
    Node.app.register_node('osc_send', OSCSendNode.factory)
    Node.app.register_node('osc_route', OSCRouteNode.factory)
    Node.app.register_node('osc_slider', OSCValueNode.factory)
    Node.app.register_node('osc_float', OSCValueNode.factory)
    Node.app.register_node('osc_int', OSCValueNode.factory)
    Node.app.register_node('osc_message', OSCValueNode.factory)
    Node.app.register_node('osc_string', OSCValueNode.factory)
    Node.app.register_node('osc_knob', OSCValueNode.factory)
    Node.app.register_node('osc_button', OSCButtonNode.factory)
    Node.app.register_node('osc_toggle', OSCToggleNode.factory)
    Node.app.register_node('osc_menu', OSCMenuNode.factory)
    Node.app.register_node('osc_radio', OSCRadioButtonsNode.factory)


# def osc_handler(address, *args):
#     if address == '/filter':
#         for arg in args:
#             print(arg)
#
#
# def osc_thread():
#     dispatcher = Dispatcher()
#     dispatcher.map('/filter', osc_handler)
#
#     OSCserver = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
#     OSCserver_thread = threading.Thread(target=OSCserver.serve_forever)
#     OSCserver_thread.start()

class OSCBase:
    osc_manager = None

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def parse_osc_address(self, data):
        t = type(data)
        if t == str:
            data = data.split(' ')
        else:
            data = any_to_list(data)
        router = any_to_string(data[0])
        split = router.split('/')
        if split[0] == '':
            split = split[1:]
        return split

    def construct_osc_address(self, address_as_list):
        if type(address_as_list) == str:
            address_as_list = address_as_list.split(' ')
        address = '/'.join(address_as_list)
        address = '/' + address
        return address


class OSCManager:
    # started = False
    def __init__(self, label: str, data, args):
        self.pending_message_buffer = 0
        self.targets = {}
        self.sources = {}
        self.send_nodes = []
        self.receive_nodes = []
        self.pending_messages = [[], []]

        OSCBase.osc_manager = self
        self.lock = threading.Lock()
        # if not self.started:
        #     osc_thread()

    # --
    def register_target(self, target):
        if target is not None:
            name = target.name
            if name != '' and name not in self.targets:
                self.targets[name] = target
                self.connect_new_target_to_send_nodes(target)

    # --
    def remove_target(self, target):
        target.disconnect_from_send_nodes()
        if target.name in self.targets:
            self.targets.pop(target.name)

    def find_target(self, name):
        if name != '' and name in self.targets:
            return self.targets[name]
        return None

    def receive_pending_message(self, source, message, args):
        if self.lock.acquire(blocking=True):
            self.pending_messages[self.pending_message_buffer].append([source, message, args])
            self.lock.release()

    def swap_pending_message_buffer(self):
        self.pending_message_buffer = 1 - self.pending_message_buffer

    def relay_pending_messages(self):
        self.swap_pending_message_buffer()
        for osc_message in self.pending_messages[1 - self.pending_message_buffer]:
            if len(osc_message) >= 3:
                source = osc_message[0]
                address = osc_message[1]
                args_ = osc_message[2]

                source.relay_osc(address, args_)
                self.pending_messages[1 - self.pending_message_buffer] = []

    def get_target_list(self):
        return list(self.targets.keys())

    def register_source(self, source):
        if source is not None:
            name = source.name
            if name != '' and name not in self.sources:
                self.sources[name] = source
                self.connect_new_source_to_receive_nodes(source)

    def remove_source(self, source):
        if source.name in self.sources:
            self.sources.pop(source.name)

    def find_source(self, name):
        if name != '' and name in self.sources:
            return self.sources[name]
        return None

    def get_source_list(self):
        return list(self.sources.keys())

    def connect_send_node_to_target(self, send_node, target):
        if target:
            target.register_send_node(send_node)
        if send_node not in self.send_nodes:
            self.send_nodes.append(send_node)

    def connect_new_target_to_send_nodes(self, target):
        for send_node in self.send_nodes:
            if send_node.name == target.name:
                target.register_send_node(send_node)

    def unregister_send_node(self, send_node):
        if send_node.target is not None:
            send_node.target.unregister_send_node(send_node)
            send_node.target = None
        if send_node in self.send_nodes:
            self.send_nodes.remove(send_node)

    def connect_receive_node_to_source(self, receive_node, source):
        if source:
            source.register_receive_node(receive_node)
        if receive_node not in self.receive_nodes:
            self.receive_nodes.append(receive_node)

    def connect_new_source_to_receive_nodes(self, source):
        for receive_node in self.receive_nodes:
            if receive_node.name == source.name:
                source.register_receive_node(receive_node)
                receive_node.source = source

    def receive_node_address_changed(self, receive_node, new_address, source):
        if source is not None:
            source.unregister_receive_node(receive_node)
        receive_node.address = new_address
        if source is not None:
            source.register_receive_node(receive_node)

    def unregister_receive_node(self, receive_node):
        if receive_node.source is not None:
            receive_node.source.unregister_receive_node(receive_node)
            receive_node.source = None
        if receive_node in self.receive_nodes:
            self.receive_nodes.remove(receive_node)

    def print_state(self):
        for name in self.sources:
            print(name, self.sources[name])
        for name in self.targets:
            print(name, self.targets[name])
        for receive_node in self.receive_nodes:
            print(receive_node.name)
        for send_node in self.send_nodes:
            print(send_node.name)


class OSCTarget(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ip = '127.0.0.1'
        self.name = 'untitled'
        self.target_port = 2500

        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.target_port = arg
                elif t == str:
                    is_name = False
                    for c in arg:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                            self.name = arg
                            is_name = True
                            break
                    if not is_name:
                        self.ip = arg

        self.osc_format = 0
        self.connected = False
        self.client = None
        self.send_nodes = {}

    def custom_create(self, from_file):
        self.create_client()
        self.osc_manager.register_target(self)

    def create_client(self):
        try:
            self.client = SimpleUDPClient(self.ip, self.target_port)
        except Exception as e:
            self.client = None
            print(e)

    def destroy_client(self):
        # self.osc_manager.remove_target(self)
        self.client = None

    def disconnect_from_send_nodes(self):
        poppers = []
        for send_address in self.send_nodes:
            send_node = self.send_nodes[send_address]
            send_node.target_going_away(self)
            poppers.append(send_address)
        for pop_address in poppers:
            self.send_nodes.pop(pop_address)

    # this is really just to allow us who might call us so that we can tell them we are gone.
    def register_send_node(self, send_node):
        send_address = send_node.address
        self.send_nodes[send_address] = send_node
        send_node.set_target(self)

    def unregister_send_node(self, send_node):
        address = send_node.address
        if address in self.send_nodes:
            self.send_nodes.pop(address)

    def send_message(self, address, args_):
        if self.client is not None:
            t = type(args_)
            if t not in [str]:
                args_ = any_to_list(args_)
            self.client.send_message(address, args_)


class OSCTargetNode(OSCTarget, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCTargetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.target_changed)
        self.target_ip_property = self.add_property('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.target_port), callback=self.target_changed)

    def target_changed(self):
        name = self.target_name_property()
        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()

        if port != self.target_port or ip != self.ip:
            self.destroy_client()
            self.target_port = port
            self.ip = ip
            self.create_client()

        if name != self.name:
            self.osc_manager.remove_target(self)
            self.name = name
            self.osc_manager.register_target(self)

    def cleanup(self):
        self.destroy_client()

    def execute(self):
        content = []
        message = ''
        if self.input.fresh_input:
            data = list(self.input())
            hybrid_list, homogenous, types = list_to_hybrid_list(data)
            if hybrid_list is not None:
                if len(hybrid_list) > 0:
                    message = hybrid_list[0]
                if len(hybrid_list) > 1:
                    content = hybrid_list[1:]
                if type(message) == list and len(message) == 1:
                    message = message[0]
                self.send_message(message, content)


class OSCSource(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.receive_nodes = {}

        self.name = ''
        self.source_port = 2500

        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.source_port = arg
                elif t == str:
                    is_name = False
                    for c in arg:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                            self.name = arg
                            is_name = True
                            break
                    if not is_name:
                        pass

        self.osc_manager.register_source(self)
        self.lock = threading.Lock()

        self.handle_in_loop = False

    def osc_handler(self, address, *args):
        if self.lock.acquire(blocking=True):
            if type(args) == tuple:
                args = list(args)
            if self.handle_in_loop:
                self.osc_manager.receive_pending_message(self, address, args)
                self.lock.release()
                return
            if address in self.receive_nodes:
                self.receive_nodes[address].receive(args)
                self.lock.release()
                return
            self.relay_osc(address, args)
            self.lock.release()

    def create_dispatcher(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.osc_handler)

    def stop_serving(self):
        for address in self.receive_nodes:
            receive_node = self.receive_nodes[address]
            receive_node.source_going_away(self)
        if self.dispatcher:
            self.dispatcher = None

    def relay_osc(self, address, args):
        if address in self.receive_nodes:
            self.receive_nodes[address].receive(args)
            return
        else:
            if '/' in address:
                sub_addresses = address.split('/')
                length = len(sub_addresses)
                for i in range(1, length):
                    temp = '/'.join(sub_addresses[:-i])
                    if temp in self.receive_nodes:
                        sub = ['/' + '/'.join(sub_addresses[-i:])] + list(args)
                        self.receive_nodes[temp].receive(sub)
                        return
        self.output_message_directly(address, args)

    def output_message_directly(self, address, args):
        pass

    def register_receive_node(self, receive_node):
        self.receive_nodes[receive_node.address] = receive_node

    def unregister_receive_node(self, receive_node):
        if receive_node.address in self.receive_nodes:
            self.receive_nodes.pop(receive_node.address)


class OSCThreadingSource(OSCSource):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def start_serving(self):
        try:
            self.create_dispatcher()

            self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.source_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.start()

        except Exception as e:
            print('start_serving', e)
            if self.dispatcher:
                self.dispatcher = None
            if self.server:
                self.server.shutdown()
            if self.server_thread:
                self.server_thread.join()
                self.server_thread = None
            self.server = None

    def destroy_server(self):
        self.stop_serving()

        if self.server is not None:
            self.server.shutdown()
        if self.server_thread is not None:
            self.server_thread.join()
        self.server = None


server_to_stop = None


def stop_server():
    global server_to_stop
    server_to_stop.destroy_server()


def start_async():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever).start()
    return loop


def submit_async(awaitable, looper):
    return asyncio.run_coroutine_threadsafe(awaitable, looper)


def stop_async(looper):
    looper.call_soon_threadsafe(looper.stop)


class OSCAsyncIOSource(OSCSource):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.transport = None
        self.protocol = None
        self.pending_dead_loop = []
        self.handle_in_loop = True
        self.async_loop = None

        self.start_serving()

    def start_serving(self):
        try:
            self.create_dispatcher()

            self.async_loop = start_async()
            submit_async(self.server_coroutine(), self.async_loop)

        except Exception as e:
            print('start_serving', e)
            if self.dispatcher:
                self.dispatcher = None

    def destroy_server(self):
        self.stop_serving()

        if self.transport is not None:
            self.transport.close()
        if self.async_loop:
            self.pending_dead_loop.append(self.async_loop)
            stop_async(self.async_loop)
            self.async_loop = None

    async def server_coroutine(self):
        self.server = osc_server.AsyncIOOSCUDPServer(('0.0.0.0', self.source_port), self.dispatcher,
                                                     asyncio.get_event_loop())
        self.transport, self.protocol = await self.server.create_serve_endpoint()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            for i in range(len(self.pending_dead_loop)):
                self.pending_dead_loop[i] = None
            self.pending_dead_loop = []


class OSCSourceNode(OSCThreadingSource, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.start_serving()

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = self.source_name_property()
        port = any_to_int(self.source_port_property())

        if port != self.source_port:
            self.destroy_server()
            self.source_port = port
            self.start_serving()

        if name != self.name:
            poppers = []
            for address in self.receive_nodes:
                receive_node = self.receive_nodes[address]
                if receive_node is not None:
                    receive_node.source_going_away(self)
                    poppers.append(address)
            for address in poppers:
                self.receive_nodes.pop(address)
            self.osc_manager.remove_source(self)
            self.name = name
            self.osc_manager.register_source(self)

        # go to osc_manager and see if any existing receiveNodes refer to this new name

    def cleanup(self):
        self.osc_manager.remove_source(self)
        global server_to_stop
        server_to_stop = self
        stop_thread = threading.Thread(target=stop_server)
        stop_thread.start()
        i = 0
        while self.server is not None:
            i += 1


class OSCAsyncIOSourceNode(OSCAsyncIOSource, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCAsyncIOSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source_name_property = self.add_input('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_input('port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.handle_in_loop_option = self.add_option('handle in main loop', widget_type='checkbox', default_value=self.handle_in_loop, callback=self.handle_in_loop_changed)
        # asyncio.run(self.init_main())

    def handle_in_loop_changed(self):
        self.handle_in_loop = self.handle_in_loop_option()

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = self.source_name_property()
        port = any_to_int(self.source_port_property())

        if port != self.source_port:
            self.stop_serving()
            self.source_port = port
            self.start_serving()
            # self.create_server()

        if name != self.name:
            poppers = []
            for address in self.receive_nodes:
                receive_node = self.receive_nodes[address]
                if receive_node is not None:
                    receive_node.source_going_away(self)
                    poppers.append(address)
            for address in poppers:
                self.receive_nodes.pop(address)
            self.osc_manager.remove_source(self)
            self.name = name
            self.osc_manager.register_source(self)

        # go to osc_manager and see if any existing receiveNodes refer to this new name

    def cleanup(self):
        self.osc_manager.remove_source(self)
        global server_to_stop
        server_to_stop = self
        stop_thread = threading.Thread(target=stop_server)
        stop_thread.start()
        i = 0
        while self.server is not None:
            i += 1


class OSCDeviceNode(OSCAsyncIOSource, OSCTarget, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCDeviceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCAsyncIOSource.__init__(self, label, data, args)
        OSCTarget.__init__(self, label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.target_changed)
        self.target_ip_property = self.add_property('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_property('target port', widget_type='text_input', default_value=str(self.target_port), callback=self.target_changed)
        self.source_port_property = self.add_property('source port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.handle_in_loop_option = self.add_option('handle in main loop', widget_type='checkbox', default_value=self.handle_in_loop, callback=self.handle_in_loop_changed)

    def target_changed(self):
        name = self.name_property()
        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()

        if port != self.target_port or ip != self.ip:
            self.destroy_client()
            self.target_port = port
            self.ip = ip
            self.create_client()

        if name != self.name:
            self.osc_manager.remove_target(self)
            self.name = name
            self.osc_manager.register_target(self)

    def source_changed(self):
        name = self.name_property()
        port = any_to_int(self.source_port_property())

        if port != self.source_port:
            self.stop_serving()
            self.source_port = port
            self.start_serving()
            # self.create_server()

        if name != self.name:
            poppers = []
            for address in self.receive_nodes:
                receive_node = self.receive_nodes[address]
                if receive_node is not None:
                    receive_node.source_going_away(self)
                    poppers.append(address)
            for address in poppers:
                self.receive_nodes.pop(address)
            self.osc_manager.remove_source(self)
            self.name = name
            self.osc_manager.register_source(self)

        # go to osc_manager and see if any existing receiveNodes refer to this new name

    def cleanup(self):
        self.destroy_client()
        self.osc_manager.remove_source(self)
        global server_to_stop
        server_to_stop = self
        stop_thread = threading.Thread(target=stop_server)
        stop_thread.start()
        i = 0
        while self.server is not None:
            i += 1

    def handle_in_loop_changed(self):
        self.handle_in_loop = self.handle_in_loop_option()

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def execute(self):
        content = []
        message = ''
        if self.input.fresh_input:
            data = list(self.input())
            hybrid_list, homogenous, types = list_to_hybrid_list(data)
            if hybrid_list is not None:
                if len(hybrid_list) > 0:
                    message = hybrid_list[0]
                if len(hybrid_list) > 1:
                    content = hybrid_list[1:]
                if type(message) == list and len(message) == 1:
                    message = message[0]
                self.send_message(message, content)



class OSCReceiver(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source = None
        self.address = ''
        self.name = 'untitled'
        self.source_name_property = None
        self.source_address_property = None

        if args is not None:
            if len(args) > 0:
                self.name = args[0]
            if len(args) > 1:
                self.address = args[1]

    def name_changed(self):
        if self.source_name_property is not None:
            new_name = self.source_name_property()
            if new_name != self.name:
                if self.source is not None:
                    self.osc_manager.unregister_receive_node(self)
                self.name = new_name
                self.find_source_node(self.name)
                self.osc_manager.connect_receive_node_to_source(self, self.source)

    def address_changed(self):
        if self.source_address_property is not None:
            new_address = self.source_address_property()
            if new_address != self.address:
                self.osc_manager.receive_node_address_changed(self, new_address, self.source)

    def find_source_node(self, name):
        if self.osc_manager is not None:
            self.source = self.osc_manager.find_source(name)
            self.osc_manager.connect_receive_node_to_source(self, self.source)
            if self.source is not None:
                return True
        return False

    def source_going_away(self, old_source):
        if self.source == old_source:
            self.source = None

    def cleanup(self):
        self.osc_manager.unregister_receive_node(self)


class OSCReceiveNode(OSCReceiver, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source_name_property = self.add_input('source name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.source_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.output = self.add_output('osc received')

    def custom_create(self, from_file):
        if self.name != '':
            self.find_source_node(self.name)

    def receive(self, data):
        if self.output:
            self.output.send(list(data))

    def cleanup(self):
        super().cleanup()


class OSCSender(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.target = None
        self.address = '/empty'
        self.name = ''

        if args is not None:
            if len(args) > 0:
                self.name = args[0]
            if len(args) > 1:
                self.address = args[1]

        self.target_name_property = None
        self.target_address_property = None

    def name_changed(self):
        if self.target_name_property is not None:
            new_name = self.target_name_property()
            if new_name != self.name:
                if self.target is not None:
                    self.osc_manager.unregister_send_node(self)
                self.name = new_name
                self.find_target_node(self.name)

    def address_changed(self):
        if self.target_address_property is not None:
            new_address = self.target_address_property()
            if new_address != self.address:
                self.osc_manager.unregister_send_node(self)
                self.address = new_address
                self.find_target_node(self.name)

    def find_target_node(self, name):
        if self.osc_manager is not None:
            self.target = self.osc_manager.find_target(name)
            if self.target is not None:
                self.osc_manager.connect_send_node_to_target(self, self.target)
                return True
            else:
                self.osc_manager.connect_send_node_to_target(self, None)
        return False

    def set_target(self, target):
        self.target = target

    def target_going_away(self, old_target):
        if self.target == old_target:
             self.target = None

    def cleanup(self):
        self.osc_manager.unregister_send_node(self)


class OSCSendNode(OSCSender, Node):

    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)

    def find_target_node(self, name):
        if self.osc_manager is not None:
            self.target = self.osc_manager.find_target(name)
            if self.target is not None:
                self.osc_manager.connect_send_node_to_target(self, self.target)
                return True
            else:
                self.osc_manager.connect_send_node_to_target(self, None)
        return False

    def cleanup(self):
        super().cleanup()

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            t = type(data)
            if t not in [str, int, float, bool, np.int64, np.double]:
                data = list(data)
                data, homogenous, types = list_to_hybrid_list(data)
            if data is not None:
                if self.target and self.address != '':
                    self.target.send_message(self.address, data)


class OSCRouteNode(OSCBase, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCRouteNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        super().__init__(label, data, args)

        self.router_count = 0

        if len(args) > 0:
            self.router_count = len(args)

        self.out_mode = 0
        self.routers = []
        self.router_options = []
        self.last_states = []
        self.current_states = []

        self.input = self.add_input('in', triggers_execution=True)

        for i in range(self.router_count):
            self.add_output(any_to_string(args[i]))
        self.miss_out = self.add_output('unmatched')
        for i in range(self.router_count):
            val, t = decode_arg(args, i)
            self.routers.append(any_to_string(val))
        for i in range(self.router_count):
            an_option = self.add_option('route address ' + str(i), widget_type='text_input', default_value=args[i], callback=self.routers_changed)
            self.router_options.append(an_option)

        self.new_routers = False

    def routers_changed(self):
        self.new_routers = True

    def update_routers(self):
        new_routers = []
        for i in range(self.router_count):
            new_routers.append(self.router_options[i]())
        for i in range(self.router_count):
            # this does not update the label
            dpg.set_item_label(self.outputs[i].uuid, label=new_routers[i])
            sel, t = decode_arg(new_routers, i)
            self.routers[i] = any_to_string(sel)

    def execute(self):
        if self.new_routers:
            self.update_routers()
            self.new_routers = False
        data = self.input()
        t = type(data)
        if t == str:
            data = data.split(' ')
        address_list = self.parse_osc_address(data)

        router = address_list[0]
        if router in self.routers:
            index = self.routers.index(router)
            if len(address_list) > 1:
                remaining_address = self.construct_osc_address(address_list[1:])
                message = [remaining_address] + data[1:]
            else:
                message = data[1:]
            if index < len(self.outputs):
                self.outputs[index].send(message)
                return
        else:
            if router[0] != '/':
                router = '/' + router
                if router in self.routers:
                    index = self.routers.index(router)
                    if len(address_list) > 1:
                        remaining_address = self.construct_osc_address(address_list[1:])
                        message = [remaining_address] + data[1:]
                    else:
                        message = data[1:]
                    if index < len(self.outputs):
                        self.outputs[index].send(message)
                        return
        self.miss_out.send(self.input())


class OSCValueNode(OSCReceiver, OSCSender, ValueNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCValueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        if label == 'osc_string':
            self.space_replacement = self.add_option('replace spaces in osc messages', widget_type='checkbox',
                                                     default_value=True)

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self)
        self.outputs[0].set_label(self.target_name_property() + ':' + self.target_address_property())

    def address_changed(self):
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.outputs[0].set_label(self.target_name_property() + ':' + self.target_address_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def receive(self, data):
        t = type(data)
        data = any_to_list(data)

        if self.label not in ['osc_string', 'osc_message']:
            if type(data[0]) == str:
                if data[0][0] == '/':
                    return
        if self.label == 'osc_string':
            data = any_to_string(data)
            if self.space_replacement():
                data = data.replace('_', ' ')
        self.inputs[0].receive_data(data)
        ValueNode.execute(self)

    def execute(self):
        ValueNode.execute(self)
        data = dpg.get_value(self.value)
        t = type(data)
        if t not in [str, int, float, bool, np.int64, np.double]:
            data = list(data)
            data, homogenous, types = list_to_hybrid_list(data)
        if self.label == 'osc_message' and t == str:
            data = data.split(' ')
            data, homogenous, types = list_to_hybrid_list(data)
        elif self.label == 'osc_string':
            data = any_to_string(data)
            if self.space_replacement():
                data = data.replace(' ', '_')
        if data is not None:
            if self.target and self.address != '':
                self.target.send_message(self.address, data)


class OSCButtonNode(OSCReceiver, OSCSender, ButtonNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCButtonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def address_changed(self):
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def receive(self, data):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        data = any_to_bool(data)
        if data:
            ButtonNode.execute(self)

    def execute(self):
        ButtonNode.execute(self)
        if self.target and self.address != '':
            self.target.send_message(self.address, self.message())


class OSCToggleNode(OSCReceiver, OSCSender, ToggleNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCToggleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def address_changed(self):
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def receive(self, data):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0] == '/':
                return
        self.value = any_to_bool(data)
        self.input.set(self.value)
        ToggleNode.execute(self)

    def execute(self):
        ToggleNode.execute(self)
        if self.target and self.address != '':
            self.target.send_message(self.address, self.value)


class OSCMenuNode(OSCReceiver, OSCSender, MenuNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCMenuNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def address_changed(self):
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def receive(self, data):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        self.choice.set(data)
        self.set_choice_internal(data)

    def execute(self):
        MenuNode.execute(self)
        if self.target and self.address != '':
            self.target.send_message(self.address, self.choice())


class OSCRadioButtonsNode(OSCReceiver, OSCSender, RadioButtonsNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCRadioButtonsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def address_changed(self):
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.output.set_label(self.target_name_property() + ':' + self.source_address_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
        self.output.set_label(self.target_name_property() + ':' + self.target_address_property())

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def receive(self, data):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        value = any_to_int(data)
        self.radio_group.set(value)
        RadioButtonsNode.execute(self)

    def execute(self):
        RadioButtonsNode.execute(self)
        if self.target and self.address != '':
            self.target.send_message(self.address, self.radio_group())



