import dearpygui.dearpygui as dpg
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from dpg_system.conversion_utils import *
import asyncio
from dpg_system.node import Node
import threading

# NOTE changing target name changed, changing target port crashed


def register_osc_nodes():
    Node.app.register_node('osc_source', OSCSourceNode.factory)
    Node.app.register_node('osc_source_async', OSCAsyncIOSourceNode.factory)
    Node.app.register_node('osc_receive', OSCReceiveNode.factory)
    Node.app.register_node('osc_target', OSCTargetNode.factory)
    Node.app.register_node('osc_send', OSCSendNode.factory)
    Node.app.register_node('osc_route', OSCRouteNode.factory)


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


class OSCManager:
    # started = False
    def __init__(self, label: str, data, args):
        self.targets = {}
        self.sources = {}
        self.send_nodes = []
        self.receive_nodes = []
        self.pending_messages = [[], []]
        self.pending_message_buffer = 0

        OSCSource.osc_manager = self
        OSCAsyncIOSourceNode.osc_manager = self
        OSCTarget.osc_manager = self
        OSCBaseNode.osc_manager = self
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
        self.pending_messages[self.pending_message_buffer].append([source, message, args])

    def swap_pending_message_buffer(self):
        self.pending_message_buffer = 1 - self.pending_message_buffer

    def relay_pending_messages(self):
        self.swap_pending_message_buffer()
        for osc_message in self.pending_messages[1 - self.pending_message_buffer]:
            source = osc_message[0]
            address = osc_message[1]
            args = osc_message[2]

            if address in source.receive_nodes:
                source.receive_nodes[address].receive(args)
            else:
                source.output_message_directly(address, args)
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



class OSCTarget:
    osc_manager = None

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.ip = '127.0.0.1'
        self.name = 'untitled'
        self.port = 2500
        self.osc_format = 0
        self.connected = False
        self.client = None
        self.send_nodes = {}

    def custom_create(self, from_file):
        self.create_client()
        self.osc_manager.register_target(self)

    def create_client(self):
        try:
            self.client = SimpleUDPClient(self.ip, self.port)
        except Exception as e:
            self.client = None
            print(e)

    def destroy_client(self):
        self.osc_manager.remove_target(self)
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

    def send_message(self, address, args):
        if self.client is not None:
            t = type(args)
            if t not in [str]:
                args = any_to_list(args)
            self.client.send_message(address, args)


class OSCTargetNode(OSCTarget, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCTargetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if self.ordered_args is not None:
            for i in range(len(self.ordered_args)):
                arg, t = decode_arg(self.ordered_args, i)
                if t == int:
                    self.port = arg
                elif t == str:
                    is_name = False
                    for c in arg:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                            self.name = arg
                            is_name = True
                            break
                    if not is_name:
                        self.ip = arg

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.target_changed)
        self.target_ip_property = self.add_property('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.port), callback=self.target_changed)

    def target_changed(self):
        name = self.target_name_property()
        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()

        if port != self.port or ip != self.ip:
            self.destroy_client()
            self.port = port
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


class OSCSource:
    osc_manager = None

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.receive_nodes = {}

        self.name = ''
        self.port = 2500
        if args is not None:
            if len(args) > 0:
                self.name = args[0]
            if len(args) > 1:
                p, t = decode_arg(args, 1)
                self.port = any_to_int(p)
        self.osc_manager.register_source(self)

    def osc_handler(self, address, *args):
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


    def output_message_directly(self, args):
        pass

    def create_server(self):
        try:
            self.dispatcher = Dispatcher()
            self.dispatcher.set_default_handler(self.osc_handler)

            self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.start()
        except Exception as e:
            print(e)
            if self.dispatcher:
                self.dispatcher = None
            if self.server:
                self.server.shutdown()
            if self.server_thread:
                self.server_thread.join()
                self.server_thread = None
            self.server = None

    def destroy_server(self):
        for address in self.receive_nodes:
            receive_node = self.receive_nodes[address]
            receive_node.source_going_away(self)
        if self.dispatcher:
            self.dispatcher = None
        if self.server is not None:
            self.server.shutdown()
        if self.server_thread is not None:
            self.server_thread.join()
        self.server = None

    def register_receive_node(self, receive_node):
        self.receive_nodes[receive_node.address] = receive_node

    def unregister_receive_node(self, receive_node):
        if receive_node.address in self.receive_nodes:
            self.receive_nodes.pop(receive_node.address)


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

class OSCAsyncIOSource:
    osc_manager = None

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.receive_nodes = {}
        self.transport = None
        self.protocol = None
        self.pending_dead_loop = []
        self.handle_in_loop = True
        self.name = ''
        self.port = 2500
        if args is not None:
            if len(args) > 0:
                self.name = args[0]
            if len(args) > 1:
                p, t = decode_arg(args, 1)
                self.port = any_to_int(p)
        self.osc_manager.register_source(self)
        self.start_serving()
        # asyncio.run(self.init_main())

    def start_serving(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.osc_handler)
        self.async_loop = start_async()
        submit_async(self.loop_coroutine(), self.async_loop)

    async def loop_coroutine(self):
        self.server = osc_server.AsyncIOOSCUDPServer(('0.0.0.0', self.port), self.dispatcher, asyncio.get_event_loop())
        self.transport, self.protocol = await self.server.create_serve_endpoint()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            for i in range(len(self.pending_dead_loop)):
                self.pending_dead_loop[i] = None
            self.pending_dead_loop = []


    def stop_serving(self):
        for address in self.receive_nodes:
            receive_node = self.receive_nodes[address]
            receive_node.source_going_away(self)
        if self.dispatcher:
            self.dispatcher = None
        if self.transport is not None:
            self.transport.close()
        if self.async_loop:
            self.pending_dead_loop.append(self.async_loop)
            stop_async(self.async_loop)
            self.async_loop = None


    def osc_handler(self, address, *args):
        # alternatively... pass message to buffer to be handled in loop
        if self.handle_in_loop:
            self.osc_manager.receive_pending_message(self, address, args)
            return
        if address in self.receive_nodes:
            self.receive_nodes[address].receive(args)
        else:
            self.output_message_directly(address, args)

    def output_message_directly(self, address, args):
        pass

    def register_receive_node(self, receive_node):
        self.receive_nodes[receive_node.address] = receive_node

    def unregister_receive_node(self, receive_node):
        if receive_node.address in self.receive_nodes:
            self.receive_nodes.pop(receive_node.address)


class OSCSourceNode(OSCSource, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.name = ''
        self.port = 2500

        if args is not None and len(args) > 0:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.port = arg
                elif t == str:
                    self.name = arg

        self.source_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.create_server()

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = self.source_name_property()
        port = any_to_int(self.source_port_property())

        if port != self.port:
            self.destroy_server()
            self.port = port
            self.create_server()

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
        self.name = ''
        self.port = 2500

        if args is not None and len(args) > 0:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.port = arg
                elif t == str:
                    self.name = arg

        self.source_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        # asyncio.run(self.init_main())

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = self.source_name_property()
        port = any_to_int(self.source_port_property())

        if port != self.port:
            self.stop_serving()

            self.port = port
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


class OSCBaseNode(Node):
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


class OSCReceiveNode(OSCBaseNode):

    @staticmethod
    def factory(name, data, args=None):
        node = OSCReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source = None
        self.address = ''
        self.name = 'untitled'

        if args is not None:
            if len(args) > 0:
                self.name = args[0]
            if len(args) > 1:
                self.address = args[1]

        self.source_name_property = self.add_property('source name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.source_address_property = self.add_property('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.output = self.add_output('osc received')

    def name_changed(self):
        new_name = self.source_name_property()
        if new_name != self.name:
            if self.source is not None:
                self.osc_manager.unregister_receive_node(self)
            self.name = new_name
            self.find_source_node(self.name)
            self.osc_manager.connect_receive_node_to_source(self, self.source)

    def address_changed(self):
        new_address = self.source_address_property()
        if new_address != self.address:
            self.osc_manager.receive_node_address_changed(self, new_address, self.source)

    def custom_create(self, from_file):
        if self.name != '':
            self.find_source_node(self.name)

    def receive(self, data):
        if self.output:
            self.output.send(list(data))

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

    def verify_source(self):
        if self.source.registered_name == self.source_name_property():
            return True
        return False

    def cleanup(self):
        self.osc_manager.unregister_receive_node(self)


class OSCSendNode(OSCBaseNode):

    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendNode(name, data, args)
        return node

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

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_property('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_property('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)

    def name_changed(self):
        new_name = self.target_name_property()
        if new_name != self.name:
            if self.target is not None:
                self.osc_manager.unregister_send_node(self)
            self.name = new_name
            self.find_target_node(self.name)

    def address_changed(self):
        new_address = self.target_address_property()
        if new_address != self.address:
            self.osc_manager.unregister_send_node(self)
            self.address = new_address
            self.find_target_node(self.name)

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

    def set_target(self, target):
        self.target = target

    def target_going_away(self, old_target):
        if self.target == old_target:
             self.target = None

    def verify_target(self):
        if self.target.registered_name == self.target_name_property():
            return True
        return False

    def cleanup(self):
        self.osc_manager.unregister_send_node(self)

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


class OSCRouteNode(OSCBaseNode):

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
