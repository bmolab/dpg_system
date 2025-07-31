import queue

import dearpygui.dearpygui as dpg
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from dpg_system.conversion_utils import *
import asyncio
from dpg_system.node import Node
import threading
from dpg_system.interface_nodes import ValueNode, ButtonNode, ToggleNode, MenuNode, RadioButtonsNode
import time
import netifaces
import socket

# what if each main patcher has a udp port for receiving OSC
# OSCSource is automatically created and all OSCReceive nodes automatically register with this
# optional OSCSources can be created
# OSCTargets are assigned through OSCQuery normally. (and invisibly in the course of creating an OSCSend or OSCUI
# OSCDevice construct is split into default OSCSource for patch and external device related OSCTargets
#

# how about create OSCSource in root patcher when an OSCReceive or OSCUI are instantiated with default name of root patcher
# and then how do we decide port number?
# as per Aidan, use the last number of the ip address as a base with 100 port addresses
# i.e. ip 10.1.1.31 would create a bank of ports 3100-3199 (or 3101-3199)
# then the program can iterate through this bank and find an empty port

# what port for 127.0.0.1 ? - get port from ip address of this computer...

# do we want osc_in and osc_out nodes that explicitly work with osc_query and keep osc_send and osc_receive nodes to be mostly like now?

# if we create an osc_send /eos/user/99/chan/1/param/red, then an OSCQuery is made looking for the registration for
# eos, verifying that the rest of the url exists, and if so, checking to see if there is already an OSCTarget for eos
# - if not, create an OSCTargetNode using the OSCQuery ip and port

# if eos is not found in the OSCQuery system, then we assume that the supplied text is a search query
# we want to supply a list of urls that satisfy the search query as a list (like in placeholder) from which the desired one can be selected
# creating the target if necessary and building the osc_send node.

# similar to the osc_source question, where do we put the auto-created osc_target?
# is there any value to having an osc tab that is auto created and those auto-created sources and targets are put there?

# add FLOW tag
#   'FLOW': 'IN' -> accepts input from sources (usually a primary actuator)
#   'FLOW': 'OUT' -> sends to external target (usually controlling a remote system)
#   'FLOW': 'BOTH' -> sends and receives (osc_device) -> a primary actuator (this is the best target to change this value (rather than a proxy)
#   'FLOW': 'PROXY' -> sends and receives but is not a primary actuator (used to control a remote system)

# NOTE OSCSource and OSCTarget cannot have the same name in the registry since this means one replaces the other and vice versa in the registry
# OSCDevice is a fabrication... the source associated with the device is not really asssociated with the device, except as a receiver
# Most importantly, the OSCSource should not broadcast any specific relationship with the target device.
# it is up to the target device to associate itself with the local source as a place to send stuff

# is there any reason to be concerned about having one OSCSource per root patcher? (for OSCSourcing)

# how does a path advertise the things that it can provide? Different from the parameter paradigm of osc query which seems
# mostly about things that can be controlled.



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
    Node.app.register_node('osc_cue', OSCCueNode.factory)
    Node.app.register_node('osc_query_json', OSCQueryJSONNode.factory)
    Node.app.register_node('osc_manager', OSCManagerNode.factory)


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

def compose_path(comp_list):
    full_path = ''
    for comp in comp_list:
        if comp is not None and comp != '':
            if comp[0] != '/':
                comp = '/' + comp
            full_path += comp
    return full_path


class OSCManager:
    def __init__(self, label: str, data, args):
        self.pending_message_buffer = 0
        self.targets = {}
        self.sources = {}
        self.send_nodes = []
        self.receive_nodes = []
        self.registry = OSCQueryRegistry()

        OSCBase.osc_manager = self
        self.lock = threading.Lock()
        self.pending_message_queue = queue.Queue()

    def  register_target(self, target):
        if target is not None:
            name = target.name
            if name != '' and name not in self.targets:
                self.targets[name] = target
                self.connect_new_target_to_send_nodes(target)

    def remove_target(self, target):
        target.disconnect_from_send_nodes()
        if target.name in self.targets:
            self.targets.pop(target.name)

    def find_target(self, name):
        if name != '' and name in self.targets:
            return self.targets[name]
        return None

    def receive_pending_message(self, source, message, args):
        self.pending_message_queue.put([source, message, args], block=False)

    def relay_pending_messages(self):
        while not self.pending_message_queue.empty():
            osc_message = None
            try:
                osc_message = self.pending_message_queue.get(block=False)
            except Exception as e:
                osc_message = None
                print('relay_pending_messages exception:')
                traceback.print_exception(e)

                return
            if osc_message:
                source = osc_message[0]
                address = osc_message[1]
                args_ = osc_message[2]

                source.relay_osc(address, args_)
            else:
                with self.pending_message_queue.mutex:
                    self.pending_message_queue.queue.clear()
                print('relay_pending_messages - no osc message')
                return

    def get_target_list(self):
        return list(self.targets.keys())

    def register_source(self, source):
        if source is not None:
            name = source.name
            if name != '' and name not in self.sources:
                self.sources[name] = source
                self.connect_new_source_to_receive_nodes(source)

    def remove_source(self, source):
        source.disconnect_from_receive_nodes()
        if source.name in self.sources:
            self.sources.pop(source.name)

    def find_source(self, name):
        if name != '' and name in self.sources:
            return self.sources[name]
        return None

    def create_source(self, name):
        if name != '' and name not in self.sources:
            editor = Node.app.get_current_editor()
            if editor is not None:
                source_port = select_root_port_address()
                print('port', source_port)
                args = [name, str(source_port)]
                source_node = Node.app.create_node_by_name('osc_source_async', None, args)
                if source_node is not None:
                    dpg.set_item_pos(source_node.uuid, [20.0, 0])
                    source_node.do_not_delete = True
                    source_node.presentation_state = 'hidden'
                    source_node.set_visibility(visibility_state='hidden')



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
                # receive_node.source = source

    def update_receive_names(self, old_name, new_name):
        for receive_node in self.receive_nodes:
            if receive_node.source_name_property() == old_name:
                receive_node.source_name_property.set(new_name)
                self.path = self.registry.change_path(receive_node.path, new_name)


    def update_send_names(self, old_name, new_name):
        for send_node in self.send_nodes:
            if send_node.target_node_property() == old_name:
                send_node.target_name_property.set(new_name)

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

class OSCQueryRegistry:
    def __init__(self):
        self.registry = {
            'DESCRIPTION': 'DPG_OSC_MANAGER',
            'CONTENTS': {}
        }

    def get_param_registry_container_for_path(self, patch_path):
        path_list = self.prepare_path_list(patch_path)
        reg = self.registry['CONTENTS']
        if reg is not None:
            for domain in path_list:
                if domain in reg:
                    reg = reg[domain]['CONTENTS']
                    if reg is None:
                        print('get_param_registry_container_for_path reg = None')
                else:
                    print('domain', domain, 'missing in OSCQueryRegistry')
                    return None
        return reg

    def prepare_path_list(self, patch_path):
        if type(patch_path) == list:
            patch_path = compose_path(patch_path)
        patch_path_list = patch_path.split('/')
        if patch_path_list[0] == '':
            patch_path_list = patch_path_list[1:]
        for ind, domain in enumerate(patch_path_list):
            if ' ' in domain:
                patch_path_list[ind] = domain.replace(' ', '_')
        if len(patch_path_list) > 1:
            if patch_path_list[0] == patch_path_list[1]:
                patch_path_list = patch_path_list[1:]
        return patch_path_list

    def remove_path_from_registry(self, patch_path):
        patch_path = self.prepare_path_list(patch_path)

        reg = self.registry['CONTENTS']
        if reg is not None:
            for domain in patch_path:
                if domain not in reg:
                    return
                if domain == patch_path[-1]:
                    reg.pop(domain)
                else:
                    reg = reg[domain]['CONTENTS']
                    if reg is None:
                        print('remove_path_from_registry reg is None')
                        break

    def add_path_to_registry(self, patch_path):
        patch_path = self.prepare_path_list(patch_path)
        reg = self.registry['CONTENTS']
        if reg is not None:
            for domain in patch_path:
                print('domain', domain)
                if domain not in reg:
                    reg[domain] = {'CONTENTS':{}}
                    print('added missing domain', domain, reg)
                reg = reg[domain]['CONTENTS']
                if reg is None:
                    print('add_path_to_registry: reg = None')
                    break
        return reg, patch_path

    # def add_target_to_registry(self, patch_name, target_name):
    #     if patch_name not in self.registry['CONTENTS']:
    #         self.add_patch_to_registry(patch_name)
    #     if target_name not in self.registry['CONTENTS'][patch_name]['CONTENTS']:
    #         self.registry['CONTENTS'][patch_name]['CONTENTS'][target_name] = {'CONTENTS':{}}

    # def remove_target_from_registry(self, patch_name, target_name):
    #     if patch_name in self.registry['CONTENTS']:
    #         if target_name in self.registry['CONTENTS'][patch_name]['CONTENTS']:
    #             self.registry['CONTENTS'][patch_name]['CONTENTS'].pop(target_name)
    #
    # def remove_path_from_registry(self, patch_path):
    #     patch_path = self.prepare_path_list(patch_path)
    #
    #     if patch_name in self.registry['CONTENTS']:
    #         if target_name in self.registry['CONTENTS'][patch_name]['CONTENTS']:
    #             if param_name in self.registry['CONTENTS'][patch_name]['CONTENTS'][target_name]['CONTENTS']:
    #                 self.registry['CONTENTS'][patch_name]['CONTENTS'][target_name]['CONTENTS'].pop(param_name)

    # def remove_path_from_registry(self, patch_name, target_name, param_name):
    #     if patch_name in self.registry['CONTENTS']:
    #         if target_name in self.registry['CONTENTS'][patch_name]['CONTENTS']:
    #             if param_name in self.registry['CONTENTS'][patch_name]['CONTENTS'][target_name]['CONTENTS']:
    #                 self.registry['CONTENTS'][patch_name]['CONTENTS'][target_name]['CONTENTS'].pop(param_name)

    def insert_param_dict_into_registry(self, param_dict):
        path_list = self.prepare_path_list(param_dict['FULL_PATH'])
        
        reg, path = self.add_path_to_registry(path_list)
        if reg is not None:
            keys = list(param_dict.keys())
            for key in keys:
                reg[key] = param_dict[key]

    # def ensure_patch_and_target_in_registry(self, patch_name, target_name):
    #     if patch_name not in self.registry['CONTENTS']:
    #         self.add_target_to_registry(patch_name)
    #     if target_name not in self.registry['CONTENTS'][patch_name]['CONTENTS']:
    #         self.add_target_to_registry(patch_name, target_name)

    def change_path(self, old_path, new_path):
        old_path = self.prepare_path_list(old_path)
        new_path = self.prepare_path_list(new_path)
        reg = self.get_param_registry_container_for_path(old_path)
        if reg is not None:
            self.remove_path_from_registry(old_path)
            reg['FULL_PATH'] = compose_path(new_path)
            self.insert_param_dict_into_registry(reg)
        else:
            print('change_path', old_path, 'not found')

    def set_flow(self, path_list, flow):
        path_list = self.prepare_path_list(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            reg['FLOW'] = flow

    def set_description(self, path_list, description):
        path_list = self.prepare_path_list(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            reg['DESCRIPTION'] = description

    def set_value(self, path_list, value):
        path_list = self.prepare_path_list(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            if type(value) == list:
                reg['VALUE'] = value
            else:
                reg['VALUE'] = any_to_list(value)

    def prepare_basic_param_dict(self, type, path_list, access=3):
        path = compose_path(path_list)
        param_dict = {'TYPE': type, 'DESCRIPTION': path_list[-1], 'ACCESS': access, 'FULL_PATH': path, 'FLOW': 'BOTH'}
        return param_dict

    def add_generic_receiver_to_registry(self, path_list):
        path_list = self.prepare_path_list(path_list)
        generic_receiver_dict = self.prepare_basic_param_dict('b', path_list, access=1)
        generic_receiver_dict['FLOW'] = 'IN'
        self.insert_param_dict_into_registry(generic_receiver_dict)
        return generic_receiver_dict['FULL_PATH']

    def add_generic_sender_to_registry(self, path_list):
        path_list = self.prepare_path_list(path_list)
        generic_sender_dict = self.prepare_basic_param_dict('b', path_list, access=2)
        generic_sender_dict['FLOW'] = 'OUT'
        self.insert_param_dict_into_registry(generic_sender_dict)
        return generic_sender_dict['FULL_PATH']

    def add_float_to_registry(self, path_list, value=0.0, min=0.0, max=1.0):
        path_list = self.prepare_path_list(path_list)
        float_param_dict = self.prepare_basic_param_dict('f', path_list)
        if float_param_dict is None:
            return
        float_param_dict['RANGE'] = [{'MIN': min, 'MAX': max}]
        float_param_dict['VALUE'] = [value]
        self.insert_param_dict_into_registry(float_param_dict)
        return float_param_dict['FULL_PATH']

    def add_int_to_registry(self, path_list, value=0, min=0, max=100):
        path_list = self.prepare_path_list(path_list)
        int_param_dict = self.prepare_basic_param_dict('i', path_list)
        if int_param_dict is None:
            return
        int_param_dict['RANGE'] = [{'MIN': min, 'MAX': max}]
        int_param_dict['VALUE'] = [value]
        self.insert_param_dict_into_registry(int_param_dict)
        return int_param_dict['FULL_PATH']

    def add_bool_to_registry(self, path_list, value=False):
        path_list = self.prepare_path_list(path_list)
        bool_param_dict = self.prepare_basic_param_dict('F', path_list)
        if bool_param_dict is None:
            return
        bool_param_dict['VALUE'] = [value]
        self.insert_param_dict_into_registry(bool_param_dict)
        return bool_param_dict['FULL_PATH']

    def add_string_to_registry(self, path_list, value=''):
        path_list = self.prepare_path_list(path_list)
        string_param_dict = self.prepare_basic_param_dict('s', path_list)
        if string_param_dict is None:
            return
        string_param_dict['VALUE'] = [value]
        self.insert_param_dict_into_registry(string_param_dict)
        return string_param_dict['FULL_PATH']

    def add_string_menu_to_registry(self, path_list, value='', choices=None):
        path_list = self.prepare_path_list(path_list)
        string_menu_param_dict = self.prepare_basic_param_dict('s', path_list)
        if string_menu_param_dict is None:
            return
        string_menu_param_dict['VALUE'] = [value]
        if choices is not None and len(choices) > 0:
            string_menu_param_dict['VALS'] = list(choices)
        self.insert_param_dict_into_registry(string_menu_param_dict)
        return string_menu_param_dict['FULL_PATH']

    def add_button_to_registry(self, path_list):
        path_list = self.prepare_path_list(path_list)
        button_param_dict = self.prepare_basic_param_dict('N', path_list, access=0)
        if button_param_dict is None:
            return
        self.insert_param_dict_into_registry(button_param_dict)
        return button_param_dict['FULL_PATH']

    def add_float_array_to_registry(self, path_list, array, min=0.0, max=1.0):
        path_list = self.prepare_path_list(path_list)
        array = any_to_list(array)
        count = len(array)
        array_param_dict = self.prepare_basic_param_dict('f' * count, path_list)
        if array_param_dict is None:
            return
        array_param_dict['VALUE'] = array

        if type(min) in [float, int]:
            array_param_dict['RANGE'] = []
            range_dict = {'MIN': min, 'MAX': max}
            for i in range(count):
                array_param_dict['RANGE'].append(range_dict)
        elif type(min) in [list, np.ndarray, torch.Tensor]:
            min = any_to_list(min)
            max = any_to_list(max)
            if len(min) == count and len(max) == count:
                for i in range(count):
                    array_param_dict['RANGE'].append({'MIN': float(min[i]), 'MAX': float(max[i])})
        self.insert_param_dict_into_registry(array_param_dict)
        return array_param_dict['FULL_PATH']

    def add_int_array_to_registry(self, path_list, array, min=0.0, max=1.0):
        path_list = self.prepare_path_list(path_list)
        array = any_to_list(array)
        count = len(array)
        array_param_dict = self.prepare_basic_param_dict('i' * count, path_list)
        if array_param_dict is None:
            return
        array_param_dict['VALUE'] = array

        if type(min) in [float, int]:
            array_param_dict['RANGE'] = []
            range_dict = {'MIN': int(min), 'MAX': int(max)}
            for i in range(count):
                array_param_dict['RANGE'].append(range_dict)
        elif type(min) in [list, np.ndarray, torch.Tensor]:
            min = any_to_list(min)
            max = any_to_list(max)
            if len(min) == count and len(max) == count:
                for i in range(count):
                    array_param_dict['RANGE'].append({'MIN': int(min[i]), 'MAX': int(max[i])})
        self.insert_param_dict_into_registry(array_param_dict)
        return array_param_dict['FULL_PATH']

    # registry structure

    # {
    #     'DESCRIPTION': 'DPG_OSC_MANAGER',
    #     'CONTENTS': {
    #         '<PATCH_TAB_NAME>': {
    #             'CONTENTS': {
    #                 '<PARAM_NAME>': {
    #                     'TYPE': <type> # f i s F?
    #                     'DESCRIPTION': '<PARAM_NAME>',
    #                     'FULL_PATH': '<PATCH_TAB_NAME>/<PARAM_NAME>',
    #                     'VALUE': [1.0],
    #                     'RANGE': [
    #                         {
    #                             'MAX': 1.0,
    #                             'MIN': 0.0,
    #                         }
    #                     ],
    #                     'ACCESS': 3
    # '
    #
    #                 },
    #                 '<PARAM_NAME>': {
    #             }
    #
    #         }
    #     }
    #
    # }

class OSCQueryJSONNode(OSCBase, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCQueryJSONNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_input('print osc query json', widget_type='button', triggers_execution=True)

    def execute(self):
        if self.osc_manager is not None:
            if self.osc_manager.registry is not None:
                print(self.osc_manager.registry.registry)


class OSCManagerNode(OSCBase, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCManagerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.source_or_target = self.add_property('type of port', widget_type='combo', callback=self.source_target_changed)
        self.source_or_target.widget.combo_items = ['source', 'target']
        self.current_port = None
        self.port_list = self.add_property('sources', widget_type='combo', callback=self.port_changed)
        self.name = self.add_property('name', widget_type='text_input', callback=self.name_changed)
        self.ip = self.add_property('ip address', widget_type='text_input', callback=self.ip_changed)
        self.port = self.add_property('port', widget_type='drag_int', callback=self.port_number_changed)

    def source_target_changed(self):
        value = self.source_or_target()
        if value == 'source':
            source_ports = list(self.osc_manager.sources.keys())
            self.port_list.widget.combo_items = source_ports
            if len(source_ports) > 0:
                name = source_ports[0]
                self.current_port = self.osc_manager.sources[name]
                self.port_list.set(name)
                self.name.set(name)
                self.ip.set('')
                self.port.set(self.current_port.source_port)

        else:
            target_ports = list(self.osc_manager.targets.keys())
            self.port_list.widget.combo_items = target_ports
            if len(target_ports) > 0:
                name = target_ports[0]
                self.current_port = self.osc_manager.targets[name]
                self.port_list.set(name)
                self.name.set(name)
                self.ip.set(self.current_port.ip)
                self.port.set(self.current_port.target_port)

    def port_changed(self):
        port_name = self.port_list()
        if self.source_or_target() == 'source':
            self.current_port = self.osc_manager.sources[port_name]
            self.name.set(port_name)
            self.ip.set('')
            self.port.set(self.current_port.source_port)
        else:
            self.current_port = self.osc_manager.targets[port_name]
            self.name.set(port_name)
            self.ip.set(self.current_port.ip)
            self.port.set(self.current_port.target_port)

    def name_changed(self):
        if self.current_port is not None:
            if self.source_or_target() == 'source':
                old_name = self.current_port.source_name_property()
                self.current_port.source_name_property.set(self.name())
                self.current_port.source_changed()
                self.osc_manager.update_receive_names(old_name, self.name())
                self.source_target_changed()
            else:
                old_name = self.current_port.target_port_name_property()
                self.current_port.target_name_property.set(self.name())
                self.current_port.target_changed()
                self.osc_manager.update_send_names(old_name, self.name())
                self.source_target_changed()

    def port_number_changed(self):
        if self.current_port is not None:
            if self.source_or_target() == 'source':
                self.current_port.source_port_property.set(self.port())
                self.current_port.source_changed()
            else:
                self.current_port.target_port_property.set(self.port())
                self.current_port.target_changed()

    def ip_changed(self):
        if self.current_port is not None:
            if self.source_or_target() == 'target':
                self.current_port.ip_property.set(self.ip())
                self.current_port.target_changed()


class OSCTarget(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ip = '127.0.0.1'

        patcher = Node.app.get_current_editor()
        name = patcher.patch_name
        name = name.replace(' ', '_')
        self.name = name
        self.node = None

        got_port = False
        self.target_port = 2500

        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.target_port = arg
                    got_port = True
                elif t == str:
                    is_name = False
                    for c in arg:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                            self.name = arg
                            is_name = True
                            break
                    if not is_name:
                        self.ip = arg

        if not got_port:
            print('no port')
            self.target_port = select_root_dest_port_address(self.ip)

        print(self.name, self.ip, self.target_port)
        self.osc_format = 0
        self.connected = False
        self.client = None
        self.send_nodes = {}

    def register(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            _, self.path = self.osc_manager.registry.add_path_to_registry([patcher_path, self.name])
            self.node.path_option.set(self.path)

    def unregister(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            self.osc_manager.registry.remove_path_from_registry(patcher_path, self.name)

    def get_registry_container(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name])
        return None

    def custom_create(self, from_file):
        self.create_client()

    def create_client(self):
        try:
            self.client = SimpleUDPClient(self.ip, self.target_port)
            self.osc_manager.register_target(self)
        except Exception as e:
            self.client = None
            print('OSCTarget.create_client:')
            traceback.print_exception(e)

    def destroy_client(self):
        self.osc_manager.remove_target(self)
        self.client = None

    # this is really just to allow us who might call us so that we can tell them we are gone.
    def register_send_node(self, send_node):
        self.send_nodes[send_node.address] = send_node
        send_node.set_target(self)

    def unregister_send_node(self, send_node):
        if send_node.address in self.send_nodes:
            self.send_nodes.pop(send_node.address)

    def disconnect_from_send_nodes(self):
        poppers = []
        for send_address in self.send_nodes:
            send_node = self.send_nodes[send_address]
            if send_node is not None:
                send_node.target_going_away(self)
                poppers.append(send_address)
        for pop_address in poppers:
            self.send_nodes.pop(pop_address)

    def handle_target_change(self, name, port, ip, force=False):
        if port != self.target_port or ip != self.ip:
            self.destroy_client()
            self.target_port = port
            self.ip = ip
            self.create_client()


        if name != self.name or force:
            self.osc_manager.remove_target(self)
            self.name = name
            self.osc_manager.register_target(self)
            self.osc_manager.connect_new_target_to_send_nodes(self)

    def send_message(self, address, args_=None):
        if self.client is not None:
            if args_ is not None:
                t = type(args_)
                if t not in [str]:
                    args_ = any_to_list(args_)
            try:
                self.client.send_message(address, args_)
            except Exception as e:
                pass


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
        self.node = self
        self.path_option = self.add_option('path', widget_type='label')

    def custom_create(self, from_file):
        OSCTarget.custom_create(self, from_file)
        self.register()
        self.outputs[0].set_label(self.path)

    # def register(self):
    #     patcher_path = self.get_patcher_path()
    #     self.path = self.osc_manager.registry.add_path_to_registry([patcher_path, self.name])
    #
    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry(patcher_path, self.name)
    #
    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name])

    def target_changed(self):
        name = self.target_name_property()
        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()
        self.handle_target_change(name, port, ip)

    def cleanup(self):
        self.destroy_client()
        # unregister

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


def get_root_ip_address():
    if 'default_ip_base' in Node.app.config:
        default_ip_base = Node.app.config['default_ip_base']
        ip_comp = default_ip_base.split('.')
        for interface in netifaces.interfaces():
            for link in netifaces.ifaddresses(interface).get(netifaces.AF_INET, ()):
                address = link['addr']
                address_comp = address.split('.')
                if address_comp[0] == ip_comp[0] and address_comp[1] == ip_comp[1] and address_comp[2] == ip_comp[2]:
                    return address

    return None

def select_root_port_address():
    ip_address = get_root_ip_address()
    print('ip_address', ip_address)
    if ip_address is not None:
        ip_comp = ip_address.split('.')
        if len(ip_comp) == 4:
            port_base = int(ip_comp[3])
            port_base *= 100
            proposed_port = find_free_udp_port(port_base, port_base + 99)
            if proposed_port is not None:
                return proposed_port
    else:
        for interface in netifaces.interfaces():
            for link in netifaces.ifaddresses(interface).get(netifaces.AF_INET, ()):
                address = link['addr']
                if address != '127.0.0.1':
                    ip_comp = address.split('.')
                    if len(ip_comp) == 4:
                        port_base = int(ip_comp[3])
                        port_base *= 100
                        proposed_port = find_free_udp_port(port_base, port_base + 99)
                        if proposed_port is not None:
                            return proposed_port
    return 2500

def select_root_dest_port_address(ip_address):
    if ip_address is not None:
        if ip_address == '127.0.0.1':
            print('ip is 127.0.0.1')
            return select_root_port_address()
        ip_comp = ip_address.split('.')
        if len(ip_comp) == 4:
            port_base = int(ip_comp[3])
            port_base *= 100
            if port_base < 1000:
                port_base += 1000
            proposed_port = find_free_udp_port(port_base, port_base + 99)
            if proposed_port is not None:
                return proposed_port
    return 2500

def find_free_udp_port(start_port=1024, end_port=65535):
    """
    Find the first available UDP port in the given range.

    Args:
        start_port (int): The port number to start searching from (default: 1024)
        end_port (int): The port number to end searching at (default: 65535)

    Returns:
        int: First available port number, or None if no ports are available
    """
    for port in range(start_port, end_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    return None


class OSCSource(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.node = None
        self.receive_nodes = {}
        self.path = None

        self.name = ''
        self.source_port = select_root_port_address()

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

        if self.name == '':
            patcher = Node.app.get_current_editor()
            name = patcher.patch_name
            name = name.replace(' ', '_')
            self.name = name

        self.osc_manager.register_source(self)
        self.lock = threading.Lock()

        self.handle_in_loop = False

    def register(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            _, self.path = self.osc_manager.registry.add_path_to_registry([patcher_path, self.name])
            self.node.path_option.set(self.path)

    def unregister(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            self.osc_manager.registry.remove_path_from_registry(patcher_path, self.name)

    def get_registry_container(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name])
        return None

    def osc_handler(self, address, *args):
        # if self.lock.acquire(blocking=True):
        if type(args) == tuple:
            args = list(args)
        if self.handle_in_loop:
            self.osc_manager.receive_pending_message(self, address, args)
            return
        if self.lock.acquire(blocking=True):
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
            self.receive_nodes[address].receive(args, address)
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
        addresses = receive_node.get_addresses()
        if type(addresses) == list:
            for address in addresses:
                self.receive_nodes[address] = receive_node
        elif type(addresses) == str:
            self.receive_nodes[addresses] = receive_node
        receive_node.set_source(self)  # would match OSCTarget

    def unregister_receive_node(self, receive_node):
        addresses = receive_node.get_addresses()
        if type(addresses) == list:
            for address in addresses:
                if address in self.receive_nodes:
                    self.receive_nodes.pop(address)

    def disconnect_from_receive_nodes(self):
        poppers = []
        for address in self.receive_nodes:
            receive_node = self.receive_nodes[address]
            if receive_node is not None:
                receive_node.source_going_away(self)
                poppers.append(address)
        for address in poppers:
            self.receive_nodes.pop(address)

    def handle_source_change(self, name, port, force=False):
        if port != self.source_port:
            self.destroy_server()
            self.source_port = port
            self.start_serving()

        if name != self.name or force:
            # manage change in registry
            self.osc_manager.remove_source(self)
            self.name = name
            self.osc_manager.register_source(self)
            self.osc_manager.connect_new_source_to_receive_nodes(self)

    def cleanup(self):
        pass


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
    if server_to_stop is not None:
        server_to_stop.destroy_server()
    else:
        print('no server to stop')


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
        self.server = None

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
        self.use_queue = self.add_option('use_queue', widget_type='checkbox', default_value=False)
        self.output = self.add_output('osc received')
        self.start_serving()
        self.node = self
        self.path_option = self.add_option('path', widget_type='label')

    def custom_create(self, from_file):
        self.register()
        self.node.outputs[0].set_label(self.path)

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = any_to_string(self.source_name_property())
        port = any_to_int(self.source_port_property())

        self.handle_source_change(name, port)
        # if port != self.source_port:
        #     self.destroy_server()
        #     self.source_port = port
        #     self.start_serving()
        #
        # if name != self.name:
        #     poppers = []
        #     for address in self.receive_nodes:
        #         receive_node = self.receive_nodes[address]
        #         if receive_node is not None:
        #             receive_node.source_going_away(self)
        #             poppers.append(address)
        #     for address in poppers:
        #         self.receive_nodes.pop(address)
        #     self.osc_manager.remove_source(self)
        #     self.name = name
        #     self.osc_manager.register_source(self)
        #
        #     self.osc_manager.connect_new_source_to_receive_nodes(self)
            # new_receivers = self.osc_manager.connect_my_receivers(self)
            # for r in new_receivers:
            #     self.receive_nodes[r.address] = r


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
        OSCSource.cleanup(self)
        self.unregister()


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
        self.node = self
        self.path_option = self.add_option('path', widget_type='text_input', callback=self.set_path)

    def set_path(self):
        path = self.path_option()
        self.path = path


    def custom_create(self, from_file):
        self.register()

    def handle_in_loop_changed(self):
        self.handle_in_loop = self.handle_in_loop_option()

    def output_message_directly(self, address, args):
        if self.output:
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def source_changed(self):
        name = any_to_string(self.source_name_property())
        port = any_to_int(self.source_port_property())

        self.handle_source_change(name, port)
        # if port != self.source_port:
        #     self.stop_serving()
        #     self.source_port = port
        #     self.start_serving()
        #     # self.create_server()
        #
        # if name != self.name:
        #     poppers = []
        #     for address in self.receive_nodes:
        #         receive_node = self.receive_nodes[address]
        #         if receive_node is not None:
        #             receive_node.source_going_away(self)
        #             poppers.append(address)
        #     for address in poppers:
        #         self.receive_nodes.pop(address)
        #     self.osc_manager.remove_source(self)
        #     self.name = name
        #     self.osc_manager.register_source(self)

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
        OSCAsyncIOSource.cleanup(self)
        self.unregister()


class OSCDeviceNode(OSCAsyncIOSource, OSCTarget, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCDeviceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        port_count = 0
        source_port_index = -1
        target_port_index = -1

        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    port_count += 1
                    if port_count == 1:
                        target_port_index = i
                    elif port_count == 2:
                        source_port_index = i
                elif t == str:
                    is_name = False
                    for c in arg:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                            self.name = arg
                            is_name = True
                            break
                    if not is_name:
                        pass

        if target_port_index != -1:
            source_args = args[:target_port_index] + args[target_port_index + 1:]
        else:
            source_args = args

        OSCAsyncIOSource.__init__(self, label, data, source_args)

        if source_port_index != -1:
            target_args = args[:source_port_index] + args[source_port_index + 1:]
        else:
            target_args = args

        OSCTarget.__init__(self, label, data, target_args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.name_property = self.add_input('name', widget_type='text_input', default_value=self.name, callback=self.target_changed)
        self.target_ip_property = self.add_input('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_input('target port', widget_type='text_input', default_value=str(self.target_port), callback=self.target_changed)
        self.source_port_property = self.add_input('source port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.handle_in_loop_option = self.add_option('handle in main loop', widget_type='checkbox', default_value=self.handle_in_loop, callback=self.handle_in_loop_changed)
        OSCAsyncIOSource.node = self
        OSCTarget.node = self
        self.path_option = self.add_option('path', widget_type='text_input', callback=self.change_path)
        self.node = self

    def change_path(self):
        path = self.path_option()
        self.unregister()
        self.path = self.osc_manager.registry.add_generic_receiver_to_registry([path])
        self.path_option.set(self.path)

    def custom_create(self, from_file):
        self.register()

    def register(self):
        patcher_path = self.get_patcher_path()
        # self.path = self.osc_manager.registry.add_generic_receiver_to_registry([patcher_path, self.name])
        self.path = self.osc_manager.registry.add_generic_receiver_to_registry([self.name])
        self.path_option.set(self.path)

    def unregister(self):
        patcher_path = self.get_patcher_path()
        self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name])

    def get_registry_container(self):
        patcher_path = self.get_patcher_path()
        return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name])

    def target_changed(self):
        name = self.name_property()
        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()

        self.handle_target_change(name, port, ip)
        # if port != self.target_port or ip != self.ip:
        #     self.destroy_client()
        #     self.target_port = port
        #     self.ip = ip
        #     self.create_client()
        #
        # if name != self.name:
        #     self.osc_manager.remove_target(self)
        #     self.name = name
        #     self.osc_manager.register_target(self)
        #     self.osc_manager.connect_new_target_to_send_nodes(self)
        #     # new_senders = self.osc_manager.connect_my_senders(self)
        #     # for s in new_senders:
        #     #     self.send_nodes[s.address] = s
        self.source_changed(force=True)

    def source_changed(self, force=False):
        name = self.name_property()
        port = any_to_int(self.source_port_property())

        self.handle_source_change(name, port, force=force)

        # if port != self.source_port:
        #     self.stop_serving()
        #     self.source_port = port
        #     self.start_serving()
        #     # self.create_server()
        #
        # if name != self.name or force:
        #     poppers = []
        #     for address in self.receive_nodes:
        #         receive_node = self.receive_nodes[address]
        #         if receive_node is not None:
        #             receive_node.source_going_away(self)
        #             poppers.append(address)
        #     for address in poppers:
        #         self.receive_nodes.pop(address)
        #     self.osc_manager.remove_source(self)
        #     self.name = name
        #     self.osc_manager.register_source(self)
        #
        #     self.osc_manager.connect_new_source_to_receive_nodes(self)
            # new_receivers = self.osc_manager.connect_new_source_to_receive_nodes(self)
            # for r in new_receivers:
            #     self.receive_nodes[r.address] = r

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
        OSCAsyncIOSource.cleanup(self)
        self.unregister()

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

        # default source name is root_patcher source node
        # how do we manage am OSCQuery search for address?

        self.source = None
        self.address = ''
        self.source_name_property = None
        self.source_address_property = None
        self.path = None

        if args is not None:
            if len(args) == 1: # assuming default source
                self.name = Node.app.get_current_root_patch().patch_name.replace(' ', '_')
                if not self.find_source_node(self.name):
                    editor = Node.app.get_current_editor()
                    self.osc_manager.create_source(self.name)
                self.address = args[0]
            elif len(args) == 2:
                self.name = args[0]
                self.address = args[1]

    def name_changed(self, force=False):
        if self.source_name_property is not None:
            new_name = any_to_string(self.source_name_property())
            if new_name != self.name or force:
                if self.source is not None:
                    self.osc_manager.unregister_receive_node(self)
                self.name = new_name
                self.find_source_node(self.name)
                self.osc_manager.connect_receive_node_to_source(self, self.source)

    def address_changed(self):
        if self.source_address_property is not None:
            new_address = any_to_string(self.source_address_property())
            if new_address != self.address:
                self.osc_manager.receive_node_address_changed(self, new_address, self.source)

    def get_addresses(self):
        if self.source_address_property is not None:
            return self.source_address_property()

    def set_source(self, source):
        self.source = source

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
        self.throttle = self.add_option('throttle (ms)', widget_type='drag_int', default_value=0)
        self.last = time.time()
        self.path_option = self.add_option('path', widget_type='label')

    def register(self):
        patcher_path = self.get_patcher_path()
        self.path = self.osc_manager.registry.add_generic_receiver_to_registry([patcher_path, self.name, self.address])
        print('OSCReceiveNode', self.path)
        self.path_option.set(self.path)

    def unregister(self):
        patcher_path = self.get_patcher_path()
        self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])

    def get_registry_container(self):
        patcher_path = self.get_patcher_path()
        return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    def custom_create(self, from_file):
        if self.name != '':
            self.find_source_node(self.name)
            self.register()
            self.outputs[0].set_label(self.path)

    def receive(self, data, address=None):
        if self.throttle() > 0:
            throttle = self.throttle()
            now = time.time()
            diff = now - self.last
            if diff * 1000 < throttle:
                return
        if self.output:
            self.output.send(list(data))
        self.last = time.time()

    def cleanup(self):
        OSCReceiver.cleanup(self)
        self.unregister()


class OSCSender(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.target = None
        self.address = ''
        self.name = ''
        self.path = None

        if args is not None:
            if len(args) == 1:
                targets = list(self.osc_manager.targets.keys())
                if len(targets) == 1:
                    self.name = targets[0]
                self.address = args[0]
            if len(args) == 2:
                if not is_number(args[0]):
                    self.name = args[0]
                if not is_number(args[1]):
                    self.address = args[1]

        self.osc_path = ''
        self.target_name_property = None
        self.target_address_property = None

    def name_changed(self, force=False):
        if self.target_name_property is not None:
            new_name = any_to_string(self.target_name_property())
            if new_name != self.name or force:
                if self.target is not None:
                    self.osc_manager.unregister_send_node(self)
                self.name = new_name
                self.find_target_node(self.name)

    def address_changed(self):
        if self.target_address_property is not None:
            new_address = any_to_string(self.target_address_property())
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
        self.path_option = self.add_option('path', widget_type='label')

    def register(self):
        patcher_path = self.get_patcher_path()
        self.path = self.osc_manager.registry.add_generic_sender_to_registry([patcher_path, self.name, self.address])
        self.path_option.set(self.path)

    def unregister(self):
        patcher_path = self.get_patcher_path()
        self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])

    def get_registry_container(self):
        patcher_path = self.get_patcher_path()
        return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.register()
            self.outputs[0].set_label(self.path)

    # def find_target_node(self, name):
    #     if self.osc_manager is not None:
    #         self.target = self.osc_manager.find_target(name)
    #         if self.target is not None:
    #             self.osc_manager.connect_send_node_to_target(self, self.target)
    #             print('found target', self.name, self.address)
    #             return True
    #         else:
    #             self.osc_manager.connect_send_node_to_target(self, None)
    #             print('did not find target')
    #     return False

    def cleanup(self):
        OSCSender.cleanup(self)
        self.unregister()

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


class OSCCueNode(OSCSender, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCCueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_int_input('cue # to send', triggers_execution=True)
        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.path = None
        self.path_option = self.add_option('path', widget_type='label')

    def register(self):
        patcher_path = self.get_patcher_path()
        self.path = self.osc_manager.registry.add_generic_sender_to_registry([patcher_path, self.name, self.address])
        self.path_option.set(self.path)

    def unregister(self):
        patcher_path = self.get_patcher_path()
        self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])

    def get_registry_container(self):
        patcher_path = self.get_patcher_path()
        return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.register()
            self.outputs[0].set_label(self.path)

    # def find_target_node(self, name):
    #     if self.osc_manager is not None:
    #         self.target = self.osc_manager.find_target(name)
    #         if self.target is not None:
    #             self.osc_manager.connect_send_node_to_target(self, self.target)
    #             return True
    #         else:
    #             self.osc_manager.connect_send_node_to_target(self, None)
    #     return False

    def cleanup(self):
        OSCSender.cleanup(self)
        self.unregister()

    def execute(self):
        cue = self.input()
        if self.target:
            self.target.send_message('/cue/' + str(cue) + '/go')


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


# OSCWidget
# Sender is remote-machine specific (Target / address usually constructed by OSCQuery)
# receiver is by default the generic receiver for this root patch
# if Receiver is specified, it can be either Source / address or just address
# normally a widget will have the same terminal name for sending and receiving

class OSCWidget(OSCReceiver, OSCSender):
    def __init__(self, label: str, data, args):
        self.node = None
        # HOW TO HANDLE VARIABLE NUMBER OF SOURCE / TARGET / ADDRESS ARGS before the non-osc version args?
        # ASSESS FIRST 4 ARGS, IF ARGS ARE ACTUAL SOURCE OR TARGET NAMES, WE KNOW WHAT TO DO WITH THEM

        super().__init__(label, data, args)

    def preprocess_args(self, args):
        # if this widget is a virtual control for a remote parameter, then it should have a target name and address
        # this can be either selected from an OSCQuery Json based menu or explicitly
        # if explicitly, then the OSCTarget must already exist, and name will be supplied as an argument
        # so if an argument corresponds to an existing OSCTarget, then it is a target name
        # if target name is supplied, then the address must be supplied as well (but could be resolved by OSCQuery also
        # if an argument corresponds to an existing OSCSource, then it is a source name
        # if source name is supplied, then the address must be supplied as well
        # default source name is root_patcher source node (create if it doesn't exist?)

        # where a Widget is bidirectional,
        # what is the relationship between send path and receive path?


        target_name = None
        receive_name = None
        target_path = None
        send_path = None
        for index in range(len(args)):
            arg, t = decode_arg(args, index)
            if t == str:
                if arg in self.osc_manager.targets:
                    target_name = arg
                elif arg in self.osc_manager.sources:
                    source_name = arg
                else:
                    if target_name is not None:
                        target_path = arg





    def register(self):
        pass

    def unregister(self):
        if self.node is None:
            return
        patcher_path = self.node.get_patcher_path()
        self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])

    def get_registry_container(self):
        if self.node is None:
            return
        patcher_path = self.node.get_patcher_path()
        return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    def name_changed(self, force=False):
        if self.node is None:
            return
        patcher_path = self.node.get_patcher_path()
        old_path = [patcher_path, self.name, self.address]
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self, force=True)
        self.node.outputs[0].set_label(self.path)
        self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])

    def address_changed(self):
        if self.node is None:
            return
        patcher_path = self.node.get_patcher_path()
        old_path = [patcher_path, self.name, self.address]
        OSCReceiver.address_changed(self)
        OSCSender.address_changed(self)
        self.node.outputs[0].set_label(self.path)
        self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
        reg = self.get_registry_container()
        if reg is not None:
            reg['DESCRIPTION'] = self.address

    def proxy_changed(self):
        if self.node is None:
            return
        patcher_path = self.node.get_patcher_path()
        proxy = self.node.proxy()
        if proxy:
            self.osc_manager.registry.set_flow([patcher_path, self.name, self.address], flow='PROXY')
        else:
            self.osc_manager.registry.set_flow([patcher_path, self.name, self.address], flow='BOTH')

    def update_value_in_registry(self):
        if self.node is None:
            return
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = self.node.input()

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)
            self.register()
            self.node.outputs[0].set_label(self.path)

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)
        self.unregister()


class OSCValueNode(OSCWidget, ValueNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCValueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.node = self
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        if label == 'osc_string':
            self.space_replacement = self.add_option('replace spaces in osc messages', widget_type='checkbox',
                                                   default_value=True)
        self.proxy = self.add_option('proxy', widget_type='checkbox', default_value=True, callback=self.proxy_changed)
        self.path_option = self.add_option('path', widget_type='label')

    # def custom_create(self, from_file):
    #     self.register()
    #
    def register(self):
        patcher_path = self.get_patcher_path()
        if self.input.widget.widget in ['drag_float', 'slider_float', 'knob_float', 'input_float']:
            self.path = self.osc_manager.registry.add_float_to_registry([patcher_path, self.name, self.address])
        elif self.input.widget.widget in ['drag_int', 'slider_int', 'knob_int', 'input_int']:
            self.path = self.osc_manager.registry.add_int_to_registry([patcher_path, self.name, self.address])
        elif self.input.widget.widget in ['text_input', 'text_editor']:
            self.path = self.osc_manager.registry.add_string_to_registry([patcher_path, self.name, self.address])
        elif self.input.widget.widget in ['checkbox']:
            self.path = self.osc_manager.registry.add_boolean_to_registry([patcher_path, self.name, self.address])
        self.path_option.set(self.path)

    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])

    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    # def name_changed(self, force=False):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.name_changed(self)
    #     OSCSender.name_changed(self, force=True)
    #     self.outputs[0].set_label(any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #
    # def address_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.address_changed(self)
    #     OSCSender.address_changed(self)
    #     self.outputs[0].set_label(any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #     reg = self.get_registry_container()
    #     if reg is not None:
    #         reg['DESCRIPTION'] = self.address

    def custom_create(self, from_file):
        OSCWidget.custom_create(self, from_file)
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = [self.input.widget.value]
            if 'RANGE' in registration:
                registration['RANGE'] = [{'MIN': self.input.widget.min, 'MAX': self.input.widget.max}]

    def update_registry_range(self):
        registration = self.get_registry_container()
        if registration is not None:
            if 'RANGE' in registration:
                registration['RANGE'] = [{'MIN': self.input.widget.min, 'MAX': self.input.widget.max}]

    def update_value_in_registry(self):
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = self.input()

    # def cleanup(self):
    #     OSCSender.cleanup(self)
    #     OSCReceiver.cleanup(self)
    #     self.unregister()

    def receive(self, data, address=None):
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

    def variable_update(self):
        if self.variable is not None:
            data = self.variable.get_value()
            self.input.set(data, propagate=False)
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
        self.update(propagate=False)

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
                self.update_value_in_registry()
                self.target.send_message(self.address, data)


class OSCButtonNode(OSCWidget, ButtonNode):
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
        self.path_option = self.add_option('path', widget_type='label')
        self.node = self

    # def custom_create(self, from_file):
    #     self.register()

    def register(self):
        patcher_path = self.get_patcher_path()
        self.path = self.osc_manager.registry.add_button_to_registry([patcher_path, self.name, self.address])
        self.path_option.set(self.path)

    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])
    #
    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])

    # def name_changed(self, force=False):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.name_changed(self)
    #     OSCSender.name_changed(self, force=True)
    #     self.outputs[0].set_label(any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])

    # def address_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     old_address = self.address
    #     OSCReceiver.address_changed(self)
    #     OSCSender.address_changed(self)
    #     self.outputs[0].set_label(any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #     reg = self.get_registry_container()
    #     if reg is not None:
    #         reg['DESCRIPTION'] = self.address

    # def custom_create(self, from_file):
    #     if self.name != '':
    #         self.find_target_node(self.name)
    #         self.find_source_node(self.name)
    #     self.output.set_label(any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))

    # def cleanup(self):
    #     OSCSender.cleanup(self)
    #     OSCReceiver.cleanup(self)
    #     self.unregister()

    def receive(self, data, address=None):
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


class OSCToggleNode(OSCWidget, ToggleNode):
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
        self.proxy = self.add_option('proxy', widget_type='checkbox', default_value=True, callback=self.proxy_changed)
        self.path_option = self.add_option('path', widget_type='label')
        self.node = self

    def register(self):
        patcher_path = self.get_patcher_path()
        if self.input.widget.widget in ['checkbox']:
            self.path = self.osc_manager.registry.add_bool_to_registry([patcher_path, self.name, self.address])
            self.osc_manager.registry.set_flow([patcher_path, self.name, self.address], flow='PROXY')
        self.path_option.set(self.path)

    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])
    #
    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])
    #
    # def update_value_in_registry(self):
    #     registration = self.get_registry_container()
    #     if registration is not None:
    #         registration['VALUE'] = self.input()

    # def proxy_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     proxy = self.proxy()
    #     if proxy:
    #         self.osc_manager.registry.set_flow([patcher_path, self.name, self.address], flow='PROXY')
    #     else:
    #         self.osc_manager.registry.set_flow([patcher_path, self.name, self.address], flow='BOTH')

    # def name_changed(self, force=False):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.name_changed(self)
    #     OSCSender.name_changed(self, force=True)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #
    # def address_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.address_changed(self)
    #     OSCSender.address_changed(self)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #     reg = self.get_registry_container()
    #     if reg is not None:
    #         reg['DESCRIPTION'] = self.address

    def custom_create(self, from_file):
        OSCWidget.custom_create(self, from_file)
        # if self.name != '':
        #     self.find_target_node(self.name)
        #     self.find_source_node(self.name)
        # self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = [self.input.widget.value]

    # def cleanup(self):
    #     OSCSender.cleanup(self)
    #     OSCReceiver.cleanup(self)
    #     self.unregister()

    def receive(self, data, address):
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
            self.update_value_in_registry()
            self.target.send_message(self.address, self.value)


class OSCMenuNode(OSCWidget, MenuNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCMenuNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        # HOW TO HANDLE VARIABLE NUMBER OF SOURCE / TARGET / ADDRESS ARGS before the non-osc version args?
        # ASSESS FIRST 4 ARGS, IF ARGS ARE ACTUAL SOURCE OR TARGET NAMES, WE KNOW WHAT TO DO WITH THEM
        super().__init__(label, data, args)
        # n.b. if we do not explicitly list a source, then this fails
        # because MenuNode removes the first 2 arguments if osc_menu
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        self.proxy = self.add_option('proxy', widget_type='checkbox', default_value=True, callback=self.proxy_changed)
        self.path_option = self.add_option('path', widget_type='label')
        self.node = self

    # should it expose its possible values in the OSCQueryRegistry?
    def register(self):
        patcher_path = self.get_patcher_path()
        self.path = self.osc_manager.registry.add_string_menu_to_registry([patcher_path, self.name, self.address], choices=self.choices)
        self.path_option.set(self.path)

    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])
    #
    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])
    #
    def update_value_in_registry(self):
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = self.choice()

    # def name_changed(self, force=False):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.name_changed(self)
    #     OSCSender.name_changed(self, force=True)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #
    # def address_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.address_changed(self)
    #     OSCSender.address_changed(self)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #     reg = self.get_registry_container()
    #     if reg is not None:
    #         reg['DESCRIPTION'] = self.address
    #
    def custom_create(self, from_file):
        OSCWidget.custom_create(self, from_file)
        # if self.name != '':
        #     self.find_target_node(self.name)
        #     self.find_source_node(self.name)
        # self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = [self.choice.widget.value]

    # def cleanup(self):
    #     OSCSender.cleanup(self)
    #     OSCReceiver.cleanup(self)
    #     self.unregister()

    def receive(self, data, address):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        self.choice.set(data)
        self.set_choice_internal()

    def execute(self):
        MenuNode.execute(self)
        if self.target and self.address != '':
            self.update_value_in_registry()
            self.target.send_message(self.address, self.choice())


class OSCRadioButtonsNode(OSCWidget, RadioButtonsNode):
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
        self.proxy = self.add_option('proxy', widget_type='checkbox', default_value=True, callback=self.proxy_changed)
        self.path_option = self.add_option('path', widget_type='label')
        self.node = self

    def register(self):
        patcher_path = self.get_patcher_path()
        choices = []
        for button in self.buttons:
            choices.append(str(button))
        self.path = self.osc_manager.registry.add_string_menu_to_registry([patcher_path, self.name, self.address], choices=choices)
        self.path_option.set(self.path)

    # def unregister(self):
    #     patcher_path = self.get_patcher_path()
    #     self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name, self.address])
    #
    # def get_registry_container(self):
    #     patcher_path = self.get_patcher_path()
    #     return self.osc_manager.registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])
    #
    def update_value_in_registry(self):
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = self.radio_group()

    # def name_changed(self, force=False):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.name_changed(self)
    #     OSCSender.name_changed(self, force=True)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #
    # def address_changed(self):
    #     patcher_path = self.get_patcher_path()
    #     old_path = [patcher_path, self.name, self.address]
    #     OSCReceiver.address_changed(self)
    #     OSCSender.address_changed(self)
    #     self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.source_address_property()))
    #     self.osc_manager.registry.change_path(old_path, [patcher_path, self.name, self.address])
    #     reg = self.get_registry_container()
    #     if reg is not None:
    #         reg['DESCRIPTION'] = self.address
    #
    def custom_create(self, from_file):
        OSCWidget.custom_create(self, from_file)
        # if self.name != '':
        #     self.find_target_node(self.name)
        #     self.find_source_node(self.name)
        # self.output.set_label( any_to_string(self.target_name_property()) + ':' + any_to_string(self.target_address_property()))
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = [self.radio_group.widget.value]

    # def cleanup(self):
    #     OSCSender.cleanup(self)
    #     OSCReceiver.cleanup(self)
    #     self.unregister()

    def receive(self, data, address):
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
            self.update_value_in_registry()
            self.target.send_message(self.address, self.radio_group())


