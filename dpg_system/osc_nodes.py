import queue

import dearpygui.dearpygui as dpg
from click import format_filename
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from dpg_system.conversion_utils import *
import asyncio
from dpg_system.node import Node
import threading
from dpg_system.interface_nodes import ValueNode, ButtonNode, ToggleNode, MenuNode, RadioButtonsNode
from dpg_system.interface_nodes import SliderNode, FloatNode, IntNode, StringNode, KnobNode, NumericValueNode, TextEditorNode
import time
import netifaces
import socket
import json

try:
    from dpg_system.oscquery_service import OSCQueryServer, OSCQueryBrowser, ServiceAliasRegistry
    HAS_OSCQUERY = True
except ImportError:
    HAS_OSCQUERY = False



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
    Node.app.register_node('pipo_range', PipoRangeSourceNode.factory)
    Node.app.register_node('pipo_motion', PipoMotionSourceNode.factory)

    # OSCQuery nodes
    from dpg_system.oscquery_nodes import register_oscquery_nodes
    register_oscquery_nodes()


class OSCBase:
    osc_manager = None

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
    def __init__(self, label: str, data, args):
        self.pending_message_buffer = 0
        self.targets = {}
        self.sources = {}
        self.send_nodes = []
        self.receive_nodes = []
        self.registry = OSCQueryRegistry()
        self.service_registries = {}  # service_name -> (OSCQueryRegistry, OSCQueryServer)

        OSCBase.osc_manager = self
        self.lock = threading.Lock()
        self.pending_message_queue = queue.Queue()

        # OSCQuery integration
        self.oscquery_server = None
        self.oscquery_browser = None
        self.alias_registry = None
        if HAS_OSCQUERY:
            self.alias_registry = ServiceAliasRegistry()
            self.oscquery_browser = OSCQueryBrowser()
            self.oscquery_browser.start()
            # Server is started lazily when first source/target registers,
            # or can be started explicitly via start_oscquery_server()

    def start_oscquery_server(self, service_name=None, osc_port=2500):
        """Start the OSCQuery HTTP server and mDNS advertisement."""
        if not HAS_OSCQUERY:
            return
        if self.oscquery_server is None:
            self.oscquery_server = OSCQueryServer(
                self.registry,
                default_osc_port=osc_port,
                service_name=service_name or 'dpg_system'
            )
            self.oscquery_server.start()
        elif service_name:
            # Advertise an additional service name
            self.oscquery_server.advertise_service(service_name, osc_port)

    def cleanup_oscquery(self):
        """Stop OSCQuery server and browser."""
        if self.oscquery_server:
            self.oscquery_server.stop()
            self.oscquery_server = None
        # Stop all per-service servers
        for name, (reg, srv) in list(self.service_registries.items()):
            if srv:
                srv.stop()
        self.service_registries.clear()
        if self.oscquery_browser:
            self.oscquery_browser.stop()
            self.oscquery_browser = None

    # --- Per-service registry management ---

    def register_service(self, service_name, osc_port):
        """Create a dedicated registry + server for a named service."""
        if not HAS_OSCQUERY:
            return None
        if service_name in self.service_registries:
            return self.service_registries[service_name][0]  # already registered

        registry = OSCQueryRegistry()
        server = OSCQueryServer(registry, default_osc_port=osc_port, service_name=service_name)
        server.start()
        self.service_registries[service_name] = (registry, server)
        print(f"OSCManager: Registered service '{service_name}' on OSC port {osc_port}")
        return registry

    def unregister_service(self, service_name):
        """Stop and remove a named service."""
        if service_name in self.service_registries:
            reg, srv = self.service_registries.pop(service_name)
            if srv:
                srv.stop()
            print(f"OSCManager: Unregistered service '{service_name}'")

    def get_registry_for_editor(self, editor):
        """Walk up the patcher hierarchy to find a service-scoped registry.

        Returns the service's OSCQueryRegistry if an oscq_host ancestor is found,
        otherwise returns the global registry.
        """
        current_editor = editor
        while current_editor is not None:
            # Check if any node in this editor is an OSCQueryHostNode
            for node in current_editor._nodes:
                if hasattr(node, '_is_oscq_host') and node._is_oscq_host:
                    service_name = node.service_name
                    if service_name in self.service_registries:
                        return self.service_registries[service_name][0]
            # Walk up to parent
            parent = current_editor.parent_patcher
            if parent is not None:
                # parent_patcher is a NodeEditor; walk up
                current_editor = parent
            else:
                break
        return self.registry  # fallback to global

    def get_service_name_for_editor(self, editor):
        """Return the service name if the editor is inside a service scope, else None."""
        current_editor = editor
        while current_editor is not None:
            for node in current_editor._nodes:
                if hasattr(node, '_is_oscq_host') and node._is_oscq_host:
                    service_name = node.service_name
                    if service_name in self.service_registries:
                        return service_name
            parent = current_editor.parent_patcher
            if parent is not None:
                current_editor = parent
            else:
                break
        return None

    def resolve_service(self, name):
        """
        Resolve a service name through aliases and discovery.
        Returns a DiscoveredService or None.
        """
        if not HAS_OSCQUERY or self.oscquery_browser is None:
            return None
        # Try alias resolution first
        canonical = name
        if self.alias_registry:
            canonical = self.alias_registry.resolve(name)
        # Look up in discovered services
        service = self.oscquery_browser.get_service(canonical)
        if service is None:
            # Try case-insensitive substring match
            for svc_name in self.oscquery_browser.get_service_names():
                if canonical.lower() in svc_name.lower():
                    service = self.oscquery_browser.get_service(svc_name)
                    break
        return service

    def get_discovered_services(self):
        """Return list of discovered service names."""
        if self.oscquery_browser:
            return self.oscquery_browser.get_service_names()
        return []

    def search_param(self, query):
        """Search for a parameter across all discovered services."""
        if self.oscquery_browser:
            return self.oscquery_browser.search_param(query)
        return []

    def register_target(self, target):
        if target is not None:
            name = target.name
            if name != '' and name not in self.targets:
                self.targets[name] = target
                self.connect_new_target_to_send_nodes(target)
                for node in OSCManagerNode.instances:
                    node.update_targets()

    def remove_target(self, target):
        target.disconnect_from_send_nodes()
        if target.name in self.targets:
            self.targets.pop(target.name)
            for node in OSCManagerNode.instances:
                node.update_targets()

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
                for node in OSCManagerNode.instances:
                    node.update_sources()

    def remove_source(self, source):
        source.disconnect_from_receive_nodes()
        if source.name in self.sources:
            self.sources.pop(source.name)
            for node in OSCManagerNode.instances:
                node.update_sources()

    def find_source(self, name):
        if name != '' and name in self.sources:
            return self.sources[name]
        return None

    def create_source(self, name):
        if name != '' and name not in self.sources:
            editor = Node.app.get_current_editor()
            if editor is not None:
                source_port = select_root_port_address()
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

    def connect_new_target_to_send_nodes(self, target, old_name='', hold_senders=None):
        if hold_senders is None:
            hold_senders = self.send_nodes
        for send_node in hold_senders:
            if send_node.name == old_name and old_name != '':
                target.register_send_node(send_node)
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

    def connect_new_source_to_receive_nodes(self, source, old_name='', hold_receivers=None):
        if hold_receivers is None:
            hold_receivers = self.receive_nodes
        for receive_node in hold_receivers:
            if receive_node.name == old_name:
                # print('connect_new_source_to_receive_nodes:reassign receiver', receive_node.source.name)
                source.register_receive_node(receive_node)
            elif receive_node.name == source.name:
                source.register_receive_node(receive_node)
                # receive_node.source = source

    def update_receive_names(self, old_name, new_name):
        for receive_node in self.receive_nodes:
            if receive_node.source_name_property() == old_name:
                receive_node.source_name_property.set(new_name)
                self.path = self.registry.change_path(receive_node.path, new_name)

    def update_send_names(self, old_name, new_name):
        for send_node in self.send_nodes:
            if send_node.target_name_property() == old_name:
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

    def rename_device(self, device_to_rename, new_name):
        old_name = device_to_rename.name
        if old_name == new_name or new_name == '':
            return

        if new_name in self.sources:
            print(f"Error: A source with the name '{new_name}' already exists.")
            if hasattr(device_to_rename, 'source_name_property'):
                device_to_rename.source_name_property.set(old_name, propagate=False)
            return

        dependent_senders = [
            node for node in self.send_nodes if hasattr(node, 'name') and node.name == old_name
        ]

        dependent_receivers = [
            node for node in self.receive_nodes if hasattr(node, 'name') and node.name == old_name
        ]

        patcher_path = device_to_rename.get_patcher_path()
        old_path_prefix = self.registry.sanitize_path_components([patcher_path, old_name])
        new_path_prefix = self.registry.sanitize_path_components([patcher_path, new_name])
        self.registry.rename_path(old_path_prefix, new_path_prefix)

        if old_name in self.sources:
            self.sources.pop(old_name)

        if old_name in self.targets:
            self.targets.pop(old_name)

        device_to_rename.name = new_name
        self.sources[new_name] = device_to_rename
        self.targets[new_name] = device_to_rename

        device_to_rename.source_name_property.set(new_name, propagate=False)
        device_to_rename.target_name_property.set(new_name, propagate=False)
        path_list = device_to_rename._get_registry_path_components()
        device_to_rename.adjust_path(path_list)

        for send_node in dependent_senders:
            if hasattr(send_node, 'update_target_name_state'):
                send_node.update_target_name_state(new_name)

        for receive_node in dependent_receivers:
            if hasattr(receive_node, 'update_source_name_state'):
                receive_node.update_source_name_state(new_name)

        for node in OSCManagerNode.instances:
            node.update_sources()
            node.update_targets()

    def rename_source(self, source_to_rename, new_name: str):
        old_name = source_to_rename.name
        if old_name == new_name or new_name == '':
            return

        if new_name in self.sources:
            print(f"Error: A source with the name '{new_name}' already exists.")
            if hasattr(source_to_rename, 'source_name_property'):
                source_to_rename.source_name_property.set(old_name, propagate=False)
            return

        # 1. CAPTURE all dependent nodes BEFORE making changes.
        dependent_receivers = [
            node for node in self.receive_nodes if hasattr(node, 'name') and node.name == old_name
        ]

        # 2. Tell the REGISTRY to move the source's main container.
        patcher_path = source_to_rename.get_patcher_path()
        old_path_prefix = self.registry.sanitize_path_components([patcher_path, old_name])
        new_path_prefix = self.registry.sanitize_path_components([patcher_path, new_name])
        self.registry.rename_path(old_path_prefix, new_path_prefix)

        # 3. Update the manager's internal state.
        if old_name in self.sources:
            self.sources.pop(old_name)
        source_to_rename.name = new_name
        self.sources[new_name] = source_to_rename
        source_to_rename.source_name_property.set(new_name, propagate=False)
        path_list = source_to_rename._get_registry_path_components()
        source_to_rename.adjust_path(path_list)

        # 4. INSTRUCT each dependent node to update itself.
        for receive_node in dependent_receivers:
            if hasattr(receive_node, 'update_source_name_state'):
                receive_node.update_source_name_state(new_name)

        # 5. Update the OSCManagerNode UI.
        for node in OSCManagerNode.instances:
            node.update_sources()

    def rename_target(self, target_to_rename, new_name: str):
        old_name = target_to_rename.name
        if old_name == new_name or new_name == '':
            return

        if new_name in self.targets:
            print(f"Error: A target with the name '{new_name}' already exists.")
            if hasattr(target_to_rename, 'target_name_property'):
                target_to_rename.target_name_property.set(old_name, propagate=False)
            return

        # 1. CAPTURE dependent nodes.
        dependent_senders = [
            node for node in self.send_nodes if hasattr(node, 'name') and node.name == old_name
        ]

        # 2. Tell the REGISTRY to move the target's main container.
        patcher_path = target_to_rename.get_patcher_path()
        old_path_prefix = self.registry.sanitize_path_components([patcher_path, old_name])
        new_path_prefix = self.registry.sanitize_path_components([patcher_path, new_name])
        self.registry.rename_path(old_path_prefix, new_path_prefix)

        # 3. Update manager's state.
        if old_name in self.targets:
            self.targets.pop(old_name)
        target_to_rename.name = new_name
        self.targets[new_name] = target_to_rename
        target_to_rename.target_name_property.set(new_name, propagate=False)
        path_list = target_to_rename._get_registry_path_components()
        target_to_rename.adjust_path(path_list)

        # 4. INSTRUCT dependent nodes.
        for send_node in dependent_senders:
            if hasattr(send_node, 'update_target_name_state'):
                send_node.update_target_name_state(new_name)

        # 5. Update UI.
        for node in OSCManagerNode.instances:
            node.update_targets()


class OSCQueryRegistry:
    def __init__(self):
        self.registry = {
            'DESCRIPTION': 'DPG_OSC_MANAGER',
            'CONTENTS': {}
        }

    @staticmethod
    def compose_path_string(path_input) -> str:
        """
        Takes ANY path-like input and returns a perfectly formatted OSC path string.
        This is the SINGLE SOURCE OF TRUTH for creating path strings.
        """
        sanitized_components = OSCQueryRegistry.sanitize_path_components(path_input)
        if not sanitized_components:
            return "/"
        return "/" + "/".join(sanitized_components)


    def get_param_registry_container_for_path(self, patch_path):
        return self.get_registry_container_for_path(patch_path)

    def get_registry_container_for_path(self, patch_path):
        path_list = self.sanitize_path_components(patch_path)
        reg = self.registry['CONTENTS']
        previous = self.registry['CONTENTS']
        if reg is not None:
            for domain in path_list:
                if domain in reg:
                    previous = reg[domain]
                    reg = previous['CONTENTS']
                    if reg is None:
                        print('get_param_registry_container_for_path reg = None')
                else:
                    return None
        return previous

    def remove_path_from_registry(self, patch_path):
        patch_list = self.sanitize_path_components(patch_path)
        if not patch_list:
            return

        def remove_recursive(current_reg, path_idx):
            domain = patch_list[path_idx]
            if domain not in current_reg:
                return
            
            if path_idx == len(patch_list) - 1:
                current_reg.pop(domain)
            else:
                child_node = current_reg[domain]
                next_reg = child_node.get('CONTENTS')
                if next_reg is not None:
                    remove_recursive(next_reg, path_idx + 1)
                    # Prune the directory itself if it is now empty and not a legitimate leaf parameter
                    if not next_reg and 'TYPE' not in child_node:
                        current_reg.pop(domain)

        reg = self.registry.get('CONTENTS')
        if reg is not None:
            remove_recursive(reg, 0)

    def add_path_to_registry(self, patch_path):
        patch_path = self.sanitize_path_components(patch_path)
        reg = self.registry['CONTENTS']
        node = None
        if reg is not None:
            for domain in patch_path:
                if domain not in reg:
                    reg[domain] = {'CONTENTS':{}}
                node = reg[domain]
                reg = node['CONTENTS']
                if reg is None:
                    break
        if type(patch_path) == list:
            patch_path = '/' + '/'.join(patch_path)
        return node, patch_path

    def insert_param_dict_into_registry(self, param_dict):
        path_list = self.sanitize_path_components(param_dict['FULL_PATH'])
        
        reg, path = self.add_path_to_registry(path_list)
        if reg is not None:
            keys = list(param_dict.keys())
            for key in keys:
                reg[key] = param_dict[key]
        return reg, param_dict['FULL_PATH']

    def change_path(self, old_path, new_path):
        old_path = self.sanitize_path_components(old_path)
        new_path = self.sanitize_path_components(new_path)
        reg = self.get_param_registry_container_for_path(old_path)
        if reg is not None:
            self.remove_path_from_registry(old_path)
            reg['FULL_PATH'] = self.compose_path_string(new_path)
            self.insert_param_dict_into_registry(reg)
        else:
            print('change_path', old_path, 'not found')

    def set_host_info(self, path, target_port=None, source_port=None, transport='UDP', ip=None):
        reg = self.get_registry_container_for_path(path)
        if reg is not None:
            info = {}
            if source_port is not None and target_port is not None:
                info['DPG_TYPE'] = 'OSC_DEVICE'
                info['OSC_SOURCE_PORT'] = source_port
                info['OSC_TARGET_PORT'] = target_port
            elif source_port is not None:
                info['DPG_TYPE'] = 'DPG_SOURCE'
                info['OSC_SOURCE_PORT'] = source_port
            elif target_port is not None:
                info['DPG_TYPE'] = 'DPG_TARGET'
                info['OSC_TARGET_PORT'] = target_port

            info['OSC_TRANSPORT'] = transport
            if ip is not None:
                info['OSC_IP'] = ip
            info['NAME'] = path[-1]
            reg['HOST_INFO'] = info

    def set_flow(self, path_list, flow):
        path_list = self.sanitize_path_components(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            reg['FLOW'] = flow

    def set_description(self, path_list, description):
        path_list = self.sanitize_path_components(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            reg['DESCRIPTION'] = description

    def set_value_for_path(self, path, value):
        if path is None or path == '':
            return

        param_container = self.get_param_registry_container_for_path(path)
        if param_container is not None:
            # The OSCQuery spec for VALUE is an array of values.
            if not isinstance(value, list):
                value = any_to_list(value)
            param_container['VALUE'] = value

    def set_value(self, path_list, value):
        path_list = self.sanitize_path_components(path_list)
        reg = self.get_param_registry_container_for_path(path_list)
        if reg is not None:
            if type(value) == list:
                reg['VALUE'] = value
            else:
                reg['VALUE'] = any_to_list(value)

    def prepare_basic_param_dict(self, type, path_list, access=3):
        path_string = self.compose_path_string(path_list)
        description = path_list[-1] if path_list else ''
        param_dict = {
            'TYPE': type,
            'DESCRIPTION': description,
            'ACCESS': access,
            'FULL_PATH': path_string,
            'FLOW': 'BOTH'
        }
        return param_dict

    def add_generic_device_to_registry(self, path_list):
        sanitized_path_list = self.sanitize_path_components(path_list)
        generic_device_dict = self.prepare_basic_param_dict('b', sanitized_path_list, access=1)
        generic_device_dict['FLOW'] = 'BOTH'
        return self.insert_param_dict_into_registry(generic_device_dict)

    def add_source_to_registry(self, path_list):
        pass

    def add_target_to_registry(self, path_list):
        pass

    def add_generic_receiver_to_registry(self, path_list):
        sanitized_path_list = self.sanitize_path_components(path_list)
        generic_receiver_dict = self.prepare_basic_param_dict('b', sanitized_path_list, access=1)
        generic_receiver_dict['FLOW'] = 'IN'
        return self.insert_param_dict_into_registry(generic_receiver_dict)

    def add_generic_sender_to_registry(self, path_list):
        sanitized_path_list = self.sanitize_path_components(path_list)
        generic_sender_dict = self.prepare_basic_param_dict('b', sanitized_path_list, access=2)
        generic_sender_dict['FLOW'] = 'OUT'
        return self.insert_param_dict_into_registry(generic_sender_dict)

    def add_float_to_registry(self, path_list, value=0.0, min=0.0, max=1.0):
        sanitized_path_list = self.sanitize_path_components(path_list)
        float_param_dict = self.prepare_basic_param_dict('f', sanitized_path_list)
        if float_param_dict is None:
            return
        float_param_dict['RANGE'] = [{'MIN': min, 'MAX': max}]
        float_param_dict['VALUE'] = [value]
        return self.insert_param_dict_into_registry(float_param_dict)

    def add_int_to_registry(self, path_list, value=0, min=0, max=100):
        sanitized_path_list = self.sanitize_path_components(path_list)
        int_param_dict = self.prepare_basic_param_dict('i', sanitized_path_list)
        if int_param_dict is None:
            return
        int_param_dict['RANGE'] = [{'MIN': min, 'MAX': max}]
        int_param_dict['VALUE'] = [value]
        return self.insert_param_dict_into_registry(int_param_dict)

    def add_bool_to_registry(self, path_list, value=False):
        sanitized_path_list = self.sanitize_path_components(path_list)
        bool_param_dict = self.prepare_basic_param_dict('F', sanitized_path_list)
        if bool_param_dict is None:
            return
        bool_param_dict['VALUE'] = [value]
        return self.insert_param_dict_into_registry(bool_param_dict)

    def add_string_to_registry(self, path_list, value=''):
        sanitized_path_list = self.sanitize_path_components(path_list)
        string_param_dict = self.prepare_basic_param_dict('s', sanitized_path_list)
        if string_param_dict is None:
            return
        string_param_dict['VALUE'] = [value]
        return self.insert_param_dict_into_registry(string_param_dict)

    def add_string_menu_to_registry(self, path_list, value='', choices=None):
        sanitized_path_list = self.sanitize_path_components(path_list)
        string_menu_param_dict = self.prepare_basic_param_dict('s', sanitized_path_list)
        if string_menu_param_dict is None:
            return
        string_menu_param_dict['VALUE'] = [value]
        if choices is not None and len(choices) > 0:
            string_menu_param_dict['VALS'] = list(choices)
        return self.insert_param_dict_into_registry(string_menu_param_dict)

    def add_button_to_registry(self, path_list):
        sanitized_path_list = self.sanitize_path_components(path_list)
        button_param_dict = self.prepare_basic_param_dict('N', sanitized_path_list, access=0)
        if button_param_dict is None:
            return
        return self.insert_param_dict_into_registry(button_param_dict)

    def add_float_array_to_registry(self, path_list, array, min=0.0, max=1.0):
        sanitized_path_list = self.sanitize_path_components(path_list)
        array = any_to_list(array)
        count = len(array)
        array_param_dict = self.prepare_basic_param_dict('f' * count, sanitized_path_list)
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
        return self.insert_param_dict_into_registry(array_param_dict)

    def add_int_array_to_registry(self, path_list, array, min=0.0, max=1.0):
        sanitized_path_list = self.sanitize_path_components(path_list)
        array = any_to_list(array)
        count = len(array)
        array_param_dict = self.prepare_basic_param_dict('i' * count, sanitized_path_list)
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
        return self.insert_param_dict_into_registry(array_param_dict)

    def get_container_for_path(self, path, create_if_missing=False):
        """
        Helper function to get a container dictionary for a given path.
        Accepts either a list of components or a path string.
        Returns the parent container and the final key.
        """
        path_list = []
        if isinstance(path, str):
            print('get_container_for_path', path.split('/'))
            path_list = self.sanitize_path_components(path.split('/'))
        elif isinstance(path, list):
            print('get_container_for_path (list)', path)
            path_list = self.sanitize_path_components(path)

        if not path_list:
            return None, None

        reg = self.registry['CONTENTS']
        for domain in path_list[:-1]:
            if domain not in reg:
                if create_if_missing:
                    reg[domain] = {'CONTENTS': {}}
                else:
                    return None, None
            reg = reg[domain]['CONTENTS']

        return reg, path_list[-1]

    def rename_path(self, old_path_prefix: list, new_path_prefix: list):
        """
        Moves an entire branch of the registry from an old path to a new path
        and recursively updates the FULL_PATH attribute of all children.
        This version operates "in-place" to ensure references are maintained.
        """
        old_path_prefix_clean = self.sanitize_path_components(old_path_prefix)
        new_path_prefix_clean = self.sanitize_path_components(new_path_prefix)

        if not old_path_prefix_clean or not new_path_prefix_clean or old_path_prefix_clean == new_path_prefix_clean:
            return

        # 1. Find the parent container and the key of the branch to move.
        old_parent_container, old_key = self.get_container_for_path(old_path_prefix)
        if old_parent_container is None or old_key not in old_parent_container:
            print(f"Registry rename failed: could not find old path {'/'.join(old_path_prefix)}")
            return

        # 2. Find the new parent container and key.
        new_parent_container, new_key = self.get_container_for_path(new_path_prefix, create_if_missing=True)
        if new_parent_container is None:
            print(f"Registry rename failed: could not create new path {'/'.join(new_path_prefix)}")
            return

        # This check is crucial if old and new parent are the same
        if old_parent_container is new_parent_container and new_key in new_parent_container:
            print(f"Registry rename failed: destination key '{new_key}' already exists in the same container.")
            return

        # Get a direct reference to the dictionary that will be moved.
        node_to_move = old_parent_container[old_key]
        new_parent_container[new_key] = node_to_move
        del old_parent_container[old_key]

        # If the old parent container is now empty, we should clean it up.
        # if 'CONTENTS' in old_parent_container and not old_parent_container['CONTENTS']:
        #     # This is a bit tricky, would need to traverse back up. Let's omit for now for simplicity
        #     # and focus on the main bug. The "empty stub" bug is a separate cleanup issue.
        #     pass
        #
        # # 4. Recursively update FULL_PATH in all descendants using the reference.
        # # This will now modify the dictionaries *in their new location* within self.registry.
        # # We need to create the full path strings for the recursive update.
        # old_full_prefix_str = '/' + '/'.join(old_path_prefix)
        # new_full_prefix_str = '/' + '/'.join(new_path_prefix)
        #
        # # Update the moved node itself
        # if 'FULL_PATH' in node_to_move:
        #     # (same logic as in the recursion)
        #     current_components = self.sanitize_path_components(node_to_move['FULL_PATH'])
        #     if current_components[:len(old_path_prefix_clean)] == old_path_prefix_clean:
        #         remainder = current_components[len(old_path_prefix_clean):]
        #         node_to_move['FULL_PATH'] = self.compose_path_string(new_path_prefix_clean + remainder)

        # And recurse into its contents
        self._recursive_path_update(node_to_move.get('CONTENTS', {}), old_path_prefix_clean, new_path_prefix_clean)

    def _recursive_path_update(self, container_dict, old_prefix_list, new_prefix_list):
        """
        Recursively updates FULL_PATH using component lists, not fragile strings.
        """

        if not isinstance(container_dict, dict):
            return

        if 'FULL_PATH' in container_dict:
            current_path_components = self.sanitize_path_components(container_dict['FULL_PATH'])
            if current_path_components[:len(old_prefix_list)] == old_prefix_list:
                remainder = current_path_components[len(old_prefix_list):]
                # Reconstruct using the official composer
                container_dict['FULL_PATH'] = self.compose_path_string(new_prefix_list + remainder)

            if 'CONTENTS' in container_dict:
                # Pass the component lists down the recursion
                self._recursive_path_update(container_dict['CONTENTS'], old_prefix_list, new_prefix_list)

    @staticmethod
    def sanitize_path_components(path_input) -> list[str]:
        """
        Takes ANY path-like input (string, list) and returns a clean list of
        string components. GUARANTEES no slashes, no spaces, no empty strings.
        This is the single source of truth for cleaning path parts.
        """
        components_to_process = []
        if isinstance(path_input, str):
            clean_input = path_input.replace(' ', '_')
            components_to_process = clean_input.split('/')
        elif isinstance(path_input, (list, tuple)):
            for item in path_input:
                if item is None: continue
                clean_item_str = str(item).strip().replace(' ', '_')
                components_to_process.extend(clean_item_str.split('/'))
        elif path_input is not None:
            components_to_process = str(path_input).replace(' ', '_').split('/')

        # Filter out any empty strings that result from leading/trailing/double slashes
        sanitized = [comp for comp in components_to_process if comp]
        return sanitized



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
                pretty_json = json.dumps(self.osc_manager.registry.registry, indent=4, sort_keys=True)
                print(pretty_json)


class OSCManagerNode(OSCBase, Node):
    instances = []

    @staticmethod
    def factory(name, data, args=None):
        node = OSCManagerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.source_or_target = self.add_property('type of port', widget_type='combo', callback=self.source_target_changed)
        self.source_or_target.widget.combo_items = ['source', 'target', '']

        self.current_port = None
        self.port_list = self.add_input('sources', widget_type='combo', callback=self.port_changed)
        self.name = self.add_input('name', widget_type='text_input', callback=self.name_changed)
        self.ip = self.add_input('ip address', widget_type='text_input', callback=self.ip_changed)
        self.port = self.add_input('port', widget_type='drag_int', callback=self.port_number_changed)
        self.instances.append(self)

    def custom_create(self, from_file):
        self.source_or_target.set('')

    def custom_cleanup(self):
        if self in self.instances:
            self.instances.remove(self)

    def update_sources(self):
        if self.source_or_target() == 'source':
            value = self.port_list()
            source_ports = list(self.osc_manager.sources.keys())
            self.port_list.widget.combo_items = source_ports
            dpg.configure_item(self.port_list.widget.uuid, items=source_ports)
            if value not in source_ports:
                self.source_target_changed()

    def update_targets(self):
        if self.source_or_target() == 'target':
            value = self.port_list()
            target_ports = list(self.osc_manager.targets.keys())
            self.port_list.widget.combo_items = target_ports
            dpg.configure_item(self.port_list.widget.uuid, items=target_ports)
            if value not in target_ports:
                self.source_target_changed()

    def source_target_changed(self):
        value = self.source_or_target()
        if value == 'source':
            source_ports = list(self.osc_manager.sources.keys())
            self.port_list.widget.combo_items = source_ports
            dpg.configure_item(self.port_list.widget.uuid, items=source_ports)
            self.port_list.set_label('source')

            if len(source_ports) > 0:
                name = source_ports[0]
                self.current_port = self.osc_manager.sources[name]
                self.port_list.set(name)
                self.name.set(name)
                self.ip.set('')
                self.port.set(self.current_port.source_port)
            else:
                self.port_list.set('')
                self.name.set('')

        elif value == 'target':
            target_ports = list(self.osc_manager.targets.keys())
            self.port_list.widget.combo_items = target_ports
            dpg.configure_item(self.port_list.widget.uuid, items=target_ports)

            self.port_list.set_label('target')

            if len(target_ports) > 0:
                name = target_ports[0]
                self.current_port = self.osc_manager.targets[name]
                self.port_list.set(name)
                self.name.set(name)
                self.ip.set(self.current_port.ip)
                self.port.set(self.current_port.target_port)
            else:
                self.port_list.set('')
                self.name.set('')

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
                old_name = self.current_port.target_name_property()
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


class OSCRegistrableMixin:
    """
    A mixin class that provides OSC registry functionality.
    It assumes it is being mixed into a class that inherits from dpg_system.node.Node.
    """

    def _registerable_init(self):
        """
        Initializer for the mixin's attributes. Call this from the main class's __init__.
        """
        self.path = None
        # self.add_option is expected to exist on the instance (from the Node class)
        self.path_option = self.add_option('path', widget_type='text_input', width=150, callback=self.set_path)

    def _registerable_custom_create(self):
        """Life-cycle hook to be called from the main class's custom_create."""
        self.register()

    def _registerable_cleanup(self):
        """Life-cycle hook to be called from the main class's cleanup."""
        self.unregister()

    def set_path(self):
        pass

    def _get_registry(self):
        """Return the correct registry for this node based on patcher hierarchy.

        Returns (registry, is_service_scoped) tuple. When is_service_scoped is True,
        callers should strip the patcher path prefix from path components since the
        service registry is already scoped by patcher name.
        """
        editor = getattr(self, 'my_editor', None)
        if editor is None and hasattr(self, 'app') and self.app is not None:
            editor = self.app.get_current_editor()

        if self.osc_manager and editor is not None:
            registry = self.osc_manager.get_registry_for_editor(editor)
            is_global = registry is self.osc_manager.registry
            return registry, not is_global
        if self.osc_manager:
            return self.osc_manager.registry, False
        return None, False

    def _should_register_in_service(self):
        """Check if this node should register in any exported OSCQuery registry.

        Non-local widgets (proxy/peer) should NOT register
        in local registries — they are remote controls, not local parameters.
        """
        if hasattr(self, 'mode_option'):
            try:
                if self.mode_option() != 'local':
                    return False
            except Exception:
                pass
        return True

    def register(self):
        """
        Registers the node with the OSCQueryRegistry by calling the abstract
        _create_registry_entry method.
        """
        if self.osc_manager is None:
            return

        registry, is_service_scoped = self._get_registry()
        if registry is None:
            return

        # Proxy widgets should not register in ANY exported OSC registries
        if not self._should_register_in_service():
            return

        raw_path_components = self._get_registry_path_components()

        # Strip patcher path prefix and service name for service-scoped registries
        # (the service is already scoped by patcher name)
        if is_service_scoped and raw_path_components:
            # First component is get_patcher_path() — always remove it
            raw_path_components = raw_path_components[1:]
            # Second component is typically self.name (target/service name).
            # If it matches the service name, strip it too — it's redundant.
            if raw_path_components:
                svc_name = None
                if hasattr(self, 'my_editor') and self.my_editor:
                    svc_name = self.osc_manager.get_service_name_for_editor(self.my_editor)
                if svc_name and raw_path_components[0] == svc_name:
                    raw_path_components = raw_path_components[1:]

        # Use the composer to get the final, clean path string
        final_path_string = registry.compose_path_string(raw_path_components)

        _, registered_path_string = self._create_registry_entry(raw_path_components)

        if registered_path_string:
            self.path = final_path_string
            # This is the key fix for the UI widget.
            self.path_option.set(self.path)
        else:
            self.path = ''
            self.path_option.set('Registration Failed')

    def unregister(self):
        """Unregisters the node by removing its path from the registry."""
        if self.osc_manager and self.path:
            registry, is_service_scoped = self._get_registry()
            if registry:
                path_components = self._get_registry_path_components()
                if is_service_scoped and path_components:
                    path_components = path_components[1:]
                    if path_components:
                        svc_name = None
                        if hasattr(self, 'my_editor') and self.my_editor:
                            svc_name = self.osc_manager.get_service_name_for_editor(self.my_editor)
                        if svc_name and path_components[0] == svc_name:
                            path_components = path_components[1:]
                            
                registry.remove_path_from_registry(path_components)
            self.path = None
            self.path_option.set('Unregistered')

    # def _update_registration(self):
    #     """A robust way to handle changes: unregister the old state, register the new."""
    #     self.unregister()
    #     self.register()

    def _update_registration(self, old_path_components: list = None):
        """
        A robust way to handle changes. It unregisters the old state and
        registers the new one.
        """
        # --- Unregister Step ---
        if self.osc_manager:
            registry, is_service_scoped = self._get_registry()
            if registry:
                path_stuff = old_path_components if old_path_components is not None else self._get_registry_path_components()
                
                if is_service_scoped and path_stuff:
                    path_stuff = path_stuff[1:]
                    if path_stuff:
                        svc_name = None
                        if hasattr(self, 'my_editor') and self.my_editor:
                            svc_name = self.osc_manager.get_service_name_for_editor(self.my_editor)
                        if svc_name and path_stuff[0] == svc_name:
                            path_stuff = path_stuff[1:]

                components_to_unregister = registry.sanitize_path_components(path_stuff)
                if components_to_unregister:
                    registry.remove_path_from_registry(components_to_unregister)

        # We nullify the path to ensure the register step creates a new one.
        self.path = None

        # --- Register Step ---
        self.register()  # This will use the NEW state of the node.

    # --- Abstract methods for subclasses to implement ---

    # ok
    def _get_registry_path_components(self) -> list:
        """Subclasses must implement this to define their OSC path."""
        raise NotImplementedError("Subclasses must implement _get_registry_path_components")

    # ok
    def _create_registry_entry(self, path_components: list) -> str:
        """Subclasses must implement this to define their registry entry type."""
        raise NotImplementedError("Subclasses must implement _create_registry_entry")


class OSCTarget(OSCBase):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ip = '127.0.0.1'

        patcher = Node.app.get_current_editor()
        name = patcher.patch_name
        name = name.replace(' ', '_')
        self.name = name
        self.node = None
        self.target_path = None

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

        self.osc_format = 0
        self.connected = False
        self.client = None
        self.send_nodes = {}

    def register(self):
        if self.osc_manager is None:
            return

        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            _, self.target_path = self.osc_manager.registry.add_path_to_registry([patcher_path, self.name])
            self.osc_manager.registry.set_host_info(path=[patcher_path, self.name], target_port=self.target_port, ip=self.ip)
            self.node.path_option.set(self.target_path)

    def unregister(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name])

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

    def target_changed(self):
        if hasattr(self, 'target_name_property'):
            name = self.target_name_property()
            if name != self.name:
                self.update_target_name_state(name)

            if hasattr(self, 'target_port_property') and hasattr(self, 'target_ip_property'):
                port = any_to_int(self.target_port_property())
                ip = self.target_ip_property()
                if port != self.target_port or ip != self.ip:
                    self.destroy_client()
                    self.target_port = port
                    self.ip = ip
                    self.create_client()

    def update_target_name_state(self, new_target_name: str):
        print('OSCTarget update_target_name_state')
        """Called by OSCManager when the source this node is listening to has been renamed."""
        self.osc_manager.rename_target(self, new_target_name)
        if isinstance(self, Node):
            self.unparsed_args = [str(self.name), str(self.ip), str(self.target_port)]
            print('target', self.unparsed_args)


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

    def cleanup(self):
        self.destroy_client()

    def set_path(self):
        if hasattr(self, 'path_option'):
            path = self.path_option()
            self.adjust_path(path)

    def adjust_path(self, path):
        path_list = self.osc_manager.registry.sanitize_path_components(path)
        final_path_string = self.osc_manager.registry.compose_path_string(path_list)
        if hasattr(self, 'path_option'):
            self.path_option.set(final_path_string)
        self.target_path = path


class OSCTargetNode(OSCTarget, OSCRegistrableMixin, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCTargetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCTarget.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.target_changed)
        self.target_ip_property = self.add_property('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.target_port), callback=self.target_changed)
        self.node = self
        self._registerable_init()

    def custom_create(self, from_file):
        OSCTarget.custom_create(self, from_file)
        self._registerable_custom_create()

    def _get_registry_path_components(self):
        return [self.get_patcher_path(), self.name]

    def cleanup(self):
        OSCTarget.cleanup(self)
        self._registerable_cleanup()

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
        self.source_path = None
        self.source_name_property = None
        self.source_port_property = None

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
            _, self.source_path = self.osc_manager.registry.add_path_to_registry([patcher_path, self.name])
            self.osc_manager.registry.set_host_info([patcher_path, self.name], source_port=self.source_port)
            self.node.path_option.set(self.source_path)

    def unregister(self):
        if self.node is not None:
            patcher_path = self.node.get_patcher_path()
            self.osc_manager.registry.remove_path_from_registry([patcher_path, self.name])

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
                for rn in self.receive_nodes[address]:
                    rn.receive(args)
                self.lock.release()
                return
            self.relay_osc(address, args)
            self.lock.release()

    def create_dispatcher(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.osc_handler)

    def stop_serving(self):
        for address in self.receive_nodes:
            for receive_node in self.receive_nodes[address]:
                receive_node.source_going_away(self)
        if self.dispatcher:
            self.dispatcher = None

    def custom_create(self, from_file):
        pass

    def output_message_directly(self, address, args):
        if hasattr(self, 'output'):
            out_list = [address]
            if args is not None and len(args) > 0:
                out_list.append(list(args))
            self.output.send(out_list)

    def relay_osc(self, address, args):
        if address in self.receive_nodes:
            for rn in self.receive_nodes[address]:
                rn.receive(args, address)
            return
        else:
            if '/' in address:
                sub_addresses = address.split('/')
                length = len(sub_addresses)
                for i in range(1, length):
                    temp = '/'.join(sub_addresses[:-i])
                    if temp in self.receive_nodes:
                        sub = ['/' + '/'.join(sub_addresses[-i:])] + list(args)
                        for rn in self.receive_nodes[temp]:
                            rn.receive(sub)
                        return
        self.output_message_directly(address, args)

    # def output_message_directly(self, address, args):
    #     pass
    #
    def register_receive_node(self, receive_node):
        addresses = receive_node.get_addresses()
        if type(addresses) == str:
            addresses = [addresses]
        if type(addresses) == list:
            for address in addresses:
                if address not in self.receive_nodes:
                    self.receive_nodes[address] = []
                if receive_node not in self.receive_nodes[address]:
                    self.receive_nodes[address].append(receive_node)
        receive_node.set_source(self)  # would match OSCTarget

    def unregister_receive_node(self, receive_node):
        addresses = receive_node.get_addresses()
        if type(addresses) == str:
            addresses = [addresses]
        if type(addresses) == list:
            for address in addresses:
                if address in self.receive_nodes:
                    if receive_node in self.receive_nodes[address]:
                        self.receive_nodes[address].remove(receive_node)
                    if not self.receive_nodes[address]:
                        del self.receive_nodes[address]

    def disconnect_from_receive_nodes(self):
        for address, node_list in list(self.receive_nodes.items()):
            for receive_node in node_list:
                if receive_node is not None:
                    receive_node.source_going_away(self)
        self.receive_nodes.clear()

    def handle_name_change(self, name, force=False):
        if name != self.name:
            self.osc_manager.rename_source(self, name)

    def set_path(self):
        if hasattr(self, 'path_option'):
            path = self.path_option()
            self.adjust_path(path)

    def adjust_path(self, path):
            path_list = self.osc_manager.registry.sanitize_path_components(path)
            final_path_string = self.osc_manager.registry.compose_path_string(path_list)
            if hasattr(self, 'path_option'):
                self.path_option.set(final_path_string)
            self.source_path = path

    def update_source_name_state(self, new_source_name: str):
        """Called by OSCManager when the source this node is listening to has been renamed."""
        self.osc_manager.rename_source(self, new_source_name)
        if isinstance(self, Node):
            self.unparsed_args = [str(self.name), str(self.source_port)]
            print('source', self.unparsed_args)

    def update_output_label(self):
        pass

    def cleanup(self):
        self.osc_manager.remove_source(self)
        global server_to_stop
        server_to_stop = self
        stop_thread = threading.Thread(target=stop_server)
        stop_thread.start()
        i = 0
        while self.server is not None:
            i += 1

    def source_changed(self):
        name = None
        if hasattr(self, 'source_name_property'):
            name = any_to_string(self.source_name_property())
        if name is not None:
            if name != self.name:
                print('source_changed')
                self.update_source_name_state(name)
            if hasattr(self, 'source_port_property'):
                port = any_to_int(self.source_port_property())
                if port != self.source_port:
                    self.destroy_server()
                    self.source_port = port
                    self.start_serving()


class OSCThreadingSource(OSCSource):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def start_serving(self):
        try:
            self.create_dispatcher()
            self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.source_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

        except Exception as e:
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
    threading.Thread(target=loop.run_forever, daemon=True).start()
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


class OSCSourceNode(OSCThreadingSource, OSCRegistrableMixin, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCThreadingSource.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.source_name_property = self.add_property('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_property('port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.use_queue = self.add_option('use_queue', widget_type='checkbox', default_value=False)
        self.output = self.add_output('osc received')
        self.start_serving()
        self.node = self
        self._registerable_init()

    def custom_create(self, from_file):
        OSCSource.custom_create(self, from_file)
        self._registerable_custom_create()

    def _get_registry_path_components(self):
        return [self.get_patcher_path(), self.name]

    def cleanup(self):
        OSCSource.cleanup(self)
        self._registerable_cleanup()


class PipoMotionSourceNode(Node):
    """Receives OSC from a PiPo-Motion sensor and outputs yaw, pitch, roll as separate floats and acceleration as an np.array."""

    @staticmethod
    def factory(name, data, args=None):
        node = PipoMotionSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.lock = threading.Lock()

        # default port
        self.source_port = 8000
        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.source_port = arg

        self.port_property = self.add_property('port', widget_type='input_int', default_value=self.source_port, callback=self.port_changed)
        self.yaw_output = self.add_output('yaw')
        self.pitch_output = self.add_output('pitch')
        self.roll_output = self.add_output('roll')
        self.acc_output = self.add_output('acc')

        self.acc = np.zeros(3, dtype=np.float32)

        self.start_serving()

    def port_changed(self):
        port = any_to_int(self.port_property())
        if port != self.source_port:
            self.destroy_server()
            self.source_port = port
            self.start_serving()

    def osc_handler(self, address, *args):
        if self.lock.acquire(blocking=True):
            try:
                if len(args) > 0:
                    value = float(args[0])
                else:
                    value = 0.0

                if address == '/motion/yaw':
                    self.yaw_output.send(value)
                elif address == '/motion/pitch':
                    self.pitch_output.send(value)
                elif address == '/motion/roll':
                    self.roll_output.send(value)
                elif address == '/motion/accX':
                    self.acc[0] = value
                elif address == '/motion/accY':
                    self.acc[1] = value
                elif address == '/motion/accZ':
                    self.acc[2] = value
                    self.acc_output.send(self.acc.copy())
            finally:
                self.lock.release()

    def start_serving(self):
        try:
            self.dispatcher = Dispatcher()
            self.dispatcher.map('/motion/yaw', self.osc_handler)
            self.dispatcher.map('/motion/pitch', self.osc_handler)
            self.dispatcher.map('/motion/roll', self.osc_handler)
            self.dispatcher.map('/motion/accX', self.osc_handler)
            self.dispatcher.map('/motion/accY', self.osc_handler)
            self.dispatcher.map('/motion/accZ', self.osc_handler)
            self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.source_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
        except Exception as e:
            print(f'PipoMotionSourceNode: failed to start server on port {self.source_port}: {e}')
            if self.server:
                self.server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=2.0)
                self.server_thread = None
            self.server = None
            self.dispatcher = None

    def destroy_server(self):
        if self.dispatcher:
            self.dispatcher = None
        server = self.server
        thread = self.server_thread
        self.server = None
        self.server_thread = None
        if server is not None:
            server.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)

    def cleanup(self):
        def _stop():
            self.destroy_server()
        stop_thread = threading.Thread(target=_stop, daemon=True)
        stop_thread.start()


class PipoRangeSourceNode(Node):
    """Receives OSC from a PiPo range sensor and outputs distance as a float."""

    @staticmethod
    def factory(name, data, args=None):
        node = PipoRangeSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.server_thread = None
        self.dispatcher = None
        self.lock = threading.Lock()

        self.source_port = 8001
        if args is not None:
            for i in range(len(args)):
                arg, t = decode_arg(args, i)
                if t == int:
                    self.source_port = arg

        self.port_property = self.add_property('port', widget_type='input_int', default_value=self.source_port, callback=self.port_changed)
        self.dist_output = self.add_output('dist')

        self.start_serving()

    def port_changed(self):
        port = any_to_int(self.port_property())
        if port != self.source_port:
            self.destroy_server()
            self.source_port = port
            self.start_serving()

    def osc_handler(self, address, *args):
        if self.lock.acquire(blocking=True):
            try:
                if len(args) > 0:
                    value = float(args[0])
                else:
                    value = 0.0
                if address == '/range/dist':
                    self.dist_output.send(value)
            finally:
                self.lock.release()

    def start_serving(self):
        try:
            self.dispatcher = Dispatcher()
            self.dispatcher.map('/range/dist', self.osc_handler)
            self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.source_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
        except Exception as e:
            print(f'PipoRangeSourceNode: failed to start server on port {self.source_port}: {e}')
            if self.server:
                self.server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=2.0)
                self.server_thread = None
            self.server = None
            self.dispatcher = None

    def destroy_server(self):
        if self.dispatcher:
            self.dispatcher = None
        server = self.server
        thread = self.server_thread
        self.server = None
        self.server_thread = None
        if server is not None:
            server.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)

    def cleanup(self):
        def _stop():
            self.destroy_server()
        stop_thread = threading.Thread(target=_stop, daemon=True)
        stop_thread.start()


class OSCAsyncIOSourceNode(OSCAsyncIOSource, OSCRegistrableMixin, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCAsyncIOSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCAsyncIOSource.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.source_name_property = self.add_input('name', widget_type='text_input', default_value=self.name, callback=self.source_changed)
        self.source_port_property = self.add_input('port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.handle_in_loop_option = self.add_option('handle in main loop', widget_type='checkbox', default_value=self.handle_in_loop, callback=self.handle_in_loop_changed)
        self.node = self
        self._registerable_init()

    def custom_create(self, from_file):
        OSCSource.custom_create(self, from_file)
        self._registerable_custom_create()

    def _get_registry_path_components(self):
        return [self.get_patcher_path(), self.name]

    def handle_in_loop_changed(self):
        self.handle_in_loop = self.handle_in_loop_option()

    def cleanup(self):
        OSCSource.cleanup(self)
        self._registerable_cleanup()


class OSCDeviceNode(OSCAsyncIOSource, OSCTarget, OSCRegistrableMixin, Node):
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
        Node.__init__(self, label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.source_name_property = self.add_input('name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_name_property = self.source_name_property
        self.target_ip_property = self.add_input('ip', widget_type='text_input', default_value=str(self.ip), callback=self.target_changed)
        self.target_port_property = self.add_input('target port', widget_type='text_input', default_value=str(self.target_port), callback=self.target_changed)
        self.source_port_property = self.add_input('source port', widget_type='text_input', default_value=str(self.source_port), callback=self.source_changed)
        self.output = self.add_output('osc received')
        self.handle_in_loop_option = self.add_option('handle in main loop', widget_type='checkbox', default_value=self.handle_in_loop, callback=self.handle_in_loop_changed)
        OSCAsyncIOSource.node = self
        OSCTarget.node = self
        self.node = self
        self._registerable_init()

    def name_changed(self):
        # This does not work because self.target_changed() changes the path, etc, so self.source_change fails

        #  target_changed
        name = self.target_name_property()
        self.osc_manager.rename_device(self, name)

        port = any_to_int(self.target_port_property())
        ip = self.target_ip_property()
        if port != self.target_port or ip != self.ip:
            self.destroy_client()
            self.target_port = port
            self.ip = ip
            self.create_client()

        #  source_changed
        # name = any_to_string(self.source_name_property())
        # if name != self.name:
        #     self.update_source_name_state(name)
        port = any_to_int(self.source_port_property())
        if port != self.source_port:
            self.destroy_server()
            self.source_port = port
            self.start_serving()

        self.unparsed_args = [str(self.name), str(self.ip), str(self.target_port), str(self.source_port)]
        print('device set_path', self.unparsed_args)

    def custom_create(self, from_file):
        OSCTarget.custom_create(self, from_file)
        OSCSource.custom_create(self, from_file)
        self._registerable_custom_create()

    # def target_changed(self):
    #     name = self.name_property()
    #     port = any_to_int(self.target_port_property())
    #     ip = self.target_ip_property()
    #     self.handle_target_change(name, port, ip)
        # self.handle_source_change(name, self.source_port)

    # def source_changed(self, force=False):
    #     name = any_to_string(self.name_property())
    #     port = any_to_int(self.source_port_property())
    #     self.handle_source_change(name, port)

    def set_path(self):
        OSCSource.set_path(self)
        OSCTarget.set_path(self)

        # path = self.path_option()
        # path_list = self.osc_manager.registry.sanitize_path_components(path)
        # path_text = '/'.join(path_list)
        # self.path_option.set(path_text)
        # self.source_path = path
        # self.target_path = path
        # self.unregister()
        # self.register()

    def _get_registry_path_components(self):
        return [self.get_patcher_path(), self.name]

    def _create_registry_entry(self, path_components: list) -> str:
        return self.osc_manager.registry.add_path_to_registry(path_components)

    def register(self):
        if self.osc_manager is None:
            return

        if self.node is not None:
            # Devices always register in the global registry (not per-service)
            _, self.source_path = self.osc_manager.registry.add_path_to_registry(self._get_registry_path_components())
            self.osc_manager.registry.set_host_info(path=self._get_registry_path_components(), target_port=self.target_port, source_port=self.source_port, ip=self.ip)

            self.node.path_option.set(self.source_path)
            self.target_path = self.source_path

        # if self.osc_manager is None:
        #     return
        #
        # raw_path_components = self._get_registry_path_components()
        # final_path_string = self.osc_manager.registry.compose_path_string(raw_path_components)
        #
        # _, registered_path_string = self._create_registry_entry(raw_path_components)
        # if registered_path_string:
        #     self.source_path = final_path_string
        #     # This is the key fix for the UI widget.
        #     self.path_option.set(self.source_path)
        # else:
        #     self.path = ''
        #     self.path_option.set('Registration Failed')
        #
        # self.target_path = self.source_path

    def unregister(self):
        if self.osc_manager and self.source_path:
            OSCSource.unregister(self)
            # path_components = self._get_registry_path_components()
            # self.osc_manager.registry.remove_path_from_registry(path_components)
            # self.source_path = None
            # self.target_path = None
            # self.path_option.set('Unregistered')

    def get_registry_container(self):
        return self.osc_manager.registry.get_param_registry_container_for_path(self._get_registry_path_components())

    def cleanup(self):
        OSCTarget.cleanup(self)
        OSCAsyncIOSource.cleanup(self)
        self._registerable_cleanup()

    def handle_in_loop_changed(self):
        self.handle_in_loop = self.handle_in_loop_option()

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


class OSCReceiver:
    def __init__(self, label: str, data, args):

        # default source name is root_patcher source node
        # how do we manage am OSCQuery search for address?

        self.source = None
        self.name = ''
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
                # self.name = new_name
                self.find_source_node(new_name)
                # self.find_source_node(self.name)
                # self.osc_manager.connect_receive_node_to_source(self, self.source)

    def update_source_name_state(self, new_source_name: str):
        """Called by OSCManager when the source this node is listening to has been renamed."""
        print('OSC_Receiver update_source_name_state')
        # 1. Capture the old path of THIS node before changing its state.
        old_path_components = self._get_registry_path_components()

        # 2. Update internal state and UI widget.
        self.name = new_source_name
        if self.source_name_property:
            self.source_name_property.set(new_source_name, propagate=False)

        # 3. Re-register THIS node under its new path.
        if isinstance(self, OSCRegistrableMixin):
            self._update_registration(old_path_components=old_path_components)
        self.update_output_label()
        if isinstance(self, Node):
            self.unparsed_args = [str(self.name), str(self.address)]
            print('receiver', self.unparsed_args)

    def update_output_label(self):
        pass

    def get_addresses(self):
        if self.source_address_property is not None:
            return self.source_address_property()

    def set_source(self, source):
        self.source = source
        if self.source_name_property is not None:
            self.source_name_property.set(source.name)
        self.name = source.name
    #     self.adjust_path()
    #
    # def adjust_path(self):
    #     pass

    def find_source_node(self, name):
        if self.osc_manager is not None:
            self.source = self.osc_manager.find_source(name)
            if self.source is not None:
                self.osc_manager.connect_receive_node_to_source(self, self.source)
                self.name = self.source.name
                return True
        return False

    def source_going_away(self, old_source):
        if self.source == old_source:
            self.source = None

    def cleanup(self):
        self.osc_manager.unregister_receive_node(self)


class OSCReceiveNode(Node, OSCBase, OSCReceiver, OSCRegistrableMixin):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)
        OSCReceiver.__init__(self, label, data, args)

        self.source_name_property = self.add_input('source name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.source_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.output = self.add_output('osc received')
        self.throttle = self.add_option('throttle (ms)', widget_type='drag_int', default_value=0)
        self._registerable_init()
        self.last = time.time()

    # --- Simplified Change Handlers ---
    def name_changed(self, force=False):
        """
        Called when the user edits the name property on THIS node.
        It disconnects from the old source and tries to connect to the new one.
        """
        # This logic comes from the original OSCReceiver.name_changed
        if self.source_name_property is not None:
            new_name = any_to_string(self.source_name_property())
            if new_name != self.name or force:
                # Capture the old path BEFORE updating self.name
                old_path_components = None
                if isinstance(self, OSCRegistrableMixin):
                    old_path_components = self._get_registry_path_components()

                # Disconnect from the current source if there is one
                if self.source is not None:
                    self.osc_manager.unregister_receive_node(self)

                # Update internal name and try to find the new source
                self.name = new_name
                self.find_source_node(self.name)  # This will re-register with the manager under the new source

                # Update the registry path
                if isinstance(self, OSCRegistrableMixin):
                    self._update_registration(old_path_components=old_path_components)

    def address_changed(self):
        """
        Handles changes to the node's OSC address, ensuring the registry is
        updated correctly.
        """
        address_property = None
        if hasattr(self, 'source_address_property'):
            address_property = self.source_address_property
        # elif hasattr(self, 'target_address_property'):
        #     address_property = self.target_address_property

        if address_property is None:
            return

        new_address = any_to_string(address_property())

        if new_address != self.address:
            # 1. CAPTURE the old path components BEFORE changing the state.
            old_path_components = self._get_registry_path_components()

            # 2. CHANGE the internal state.
            # This is the logic that was in the base OSCReceiver/OSCSender.
            if self.source is not None:
                self.source.unregister_receive_node(self)  # For receivers
            # elif self.target is not None:
            #     self.target.unregister_send_node(self)  # For senders

            self.address = new_address

            # Re-register with the source/target under the new address
            if self.source is not None:
                self.source.register_receive_node(self)
            # elif self.target is not None:
            #     self.target.register_send_node(self)

            # 3. UPDATE the registry, passing in the captured old path.
            self._update_registration(old_path_components=old_path_components)

     # --- Implement required methods from OSCRegistrableNode ---
    def _get_registry_path_components(self) -> list:
        return [self.get_patcher_path(), self.name, self.address]

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_generic_receiver_to_registry(path_components)

    def custom_create(self, from_file):
        self._needs_deferred_connect = False
        if self.name != '':
            source_found = self.find_source_node(self.name)
            if from_file and not source_found:
                self._needs_deferred_connect = True
        self._registerable_custom_create()

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

    def post_load_callback(self):
        if not getattr(self, '_needs_deferred_connect', False):
            return
        self._needs_deferred_connect = False
        if self.name != '' and self.source is None:
            self.find_source_node(self.name)
        if not self.path:
            self._registerable_custom_create()

    def cleanup(self):
        OSCReceiver.cleanup(self)
        self._registerable_cleanup()


class OSCSender:
    def __init__(self, label: str, data, args):
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
            if len(args) >= 2:
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
        self.target_name_property.set(target.name)
        self.name = self.target.name

    def target_going_away(self, old_target):
        if self.target == old_target:
             self.target = None

    def cleanup(self):
        self.osc_manager.unregister_send_node(self)

    def update_target_name_state(self, new_target_name: str):
        """Called by OSCManager when the target this node is sending to has been renamed."""
        # 1. Capture the old path of THIS node.
        old_path_components = self._get_registry_path_components()

        # 2. Update internal state and UI widget.
        self.name = new_target_name
        if hasattr(self, 'target_name_property'):
            self.target_name_property.set(new_target_name, propagate=False)

        # 3. Re-register THIS node under its new path.
        if isinstance(self, OSCRegistrableMixin):
            self._update_registration(old_path_components=old_path_components)
        if isinstance(self, Node):
            self.unparsed_args = [str(self.name), str(self.address)]
            print('sender', self.unparsed_args)


class OSCSendNode(Node, OSCBase, OSCSender, OSCRegistrableMixin):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)
        OSCSender.__init__(self, label, data, args)

        self.input = self.add_input('osc to send', triggers_execution=True)

        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self._registerable_init()
        # self.path_option = self.add_option('path', widget_type='label')

    # --- Simplified Change Handlers ---
    def name_changed(self, force=False):
        if self.target_name_property is not None:
            new_name = any_to_string(self.target_name_property())
            if new_name != self.name or force:
                old_path_components = None
                if isinstance(self, OSCRegistrableMixin):
                    old_path_components = self._get_registry_path_components()

                if self.target is not None:
                    self.osc_manager.unregister_send_node(self)

                self.name = new_name
                self.find_target_node(self.name)
                if isinstance(self, OSCRegistrableMixin):
                    self._update_registration(old_path_components=old_path_components)


        # The base OSCReceiver logic to find the new source
        super().name_changed(force)
        # The registration logic is now just one call
        self._update_registration()

    def address_changed(self):
        """
        Handles changes to the node's OSC address, ensuring the registry is
        updated correctly.
        """
        address_property = None
        if hasattr(self, 'target_address_property'):
            address_property = self.target_address_property

        if address_property is None:
            return

        new_address = any_to_string(address_property())

        if new_address != self.address:
            # 1. CAPTURE the old path components BEFORE changing the state.
            old_path_components = self._get_registry_path_components()

            # 2. CHANGE the internal state.
            # This is the logic that was in the base OSCReceiver/OSCSender.
            if self.target is not None:
                self.target.unregister_send_node(self)  # For senders

            self.address = new_address

            # Re-register with the source/target under the new address
            if self.target is not None:
                self.target.register_send_node(self)

            # 3. UPDATE the registry, passing in the captured old path.
            self._update_registration(old_path_components=old_path_components)

    # --- Implement required methods from OSCRegistrableNode ---
    def _get_registry_path_components(self) -> list:
        return [self.get_patcher_path(), self.name, self.address]

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_generic_sender_to_registry(path_components)

    def custom_create(self, from_file):
        self._needs_deferred_connect = False
        if self.name != '':
            target_found = self.find_target_node(self.name)
            if from_file and not target_found:
                self._needs_deferred_connect = True
        self._registerable_custom_create()

    def post_load_callback(self):
        if not getattr(self, '_needs_deferred_connect', False):
            return
        self._needs_deferred_connect = False
        if self.name != '' and self.target is None:
            self.find_target_node(self.name)
        if not self.path:
            self._registerable_custom_create()

    def cleanup(self):
        OSCSender.cleanup(self)
        self._registerable_cleanup()

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


class OSCCueNode(Node, OSCBase, OSCSender, OSCRegistrableMixin):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCCueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)
        OSCSender.__init__(self, label, data, args)

        self.input = self.add_int_input('cue # to send', triggers_execution=True)
        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        # self.target_address_property = self.add_input('address', widget_type='text_input', default_value=self.address)
        self.address = 'cue'
        self._registerable_init()

    def name_changed(self, force=False):
        if self.target_name_property is not None:
            new_name = any_to_string(self.target_name_property())
            if new_name != self.name or force:
                old_path_components = None
                if isinstance(self, OSCRegistrableMixin):
                    old_path_components = self._get_registry_path_components()

                if self.target is not None:
                    self.osc_manager.unregister_send_node(self)

                self.name = new_name
                self.find_target_node(self.name)
                if isinstance(self, OSCRegistrableMixin):
                    self._update_registration(old_path_components=old_path_components)


        # The base OSCReceiver logic to find the new source
        super().name_changed(force)
        # The registration logic is now just one call
        self._update_registration()

    def address_changed(self):
        """
        Handles changes to the node's OSC address, ensuring the registry is
        updated correctly.
        """
        address_property = None
        if hasattr(self, 'target_address_property'):
            address_property = self.target_address_property

        if address_property is None:
            return

        new_address = any_to_string(address_property())

        if new_address != self.address:
            # 1. CAPTURE the old path components BEFORE changing the state.
            old_path_components = self._get_registry_path_components()

            # 2. CHANGE the internal state.
            # This is the logic that was in the base OSCReceiver/OSCSender.
            if self.target is not None:
                self.target.unregister_send_node(self)  # For senders

            self.address = new_address

            # Re-register with the source/target under the new address
            if self.target is not None:
                self.target.register_send_node(self)

            # 3. UPDATE the registry, passing in the captured old path.
            self._update_registration(old_path_components=old_path_components)

    # --- Implement required methods from OSCRegistrableNode ---
    def _get_registry_path_components(self) -> list:
        return [self.get_patcher_path(), self.name, self.address]

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_generic_sender_to_registry(path_components)

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
        self._registerable_custom_create()

    def cleanup(self):
        OSCSender.cleanup(self)
        self._registerable_cleanup()

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

class OSCWidget(OSCBase, OSCReceiver, OSCSender, OSCRegistrableMixin):
    def __init__(self, label: str, data, args):
        # HOW TO HANDLE VARIABLE NUMBER OF SOURCE / TARGET / ADDRESS ARGS before the non-osc version args?
        # ASSESS FIRST 4 ARGS, IF ARGS ARE ACTUAL SOURCE OR TARGET NAMES, WE KNOW WHAT TO DO WITH THEM

        OSCReceiver.__init__(self, label, data, args)
        OSCSender.__init__(self, label, data, args)

    # --- Life-cycle Methods ---
    # These will be called by the final Node class's own life-cycle methods.

    """
    Template method for node creation. It orchestrates the setup process
    and calls a hook for subclass-specific logic.
    """

    def custom_create(self, from_file):
        """Handles finding source/target and then triggers registration."""
        self._needs_deferred_connect = False
        self._setup_mode_combo()
        
        # 1. Auto-assign service name as target if in a service-scoped editor
        if self.name == '' and hasattr(self, 'my_editor') and self.my_editor and self.osc_manager:
            svc_name = self.osc_manager.get_service_name_for_editor(self.my_editor)
            if svc_name:
                self.name = svc_name
                # Update the UI property
                if hasattr(self, 'target_name_property') and self.target_name_property is not None:
                    self.target_name_property.set(self.name)
                if hasattr(self, 'source_name_property') and self.source_name_property is not None:
                    self.source_name_property.set(self.name)

        # 2. Connect to source/target
        target_found = False
        source_found = False
        if self.name != '':
            target_found = self.find_target_node(self.name)
            source_found = self.find_source_node(self.name)

        # If connections failed and we're loading from file, defer to post_load_callback
        if from_file and self.name != '' and (not target_found or not source_found):
            self._needs_deferred_connect = True

        # 3. Trigger registration (from OSCRegistrableMixin)
        self._registerable_custom_create()  # This calls self.register()

        # 4. Call the hook for subclass-specific post-registration logic
        if self.path: # Only run the hook if registration was successful
            self._on_registration_complete()
        self.update_output_label()

    # --- New Hook Method ---
    def _on_registration_complete(self):
        """
        Hook method for subclasses to implement. Called after the node has been
        successfully registered. This is the place to sync initial state like
        value, range, etc., to the registry.
        """
        pass # Default implementation does nothing.

    def cleanup(self):
        """Handles OSC cleanup and then triggers unregistration."""
        self._stop_proxy_polling()
        self._stop_peer_subscription()
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)
        self._registerable_cleanup()  # This calls self.unregister()

    def post_load_callback(self):
        """Called after ALL nodes and links in all patchers are loaded."""
        # Backward compat: migrate old boolean proxy values to mode strings
        if hasattr(self, 'mode_option'):
            current = self.mode_option()
            if current is True:
                self.mode_option.widget.set('proxy')
            elif current is False:
                self.mode_option.widget.set('local')

        is_remote = hasattr(self, 'mode_option') and self.mode_option() != 'local'
        
        # 1. Proxy isolation: unregister widgets that were incorrectly registered
        if is_remote:
            if self.path:
                self.unregister()
            # Start appropriate remote feedback mechanism
            mode = self.mode_option() if hasattr(self, 'mode_option') else 'proxy'
            if mode in ('proxy', 'peer'):
                self._start_proxy_polling()
            if mode == 'peer':
                self._start_peer_subscription()
        else:
            # Non-proxy: ALWAYS unregister and re-register.
            # During custom_create, the widget may have registered in the GLOBAL
            # registry because oscq_host hadn't started its service yet. Now that
            # all nodes are loaded, the service-scoped registry should be available.
            if self.path:
                self.unregister()
            self._registerable_custom_create()
            if self.path:
                self._on_registration_complete()
                self.update_output_label()

        # 2. Re-attempt source/target connections
        if self.name != '':
            if self.target is None:
                self.find_target_node(self.name)
            if self.source is None:
                self.find_source_node(self.name)

    # --- Proxy Polling ---

    def _start_proxy_polling(self):
        """Start background HTTP polling for remote value changes."""
        self._stop_proxy_polling()  # Stop any existing polling
        
        if not self.osc_manager or not self.osc_manager.oscquery_browser:
            return
        if not self.name:
            return

        self._proxy_poll_service = None
        self._proxy_poll_value = None
        self._proxy_poll_has_new = False
        self._proxy_poll_running = True
        self._proxy_poll_thread = threading.Thread(
            target=self._proxy_poll_loop, daemon=True
        )
        self._proxy_poll_thread.start()
        self.add_frame_task()

    def _stop_proxy_polling(self):
        """Stop background HTTP polling."""
        self._proxy_poll_running = False
        if hasattr(self, '_proxy_poll_thread') and self._proxy_poll_thread is not None:
            self._proxy_poll_thread = None
        if hasattr(self, 'has_frame_task') and self.has_frame_task:
            self.remove_frame_tasks()

    def _proxy_poll_loop(self):
        """Background thread: discover service, then fetch remote VALUE periodically."""
        import time
        while getattr(self, '_proxy_poll_running', False):
            svc = getattr(self, '_proxy_poll_service', None)

            # Phase 1: Wait for service discovery
            if svc is None:
                browser = getattr(self.osc_manager, 'oscquery_browser', None)
                if browser:
                    svc = browser.get_service(self.name)
                    if svc:
                        self._proxy_poll_service = svc
                        self._proxy_device_needs_update = True
                        print(f"OSCWidget: proxy poll connected to '{self.name}' at {svc.ip}:{svc.http_port} (OSC:{svc.osc_port})")
                    else:
                        time.sleep(2.0)  # Retry discovery every 2s
                        continue
                else:
                    time.sleep(2.0)
                    continue

            # Phase 2: Poll value
            try:
                if self.address:
                    param = svc.fetch_param(self.address)
                    if param and 'VALUE' in param:
                        raw = param['VALUE']
                        if isinstance(raw, list) and len(raw) == 1:
                            raw = raw[0]
                        self._proxy_poll_value = raw
                        self._proxy_poll_has_new = True
            except Exception:
                # Service may have gone away — reset and retry discovery
                self._proxy_poll_service = None
            time.sleep(0.1)  # ~10 Hz

    def frame_task(self):
        """Main-thread: apply polled value and update device port if needed."""
        # Update device target port when service (re)connects
        if getattr(self, '_proxy_device_needs_update', False):
            self._proxy_device_needs_update = False
            self._update_device_port()

        if getattr(self, '_proxy_poll_has_new', False):
            self._proxy_poll_has_new = False
            value = self._proxy_poll_value
            if value is not None and hasattr(self, 'input'):
                try:
                    current = self.input()
                    if current != value:
                        self.input.widget.set(value, propagate=False)
                except Exception:
                    pass

    def _update_device_port(self):
        """Update the osc_device's target port to match the current service port."""
        svc = getattr(self, '_proxy_poll_service', None)
        if svc is None:
            return
        target = self.osc_manager.targets.get(self.name)
        if target is None:
            return
        if target.target_port != svc.osc_port or target.ip != svc.ip:
            print(f"OSCWidget: updating device '{self.name}' port {target.target_port}→{svc.osc_port}, ip {target.ip}→{svc.ip}")
            target.destroy_client()
            target.target_port = svc.osc_port
            target.ip = svc.ip
            target.create_client()

    # --- Peer Subscription ---

    def _start_peer_subscription(self):
        """Subscribe to the remote service for OSC push updates."""
        self._stop_peer_subscription()

        if not self.osc_manager or not hasattr(self.osc_manager, 'oscquery_browser'):
            return
        svc = self.osc_manager.oscquery_browser.get_service(self.name)
        if svc is None:
            return

        # Ensure the osc_device has a source (listening) port
        source = self.osc_manager.sources.get(self.name)
        if source is None:
            return

        local_port = getattr(source, 'source_port', None)
        if local_port is None:
            return

        # Send subscribe request to remote HTTP server
        try:
            import socket as _socket
            local_ip = self._get_local_ip()
            url = f"http://{svc.ip}:{svc.http_port}/subscribe?ip={local_ip}&port={local_port}"
            from urllib.request import urlopen
            urlopen(url, timeout=3)
            self._peer_subscribed_service = svc
            self._peer_local_port = local_port
        except Exception as e:
            print(f"OSCWidget: Failed to subscribe to {self.name}: {e}")

    def _stop_peer_subscription(self):
        """Unsubscribe from the remote service."""
        svc = getattr(self, '_peer_subscribed_service', None)
        if svc is None:
            return
        local_port = getattr(self, '_peer_local_port', None)
        if local_port is None:
            return
        try:
            local_ip = self._get_local_ip()
            url = f"http://{svc.ip}:{svc.http_port}/unsubscribe?ip={local_ip}&port={local_port}"
            from urllib.request import urlopen
            urlopen(url, timeout=2)
        except Exception:
            pass
        self._peer_subscribed_service = None
        self._peer_local_port = None

    @staticmethod
    def _get_local_ip():
        """Get the local IP address of this machine."""
        import socket as _socket
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _notify_peers(self, osc_address, value):
        """Push a value change to all subscribed peer clients via the host's OSCQueryServer."""
        if not self.osc_manager or not osc_address:
            return
        # Only local widgets should push to peers
        if hasattr(self, 'mode_option') and self.mode_option() != 'local':
            return
        # Find the OSCQueryServer for the service this widget belongs to
        svc_name = self.osc_manager.get_service_name_for_editor(self.my_editor) if hasattr(self, 'my_editor') and self.my_editor else None
        if svc_name and svc_name in self.osc_manager.service_registries:
            _, srv = self.osc_manager.service_registries[svc_name]
            if srv and hasattr(srv, 'notify_subscribers'):
                srv.notify_subscribers(osc_address, value)

    # --- Mode Combo Setup ---

    def _setup_mode_combo(self):
        """Configure backward compat for the mode combo (combo_items set in __init__)."""
        if hasattr(self, 'mode_option'):
            # Backward compat: old saved patches stored this as 'proxy' (checkbox, bool)
            self.mode_option.name_archive = ['proxy']

    def patcher_name_changed(self, old_name, new_name):
        """Called by the editor when the patcher is saved under a new name."""
        if hasattr(self, 'mode_option') and hasattr(self, 'target_name_property'):
            if self.mode_option() != 'local':
                # If we are a proxy, and we were specifically tracking the old patcher's service name,
                # auto-update to the new patcher's service name so we don't disconnect.
                if self.target_name_property() == old_name:
                    self.target_name_property.set(new_name)

    # --- Change Handlers ---

    def name_changed(self, force=False):
        """
        Called when the user edits the name property on THIS node.
        It disconnects from the old source and tries to connect to the new one.
        """
        # This logic comes from the original OSCReceiver.name_changed
        if self.source_name_property is not None:
            new_name = any_to_string(self.source_name_property())
            if new_name != self.name or force:
                # Capture the old path BEFORE updating self.name
                old_path_components = None
                if isinstance(self, OSCRegistrableMixin):
                    old_path_components = self._get_registry_path_components()

                # Disconnect from the current source if there is one
                if self.source is not None:
                    self.osc_manager.unregister_receive_node(self)

                # Update internal name and try to find the new source
                self.name = new_name
                self.find_source_node(self.name)  # This will re-register with the manager under the new source

                # Update the registry path
                if isinstance(self, OSCRegistrableMixin):
                    self._update_registration(old_path_components=old_path_components)
            self.update_output_label()
            if isinstance(self, Node):
                self.unparsed_args = [str(self.name), str(self.address)]

    def update_output_label(self):
        if hasattr(self, 'output'):
            # Build the full path for reference
            if len(self.address) > 0 and self.address[0] == '/':
                full_label = self.name + self.address
            else:
                full_label = self.name + '/' + self.address

            # Check OSCQuery registry for custom SHORT_NAME first
            short_name = None
            if hasattr(self, 'osc_manager') and self.osc_manager and hasattr(self.osc_manager, 'registry'):
                try:
                    if hasattr(self, '_get_registry_path_components'):
                        path_comps = self._get_registry_path_components()
                        param_container = self.osc_manager.registry.get_registry_container_for_path(path_comps)
                        if param_container and 'SHORT_NAME' in param_container:
                            short_name = param_container['SHORT_NAME']
                except Exception:
                    pass

            if short_name is not None:
                self.output.set_label(short_name)
                return

            # Extract a sensible shortened label
            segments = [s for s in full_label.split('/') if s]
            
            if not segments:
                self.output.set_label(full_label)
                return

            leaf = segments[-1]
            
            # Look for the right-most integer in the preceding segments
            int_idx = -1
            for i in range(len(segments) - 2, -1, -1):
                if segments[i].lstrip('-').isdigit():
                    int_idx = i
                    break
                    
            if int_idx != -1:
                # We found an integer index. Include the context right before it if it exists.
                # e.g., ['eos', 'chan', '1', 'param', 'intensity'] -> 'chan/1/intensity'
                if int_idx > 0:
                    short_label = f"{segments[int_idx-1]}/{segments[int_idx]}/{leaf}"
                else:
                    short_label = f"{segments[int_idx]}/{leaf}"
            else:
                # No integer found, default to last two segments: e.g., 'reverb/wet'
                if len(segments) > 2:
                    short_label = '/'.join(segments[-2:])
                else:
                    short_label = '/'.join(segments)
                    
            self.output.set_label(short_label)

    def address_changed(self):
        """
        Handles changes to the node's OSC address, ensuring the registry is
        updated correctly.
        """
        address_property = None
        if hasattr(self, 'source_address_property'):
            address_property = self.source_address_property
        elif hasattr(self, 'target_address_property'):
            address_property = self.target_address_property

        if address_property is None:
            return

        new_address = any_to_string(address_property())

        if new_address != self.address:
            # 1. CAPTURE the old path components BEFORE changing the state.
            old_path_components = self._get_registry_path_components()

            # 2. CHANGE the internal state.
            # This is the logic that was in the base OSCReceiver/OSCSender.
            if self.source is not None:
                self.source.unregister_receive_node(self)  # For receivers
            elif self.target is not None:
                self.target.unregister_send_node(self)  # For senders

            self.address = new_address

            # Re-register with the source/target under the new address
            if self.source is not None:
                self.source.register_receive_node(self)
            elif self.target is not None:
                self.target.register_send_node(self)

            # 3. UPDATE the registry, passing in the captured old path.
            self._update_registration(old_path_components=old_path_components)

        self.update_output_label()

    # --- Implement one of the abstract methods from the mixin ---

    def _get_registry_path_components(self) -> list:
        # # Assumes `self.get_patcher_path` exists because self is a Node instance.
        return [self.get_patcher_path(), self.name, self.address]


    # The second abstract method, _create_registry_entry, is intentionally
    # left for the final concrete classes (OSCValueNode, etc.) to implement.

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

    def get_registry_container(self):
        registry, _ = self._get_registry()
        if registry is None:
            return
        if hasattr(self, 'get_patcher_path'):
            patcher_path = self.get_patcher_path()
            return registry.get_param_registry_container_for_path([patcher_path, self.name, self.address])
        return None

    def mode_changed(self):
        mode = self.mode_option()

        # Registry operations (may be unavailable for remote-only widgets)
        registry, _ = self._get_registry()
        if registry is not None:
            # When mode changes, we might need to unregister from the 
            # exported OSC registry since non-local widgets shouldn't be publicly served.
            old_path_components = None
            if hasattr(self, '_get_registry_path_components'):
                old_path_components = self._get_registry_path_components()
                
            if hasattr(self, '_update_registration'):
                self._update_registration(old_path_components=old_path_components)

            if hasattr(self, 'get_patcher_path'):
                patcher_path = self.get_patcher_path()
                if mode != 'local':
                    registry.set_flow([patcher_path, self.name, self.address], flow='PROXY')
                else:
                    registry.set_flow([patcher_path, self.name, self.address], flow='BOTH')

        # Start/stop remote feedback based on mode (ALWAYS runs)
        self._stop_proxy_polling()
        self._stop_peer_subscription()
        if mode in ('proxy', 'peer'):
            self._start_proxy_polling()
        if mode == 'peer':
            self._start_peer_subscription()

    def update_value_in_registry(self):
        """
        Updates the value for this widget's path in the OSCQueryRegistry.
        It calls the abstract _get_registry_value to get the current value.
        Also notifies any subscribed peers via the host's OSCQueryServer.
        """
        if self.osc_manager and self.path:
            registry, _ = self._get_registry()
            if registry is None:
                return
            current_value = self._get_registry_value()
            if current_value is not None:
                registry.set_value_for_path(self.path, current_value)
                # Notify subscribed peers via OSCQueryServer
                self._notify_peers(self.address, current_value)

    def _get_registry_value(self):
        """
        Abstract method. Concrete widget nodes (OSCValueNode, OSCToggleNode, etc.)
        must implement this to return their current value for the registry.
        """
        raise NotImplementedError("Concrete widget class must implement _get_registry_value")

    # def update_source_name_state(self, new_source_name: str):
    #     print('OSCWidget update_source_name_state')
    #     OSCSource.update_source_name_state(self, new_source_name)
    #     self.update_output_label()


class OSCValueNode(ValueNode, OSCWidget):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCValueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        # Pre-initialize NumericValueNode attributes for numeric types
        base_name = label.split('_')[-1] if label else ''
        if base_name in ('slider', 'float', 'int', 'knob'):
            self.min = None
            self.max = None
            self.start_value = None
            self.format = '%.3f'
            self.min_property = None
            self.max_property = None
            self.speed_property = None
            self.format_property = None
        ValueNode.__init__(self, label, data, args)
        OSCWidget.__init__(self, label, data, args)

        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        if label == 'osc_string':
            self.space_replacement = self.add_option('replace spaces in osc messages', widget_type='checkbox',
                                                   default_value=True)
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='local', callback=self.mode_changed)
        self.mode_option.widget.combo_items = ['local', 'proxy', 'peer']
        self._registerable_init()

    def setup_specific_ui(self, args):
        """Dispatch to the correct concrete ValueNode subclass's setup."""
        base_name = self.label.split('_')[-1] if self.label else ''
        if base_name == 'slider':
            SliderNode.setup_specific_ui(self, args)
        elif base_name == 'float':
            FloatNode.setup_specific_ui(self, args)
        elif base_name == 'int':
            IntNode.setup_specific_ui(self, args)
        elif base_name == 'knob':
            KnobNode.setup_specific_ui(self, args)
        elif base_name in ('string', 'message', 'list'):
            StringNode.setup_specific_ui(self, args)
        elif base_name == 'text':
            TextEditorNode.setup_specific_ui(self, args)
        else:
            StringNode.setup_specific_ui(self, args)

    def create_numeric_options(self):
        """Delegate to NumericValueNode for slider/float/int/knob types."""
        NumericValueNode.create_numeric_options(self)

    def options_changed(self):
        """Delegate to NumericValueNode for proper min/max/speed handling."""
        NumericValueNode.options_changed(self)

    def _create_registry_entry(self, path_components: list) -> str:
        """
        This is the only piece of registration logic specific to this node class.
        """
        registry, _ = self._get_registry()
        widget_type = self.input.widget.widget
        if widget_type in ['drag_float', 'slider_float', 'knob_float', 'input_float']:
            return registry.add_float_to_registry(path_components)
        elif widget_type in ['drag_int', 'slider_int', 'input_int']:
            return registry.add_int_to_registry(path_components)
        elif widget_type in ['text_input', 'text_editor']:
            return registry.add_string_to_registry(path_components)
        elif widget_type in ['checkbox']:
            return registry.add_bool_to_registry(path_components)
        return None, ''  # Return None on failure

    def _on_registration_complete(self):
        """
               This hook is called automatically by OSCWidget.custom_create.
               We use it to sync our initial widget state to the registry.
               """
        if self.osc_manager and self.path:
            # Update the initial value in the registry
            self.update_value_in_registry()

            # Additionally, sync metadata like range
            registry, _ = self._get_registry()
            if registry:
                param_container = registry.get_param_registry_container_for_path(self.path)
                if param_container and 'RANGE' in param_container:
                    # Assuming input.widget has min/max attributes
                    if hasattr(self.input.widget, 'min') and hasattr(self.input.widget, 'max'):
                        param_container['RANGE'] = [{'MIN': self.input.widget.min, 'MAX': self.input.widget.max}]

    def update_registry_range(self):
        registration = self.get_registry_container()
        if registration is not None:
            if 'RANGE' in registration:
                registration['RANGE'] = [{'MIN': self.input.widget.min, 'MAX': self.input.widget.max}]

    def _get_registry_value(self):
        """Returns the current value from the DPG widget."""
        return self.input()

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
        self.update_value_in_registry()

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

    def custom_create(self, from_file):
        ValueNode.custom_create(self, from_file)
        OSCWidget.custom_create(self, from_file)

    def post_load_callback(self):
        OSCWidget.post_load_callback(self)

    def cleanup(self):
        OSCWidget.cleanup(self)

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
            if hasattr(self, 'space_replacement') and self.space_replacement():
                data = data.replace(' ', '_')
        if data is not None:
            self.update_value_in_registry()
            if self.target and self.address != '':
                self.target.send_message(self.address, data)


class OSCButtonNode(ButtonNode, OSCWidget):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCButtonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        ButtonNode.__init__(self, label, data, args)
        OSCWidget.__init__(self, label, data, args)

        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        self._registerable_init()

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_button_to_registry(path_components)

    def receive(self, data, address=None):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        data = any_to_bool(data)
        if data:
            ButtonNode.execute(self)

    def custom_create(self, from_file):
        ButtonNode.custom_create(self, from_file)
        OSCWidget.custom_create(self, from_file)

    def post_load_callback(self):
        OSCWidget.post_load_callback(self)

    def cleanup(self):
        OSCWidget.cleanup(self)

    def frame_task(self):
        ButtonNode.frame_task(self)
        OSCWidget.frame_task(self)

    def execute(self):
        ButtonNode.execute(self)
        self.update_value_in_registry()
        if self.target and self.address != '':
            self.target.send_message(self.address, self.message())


class OSCToggleNode(ToggleNode, OSCWidget):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCToggleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        ToggleNode.__init__(self, label, data, args)
        OSCWidget.__init__(self, label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='local', callback=self.mode_changed)
        self.mode_option.widget.combo_items = ['local', 'proxy', 'peer']
        self._registerable_init()

    def _get_registry_value(self):
        """Returns the current boolean state of the toggle."""
        return self.input()

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_bool_to_registry(path_components)

    def _on_registration_complete(self):
        """
               This hook is called automatically by OSCWidget.custom_create.
               We use it to sync our initial widget state to the registry.
               """
        if self.osc_manager and self.path:
            # Update the initial value in the registry
            self.update_value_in_registry()

            # Additionally, sync metadata like range
            registry, _ = self._get_registry()
            if registry:
                param_container = registry.get_param_registry_container_for_path(self.path)


    def receive(self, data, address):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0] == '/':
                return
        self.value = any_to_bool(data)
        self.input.set(self.value)
        ToggleNode.execute(self)

    def custom_create(self, from_file):
        ToggleNode.custom_create(self, from_file)
        OSCWidget.custom_create(self, from_file)

    def post_load_callback(self):
        OSCWidget.post_load_callback(self)

    def cleanup(self):
        OSCWidget.cleanup(self)

    def execute(self):
        ToggleNode.execute(self)
        self.update_value_in_registry()
        if self.target and self.address != '':
            self.target.send_message(self.address, self.value)


class OSCMenuNode(MenuNode, OSCWidget):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCMenuNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        # HOW TO HANDLE VARIABLE NUMBER OF SOURCE / TARGET / ADDRESS ARGS before the non-osc version args?
        # ASSESS FIRST 4 ARGS, IF ARGS ARE ACTUAL SOURCE OR TARGET NAMES, WE KNOW WHAT TO DO WITH THEM
        MenuNode.__init__(self, label, data, args)
        OSCWidget.__init__(self, label, data, args)
        # n.b. if we do not explicitly list a source, then this fails
        # because MenuNode removes the first 2 arguments if osc_menu
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='local', callback=self.mode_changed)
        self.mode_option.widget.combo_items = ['local', 'proxy', 'peer']
        self._registerable_init()

    def _get_registry_value(self):
        """Returns the currently selected choice from the menu."""
        return self.choice()

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_string_menu_to_registry(path_components)

    def _on_registration_complete(self):
        """Syncs the initial menu state to the registry."""
        if self.osc_manager and self.path:
            # Update the initial selected value
            self.update_value_in_registry()

            # Update the list of available choices
            registry, _ = self._get_registry()
            if registry:
                param_container = registry.get_param_registry_container_for_path(self.path)
                if param_container:
                    param_container['VALS'] = self.choices

    def receive(self, data, address):
        data = any_to_list(data)
        if type(data[0]) == str:
            if data[0][0] == '/':
                return
        self.choice.set(data)
        self.set_choice_internal()

    def custom_create(self, from_file):
        MenuNode.custom_create(self, from_file)
        OSCWidget.custom_create(self, from_file)

    def post_load_callback(self):
        OSCWidget.post_load_callback(self)

    def cleanup(self):
        OSCWidget.cleanup(self)

    def execute(self):
        MenuNode.execute(self)
        self.update_value_in_registry()
        if self.target and self.address != '':
            self.target.send_message(self.address, self.choice())


class OSCRadioButtonsNode(RadioButtonsNode, OSCWidget):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCRadioButtonsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        RadioButtonsNode.__init__(self, label, data, args)
        OSCWidget.__init__(self, label, data, args)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_option('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.source_name_property = self.target_name_property
        self.source_address_property = self.target_address_property
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='local', callback=self.mode_changed)
        self.mode_option.widget.combo_items = ['local', 'proxy', 'peer']
        self._registerable_init()

    def _get_registry_value(self):
        """Returns the currently selected choice from the menu."""
        return self.radio_group()

    def _create_registry_entry(self, path_components: list) -> str:
        return self._get_registry()[0].add_string_menu_to_registry(path_components)

    def custom_create(self, from_file):
        RadioButtonsNode.custom_create(self, from_file)
        OSCWidget.custom_create(self, from_file)
        registration = self.get_registry_container()
        if registration is not None:
            registration['VALUE'] = [self.radio_group.widget.value]

    def post_load_callback(self):
        OSCWidget.post_load_callback(self)

    def cleanup(self):
        OSCWidget.cleanup(self)

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
        self.update_value_in_registry()
        if self.target and self.address != '':
            self.target.send_message(self.address, self.radio_group())


