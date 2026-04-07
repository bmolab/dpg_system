"""
OSCQuery Node Types for dpg_system.

Provides two node types:
- oscq_service: Quick connect to a named OSCQuery service
- oscq_browse: Interactive browser with search, drill-down, and auto-instantiation
"""

import re
import dearpygui.dearpygui as dpg
import threading
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.osc_nodes import OSCBase

try:
    from dpg_system.convert_osc_query_to_dpg_json import (
        param_to_node_spec, walk_tree, search_tree,
        detect_channel_pattern, get_contents_list, navigate_to_path
    )
    HAS_CONVERTER = True
except ImportError:
    HAS_CONVERTER = False


def register_oscquery_nodes():
    """Register OSCQuery node types with the app."""
    Node.app.register_node('oscq_service', OSCQueryServiceNode.factory)
    Node.app.register_node('oscq_browse', OSCQueryBrowseNode.factory)
    Node.app.register_node('oscq_host', OSCQueryHostNode.factory)


# ---------------------------------------------------------------------------
# OSCQueryServiceNode — quick connect by name
# ---------------------------------------------------------------------------

class OSCQueryServiceNode(Node, OSCBase):
    """
    Quick-connect node: takes a service name (or alias) and auto-discovers
    the corresponding OSCQuery service, creating the appropriate osc_device
    node with the correct IP/port settings.

    Usage: oscq_service speech_to_text
           oscq_service lights
    """

    @staticmethod
    def factory(name, data, args=None):
        node = OSCQueryServiceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Determine requested service name from args
        self.requested_service_name = ''
        if args and len(args) > 0:
            self.requested_service_name = any_to_string(args[0])

        # UI elements
        self.service_name_input = self.add_input(
            'service name', widget_type='text_input',
            default_value=self.requested_service_name,
            triggers_execution=True
        )

        # Dropdown of discovered services
        self.service_selector = self.add_input(
            'available services', widget_type='combo',
            default_value='(none)',
            callback=self.on_service_selected
        )

        self.status_label = self.add_property('status', widget_type='label')
        self.refresh_button = self.add_input('refresh', widget_type='button',
                                             callback=self.refresh_services)

        # Outputs
        self.service_info_out = self.add_output('service info')
        self.address_space_out = self.add_output('address space')

        # Internal state
        self.connected_service = None
        self.created_device_name = None

    def custom_create(self, from_file):
        self.refresh_services()
        if self.requested_service_name:
            self.connect_to_service(self.requested_service_name)

    def refresh_services(self):
        """Update the combo box with currently discovered services."""
        services = self.osc_manager.get_discovered_services()
        if services:
            dpg.configure_item(self.service_selector.widget.uuid,
                               items=['(none)'] + services)
        else:
            dpg.configure_item(self.service_selector.widget.uuid,
                               items=['(none)', '(no services found)'])

    def on_service_selected(self):
        """Handle selection from the services dropdown."""
        selected = self.service_selector()
        if selected and selected not in ('(none)', '(no services found)'):
            self.service_name_input.widget.set(selected)
            self.connect_to_service(selected)

    def connect_to_service(self, name):
        """Attempt to discover and connect to the named service."""
        if not name or name.strip() == '':
            self.set_status('No service name specified')
            return

        service = self.osc_manager.resolve_service(name)
        if service is None:
            self.set_status(f"Service '{name}' not found")
            return

        self.connected_service = service
        target_name = service.name

        # Create osc_device if it doesn't exist
        if target_name not in self.osc_manager.targets:
            args = [target_name, str(service.ip), str(service.osc_port)]
            try:
                device_node = Node.app.create_node_by_name('osc_device', None, args)
                if device_node:
                    device_node.set_visibility('hidden')
            except Exception as e:
                self.set_status(f"Failed to create device: {e}")
                return

        self.created_device_name = target_name
        self.set_status(f"Connected: {service.ip}:{service.osc_port}")

        # Output service info
        self.service_info_out.send([service.name, service.ip, service.osc_port])

        # Output address space if available
        if service.json_tree:
            self.address_space_out.send(service.json_tree)

    def set_status(self, text):
        """Update the status label."""
        if self.status_label and self.status_label.widget:
            self.status_label.widget.set(text)

    def execute(self):
        if self.service_name_input.fresh_input:
            name = self.service_name_input()
            self.connect_to_service(name)


# ---------------------------------------------------------------------------
# OSCQueryBrowseNode — browse + search + instantiate
# ---------------------------------------------------------------------------

class OSCQueryBrowseNode(Node, OSCBase):
    """
    Interactive browser for discovered OSCQuery services.

    Features:
    - Service selector combo
    - Cross-service parameter search (via on_edit)
    - Tree drill-down navigation via list_box (via on_deactivate)
    - Instantiate widgets at any level (leaf, container, or root)
    - Channel-aware instantiation for repeating indexed structures
    """

    @staticmethod
    def factory(name, data, args=None):
        node = OSCQueryBrowseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # --- UI Elements ---

        # Search box — uses on_edit for live filtering
        self.search_input = self.add_input(
            '##search', widget_type='text_input', widget_width=200,
            default_value=''
        )

        # Clickable path bar is injected dynamically in custom_create
        self._current_search_query = ''

        # Navigation list
        self.nav_list = self.add_property('###browser', widget_type='list_box', width=280)

        # Refresh button (right under the list)
        self.refresh_button = self.add_input('refresh', widget_type='button',
                                             callback=self.refresh_services)

        # Subset filter (e.g. '1-3' to filter numbered entries)
        self.channel_input = self.add_input(
            'subset', widget_type='text_input', widget_width=100,
            default_value='',
        )

        # Action buttons
        self.create_button = self.add_input('create', widget_type='button',
                                             callback=self.instantiate_selected)
        self.create_all_button = self.add_input('create all', widget_type='button',
                                                 callback=self.instantiate_all)
        self.go_to_button = self.add_input('go to', widget_type='button',
                                           callback=self.go_to_selected)

        # Layout selector (under create_all)
        self.layout_selector = self.add_input(
            'layout', widget_type='combo',
            default_value='horizontal',
        )

        # --- Outputs ---
        self.selected_path_out = self.add_output('selected path')
        self.service_info_out = self.add_output('service info')
        self.param_info_out = self.add_output('param info')

        # Close button (under the outputs)
        self.close_button = self.add_input('close', widget_type='button',
                                           callback=self.close_browser)

        # --- Internal State ---
        self.current_service = None
        self.current_service_name = ''
        self.current_path = ''
        self.current_items = []
        self.search_results = []
        self.in_search_mode = False
        self.viewing_services = True  # start in services list
        self.channel_patterns = []
        self.filtered_list = []
        self._last_click_time = 0
        self._last_click_index = -1
        self._cumulative_y_offset = 0  # tracks vertical position for sequential instantiations

    def close_browser(self):
        """Remove this browser node from the patcher."""
        if hasattr(self, 'my_editor') and self.my_editor:
            self.my_editor.remove_node(self)

    def custom_create(self, from_file):
        self.layout_selector.widget.combo_items = ['horizontal', 'vertical', 'tree']
        dpg.configure_item(self.layout_selector.widget.uuid, items=['horizontal', 'vertical', 'tree'])
        self.path_group_uuid = dpg.add_group(horizontal=True, before=self.nav_list.widget.uuid)
        self.refresh_services()
        
        # Register for automatic UI updates when services arrive/depart
        self.needs_refresh = False
        if self.osc_manager and self.osc_manager.oscquery_browser:
            self.osc_manager.oscquery_browser.add_callback(self._on_service_event)
            self.add_frame_task()

    def _on_service_event(self, event, service_name):
        """Called from a background thread when Zeroconf discovers/loses a service."""
        self.needs_refresh = True

    def frame_task(self):
        """Executes on the main DPG thread safely."""
        if getattr(self, 'needs_refresh', False):
            self.needs_refresh = False
            if self.viewing_services and not self.in_search_mode:
                self.refresh_services()

    def cleanup(self):
        if self.osc_manager and getattr(self.osc_manager, 'oscquery_browser', None):
            self.osc_manager.oscquery_browser.remove_callback(self._on_service_event)
        if hasattr(self, 'app') and hasattr(self.app, 'remove_frame_task'):
            self.app.remove_frame_task(self)
        super().cleanup()

    def _breadcrumb_nav_callback(self, sender, app_data, user_data):
        action, value = user_data
        if action == 'services':
            self.show_services_list()
        elif action == 'service_root':
            svc_name = value
            svc = self.osc_manager.oscquery_browser.get_service(svc_name)
            if svc:
                # Always re-fetch to get the latest registry state
                svc.fetch_json()
                self.current_service = svc
                self.current_service_name = svc_name
            self.viewing_services = False
            self.in_search_mode = False
            self.current_path = ''
            self.update_navigation()
        elif action == 'navigate':
            self.viewing_services = False
            self.in_search_mode = False
            self.current_path = value
            self.update_navigation()

    def update_path_bar(self):
        """Rebuild the clickable path bar above the list."""
        if not hasattr(self, 'path_group_uuid') or not dpg.does_item_exist(self.path_group_uuid):
            return
            
        dpg.delete_item(self.path_group_uuid, children_only=True)
        self.leaf_name_uuid = None
        
        # Always have a Services button
        dpg.add_button(label="Services", parent=self.path_group_uuid, user_data=('services', ''), callback=self._breadcrumb_nav_callback)
        
        if self.in_search_mode:
            dpg.add_text(">", parent=self.path_group_uuid)
            dpg.add_text(f"Search: '{self._current_search_query}' ({len(self.filtered_list)} results)", parent=self.path_group_uuid)
            return
            
        if self.viewing_services or not self.current_service_name:
            self.leaf_name_uuid = dpg.add_text("", parent=self.path_group_uuid, color=(150, 150, 150))
            return
            
        # Add service name button
        dpg.add_text(">", parent=self.path_group_uuid)
        dpg.add_button(label=self.current_service_name, parent=self.path_group_uuid, user_data=('service_root', self.current_service_name), callback=self._breadcrumb_nav_callback)
        
        if self.current_path:
            # Add intermediate segments
            parts = [p for p in self.current_path.split('/') if p]
            for i, part in enumerate(parts):
                dpg.add_text(">", parent=self.path_group_uuid)
                segment_path = '/' + '/'.join(parts[:i + 1])
                dpg.add_button(label=part, parent=self.path_group_uuid, user_data=('navigate', segment_path), callback=self._breadcrumb_nav_callback)

        self.leaf_name_uuid = dpg.add_text("", parent=self.path_group_uuid, color=(150, 150, 150))

    def update_leaf_name(self, name):
        """Safely update leaf name display without rebuilding the path bar (which breaks double-clicks)."""
        if hasattr(self, 'leaf_name_uuid') and getattr(self, 'leaf_name_uuid', None) and dpg.does_item_exist(self.leaf_name_uuid):
            if name:
                dpg.set_value(self.leaf_name_uuid, f"> {name}")
            else:
                dpg.set_value(self.leaf_name_uuid, "")

    def refresh_services(self, *args, **kwargs):
        """Refresh and show the services list."""
        self.viewing_services = True
        self.in_search_mode = False
        self.current_service = None
        self.current_service_name = ''
        self.current_path = ''
        self.show_services_list()

    def show_services_list(self):
        """Populate the list with all discovered services."""
        services = self.osc_manager.get_discovered_services()
        self.filtered_list = []
        self.current_items = []

        if not services:
            self.filtered_list = ['(no services found)']
            self.current_items = [{'name': '(none)', 'path': '', 'is_container': False, 'type': '', 'description': '', 'service_name': ''}]
        else:
            for svc_name in services:
                svc = self.osc_manager.oscquery_browser.get_service(svc_name)
                desc = ''
                if svc and svc.json_tree and 'DESCRIPTION' in svc.json_tree:
                    desc = f"  ({svc.json_tree['DESCRIPTION']})"
                self.filtered_list.append(f"[S] {svc_name}{desc}")
                self.current_items.append({
                    'name': svc_name,
                    'path': '',
                    'is_container': True,
                    'type': '',
                    'description': desc,
                    'service_name': svc_name,
                })

        dpg.configure_item(self.nav_list.widget.uuid, items=self.filtered_list)
        if self.filtered_list:
            self.nav_list.set(self.filtered_list[0])
        self.update_path_bar()

    # -----------------------------------------------------------------------
    # on_edit / on_deactivate
    # -----------------------------------------------------------------------

    def on_edit(self, widget):
        """Called when the search text changes — do live fuzzy search."""
        if widget == self.search_input.widget:
            query = dpg.get_value(self.search_input.widget.uuid)
            if query and query.strip():
                self.do_search(query.strip())
            else:
                # Empty search — go back to previous view
                if self.viewing_services:
                    self.show_services_list()
                else:
                    self.in_search_mode = False
                    self.update_navigation()

    def on_deactivate(self, widget):
        """Called when a list_box item is selected. Double-click drills down."""
        if widget == self.nav_list.widget:
            import time as _time
            selected = dpg.get_value(self.nav_list.widget.uuid)
            if selected and selected in self.filtered_list:
                index = self.filtered_list.index(selected)
                if index < len(self.current_items):
                    item = self.current_items[index]

                    # Output path
                    full_path = ''
                    svc_name = item.get('service_name', self.current_service_name)
                    if svc_name and item['path']:
                        full_path = f"{svc_name}:{item['path']}"
                    elif svc_name:
                        full_path = svc_name
                    if full_path:
                        self.selected_path_out.send(full_path)
                        self.update_leaf_name(item.get('name'))

                    # Output param info for leaves
                    if not item['is_container'] and item.get('service_name'):
                        svc = self.osc_manager.oscquery_browser.get_service(item['service_name'])
                        if svc and svc.json_tree:
                            node_at_path = navigate_to_path(svc.json_tree, item['path'])
                            if node_at_path:
                                self.param_info_out.send(node_at_path)

                    # Double-click detection
                    now = _time.time()
                    if index == self._last_click_index and (now - self._last_click_time) < 0.4:
                        # Double-click — drill down for containers, create for leaves
                        self.open_selected()
                        self._last_click_time = 0
                        self._last_click_index = -1
                    else:
                        self._last_click_time = now
                        self._last_click_index = index
        elif widget == self.search_input.widget:
            pass

    def open_selected(self):
        """Drill down into the currently selected item or navigate via breadcrumb."""
        selected = dpg.get_value(self.nav_list.widget.uuid)
        if not selected or selected not in self.filtered_list:
            return
        index = self.filtered_list.index(selected)
        if index >= len(self.current_items):
            return
        item = self.current_items[index]

        # Handle navigation into a container
        if not item['is_container']:
            # Double-click on a leaf triggers widget creation
            self.instantiate_selected()
            return

        # Determine which service to use
        svc_name = item.get('service_name', self.current_service_name)
        if svc_name:
            svc = self.osc_manager.oscquery_browser.get_service(svc_name)
            if svc:
                # Always re-fetch to pick up newly registered parameters
                svc.fetch_json()
                self.current_service = svc
                self.current_service_name = svc_name

        self.viewing_services = False
        self.in_search_mode = False
        self.current_path = item['path']
        self.update_navigation()

    # -----------------------------------------------------------------------
    # Navigation helpers
    # -----------------------------------------------------------------------

    def update_navigation(self):
        """Update the list_box to show contents of current path."""
        if self.current_service is None:
            return
        
        # Lazy fetch: if the initial background fetch failed, retry now
        if self.current_service.json_tree is None:
            self.current_service.fetch_json(retries=2, delay=0.3)
            if self.current_service.json_tree is None:
                print(f"OSCQueryBrowse: Could not fetch tree from {self.current_service.name} ({self.current_service.ip}:{self.current_service.http_port})")
                return

        if not HAS_CONVERTER:
            return

        # Get contents at current path
        contents_items = get_contents_list(self.current_service.json_tree, self.current_path)

        self.current_items = contents_items
        self.filtered_list = []

        for item in contents_items:
            prefix = '[+] ' if item['is_container'] else '  - '
            type_suffix = f" [{item['type']}]" if item['type'] else ''
            self.filtered_list.append(prefix + item['name'] + type_suffix)

        dpg.configure_item(self.nav_list.widget.uuid, items=self.filtered_list)
        if self.filtered_list:
            self.nav_list.set(self.filtered_list[0])
        self.in_search_mode = False
        self.update_path_bar()

    def navigate_back(self):
        """Go up one level in the tree, or back to services list."""
        if self.in_search_mode:
            self.in_search_mode = False
            if self.viewing_services:
                self.show_services_list()
            else:
                self.update_navigation()
            return

        if self.current_path:
            # Go up one level within the service
            parts = self.current_path.rsplit('/', 1)
            self.current_path = parts[0] if len(parts) > 1 else ''
            self.update_navigation()
        else:
            # Already at root of service — go back to services list
            self.viewing_services = True
            self.show_services_list()

    def do_search(self, query):
        """Fuzzy search across all services.

        Splits query into words and matches paths where ALL words appear
        somewhere in the full path (case-insensitive). e.g. 'channel volume'
        matches '/channel/1/volume'.
        """
        self.in_search_mode = True
        words = query.lower().split()

        # Use the browser's search for each word, then intersect
        browser = self.osc_manager.oscquery_browser
        all_results = []

        with browser._lock:
            for svc_name, svc in browser.services.items():
                if svc.json_tree is None:
                    continue
                leaves = self._collect_all_paths(svc.json_tree, '')
                svc_name_lower = svc_name.lower()
                for path, node_dict in leaves:
                    search_string = f"{svc_name_lower}:{path.lower()}"
                    # All words must appear somewhere in the service or path
                    if all(word in search_string for word in words):
                        all_results.append((svc_name, path, node_dict))

        # Build display list
        self.filtered_list = []
        self.current_items = []
        for svc_name, path, param_dict in all_results:
            display_label = f"{svc_name}:{path}"
            self.filtered_list.append(display_label)
            is_container = 'CONTENTS' in param_dict if isinstance(param_dict, dict) else False
            self.current_items.append({
                'name': path.split('/')[-1],
                'path': path,
                'is_container': is_container,
                'type': param_dict.get('TYPE', '') if isinstance(param_dict, dict) else '',
                'description': param_dict.get('DESCRIPTION', '') if isinstance(param_dict, dict) else '',
                'service_name': svc_name,
            })

        dpg.configure_item(self.nav_list.widget.uuid, items=self.filtered_list)
        if self.filtered_list:
            self.nav_list.set(self.filtered_list[0])

        self._current_search_query = query
        self.update_path_bar()

    def _collect_all_paths(self, tree_node, current_path):
        """Collect all nodes (containers and leaves) for search indexing."""
        results = []
        if not isinstance(tree_node, dict):
            return results

        if 'TYPE' in tree_node:
            results.append((current_path, tree_node))
        if 'CONTENTS' in tree_node:
            if current_path:  # don't add root
                results.append((current_path, tree_node))
            for key, child in tree_node['CONTENTS'].items():
                child_path = current_path + '/' + key
                results.extend(self._collect_all_paths(child, child_path))
        return results

    # -----------------------------------------------------------------------
    # Instantiation
    # -----------------------------------------------------------------------

    def go_to_selected(self):
        """Transition from the search list to the parent directory of the selected item."""
        selected = dpg.get_value(self.nav_list.widget.uuid)
        if not selected or selected not in self.filtered_list:
            return
            
        index = self.filtered_list.index(selected)
        if index >= len(self.current_items):
            return
            
        item = self.current_items[index]
        
        svc_name = item.get('service_name', self.current_service_name)
        if not svc_name:
            return
            
        svc = self.osc_manager.oscquery_browser.get_service(svc_name)
        if not svc:
            return
            
        # Transition out of search mode
        self.in_search_mode = False
        self.viewing_services = False
        self.current_service = svc
        self.current_service_name = svc_name
        
        path = item.get('path', '')
        if not path or path == '/':
            self.current_path = '/'
        else:
            if item.get('is_container'):
                self.current_path = path
            else:
                parts = [p for p in path.split('/') if p]
                if len(parts) > 1:
                    self.current_path = '/' + '/'.join(parts[:-1])
                else:
                    self.current_path = '/'
                    
        self.update_path_bar()
        self.update_navigation()


    def instantiate_selected(self):
        """Create dpg_system nodes for the currently selected item."""
        selected_text = dpg.get_value(self.nav_list.widget.uuid)
        if not selected_text or selected_text not in self.filtered_list:
            return

        index = self.filtered_list.index(selected_text)
        if index >= len(self.current_items):
            return

        item = self.current_items[index]

        # Determine the service
        svc_name = item.get('service_name', self.current_service_name)
        if not svc_name:
            return

        service = self.osc_manager.oscquery_browser.get_service(svc_name)
        if service is None:
            return

        # Ensure osc_device exists for this service
        self._ensure_device(service)

        # Get the JSON tree
        json_tree = service.json_tree
        if json_tree is None:
            return

        target_path = item['path']

        # Channel filter
        channel_spec = self.channel_input()
        channels = self._parse_channel_spec(channel_spec) if channel_spec else None

        if target_path:
            # Navigate to the specific node
            node_at_path = navigate_to_path(json_tree, target_path)
        else:
            # Entire service — use the root tree
            node_at_path = json_tree

        if node_at_path is None:
            return

        # Check for TYPE — can be at top level (external format)
        # or inside CONTENTS (internal registry format)
        is_leaf = 'TYPE' in node_at_path
        if not is_leaf and 'CONTENTS' in node_at_path and isinstance(node_at_path['CONTENTS'], dict):
            is_leaf = 'TYPE' in node_at_path['CONTENTS']
            # If it's a leaf in internal format, merge CONTENTS metadata up
            if is_leaf:
                merged = dict(node_at_path['CONTENTS'])
                merged['CONTENTS'] = {}  # preserve structure
                node_at_path = merged

        if is_leaf:
            node = self._create_widget_for_param(svc_name, target_path, node_at_path, 0)
            if node:
                self._select_and_drag_nodes([node])
        elif 'CONTENTS' in node_at_path:
            self._create_widgets_for_subtree(svc_name, target_path, node_at_path, channels)

    def instantiate_all(self):
        """Create widgets for the entire current directory context or all search results."""
        if not self.current_items:
            return

        if self.in_search_mode:
            created_nodes = []
            layout = self.layout_selector()
            offset_x = 0
            
            for item in self.current_items:
                svc_name = item.get('service_name')
                if not svc_name:
                    continue
                svc = self.osc_manager.oscquery_browser.get_service(svc_name)
                if svc is None or svc.json_tree is None:
                    continue
                self._ensure_device(svc)
                target_path = item['path']
                node_at_path = navigate_to_path(svc.json_tree, target_path)
                if node_at_path:
                    # check if leaf
                    is_leaf = 'TYPE' in node_at_path
                    if not is_leaf and 'CONTENTS' in node_at_path and isinstance(node_at_path['CONTENTS'], dict):
                        is_leaf = 'TYPE' in node_at_path['CONTENTS']
                        if is_leaf:
                            merged = dict(node_at_path['CONTENTS'])
                            merged['CONTENTS'] = {}
                            node_at_path = merged
                    if is_leaf:
                        node = self._create_widget_for_param(svc_name, target_path, node_at_path, offset_x=offset_x, offset_y=0)
                        if node:
                            created_nodes.append(node)
                            if layout == 'horizontal':
                                offset_x += 150
                            else:
                                self._cumulative_y_offset += 40
                                
            if layout == 'horizontal' and created_nodes:
                self._cumulative_y_offset += 40
                
            if created_nodes:
                self._select_and_drag_nodes(created_nodes)
            return

        # Determine the service
        svc_name = self.current_service_name
        if not svc_name:
            for item in reversed(self.current_items):
                if item.get('service_name'):
                    svc_name = item['service_name']
                    break
                    
        if not svc_name:
            return

        service = self.osc_manager.oscquery_browser.get_service(svc_name)
        if service is None or service.json_tree is None:
            return

        self._ensure_device(service)

        target_path = self.current_path
        
        # Channel filter
        channel_spec = self.channel_input()
        channels = self._parse_channel_spec(channel_spec) if channel_spec else None

        if target_path and target_path != '/':
            node_at_path = navigate_to_path(service.json_tree, target_path)
        else:
            node_at_path = service.json_tree
            target_path = ''

        if node_at_path is None:
            return

        is_leaf = 'TYPE' in node_at_path
        if not is_leaf and 'CONTENTS' in node_at_path and isinstance(node_at_path['CONTENTS'], dict):
            is_leaf = 'TYPE' in node_at_path['CONTENTS']
            if is_leaf:
                merged = dict(node_at_path['CONTENTS'])
                merged['CONTENTS'] = {}
                node_at_path = merged

        if is_leaf:
            node = self._create_widget_for_param(svc_name, target_path, node_at_path, 0)
            if node:
                self._select_and_drag_nodes([node])
        elif 'CONTENTS' in node_at_path:
            self._create_widgets_for_subtree(svc_name, target_path, node_at_path, channels)

    def _ensure_device(self, service):
        """Ensure osc_device exists for this service, and its port is up to date."""
        device_name = service.name
        existing = self.osc_manager.targets.get(device_name)
        if existing is None:
            args = [device_name, str(service.ip), str(service.osc_port)]
            try:
                device_node = Node.app.create_node_by_name('osc_device', None, args)
                if device_node:
                    device_node.set_visibility('hidden')
                    try:
                        dpg.set_item_pos(device_node.uuid, [-10000, -10000])
                    except Exception:
                        pass
            except Exception as e:
                print(f"oscq_browse: Failed to create device for {device_name}: {e}")
        elif existing.target_port != service.osc_port or existing.ip != service.ip:
            # Device exists but port/ip is stale — update it
            existing.destroy_client()
            existing.target_port = service.osc_port
            existing.ip = service.ip
            existing.create_client()
            if hasattr(existing, 'target_port_property'):
                existing.target_port_property.set(str(service.osc_port))
            print(f"oscq_browse: Updated device '{device_name}' to {service.ip}:{service.osc_port}")

    @staticmethod
    def _short_label_from_path(osc_path):
        """Extract a concise label from an OSC path.

        e.g. '/channels/1/gain' → '1/gain'
             '/master/volume' → 'volume'
             '/stt/result' → 'result'
        """
        segments = [s for s in osc_path.split('/') if s]
        if len(segments) >= 2:
            return '/'.join(segments[-2:])
        elif segments:
            return segments[-1]
        return osc_path

    def _create_widget_for_param(self, service_name, osc_path, param_dict, offset_x=0, offset_y=0):
        """Create an osc widget node for a parameter."""
        if not HAS_CONVERTER:
            return

        # Fetch the live parameter state from the server (not cached)
        if self.in_search_mode and len(self.current_items) > 0 and 'service_name' in self.current_items[0]:
            svc = self.osc_manager.oscquery_browser.get_service(service_name)
        else:
            svc = self.current_service
        if svc:
            live_param = svc.fetch_param(osc_path)
            if live_param and 'TYPE' in live_param:
                param_dict = live_param

        spec = param_to_node_spec(param_dict)
        if spec is None or spec['node_type'] is None:
            return

        node_type = spec['node_type']

        # Build args: [target_name, address, ...extra]
        args = [service_name, osc_path]
        # osc_menu requires choices as additional args
        if node_type == 'osc_menu' and spec.get('choices'):
            args.extend(spec['choices'])

        # Calculate position
        try:
            my_pos = dpg.get_item_pos(self.uuid)
            pos = [my_pos[0] + offset_x, my_pos[1] + 300 + self._cumulative_y_offset + offset_y]
        except Exception:
            pos = None

        try:
            created_node = Node.app.create_node_by_name(node_type, None, args, pos)
            # Set initial value safely without triggering an OSC broadcast
            if created_node and spec.get('value') is not None:
                if getattr(created_node, 'input', None) is not None:
                    created_node.input.widget.set(spec['value'], propagate=False)
            # Mark as proxy since this widget controls a remote service
            if created_node and hasattr(created_node, 'mode_option'):
                created_node.mode_option.set('proxy')
                created_node._apply_mode('proxy')  # canonical lifecycle entry point
            return created_node
        except Exception as e:
            osc_type = param_dict.get('TYPE', '?')
            print(f"oscq_browse: Failed to create {node_type} for {osc_path} (TYPE={osc_type}): {e}")
            return None

    def _create_widgets_for_subtree(self, service_name, base_path, tree_node, channels=None):
        """Create widgets following the selected layout mode.

        Layout modes:
        - horizontal: groups are rows, leaves left-to-right, rows stacked vertically
        - vertical: all leaves in one column, stacked top-to-bottom
        - tree: first-level children are columns, leaves stack vertically in each
        """
        if not HAS_CONVERTER:
            return

        layout = self.layout_selector() if self.layout_selector else 'horizontal'

        if 'CONTENTS' not in tree_node:
            if 'TYPE' in tree_node:
                self._create_widget_for_param(service_name, base_path, tree_node, 0, 0)
            return

        contents = tree_node['CONTENTS']

        try:
            base_pos = dpg.get_item_pos(self.uuid)
            start_x = base_pos[0]
            start_y = base_pos[1] + 300 + self._cumulative_y_offset
        except Exception:
            start_x, start_y = 100, 400

        # Gather groups: each group is a list of created nodes
        groups = []

        for child_key, child_node in contents.items():
            # Skip non-dict entries (metadata like ACCESS, TYPE at this level)
            if not isinstance(child_node, dict):
                continue
            child_path = base_path + '/' + child_key

            if channels is not None:
                if not self._path_matches_channels(child_path, base_path, channels):
                    continue

            if 'TYPE' in child_node:
                # Direct leaf — add as a single-element group
                node = self._create_widget_at_pos(
                    service_name, child_path, child_node, [start_x, start_y]
                )
                if node:
                    groups.append([node])
            elif 'CONTENTS' in child_node:
                leaves = self._collect_leaves_with_paths(child_node, child_path)
                group_nodes = []
                for leaf_path, leaf_node in leaves:
                    node = self._create_widget_at_pos(
                        service_name, leaf_path, leaf_node, [start_x, start_y]
                    )
                    if node:
                        group_nodes.append(node)
                if group_nodes:
                    groups.append(group_nodes)

        if channels is not None:
            # Per-channel grouping: each channel is a separate group
            # horizontal: each channel = row (widgets left-to-right), channels stack vertically
            # vertical: each channel = column (widgets top-to-bottom), channels go left-to-right
            if layout == 'vertical':
                layout = 'tree'  # tree mode already does columns side-by-side
            # horizontal mode already does rows stacked vertically — no change needed
        elif layout == 'vertical':
            layout = 'tree'  # use column layout for groups by default

        # Store for deferred repositioning
        if groups:
            self._pending_groups = groups
            self._all_created_nodes = [n for g in groups for n in g]  # flat list for drag selection
            self._layout_mode = layout
            self._layout_start_x = start_x
            self._layout_start_y = start_y
            self._layout_frames_remaining = 2
            Node.app.add_frame_task(self)

    def _create_widget_at_pos(self, service_name, osc_path, param_dict, pos):
        """Create a widget node at an explicit position. Returns the created node or None."""
        if not HAS_CONVERTER:
            return None

        svc = self.current_service
        if svc:
            live_param = svc.fetch_param(osc_path)
            if live_param and 'TYPE' in live_param:
                param_dict = live_param

        spec = param_to_node_spec(param_dict)
        if spec is None or spec['node_type'] is None:
            return None

        node_type = spec['node_type']
        args = [service_name, osc_path]
        # osc_menu requires choices as additional args
        if node_type == 'osc_menu' and spec.get('choices'):
            args.extend(spec['choices'])

        try:
            created_node = Node.app.create_node_by_name(node_type, None, args, pos)
            if created_node and spec.get('value') is not None:
                if getattr(created_node, 'input', None) is not None:
                    # Set initial visual value safely
                    created_node.input.widget.set(spec['value'], propagate=False)
            # Mark as proxy since this widget controls a remote service
            if created_node and hasattr(created_node, 'mode_option'):
                created_node.mode_option.set('proxy')
                created_node.mode_changed()  # .set() doesn't trigger DPG callbacks
            return created_node
        except Exception as e:
            print(f"oscq_browse: Failed to create {node_type} for {osc_path}: {e}")
            return None

    def frame_task(self):
        """Deferred reposition based on measured sizes and layout mode."""
        if not hasattr(self, '_pending_groups') or not self._pending_groups:
            Node.app.remove_frame_task(self)
            return

        self._layout_frames_remaining -= 1
        if self._layout_frames_remaining > 0:
            return

        groups = self._pending_groups
        mode = self._layout_mode
        start_x = self._layout_start_x
        start_y = self._layout_start_y
        gap = 8

        if mode == 'horizontal':
            # Groups are rows: items left-to-right, rows stacked vertically
            current_y = start_y
            for row_nodes in groups:
                current_x = start_x
                row_height = 0
                for node in row_nodes:
                    try:
                        dpg.set_item_pos(node.uuid, [current_x, current_y])
                        size = dpg.get_item_rect_size(node.uuid)
                        w = size[0] if size[0] > 0 else 150
                        h = size[1] if size[1] > 0 else 40
                        row_height = max(row_height, h)
                        current_x += w + gap
                    except Exception:
                        current_x += 160
                        row_height = max(row_height, 45)
                current_y += row_height + gap
        else:
            # tree / vertical: groups are columns, items top-to-bottom
            current_x = start_x
            for col_nodes in groups:
                current_y = start_y
                col_width = 0
                for node in col_nodes:
                    try:
                        dpg.set_item_pos(node.uuid, [current_x, current_y])
                        size = dpg.get_item_rect_size(node.uuid)
                        w = size[0] if size[0] > 0 else 150
                        h = size[1] if size[1] > 0 else 40
                        col_width = max(col_width, w)
                        current_y += h + gap
                    except Exception:
                        current_y += 45
                        col_width = max(col_width, 150)
                current_x += col_width + gap
            max_y = max(current_y for _ in [0])  # current_y from last column not useful
            # Track the tallest column
            total_height = 0
            for col_nodes in groups:
                col_h = 0
                for node in col_nodes:
                    try:
                        size = dpg.get_item_rect_size(node.uuid)
                        col_h += (size[1] if size[1] > 0 else 40) + gap
                    except Exception:
                        col_h += 45
                total_height = max(total_height, col_h)

        # Advance cumulative offset by the height of this layout block
        if mode == 'horizontal':
            total_height = current_y - start_y
        self._cumulative_y_offset += total_height + 20

        # Select all created nodes as a draggable group
        all_nodes = getattr(self, '_all_created_nodes', [])
        self._select_and_drag_nodes(all_nodes)
        self._all_created_nodes = []

        self._pending_groups = None
        Node.app.remove_frame_task(self)

    def _select_and_drag_nodes(self, nodes):
        """Make a group of nodes selected and draggable (like paste/duplicate)."""
        if not nodes:
            return
        editor = Node.app.get_current_editor()
        if not editor:
            return
        
        Node.app.created_nodes = {}
        Node.app.drag_starts = {}
        for node in nodes:
            Node.app.created_nodes[node.uuid] = node
            Node.app.drag_starts[node.uuid] = dpg.get_item_pos(node.uuid)
            
        if len(Node.app.drag_starts) > 0:
            sort = sorted(Node.app.drag_starts.items(), key=lambda item: item[1][0])
            left_most = sort[0][1][0]
            sort = sorted(Node.app.drag_starts.items(), key=lambda item: item[1][1])
            top_most = sort[0][1][1]
            left_top = [left_most + 30, top_most + 10]
            Node.app.dragging_ref = editor.editor_pos_to_global_pos(left_top)
            Node.app.dragging_created_nodes = True
            Node.app.drag_create_nodes()

    def _collect_leaves_with_paths(self, tree_node, base_path):
        """Collect all leaf parameters (with TYPE) under a tree node, depth-first."""
        results = []
        if 'TYPE' in tree_node:
            results.append((base_path, tree_node))
        if 'CONTENTS' in tree_node:
            for key, child in tree_node['CONTENTS'].items():
                child_path = base_path + '/' + key
                results.extend(self._collect_leaves_with_paths(child, child_path))
        return results

    def _parse_channel_spec(self, spec_str):
        """
        Parse a channel specification string.
        Supports: "1", "1-5", "1,3,5", "1-3,7,9-11"
        Returns a list of channel indices.
        """
        if not spec_str or not spec_str.strip():
            return None

        channels = []
        parts = spec_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start_end = part.split('-')
                if len(start_end) == 2:
                    try:
                        start = int(start_end[0])
                        end = int(start_end[1])
                        channels.extend(range(start, end + 1))
                    except ValueError:
                        pass
            else:
                try:
                    channels.append(int(part))
                except ValueError:
                    pass
        return channels if channels else None

    def _path_matches_channels(self, path, base_path, channels):
        """
        Check if a path corresponds to any of the specified channel indices.
        Looks for numeric path segments that match the channel list.
        """
        if path.startswith(base_path):
            rel = path[len(base_path):]
        else:
            rel = path

        segments = [s for s in rel.split('/') if s]
        for seg in segments:
            if re.match(r'^\d+$', seg):
                if int(seg) in channels:
                    return True
        return False

    def execute(self):
        """No-op — interactions are handled by on_edit and on_deactivate."""
        pass


# ---------------------------------------------------------------------------
# OSCQueryHostNode — declares a subpatcher as an OSCQuery service
# ---------------------------------------------------------------------------

class OSCQueryHostNode(Node, OSCBase):
    """
    Place this node inside a subpatcher to declare it as an OSCQuery service.
    The service name is derived from the containing patcher's name.

    Usage:
        oscq_host          — auto-assign OSC port
        oscq_host 8000     — use OSC port 8000
    """

    @staticmethod
    def factory(name, data, args=None):
        node = OSCQueryHostNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Marker so get_registry_for_editor can find us
        self._is_oscq_host = True
        self.service_name = ''
        self.osc_port = 0
        self._owned_device_node = None

        # Parse optional port and service_name from args
        if args:
            for arg in args:
                if type(arg) in [int, float] or (isinstance(arg, str) and arg.isdigit()):
                    self.osc_port = int(arg)
                elif isinstance(arg, str):
                    self.service_name = arg

        self._service_name_property = self.add_option('service name', widget_type='text_input', default_value=self.service_name, callback=self.service_name_changed)

        # UI
        self.service_label = self.add_property('service', widget_type='label')
        self.port_label = self.add_property('port', widget_type='label')
        self.status_label = self.add_property('status', widget_type='label')

    def custom_create(self, from_file):
        """Start the service once the node is fully created."""
        self._start_service()

    def _start_service(self):
        """Derive service name safely and start the server."""
        prop_val = self._service_name_property() if hasattr(self, '_service_name_property') else ''
        if prop_val and prop_val != '':
            self.service_name = prop_val
        else:
            # First initialization fallback (if no parameter exists yet)
            editor = self.my_editor
            if editor is not None and hasattr(editor, 'patch_name'):
                self.service_name = editor.patch_name.replace(' ', '_')
            else:
                self.service_name = 'dpg_service'
            
            # Permanently lock it so dynamic Save As renames don't break network integrations
            if hasattr(self, '_service_name_property'):
                self._service_name_property.set(self.service_name)

        # Auto-assign port if not specified
        if self.osc_port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', 0))
                self.osc_port = s.getsockname()[1]

        # Register with OSCManager
        registry = self.osc_manager.register_service(self.service_name, self.osc_port)
        if registry is not None:
            self.service_label.widget.set(self.service_name)
            self.port_label.widget.set(f'OSC port: {self.osc_port}')
            self.status_label.widget.set('active')

            # Ensure osc_device for this service exists and has the correct port
            existing_target = self.osc_manager.targets.get(self.service_name)
            if existing_target is None:
                try:
                    device_args = [self.service_name, '127.0.0.1', str(self.osc_port), str(self.osc_port)]
                    # Create at far off-screen position to prevent DPG focus handler crash
                    device_node = Node.app.create_node_by_name('osc_device', None, device_args, pos=[-10000, -10000])
                    if device_node:
                        device_node.set_visibility('hidden')
                        self._owned_device_node = device_node
                except Exception as e:
                    print(f"oscq_host: Failed to create device for {self.service_name}: {e}")
            else:
                # Device loaded from file with potentially stale port — update it
                needs_update = False
                if existing_target.target_port != self.osc_port:
                    existing_target.destroy_client()
                    existing_target.target_port = self.osc_port
                    existing_target.ip = '127.0.0.1'
                    existing_target.create_client()
                    needs_update = True
                # Also update the source (listener) port
                if hasattr(existing_target, 'source_port') and existing_target.source_port != self.osc_port:
                    existing_target.destroy_server()
                    existing_target.source_port = self.osc_port
                    existing_target.start_serving()
                    needs_update = True
                if needs_update:
                    # Update UI properties if they exist
                    if hasattr(existing_target, 'target_port_property'):
                        existing_target.target_port_property.set(str(self.osc_port))
                    if hasattr(existing_target, 'source_port_property'):
                        existing_target.source_port_property.set(str(self.osc_port))
                    print(f"oscq_host: Updated device '{self.service_name}' to port {self.osc_port}")
        else:
            self.service_label.widget.set(self.service_name)
            self.port_label.widget.set('—')
            self.status_label.widget.set('failed (no oscquery)')

    def custom_cleanup(self):
        """Stop the service when the node is removed."""
        if self.service_name:
            self.osc_manager.unregister_service(self.service_name)

    def service_name_changed(self):
        """When the user explicitly changes the service name option, dynamically restart the service."""
        new_name = self._service_name_property() if hasattr(self, '_service_name_property') else ''
        if new_name != self.service_name:
            old_name = self.service_name
            # Rename the existing osc_device rather than orphaning it
            if old_name and old_name in self.osc_manager.targets:
                existing_device = self.osc_manager.targets[old_name]
                self.osc_manager.rename_device(existing_device, new_name)
            if old_name:
                self.osc_manager.unregister_service(old_name)
            self._start_service()

    def patcher_name_changed(self, old_name, new_name):
        """Purposefully disabled: the user specifically requested that dynamic Save As renames 
        do NOT shred active OSCQuery services, decoupling file name from HTTP server name."""
        pass

    def frame_task(self):
        """Create the osc_device on the next frame after host is fully initialized."""
        Node.app.remove_frame_task(self)
        if hasattr(self, '_pending_device_name'):
            name = self._pending_device_name
            port = self._pending_device_port
            del self._pending_device_name
            del self._pending_device_port
            if name not in self.osc_manager.targets:
                try:
                    # Pass port twice: first=target_port, second=source_port
                    device_args = [name, '127.0.0.1', str(port), str(port)]
                    device_node = Node.app.create_node_by_name('osc_device', None, device_args)
                    if device_node:
                        device_node.set_visibility('hidden')
                except Exception as e:
                    print(f"oscq_host: Failed to create device for {name}: {e}")

    def execute(self):
        pass
