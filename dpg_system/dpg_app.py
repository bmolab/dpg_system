import dearpygui.dearpygui as dpg
import time
import numpy as np

from dpg_system.node import Node, InputNodeAttribute, OutputNodeAttribute, Variable, PlaceholderNode
from dpg_system.node_editor import *
from os.path import exists
import json
import os
import platform as platform_

imported = []
try:
    from dpg_system.basic_nodes import *
    imported.append('basic_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.math_nodes import *
    imported.append('math_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.signal_nodes import *
    imported.append('signal_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.interface_nodes import *
    imported.append('interface_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.quaternion_nodes import *
    imported.append('quaternion_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.osc_nodes import *
    imported.append('osc_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.spacy_nodes import *
    imported.append('spacy_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.motion_cap_nodes import *
    imported.append('motion_cap_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.matrix_nodes import *
    imported.append('matrix_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.gl_nodes import *
    imported.append('gl_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.clip_nodes import *
    imported.append('clip_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.numpy_nodes import *
    imported.append('numpy_nodes.py')
except ModuleNotFoundError:
    pass
try:
    from dpg_system.torch_nodes import *
    imported.append('torch_nodes.py')
except ModuleNotFoundError:
    pass

# import additional node files in folder

for entry in os.scandir('dpg_system'):
    if entry.is_file():
        if entry.name[-8:] == 'nodes.py':
            if entry.name not in imported:
                name = entry.name[:-3]
                string = f'from dpg_system.{name} import *'
                exec(string)

if os.path.exists('dpg_system/plugins'):
    for entry in os.scandir('dpg_system/plugins'):
        if entry.is_file():
            if entry.name[-8:] == 'nodes.py':
                if entry.name not in imported:
                    name = entry.name[:-3]
                    string = f'from dpg_system.plugins.{name} import *'
                    exec(string)
        else:
            for subentry in os.scandir('dpg_system/plugins/' + entry.name):
                if subentry.is_file():
                    if subentry.name[-8:] == 'nodes.py':
                        if subentry.name not in imported:
                            name = subentry.name[:-3]
                            string = f'from dpg_system.plugins.{entry.name}.{subentry} import *'
                            exec(string)




def widget_active(source, data, user_data):
    pass


def widget_hovered(source, data, user_data):
    if dpg.does_item_exist(data):
        Node.app.hovered_item = dpg.get_item_user_data(data)


def widget_activated(source, data, user_data):
    if dpg.does_item_exist(data):
        item_type = dpg.get_item_type(data)
        # print(item_type, type(item_type))
        if item_type not in ['mvAppItemType::mvCombo', 'mvAppItemType::mvButton']:
            Node.app.active_widget = data
        else:
            Node.app.active_widget = -1
        # print("activated", Node.app.active_widget)
    else:
        Node.app.active_widget = -1


def widget_deactive(source, data, user_data):
    if dpg.does_item_exist(data):
        node = None
        if Node.app.return_pressed:
            Node.app.return_pressed = False
            item = dpg.get_item_user_data(data)
            if item is not None:
                if isinstance(item, Node):
                    node = item
                else:
                    node = item.node
                if node is not None and node.label == 'New Node':
                    node.execute()
        elif dpg.is_item_ok(data):
            item = dpg.get_item_user_data(data)
            if item is not None:
                if isinstance(item, Node):
                    node = item
                else:
                    node = item.node
                if node is not None:
                    node.on_deactivate(item)
    Node.app.active_widget = -1
    Node.app.focussed_widget = -1


def widget_edited(source, data, user_data):
    node = None
    if user_data is not None:
        if isinstance(user_data, Node):
            user_data.on_edit()
        # print("edited+", user_data, data)
    else:
        if dpg.does_item_exist(data):
            item = dpg.get_item_user_data(data)
            # print('edited', data, item)
            if item is not None:
                if isinstance(item, Node):
                    node = item
                else:
                    node = item.node
                if node is not None:
                    # print("edited", data)
                    node.on_edit(item)


def widget_deactive_after_edit(source, data, user_data):
    if user_data is not None:
        user_data.execute()
        # print("after edit+", user_data, data)
    else:
        if dpg.does_item_exist(data):
            item = dpg.get_item_user_data(data)
            if item is not None:
                item.value_changed(data, force=True)
    Node.app.active_widget = -1
    Node.app.focussed_widget = -1


def widget_focus(source, data, user_data):
    Node.app.focussed_widget = data


def widget_clicked(source, data, user_data):
    # print('widget clicked', data)
    pass


load_path = None
save_path = None

def cancel_callback(sender, app_data):
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1

def load_patches_callback(sender, app_data):
    global load_path

    if app_data is not None and 'file_path_name' in app_data:
        load_path = app_data['file_path_name']
        if load_path != '':
            Node.app.load_from_file(load_path)
    else:
        print('no file chosen')
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1


def save_file_callback(sender, app_data):
    global save_path
    if app_data is not None and 'file_path_name' in app_data:
        save_path = app_data['file_path_name']
        if save_path != '':
            Node.app.save_patch(save_path)
            if Node.app.saving_to_lib:
                Node.app.register_patcher(Node.app.patches_name)
                Node.app.saving_to_lib = False
    else:
        print('no file chosen')
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1


def save_patches_callback(sender, app_data):
    global save_path
    if app_data is not None and 'file_path_name' in app_data:
        save_path = app_data['file_path_name']
        if save_path != '':
            Node.app.save_setup(save_path)
    else:
        print('no file chosen')
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1


class App:
    def __init__(self):
        self.viewport = None
        self.main_window_id = -1
        self.loading = False
        self.setup_dpg()
        self.verbose = False
        self.verbose_menu_item = -1
        self.minimap_menu_item = -1
        self.window_padding = [4, 3]
        self.frame_padding = [4, 0]
        self.cell_padding = [4, 2]
        self.item_spacing = [5, 2]
        self.setup_themes()
        self.node_factory_container = NodeFactoryContainer("Modifiers", 150, -1)
        self.side_panel = dpg.generate_uuid()
        self.center_panel = dpg.generate_uuid()
        self.tab_bar = None
        self.node_list = []
        self.tabs = []
        Node.app = self
        self.register_nodes()

        self.patchers = []

        self.node_editors = []
        self.current_node_editor = 0
        self.frame_tasks = []
        self.variables = {}
        self.conduits = {}
        self.actions = {}
        self.loaded_patcher_nodes = []

        self.osc_manager = OSCManager(label='osc_manager', data=0, args=None)

        self.setup_menus()
        self.patches_path = ''
        self.patches_name = ''
        self.links_containers = {}
        self.created_nodes = {}
        self.drag_starts = {}

        self.global_theme = None
        self.borderless_child_theme = None
        self.active_widget = -1
        self.focussed_widget = -1
        self.hovered_item = None
        self.return_pressed = False
        self.frame_time_variable = self.add_variable(variable_name='frame_time')
        self.frame_number = 0
        self.frame_variable = self.add_variable(variable_name='frame')
        self.font_scale_variable = self.add_variable(variable_name='font_scale', setter=self.update_font_scale, default_value=0.5)
        self.gui_scale_variable = self.add_variable(variable_name='gui_scale', setter=self.update_gui_scale, default_value=1.0)

        self.dragging_created_nodes = False
        self.dragging_ref = [0, 0]
        self.clipboard = None
        self.saving_to_lib = False

        self.register_patchers()
        self.handler = dpg.item_handler_registry(tag="widget handler")
        with self.handler:
            dpg.add_item_active_handler(callback=widget_active)
            dpg.add_item_activated_handler(callback=widget_activated)
            dpg.add_item_deactivated_handler(callback=widget_deactive)
            dpg.add_item_deactivated_after_edit_handler(callback=widget_deactive_after_edit)
            dpg.add_item_edited_handler(callback=widget_edited)
            dpg.add_item_focus_handler(callback=widget_focus)
            dpg.add_item_clicked_handler(callback=widget_clicked)
            dpg.add_item_hover_handler(callback=widget_hovered)
        self.action = self.add_action('do_it', self.reset_frame_count)

    def reset_frame_count(self):
        self.frame_number = 0

    def setup_dpg(self):
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)
        if 'macOS' in platform_.platform():
            with dpg.font_registry():
                if os.path.exists('Inconsolata-g.otf'):
                    default_font = dpg.add_font("Inconsolata-g.otf", 24)
                    dpg.bind_font(default_font)
                    dpg.set_global_font_scale(0.5)
        self.viewport = dpg.create_viewport()
        # print(self.viewport)
        dpg.setup_dearpygui()

    def register_patcher(self, name):
        self.patchers.append(name)

    def register_patchers(self):
        if os.path.exists('dpg_system/patcher_library'):
            for entry in os.scandir('dpg_system/patcher_library'):
                if entry.is_file():
                    if entry.name[-5:] == '.json':
                        self.register_patcher(entry.name[:-5])

    def position_viewport(self, x, y):
        dpg.configure_viewport(self.viewport, x_pos=x, y_pos=y)

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.window_padding[0], self.window_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, self.frame_padding[0], self.frame_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, self.cell_padding[0], self.cell_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, self.item_spacing[0], self.item_spacing[1], category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.borderless_child_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Border, [0, 0, 0, 0])

    def find_loaded_parent(self, loaded_parent_node_uuid):
        for loaded_patcher_node in self.loaded_patcher_nodes:
            if loaded_patcher_node.loaded_uuid == loaded_parent_node_uuid:
                parent = loaded_patcher_node
                return parent
        return None

    def find_orphaned_subpatch(self, editor_name, loaded_uuid):
        for editor in self.node_editors:
            if editor.patch_name == editor_name and editor.loaded_uuid == loaded_uuid:
                if editor.parent_patcher is None:
                    return editor
                else:
                    print('sub-patcher already claimed')

        return None

    def find_editor(self, editor_name):
        for editor in self.node_editors:
            if editor.patch_name == editor_name:
                return editor
        return None

    def get_current_editor(self):
        return self.node_editors[self.current_node_editor]

    def set_verbose(self):
        if self.verbose_menu_item != -1:
            self.verbose = dpg.get_value(self.verbose_menu_item)

    def show_minimap(self):
        if self.minimap_menu_item != -1:
            show = dpg.get_value(self.minimap_menu_item)
            self.get_current_editor().show_minimap(show)

    def containerize_patch(self, editor, container=None):
        if container is None:
            container = {}

        if len(editor.subpatches) > 0:
            for index, node_editor in enumerate(editor.subpatches):
                self.containerize_patch(node_editor, container)

        container[len(container)] = editor.save_into()
        return container

    def save_patch(self, save_path):
        current_editor = self.get_current_editor()
        if len(current_editor.subpatches) == 0:
            current_editor.save(save_path)
            self.patches_name = self.get_current_editor().patch_name
            print(self.patches_name)
        else:
            with open(save_path, 'w') as f:
                self.patches_path = save_path

                patch_name = save_path.split('/')[-1]
                if '.' in patch_name:
                    parts = patch_name.split('.')
                    if len(parts) == 2:
                        if parts[1] == 'json':
                            patch_name = parts[0]

                self.patches_name = patch_name
                file_container = {}
                file_container['name'] = self.patches_name
                file_container['path'] = self.patches_path
                file_container['patches'] = self.containerize_patch(current_editor)

                json.dump(file_container, f, indent=4)

    def save_setup(self, save_path):
        with open(save_path, 'w') as f:
            self.patches_path = save_path

            patch_name = save_path.split('/')[-1]
            if '.' in patch_name:
                parts = patch_name.split('.')
                if len(parts) == 2:
                    if parts[1] == 'json':
                        patch_name = parts[0]

            self.patches_name = patch_name

            file_container = {}
            file_container['name'] = self.patches_name
            file_container['path'] = self.patches_path

            patches_container = {}
            for index, node_editor in enumerate(self.node_editors):
                patches_container = self.containerize_patch(node_editor, patches_container)

            file_container['patches'] = patches_container
            json.dump(file_container, f, indent=4)

    def setup_menus(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label='New Patch (N)', callback=self.add_node_editor)
                dpg.add_menu_item(label="Open (O)", callback=self.load_nodes)
                dpg.add_menu_item(label='Close Current Patch (W)', callback=self.close_current_node_editor)
                dpg.add_separator()
                dpg.add_menu_item(label="Save Patch (S)", callback=self.save_nodes)
                dpg.add_menu_item(label="Save Patch As", callback=self.save_as_nodes)
                dpg.add_separator()
                dpg.add_menu_item(label="Save to Library", callback=self.save_to_library)
                dpg.add_separator()
                dpg.add_menu_item(label="Save Setup", callback=self.save_patches)
                dpg.add_menu_item(label="Save Setup As", callback=self.save_patches)
            with dpg.menu(label='Edit'):
                dpg.add_menu_item(label="Cut (X)", callback=self.cut_selected)
                dpg.add_menu_item(label="Copy (C)", callback=self.copy_selected)
                dpg.add_menu_item(label="Paste (V)", callback=self.paste_selected)
                dpg.add_menu_item(label="Duplicate (D)", callback=self.duplicate_handler)
                dpg.add_separator()
                dpg.add_menu_item(label="Connect Selected", callback=self.connect_selected)
                dpg.add_menu_item(label="Align Selected", callback=self.align_selected)
                dpg.add_menu_item(label="Align and Distribute Selected", callback=self.align_distribute_selected)
                dpg.add_menu_item(label="Space Out Selected", callback=self.space_out_selected)
                dpg.add_menu_item(label="Tighten Selected", callback=self.tighten_selected)

            with dpg.menu(label='Options'):
                dpg.add_menu_item(label="Show Style Editor", callback=self.show_style)
                dpg.add_menu_item(label="Show Demo", callback=self.show_demo)
                dpg.add_separator()
                self.verbose_menu_item = dpg.add_menu_item(label="verbose logging", check=True, callback=self.set_verbose)
                dpg.add_menu_item(label='osc status', callback=self.print_osc_state)
                self.minimap_menu_item = dpg.add_menu_item(label='minimap', callback=self.show_minimap, check=True)

    def print_osc_state(self):
        if self.osc_manager:
            self.osc_manager.print_state()

    def register_nodes(self):
        if 'register_basic_nodes' in globals():
            register_basic_nodes()

        if 'register_osc_nodes' in globals():
            register_osc_nodes()

        if 'register_gl_nodes' in globals():
            register_gl_nodes()

        if 'register_interface_nodes' in globals():
            register_interface_nodes()

        if 'register_math_nodes' in globals():
            register_math_nodes()

        if 'register_matrix_nodes' in globals():
            register_matrix_nodes()

        if 'register_mocap_nodes' in globals():
            register_mocap_nodes()

        if 'register_quaternion_nodes' in globals():
            register_quaternion_nodes()

        if 'register_signal_nodes' in globals():
            register_signal_nodes()

        if 'register_spacy_nodes' in globals():
            register_spacy_nodes()

        if 'register_clip_nodes' in globals():
            register_clip_nodes()

        if 'register_numpy_nodes' in globals():
            register_numpy_nodes()

        if 'register_torch_nodes' in globals():
            register_torch_nodes()

        if 'register_base_nodes' in globals():
            register_base_nodes()

    def get_variable_list(self):
        v_list = list(self.variables.keys())
        return v_list

    def get_conduit_list(self):
        c_list = list(self.conduit.keys())
        return c_list

    def update_font_scale(self, value):
        dpg.set_global_font_scale(value)

    def update_gui_scale(self, value):
        for editor in self.node_editors:
            editor.scale_nodes(value)
        pass

    def add_conduit(self, conduit_name='untitled'):
        c = Conduit(label=conduit_name)
        self.conduits[conduit_name] = c
        return c

    def find_conduit(self, conduit_name):
        if conduit_name in self.conduits:
            return self.conduits[conduit_name]
        return None

    def add_variable(self, variable_name='untitled', default_value=None, getter=None, setter=None):
        v = Variable(label=variable_name, default_value=default_value, getter=getter, setter=setter)
        self.variables[variable_name] = v
        return v

    def add_action(self, action_name, action_function):
        a = Action(label=action_name, action_function=action_function)
        self.actions[action_name] = action_function

    def find_variable(self, variable_name):
        if variable_name in self.variables:
            return self.variables[variable_name]
        return None

    def find_action(self, action_name):
        if action_name in self.actions:
            return self.actions[action_name]
        return None

    def create_node_by_name_from_file(self, node_name, pos, args=[]):
        node_model = self.node_factory_container.locate_node_by_name(node_name)
        if node_model is not None:
            new_node = self.create_node_from_model(node_model, pos, args=args, from_file=True)
            return new_node
        else:
            new_node = self.create_var_node_for_variable(node_name, pos)
            return new_node

    def create_var_node_for_variable(self, node_name, pos):
        v = self.find_variable(node_name)
        if v:
            args = [node_name]
            node_name = 'var'
            new_node = self.create_node_by_name(node_name, None, args, pos)
            return new_node
        return None

    def create_node_by_name(self, node_name, placeholder, args=[], pos=None):
        node_model = self.node_factory_container.locate_node_by_name(node_name)
        if placeholder:
            pos = dpg.get_item_pos(placeholder.uuid)
        if not pos:
            pos = dpg.get_mouse_pos()
        if node_model:
            new_node = self.create_node_from_model(node_model, pos, args=args)
        else:
            new_node = self.create_var_node_for_variable(node_name, pos)
        if placeholder:
            self.get_current_editor().remove_node(placeholder)
        return new_node

    def create_node_from_model(self, model, pos, name=None, args=[], from_file=False):
        node = model.create(name, args)
        node.submit(self.get_current_editor().uuid, pos=pos, from_file=from_file)
        self.get_current_editor().add_node(node)
        return node

    def remove_frame_task(self, remove_node):
        for index, node in enumerate(self.frame_tasks):
            if node == remove_node:
                self.frame_tasks[index:] = self.frame_tasks[index + 1:]

    def del_handler(self):
        if self.active_widget == -1:
            node_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for node_uuid in node_uuids:
                # somehow we have to connect to the actual Node object
                self.get_current_editor().node_cleanup(node_uuid)
            link_uuids = dpg.get_selected_links(self.get_current_editor().uuid)
            for link_uuid in link_uuids:
                dat = dpg.get_item_user_data(link_uuid)
                out = dat[0]
                child = dat[1]
                out.remove_link(link_uuid, child)

    def return_handler(self):
        # print('return')
        self.return_pressed = True
        pass

    def place_node(self, node):
        mouse_pos = dpg.get_mouse_pos(local=False)
        panel_pos = dpg.get_item_pos(self.center_panel)
        mouse_pos[0] -= (panel_pos[0] + 8)
        mouse_pos[1] -= (panel_pos[1] + 8)
        node.submit(self.get_current_editor().uuid, pos=mouse_pos)
        self.get_current_editor().add_node(node)

    def int_handler(self):
        if self.active_widget == -1:
            node = ValueNode.factory("int", None)
            self.place_node(node)
            node.name_id = dpg.generate_uuid()
            self.set_widget_focus(node.uuid)

    def float_handler(self):
        if self.active_widget == -1:
            node = ValueNode.factory("float", None)
            self.place_node(node)
            self.set_widget_focus(node.uuid)

    def vector_handler(self):
        if self.active_widget == -1:
            node = VectorNode.factory("vector", None, args=['4'])
            self.place_node(node)
            self.set_widget_focus(node.uuid)

    def toggle_handler(self):
        if self.active_widget == -1:
            node = ToggleNode.factory("toggle", None)
            self.place_node(node)

    def button_handler(self):
        if self.active_widget == -1:
            node = ButtonNode.factory("b", None)
            self.place_node(node)

    def message_handler(self):
        if self.active_widget == -1:
            node = ValueNode.factory("message", None)
            self.place_node(node)
            self.set_widget_focus(node.input.widget.uuid)

    def options_handler(self):
        if self.active_widget == -1:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node_object = dpg.get_item_user_data(selected_nodes_uuid)
                node_object.toggle_show_hide_options()

    def C_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.clipboard = self.get_current_editor().copy_selection()

    def X_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.clipboard = self.get_current_editor().cut_selection()

    def S_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.save_nodes()

    def O_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.load_nodes()
        else:
            self.options_handler()

    def N_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.add_node_editor()
        else:
            self.new_handler()

    def W_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.close_current_node_editor()

    def V_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            self.paste_selected()
        else:
            self.vector_handler()

    def D_handler(self):
        if dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LWin):
            if self.active_widget == -1:
                self.get_current_editor().duplicate_selection()

    def duplicate_handler(self):
        if self.active_widget == -1:
            self.get_current_editor().duplicate_selection()
    def cut_selected(self):
        if self.active_widget == -1:
            self.get_current_editor().cut_selection()

    def copy_selected(self):
        if self.active_widget == -1:
            self.get_current_editor().copy_selection()

    def mouse_down_handler(self):
        self.dragging_created_nodes = False

    def drag_create_nodes(self):
        if self.dragging_created_nodes:
            if dpg.is_mouse_button_down(0):
                self.dragging_created_nodes = False
            else:
                mouse_pos = dpg.get_mouse_pos()
                delta = [0, 0]
                delta[0] = mouse_pos[0] - self.dragging_ref[0]
                delta[1] = mouse_pos[1] - self.dragging_ref[1]
                for node_uuid in self.created_nodes:
                    node = self.created_nodes[node_uuid]
                    start = self.drag_starts[node_uuid]
                    dest = [0, 0]
                    dest[0] = start[0] + delta[0]
                    dest[1] = start[1] + delta[1]
                    dpg.set_item_pos(node.uuid, dest)

    def set_widget_focus(self, widget_uuid):
        dpg.focus_item(widget_uuid)
        self.focussed_widget = widget_uuid

    # def mouse_drag_handler(self):
    #     print('dragged', self.active_widget)

    def up_handler(self):
        handled = False
        if self.hovered_item is not None:
            uuid = self.hovered_item.uuid
            if dpg.does_item_exist(uuid):
                if dpg.is_item_hovered(uuid):
                    handled = True
                    self.hovered_item.node.increment_widget(self.hovered_item)
                    if self.hovered_item.callback is not None:
                        self.hovered_item.callback()
        if not handled and self.active_widget != -1:
            if dpg.does_item_exist(self.active_widget):
                widget = dpg.get_item_user_data(self.active_widget)
                if widget is not None:
                    if widget.node is not None:
                        widget.node.increment_widget(widget)
                        if widget.callback is not None:
                            widget.callback()

    def create_gui(self):
        pass

    def down_handler(self):
        handled = False
        if self.hovered_item is not None:
            uuid = self.hovered_item.uuid
            if dpg.does_item_exist(uuid):
                if dpg.is_item_hovered(uuid):
                    handled = True
                    self.hovered_item.node.decrement_widget(self.hovered_item)
                    if self.hovered_item.callback is not None:
                        self.hovered_item.callback()
        if not handled and self.active_widget != -1:
            if dpg.does_item_exist(self.active_widget):
                widget = dpg.get_item_user_data(self.active_widget)
                if widget is not None:
                    if widget.node is not None:
                        widget.node.decrement_widget(widget)
                        if widget.callback is not None:
                            widget.callback()

    def new_handler(self):
        origin = self.get_current_editor().origin

        if self.active_widget == -1:
            node = PlaceholderNode.factory("New Node", None)
            mouse_pos = dpg.get_mouse_pos(local=False)
            panel_pos = dpg.get_item_pos(self.center_panel)
            origin_pos = dpg.get_item_pos(origin.ref_property.widget.uuid)
            origin_node_pos = dpg.get_item_pos(origin.uuid)

            mouse_pos[0] -= (panel_pos[0] + 8 + (origin_pos[0] - origin_node_pos[0]) - 4)
            mouse_pos[1] -= (panel_pos[1] + 8 + (origin_pos[1] - origin_node_pos[1]) - 15)
            node.submit(self.get_current_editor().uuid, pos=mouse_pos)
            self.get_current_editor().add_node(node)
            self.set_widget_focus(node.name_property.widget.uuid)

    def update(self):
        pass

    def load_from_file(self, path):
        self.loading = True
        self.loaded_patcher_nodes = []
        self.links_containers = {}
        self.created_nodes = {}
        hold_current_editor = self.current_node_editor
        try:
            with open(path, 'r') as f:
                patch_name = path.split('/')[-1]
                if '.' in patch_name:
                    parts = patch_name.split('.')
                    if len(parts) == 2:
                        if parts[1] == 'json':
                            patch_name = parts[0]
                file_container = json.load(f)

                patch_count = 0
                if 'patches' in file_container:
                    patches_container = file_container['patches']
                    patch_count = len(patches_container)

                patch_assign = {}
                available_editors = {}

                for editor_index, editor in enumerate(self.node_editors):
                    if editor is not None:
                        if editor.num_nodes <= 1:
                            available_editors[len(list(available_editors.keys()))] = (editor_index, editor)
                for i in range(patch_count):
                    if len(list(available_editors.keys())) > 0:
                        patch_assign[i] = available_editors[0]
                        del available_editors[0]
                    else:
                        patch_assign[i] = (len(self.node_editors), self.add_node_editor())

                if 'patches' in file_container:
                    self.patches_path = path
                    self.patches_name = patch_name
                    patches_container = file_container['patches']

                    patch_count = len(patch_assign)
                    for index, patch_index in enumerate(patches_container):
                        nodes_container = patches_container[patch_index]
                        editor_index, editor = patch_assign[patch_count - index - 1]

                        if editor is not None:
                            self.current_node_editor = editor_index
                            editor.load_(nodes_container)
                else:  # single patch
                    self.get_current_editor().load_(file_container, path, patch_name)

                for node_editor_uuid in self.links_containers:
                    links_container = self.links_containers[node_editor_uuid]
                    for index, link_index in enumerate(links_container):
                        source_node = None
                        dest_node = None
                        link_container = links_container[link_index]
                        source_node_loaded_uuid = link_container['source_node']
                        if source_node_loaded_uuid in self.created_nodes:
                            source_node = self.created_nodes[source_node_loaded_uuid]
                        dest_node_loaded_uuid = link_container['dest_node']
                        if dest_node_loaded_uuid in self.created_nodes:
                            dest_node = self.created_nodes[dest_node_loaded_uuid]
                        if source_node is not None and dest_node is not None:
                            source_output_index = link_container['source_output_index']
                            dest_input_index = link_container['dest_input_index']
                            if source_output_index < len(source_node.outputs):
                                source_output = source_node.outputs[source_output_index]
                                if dest_input_index < len(dest_node.inputs):
                                    dest_input = dest_node.inputs[dest_input_index]
                                    source_output.add_child(dest_input, node_editor_uuid)

                for uuid in self.created_nodes:
                    node = self.created_nodes[uuid]
                    if node is not None:
                        node.post_load_callback()
                for uuid in self.created_nodes:
                    node = self.created_nodes[uuid]
                    node.loaded_uuid = -1
                for editor in self.node_editors:
                    editor.loaded_uuid = -1
                    editor.loaded_parent_node_uuid = -1
                # now reorder deeper subpatchers?

        except Exception as exc_:
            print(exc_)
            print('load failed')
        self.current_node_editor = hold_current_editor
        self.loading = False

    def load_nodes(self):
        self.load('')

    def load(self, path=''):
        if path != '':
            self.get_current_editor().load(path)
            return
        self.active_widget = 1
        # print('before file_dialog')
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, callback=load_patches_callback, cancel_callback=cancel_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_as_nodes(self):
        self.save('')

    def save_to_library(self):
        if not os.path.exists('dpg_system/patcher_library'):
            os.makedirs('dpg_system/patcher_library')
        if os.path.exists('dpg_system/patcher_library'):
            self.saving_to_lib = True
            self.save('', default_directory='dpg_system/patcher_library')
            self.patchers.append(self.patches_name)


    def save_nodes(self):
        # needs to save sub-patches
        if exists(self.get_current_editor().file_path):
            self.get_current_editor().save(self.get_current_editor().file_path)
        else:
            self.save_as_nodes()

    def save_as_patches(self):
        self.save_patches('')

    def connect_selected(self):
        self.get_current_editor().connect_selected()


    def paste_selected(self):
        if self.clipboard is not None:
            self.get_current_editor().paste(self.clipboard)
        else:
            print('clipboard is empty')

    def align_selected(self):
        self.get_current_editor().align_selected()

    def space_out_selected(self):
        self.get_current_editor().space_out_selected(1.1)

    def tighten_selected(self):
        self.get_current_editor().space_out_selected(0.9)

    def align_distribute_selected(self):
        self.get_current_editor().align_and_distribute_selected()

    def show_style(self):
        dpg.show_style_editor()

    def show_demo(self):
        dpg.show_imgui_demo()

    def save_with_path(self, sender, data):
        filename = os.sep.join(data)
        for i in open(filename, "rt"):
            self.get_current_editor().save(filename)

    def save(self, path='', default_directory=''):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, callback=save_file_callback, cancel_callback=cancel_callback, default_path=default_directory, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_patches(self, path=''):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, callback=save_patches_callback, cancel_callback=cancel_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def add_frame_task(self, dest):
        self.frame_tasks.append(dest)

    def register_node(self, label, factory, data=None):
        self.node_factory_container.add_node_factory(NodeFactory(label, factory, data))
        self.node_list.append(label)

    def set_current_tab_title(self, title):
        if len(self.tabs) > self.current_node_editor:
            dpg.configure_item(self.tabs[self.current_node_editor], label=title)

    def set_tab_title(self, tab_index, title):
        if len(self.tabs) > tab_index >= 0:
            dpg.configure_item(self.tabs[tab_index], label=title)

    def selected_tab(self):
        chosen_tab_uuid = dpg.get_value(self.tab_bar)
        chosen_tab_index = dpg.get_item_user_data(chosen_tab_uuid)
        self.current_node_editor = chosen_tab_index
        dpg.set_value(self.minimap_menu_item, self.get_current_editor().mini_map)

    def remove_node_editor(self, stale_editor):
        if stale_editor is None:
            return
        for i, editor in enumerate(self.node_editors):
            if editor == stale_editor:
                if editor.modified:
                    self.save_as_nodes()
                if stale_editor.parent_patcher is not None:
                    stale_editor.parent_patcher.subpatches.remove(stale_editor)
                editor.remove_all_nodes()
                stale_tab = self.tabs[i]
                self.tabs.remove(stale_tab)
                dpg.delete_item(stale_tab)
                del editor
                self.node_editors.remove(stale_editor)
                if self.current_node_editor >= len(self.node_editors):
                    self.current_node_editor = len(self.node_editors) - 1
                break

    def close_current_node_editor(self):
        self.remove_node_editor(self.get_current_editor())

    def add_node_editor(self, editor_name=None):
        if editor_name is None:
            editor_number = len(self.node_editors)
            editor_name = 'editor ' + str(editor_number)
        with dpg.tab(label=editor_name, parent=self.tab_bar, user_data=len(self.tabs)) as tab:
            self.tabs.append(tab)
            panel_uuid = dpg.generate_uuid()
            with dpg.group(id=panel_uuid):
                new_editor = NodeEditor()
                if editor_name is not None:
                    new_editor.patch_name = editor_name
                self.node_editors.append(new_editor)
                self.node_editors[len(self.node_editors) - 1].submit(panel_uuid)
        return new_editor

    def start(self):
        dpg.set_viewport_title("Untitled")
        self.node_editors = [NodeEditor()]

        with dpg.window() as main_window:
            global dpg_app
            self.main_window_id = main_window
            dpg.bind_item_theme(main_window, self.global_theme)
            dpg.add_spacer(height=14)
            with dpg.tab_bar(callback=self.selected_tab) as self.tab_bar:
                with dpg.tab(label='node editor', user_data=len(self.tabs)) as tab:
                    self.tabs.append(tab)
                    with dpg.group(id=self.center_panel):
                        self.node_editors[0].submit(self.center_panel)
                        with dpg.handler_registry():
                            # dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.up_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.down_handler)
                            # dpg.add_key_press_handler(dpg.mvKey_N, callback=self.new_handler)
                            dpg.add_key_press_handler(dpg.mvKey_I, callback=self.int_handler)
                            dpg.add_key_press_handler(dpg.mvKey_F, callback=self.float_handler)
                            dpg.add_key_press_handler(dpg.mvKey_T, callback=self.toggle_handler)
                            dpg.add_key_press_handler(dpg.mvKey_B, callback=self.button_handler)
                            dpg.add_key_press_handler(dpg.mvKey_M, callback=self.message_handler)
                            # dpg.add_key_press_handler(dpg.mvKey_V, callback=self.vector_handler)
                            # dpg.add_key_press_handler(dpg.mvKey_O, callback=self.options_handler)

                            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.D_handler)
                            dpg.add_key_press_handler(dpg.mvKey_C, callback=self.C_handler)
                            dpg.add_key_press_handler(dpg.mvKey_V, callback=self.V_handler)
                            dpg.add_key_press_handler(dpg.mvKey_X, callback=self.X_handler)
                            dpg.add_key_press_handler(dpg.mvKey_W, callback=self.W_handler)
                            dpg.add_key_press_handler(dpg.mvKey_O, callback=self.O_handler)
                            dpg.add_key_press_handler(dpg.mvKey_S, callback=self.S_handler)
                            dpg.add_key_press_handler(dpg.mvKey_N, callback=self.N_handler)

                            dpg.add_key_press_handler(dpg.mvKey_Back, callback=self.del_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Return, callback=self.return_handler)
                            dpg.add_mouse_move_handler(callback=self.drag_create_nodes)
                            dpg.add_mouse_click_handler(callback=self.mouse_down_handler)

        dpg.set_primary_window(main_window, True)
        dpg.show_viewport()

    def run_loop(self):
        elapsed = 0
        while dpg.is_dearpygui_running():
            now = time.time()
            for node_editor in self.node_editors:
                node_editor.reset_pins()
            jobs = dpg.get_callback_queue()  # retrieves and clears queue
            try:
                for task in self.frame_tasks:
                    if task.created:
                        task.frame_task()
                dpg.run_callbacks(jobs)
            except Exception as exc_:
                print(exc_)
            self.frame_number += 1
            self.frame_variable.set(self.frame_number)
            self.frame_time_variable.set(elapsed)
            dpg.render_dearpygui_frame()
            then = time.time()
            elapsed = then - now

            if 'GLContextNode' in globals():
                GLContextNode.maintenance_loop()

