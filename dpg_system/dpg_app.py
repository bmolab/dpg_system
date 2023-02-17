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


def widget_active(source, data, user_data):
    pass


def widget_hovered(source, data, user_data):
    if dpg.does_item_exist(data):
        Node.app.hovered_item = dpg.get_item_user_data(data)


def widget_activated(source, data, user_data):
    item_type = dpg.get_item_type(data)
    # print(item_type, type(item_type))
    if item_type not in ['mvAppItemType::mvCombo', 'mvAppItemType::mvButton']:
        Node.app.active_widget = data
    else:
        Node.app.active_widget = -1
    # print("activated", Node.app.active_widget)


def widget_deactive(source, data, user_data):
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


def load_patches_callback(sender, app_data):
    global load_path
    if 'file_path_name' in app_data:
        load_path = app_data['file_path_name']
        if load_path != '':
            Node.app.load_from_file(load_path)
    else:
        print('no file chosen')
    dpg.delete_item(sender)
    Node.app.active_widget = -1


def save_file_callback(sender, app_data):
    global save_path
    if 'file_path_name' in app_data:
        save_path = app_data['file_path_name']
        if save_path != '':
            Node.app.node_editors[Node.app.current_node_editor].save(save_path)
    else:
        print('no file chosen')
    dpg.delete_item(sender)
    Node.app.active_widget = -1


def save_patches_callback(sender, app_data):
    global save_path
    if 'file_path_name' in app_data:
        save_path = app_data['file_path_name']
        if save_path != '':
            with open(save_path, 'w') as f:
                file_container = {}
                patches_container = {}
                Node.app.patches_path = save_path

                patch_name = save_path.split('/')[-1]
                if '.' in patch_name:
                    parts = patch_name.split('.')
                    if len(parts) == 2:
                        if parts[1] == 'json':
                            patch_name = parts[0]

                Node.app.patches_name = patch_name
                for index, node_editor in enumerate(Node.app.node_editors):
                    patch_container = {}
                    node_editor.save_into(patch_container)
                    patches_container[index] = patch_container
                file_container['name'] = Node.app.patches_name
                file_container['path'] = Node.app.patches_path
                file_container['patches'] = patches_container
                json.dump(file_container, f, indent=4)
    else:
        print('no file chosen')
    dpg.delete_item(sender)
    Node.app.active_widget = -1


class App:
    def __init__(self):
        self.viewport = None
        self.main_window_id = -1
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

        self.node_editors = []
        self.current_node_editor = 0
        self.frame_tasks = []
        self.variables = {}

        self.osc_manager = OSCManager(label='osc_manager', data=0, args=None)

        self.setup_menus()
        self.patches_path = ''
        self.patches_name = ''

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

    def set_verbose(self):
        if self.verbose_menu_item != -1:
            self.verbose = dpg.get_value(self.verbose_menu_item)

    def show_minimap(self):
        if self.minimap_menu_item != -1:
            show = dpg.get_value(self.minimap_menu_item)
            self.node_editors[self.current_node_editor].show_minimap(show)

    def save_setup(self):
        pass

    def save_setup_as(self):
        pass

    def setup_menus(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Save Setup", callback=self.save_patches)
                dpg.add_menu_item(label="Save Setup As", callback=self.save_patches)
                dpg.add_menu_item(label="Save Nodes", callback=self.save_nodes)
                dpg.add_menu_item(label="Save Nodes As", callback=self.save_as_nodes)
                dpg.add_menu_item(label="Load", callback=self.load_nodes)
                dpg.add_menu_item(label='New Node Editor', callback=self.add_node_editor)
                dpg.add_menu_item(label="Show Style Editor", callback=self.show_style)
                dpg.add_menu_item(label="Show Demo", callback=self.show_demo)
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

    def get_variable_list(self):
        v_list = list(self.variables.keys())
        return v_list

    def update_font_scale(self, value):
        dpg.set_global_font_scale(value)

    def update_gui_scale(self, value):
        for editor in self.node_editors:
            editor.scale_nodes(value)
        pass

    def add_variable(self, variable_name='untitled', default_value=None, getter=None, setter=None):
        v = Variable(label=variable_name, default_value=default_value, getter=getter, setter=setter)
        self.variables[variable_name] = v
        return v

    def find_variable(self, variable_name):
        if variable_name in self.variables:
            return self.variables[variable_name]
        return None

    def create_node_by_name_from_file(self, node_name, pos, args=[]):
        node_model = self.node_factory_container.locate_node_by_name(node_name)
        if node_model is not None:
            new_node = self.create_node_from_model(node_model, pos, args=args)
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
            self.node_editors[self.current_node_editor].remove_node(placeholder)
        return new_node

    def create_node_from_model(self, model, pos, name=None, args=[]):
        node = model.create(name, args)
        node.submit(self.node_editors[self.current_node_editor].uuid, pos=pos)
        self.node_editors[self.current_node_editor].add_node(node)
        return node

    def remove_frame_task(self, remove_node):
        for index, node in enumerate(self.frame_tasks):
            if node == remove_node:
                self.frame_tasks[index:] = self.frame_tasks[index + 1:]

    def del_handler(self):
        if self.active_widget == -1:
            node_uuids = dpg.get_selected_nodes(self.node_editors[self.current_node_editor].uuid)
            for node_uuid in node_uuids:
                # somehow we have to connect to the actual Node object
                self.node_editors[self.current_node_editor].node_cleanup(node_uuid)
            link_uuids = dpg.get_selected_links(self.node_editors[self.current_node_editor].uuid)
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
        node.submit(self.node_editors[self.current_node_editor].uuid, pos=mouse_pos)
        self.node_editors[self.current_node_editor].add_node(node)

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
            selected_nodes_uuids = dpg.get_selected_nodes(self.node_editors[self.current_node_editor].uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node_object = dpg.get_item_user_data(selected_nodes_uuid)
                node_object.toggle_show_hide_options()

    def duplicate_handler(self):
        if self.active_widget == -1:
            self.node_editors[self.current_node_editor].duplicate_selection()

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
        if self.active_widget == -1:
            node = PlaceholderNode.factory("New Node", None)
            mouse_pos = dpg.get_mouse_pos(local=False)
            panel_size = dpg.get_item_rect_size(self.center_panel)
            panel_pos = dpg.get_item_pos(self.center_panel)
            mouse_pos[0] -= (panel_pos[0] + 8)
            mouse_pos[1] -= (panel_pos[1] + 8)
            node.submit(self.node_editors[self.current_node_editor].uuid, pos=mouse_pos)
            self.node_editors[self.current_node_editor].add_node(node)
            self.set_widget_focus(node.name_property.widget.uuid)

    def update(self):
        pass

    def load_from_file(self, path):
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
                if 'patches' in file_container:
                    self.patches_path = path
                    self.patches_name = patch_name
                    patches_container = file_container['patches']
                    for index, patch_index in enumerate(patches_container):
                        nodes_container = patches_container[patch_index]
                        loaded = False
                        for editor_index, editor in enumerate(self.node_editors):
                            if editor is not None:
                                if editor.num_nodes == 0:
                                    self.current_node_editor = editor_index
                                    editor.load_(nodes_container)
                                    loaded = True
                                    break
                        if not loaded:
                            editor_index = len(self.node_editors)
                            self.add_node_editor()
                            self.current_node_editor = editor_index
                            self.node_editors[editor_index].load_(nodes_container)
                else:  # single patch
                    self.node_editors[self.current_node_editor].load_(file_container, path, patch_name)

        except Exception as exc_:
            print(exc_)
            print('load failed')
        self.current_node_editor = hold_current_editor

    def load_nodes(self):
        self.load('')

    def load(self, path=''):
        if path != '':
            self.node_editors[self.current_node_editor].load(path)
            return
        self.active_widget = 1
        # print('before file_dialog')
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, callback=load_patches_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_as_nodes(self):
        self.save('')

    def save_nodes(self):
        if exists(self.node_editors[self.current_node_editor].file_path):
            self.node_editors[self.current_node_editor].save(self.node_editors[self.current_node_editor].file_path)
        else:
            self.save_as_nodes()

    def save_as_patches(self):
        self.save_patches('')

    def show_style(self):
        dpg.show_style_editor()

    def show_demo(self):
        dpg.show_imgui_demo()

    def save_with_path(self, sender, data):
        filename = os.sep.join(data)
        for i in open(filename, "rt"):
            self.node_editors[self.current_node_editor].save(filename)

    def save(self, path=''):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, callback=save_file_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_patches(self, path=''):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, callback=save_patches_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def add_frame_task(self, dest):
        self.frame_tasks.append(dest)

    def register_node(self, label, factory, data=None):
        self.node_factory_container.add_node_factory(NodeFactory(label, factory, data))
        self.node_list.append(label)

    def set_current_tab_title(self, title):
        if len(self.tabs) > self.current_node_editor:
            dpg.configure_item(self.tabs[self.current_node_editor], label=title)

    def selected_tab(self):
        chosen_tab_uuid = dpg.get_value(self.tab_bar)
        chosen_tab_index = dpg.get_item_user_data(chosen_tab_uuid)
        self.current_node_editor = chosen_tab_index
        dpg.set_value(self.minimap_menu_item, self.node_editors[self.current_node_editor].mini_map)

    def add_node_editor(self):
        # conf = dpg.get_item_configuration(self.tab_bar)
        # print(conf)
        editor_number = len(self.node_editors)
        with dpg.tab(label='editor ' + str(editor_number), parent=self.tab_bar, user_data=len(self.tabs)) as tab:
            self.tabs.append(tab)
            panel_uuid = dpg.generate_uuid()
            with dpg.group(id=panel_uuid):
                new_editor = NodeEditor()
                self.node_editors.append(new_editor)
                self.node_editors[len(self.node_editors) - 1].submit(panel_uuid)

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
                            dpg.add_key_press_handler(dpg.mvKey_N, callback=self.new_handler)
                            dpg.add_key_press_handler(dpg.mvKey_I, callback=self.int_handler)
                            dpg.add_key_press_handler(dpg.mvKey_F, callback=self.float_handler)
                            dpg.add_key_press_handler(dpg.mvKey_T, callback=self.toggle_handler)
                            dpg.add_key_press_handler(dpg.mvKey_B, callback=self.button_handler)
                            dpg.add_key_press_handler(dpg.mvKey_M, callback=self.message_handler)
                            dpg.add_key_press_handler(dpg.mvKey_V, callback=self.vector_handler)
                            dpg.add_key_press_handler(dpg.mvKey_O, callback=self.options_handler)
                            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.duplicate_handler)

                            dpg.add_key_press_handler(dpg.mvKey_Back, callback=self.del_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Return, callback=self.return_handler)

                with dpg.tab(label='second editor', user_data=len(self.tabs)) as tab:
                    self.tabs.append(tab)
                    with dpg.group(id=self.side_panel):
                        self.create_gui()
                        new_editor = NodeEditor()
                        self.node_editors.append(new_editor)
                        self.node_editors[1].submit(self.side_panel)
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

