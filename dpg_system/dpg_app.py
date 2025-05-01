# import dearpygui.dearpygui as dpg
import time
import numpy as np
from importlib import import_module
import dpg_system.dpg_app
from dpg_system.node import Node, NodeInput, NodeOutput, Variable, PlaceholderNode
from dpg_system.node_editor import *
from dpg_system.node import *
from os.path import exists
import json
import os
import platform as platform_
import traceback
import threading
import _thread

import dpg_system.basic_nodes as basic_nodes
import dpg_system.math_nodes as math_nodes
import dpg_system.signal_nodes as signal_nodes
import dpg_system.interface_nodes as interface_nodes
import dpg_system.quaternion_nodes as quaternion_nodes
import dpg_system.numpy_nodes as numpy_nodes
import dpg_system.matrix_nodes as matrix_nodes
import dpg_system.osc_nodes as osc_nodes


to_import = [
    'basic_nodes',
    'math_nodes',
    'signal_nodes',
    'interface_nodes',
    'plot_nodes',
    'quaternion_nodes',
    'numpy_nodes',
    'matrix_nodes',
    'osc_nodes'
]

optional_import = [
    'gl_nodes',
    'opencv_nodes',
    'elevenlabs_nodes',
    'movesense_nodes',
    'motion_cap_nodes',
    'spacy_nodes',
    'clip_nodes',
    'torch_nodes',
    'pybullet_modes',
    'smpl_nodes',
    'prompt_nodes',
    'socket_nodes',
    'ultracwt_nodes',
    'midi_nodes',
    'depthanything_nodes',
    'vae_nodes',
    'vive_tracker_nodes',
    'google_translate_nodes',
    'layout_nodes',
    'digico_nodes',
    'monitor_nodes',
    'eos_nodes',
    'orbbec_nodes'
]

imported = []
to_be_imported = []

def import_core(file_name):
    try:
        name = 'dpg_system.' + file_name
        to_be_imported.append(file_name)
        imported.append(file_name)
    except ModuleNotFoundError:
        print('No module named ' + file_name)

def import_test(file_name):
    if file_name in config:
        if config[file_name]:
            import_core(file_name)

with open('dpg_system_config.json', 'r') as f:
    config = json.load(f)

    for basic_import in to_import:
        import_core(basic_import)
    for try_import in optional_import:
        import_test(try_import)

if 'gl_nodes' in to_be_imported:
    opengl_active = True
    import glfw
else:
    opengl_active = False

for import_name in to_be_imported:
    try:
        globals()[import_name] = import_module('dpg_system.' + import_name)
        print('Imported ' + import_name)
        imported.append(import_name)
    except ModuleNotFoundError:
        print('No module named ' + import_name)

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
            if item is not None:
                if isinstance(item, Node):
                    node = item
                else:
                    node = item.node
                if node is not None:
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
            Node.app.save_internal(save_path)
            # Node.app.save_patch(save_path)
            # if Node.app.saving_to_lib:
            #     Node.app.register_patcher(Node.app.patches_name)
            #     Node.app.saving_to_lib = False
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
        self.config = None
        self.easy_mode = False
        with open('dpg_system_config.json', 'r') as f:
            self.config = json.load(f)
            if 'easy' in config:
                self.easy_mode = config['easy']
        self.torch_available = False
        self.viewport = None
        self.main_window_id = -1
        self.loading = False
        self.large_font = None
        self.default_font = None
        self.setup_dpg()
        self.verbose = False
        self.verbose_menu_item = -1
        self.minimap_menu_item = -1
        self.colour_code_pins_menu_item = -1
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
        NodeEditor.app = self
        self.register_nodes()
        self.new_patcher_index = 1
        self.patchers = []
        self.do_exit = False

        self.node_editors = []
        self.current_node_editor = 0
        self.frame_tasks = []
        self.variables = {}
        self.conduits = {}
        self.actions = {}
        self.loaded_patcher_nodes = []
        self.recent_files = {}
        self.recent_menus = []
        self.osc_manager = osc_nodes.OSCManager(label='osc_manager', data=0, args=None)

        self.recent_menu = None
        self.presentation_edit_menu_item = -1
        self.setup_menus()
        self.patches_path = ''
        self.patches_name = ''
        self.links_containers = {}
        self.created_nodes = {}
        self.drag_starts = {}

        self.color_code_pins = True
        # self.global_theme = None
        # self.borderless_child_theme = None
        self.active_widget = -1
        self.focussed_widget = -1
        self.hovered_item = None
        self.return_pressed = False
        self.frame_time_variable = self.add_variable(variable_name='frame_time')
        self.frame_number = 0
        self.frame_variable = self.add_variable(variable_name='frame')
        self.frame_clock_conduit = self.add_conduit('frame_clock')
        self.font_scale_variable = self.add_variable(variable_name='font_scale', setter=self.update_font_scale, default_value=0.5)
        self.gui_scale_variable = self.add_variable(variable_name='gui_scale', setter=self.update_gui_scale, default_value=1.0)
        self.link_thickness_variable = self.add_variable(variable_name='link_thickness', setter=self.update_link_thickness, default_value=1.0)
        self.dragging_created_nodes = False
        self.dragging_ref = [0, 0]
        self.clipboard = None
        self.saving_to_lib = False
        self.project_name = os.path.basename(__file__).split('.')[0]
        self.currently_loading_patch_name = ''
        self.currently_loading_node_name = ''

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
        self.window_context = None
        self.fresh_patcher = True
        self.pausing = False
        self.load_recent_patchers_list()
        self.gl_on_separate_thread = False

    def get_local_project_name(self):
        self.project_name = os.path.basename(__file__).split('.')[0]

    def load_recent_patchers_list(self):
        self.get_local_project_name()
        if os.path.exists(self.project_name + '_recent_patchers.json'):
            with open(self.project_name + '_recent_patchers.json', 'r') as f:
                self.recent_files = json.load(f)
            self.update_recent_menu()

    def clear_remembered_ids(self):
        self.active_widget = -1
        self.focussed_widget = -1
        self.hovered_item = None
        self.return_pressed = False

    def reset_frame_count(self):
        self.frame_number = 0

    def pause(self):
        self.pausing = True

    def resume(self):
        self.pausing = False

    def setup_dpg(self):
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)
        if 'macOS' in platform_.platform():
            with dpg.font_registry():
                if os.path.exists('Inconsolata-g.otf'):
                    self.default_font = dpg.add_font("Inconsolata-g.otf", 24)
                    dpg.bind_font(self.default_font)
                    dpg.set_global_font_scale(0.5)
                    self.large_font = dpg.add_font("Inconsolata-g.otf", 48)
        # handle other platforms...
        self.viewport = dpg.create_viewport()
        dpg.setup_dearpygui()

    def register_patcher(self, name):
        self.patchers.append(name)

    def register_patchers(self):
        if os.path.exists('dpg_system/patch_library'):
            for entry in os.scandir('dpg_system/patch_library'):
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
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 4, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 255, 0, 128), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.do_not_delete_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.window_padding[0], self.window_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, self.frame_padding[0], self.frame_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, self.cell_padding[0], self.cell_padding[1], category=dpg.mvThemeCat_Core)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (64, 0, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (128, 0, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (192, 0, 0, 255), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, self.item_spacing[0], self.item_spacing[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 4, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 255, 0, 128), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.locked_position_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.window_padding[0], self.window_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, self.frame_padding[0], self.frame_padding[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, self.cell_padding[0], self.cell_padding[1], category=dpg.mvThemeCat_Core)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (32, 32, 32, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (64, 64, 64, 255), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, self.item_spacing[0], self.item_spacing[1], category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 4, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 255, 0, 128), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 0, 255), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.invisible_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)

                dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)

                dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.widget_only_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

        with dpg.theme() as self.widget_only_node_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)

                dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

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
        if len(self.node_editors) > 0:
            return self.node_editors[self.current_node_editor]
        else:
            return None

    def get_current_root_patch(self):
        editor = self.get_current_editor()
        while editor.parent_patcher is not None:
            editor = editor.parent_patcher
        return editor

    def set_verbose(self):
        if self.verbose_menu_item != -1:
            self.verbose = dpg.get_value(self.verbose_menu_item)

    def set_colour_code_pines(self):
        if self.colour_code_pins_menu_item != -1:
            self.color_code_pins = dpg.get_value(self.colour_code_pins_menu_item)

    def show_log(self):
        if self.show_log_menu_item != -1:
            show = dpg.get_value(self.show_log_menu_item)

    def show_minimap(self):
        if self.minimap_menu_item != -1:
            show = dpg.get_value(self.minimap_menu_item)
            editor = self.get_current_editor()
            if editor is not None:
                editor.show_minimap(show)

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
        if current_editor is not None:
            if len(current_editor.subpatches) == 0:
                current_editor.set_path(save_path)
                current_editor.save(save_path)
                self.patches_name = self.get_current_editor().patch_name
                self.patches_path = save_path
                self.add_to_recent(self.patches_name, self.patches_path)
            else:
                if save_path is not None and save_path != '':
                    with open(save_path, 'w') as f:
                        self.patches_path = save_path

                        patch_name = save_path.split('/')[-1]
                        if '.' in patch_name:
                            parts = patch_name.split('.')
                            if len(parts) == 2:
                                if parts[1] == 'json':
                                    patch_name = parts[0]

                        self.patches_name = patch_name
                        current_editor.set_name(patch_name)
                        current_editor.set_path(save_path)
                        file_container = {}
                        file_container['name'] = self.patches_name
                        file_container['path'] = self.patches_path
                        file_container['patches'] = self.containerize_patch(current_editor)

                        json.dump(file_container, f, indent=4)
                        self.add_to_recent(self.patches_name, self.patches_path)
            if current_editor.patcher_node is not None:
                current_editor.patcher_node.name_option.set(self.patches_name)
                current_editor.patcher_node.name_changed()



    def save_setup(self, save_path):
        if os.path.exists(save_path):
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
                self.add_to_recent(self.patches_name, self.patches_path)

    def recent_1_callback(self):
        self.recent_callback(0)

    def recent_2_callback(self):
        self.recent_callback(1)

    def recent_3_callback(self):
        self.recent_callback(2)

    def recent_4_callback(self):
        self.recent_callback(3)

    def recent_5_callback(self):
        self.recent_callback(4)

    def recent_6_callback(self):
        self.recent_callback(5)

    def recent_7_callback(self):
        self.recent_callback(6)

    def recent_8_callback(self):
        self.recent_callback(7)

    def recent_9_callback(self):
        self.recent_callback(8)

    def recent_10_callback(self):
        self.recent_callback(9)

    def recent_callback(self, which):
        recents = list(self.recent_files.keys())
        name = recents[which]
        if name != '':
            path = self.recent_files[name]
            self.load_from_file(path)

    def add_ui(self):
        pass

    def setup_menus(self):
        recent_callback = []
        recent_callback.append(self.recent_1_callback)
        recent_callback.append(self.recent_2_callback)
        recent_callback.append(self.recent_3_callback)
        recent_callback.append(self.recent_4_callback)
        recent_callback.append(self.recent_5_callback)
        recent_callback.append(self.recent_6_callback)
        recent_callback.append(self.recent_7_callback)
        recent_callback.append(self.recent_8_callback)
        recent_callback.append(self.recent_9_callback)
        recent_callback.append(self.recent_10_callback)
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label='New Patch (N)', callback=self.add_node_editor)
                dpg.add_menu_item(label='Add UI', callback=self.add_ui)
                dpg.add_separator()

                dpg.add_menu_item(label="Open in this Patcher", callback=self.load_nodes_in_patcher)
                dpg.add_menu_item(label="Open in new Patcher (O)", callback=self.load_nodes)
                with dpg.menu(label='Open Recent') as self.recent_menu:
                    for i in range(10):
                        self.recent_menus.append(dpg.add_menu_item(label="...", show=False, callback=recent_callback[i]))
                    dpg.add_separator()
                    dpg.add_menu_item(label="Clear Recent", callback=self.clear_recent)
                dpg.add_menu_item(label="Open Example", callback=self.load_example)

                dpg.add_separator()
                dpg.add_menu_item(label='Close Current Patch (W)', callback=self.close_current_node_editor)
                dpg.add_separator()
                dpg.add_menu_item(label="Save Patch (S)", callback=self.save_nodes)
                dpg.add_menu_item(label="Save Patch As", callback=self.save_as_nodes)
                dpg.add_separator()
                dpg.add_menu_item(label='Set As Default Patch', callback=self.set_as_default_patch)
                dpg.add_menu_item(label='No Default Patch', callback=self.clear_default_patch)
                dpg.add_separator()
                dpg.add_menu_item(label="Save to Library", callback=self.save_to_library)
                dpg.add_separator()
                dpg.add_menu_item(label="Save All Patches", callback=self.save_patches)
                dpg.add_menu_item(label="Save All Patches As", callback=self.save_patches)
                dpg.add_separator()
                dpg.add_menu_item(label="Quit (Q)", callback=self.quit)
            with dpg.menu(label='Edit'):
                dpg.add_menu_item(label="Cut (X)", callback=self.cut_selected)
                dpg.add_menu_item(label="Copy (C)", callback=self.copy_selected)
                dpg.add_menu_item(label="Paste (V)", callback=self.paste_selected)
                dpg.add_menu_item(label="Duplicate (D)", callback=self.duplicate_handler)
                dpg.add_separator()
                dpg.add_menu_item(label="Patchify (P)", callback=self.patchify_handler)
                dpg.add_separator()
                dpg.add_menu_item(label="Connect Selected (K)", callback=self.connect_selected)
                dpg.add_menu_item(label="Align Selected", callback=self.align_selected)
                dpg.add_menu_item(label="Align Center and Distribute Selected (Y)", callback=self.align_distribute_selected)
                dpg.add_menu_item(label="Align Edge and Distribute Selected", callback=self.align_distribute_selected_top)
                dpg.add_menu_item(label="Space Out Selected (+)", callback=self.space_out_selected)
                dpg.add_menu_item(label="Tighten Selected (-)", callback=self.tighten_selected)
                dpg.add_separator()
                dpg.add_menu_item(label="Reset Origin", callback=self.reset_node_editor_origin)
            with dpg.menu(label='Visibility'):
                dpg.add_menu_item(label="Hide Selected", callback=self.hide_selected)
                dpg.add_menu_item(label="Widget Only for Selected", callback=self.show_widget_only_for_selected)
                dpg.add_menu_item(label="Reveal Hidden", callback=self.reveal_hidden)
                dpg.add_separator()
                dpg.add_menu_item(label="Toggle Lock Position for Selected", callback=self.lock_position_for_selected)
                dpg.add_menu_item(label="Show / Hide Options for Selected", callback=self.options_for_selected)
                dpg.add_menu_item(label='Toggle Do Not Delete for Selected', callback=self.protect_selected)
                dpg.add_separator()
                dpg.add_menu_item(label="Set As Presentation", callback=self.set_presentation)
                self.presentation_edit_menu_item = dpg.add_menu_item(label="Presentation Mode", check=True, callback=self.toggle_presentation)


            with dpg.menu(label='Options'):
                dpg.add_menu_item(label="Show Style Editor", callback=self.show_style)
                dpg.add_menu_item(label="Show Demo", callback=self.show_demo)
                dpg.add_separator()
                self.verbose_menu_item = dpg.add_menu_item(label="verbose logging", check=True, callback=self.set_verbose)
                dpg.add_separator()

                self.colour_code_pins_menu_item = dpg.add_menu_item(label="Colour Code Pins", check=True,
                                                                    callback=self.set_colour_code_pines)
                dpg.add_separator()

                dpg.add_menu_item(label='osc status', callback=self.print_osc_state)
                self.minimap_menu_item = dpg.add_menu_item(label='minimap', callback=self.show_minimap, check=True)

    def print_osc_state(self):
        if self.osc_manager:
            self.osc_manager.print_state()

    def quit(self):
        self.do_exit = True

    def register_nodes(self):
        register_base_nodes()

        for source_file in imported:
            register_call_name = 'register_' + source_file
            if source_file in globals():
                register_function = getattr(globals()[source_file], register_call_name)
                register_function()

    def get_variable_list(self):
        v_list = list(self.variables.keys())
        return v_list

    def get_conduit_list(self):
        c_list = list(self.conduit.keys())
        return c_list

    def update_font_scale(self, value):
        dpg.set_global_font_scale(value)

    def update_link_thickness(self, value):
        for editor in self.node_editors:
            editor.set_line_thickness(value)
        pass

    def update_gui_scale(self, value):
        for editor in self.node_editors:
            editor.scale_nodes(value)
        pass

    def add_conduit(self, conduit_name='untitled'):
        c = self.find_conduit(conduit_name)
        if c is None:
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
            editor = placeholder.my_editor
            if editor is not None:
                editor.remove_node(placeholder)
        return new_node

    def create_node_from_model(self, model, pos, name=None, args=[], from_file=False):
        node = model.create(name, args)
        editor = self.get_current_editor()
        if editor is not None:
            node.create(editor.uuid, pos=pos, from_file=from_file)
            editor.add_node(node)
            if not from_file:
                node.post_creation_callback()
        return node

    def remove_frame_task(self, remove_node):
        for index, node in enumerate(self.frame_tasks):
            if node == remove_node:
                self.frame_tasks[index:] = self.frame_tasks[index + 1:]

    def get_links_into_selection(self):
        editor = self.get_current_editor()
        external_sources = []
        external_targets = []
        if editor is not None:
            node_uuids = dpg.get_selected_nodes(editor.uuid)
            for uuid in node_uuids:
                node = dpg.get_item_user_data(uuid)
                if node is not None:
                    for index, in_ in enumerate(node.inputs):
                        parents = in_.get_parents()
                        if parents is not None and len(parents) > 0:
                            for parent in parents:
                                source_output = dpg.get_item_user_data(parent.uuid)
                                source_node = source_output.node
                                if source_node.uuid not in node_uuids:
                                    external_sources.append([parent, in_, uuid, index])
                    for index, out_ in enumerate(node.outputs):
                        children = out_.get_children()
                        if children is not None and len(children) > 0:
                            for child in children:
                                dest_input = dpg.get_item_user_data(child.uuid)
                                dest_node = dest_input.node
                                if dest_node.uuid not in node_uuids:
                                    external_targets.append([child, out_, uuid, index])
        return external_sources, external_targets

    def centre_of_selection(self):
        editor = self.get_current_editor()
        if editor is not None:
            node_uuids = dpg.get_selected_nodes(editor.uuid)
            centre_acc = [0, 0]
            centre_count = 0
            for uuid in node_uuids:
                pos = dpg.get_item_pos(uuid)
                size = [dpg.get_item_width(uuid), dpg.get_item_height(uuid)]
                centre = [pos[0] + size[0] / 2, pos[1] + size[1] / 2]
                centre_acc[0] += centre[0]
                centre_acc[1] += centre[1]
                centre_count += 1
            if centre_count > 0:
                return [centre_acc[0] / centre_count, centre_acc[1] / centre_count]

    def del_handler(self):
        if self.active_widget == -1:
            editor = self.get_current_editor()
            if editor is not None and not editor.presenting:
                editor.delete_selected_items()

    def return_handler(self):
        self.return_pressed = True
        pass

    def place_node(self, node):
        mouse_pos = dpg.get_mouse_pos(local=False)
        editor = self.get_current_editor()
        if editor is not None:
            try:
                editor_mouse_pos = editor.global_pos_to_editor_pos(mouse_pos)
                node.create(editor.uuid, pos=editor_mouse_pos)
                editor.add_node(node)
            except Exception as e:
                print('place_node', e)

    def int_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ValueNode.factory("int", None)
                self.place_node(node)
                node.name_id = dpg.generate_uuid()
                self.set_widget_focus(node.uuid)

    def float_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ValueNode.factory("float", None)
                self.place_node(node)
                self.set_widget_focus(node.uuid)

    def vector_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.VectorNode.factory("vector", None, args=['4'])
                self.place_node(node)
                self.set_widget_focus(node.uuid)

    def comment_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.new_handler('comment')

    def toggle_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ToggleNode.factory("toggle", None)
                self.place_node(node)

    def button_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ButtonNode.factory("b", None)
                self.place_node(node)

    def message_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ValueNode.factory("message", None)
                self.place_node(node)
                self.set_widget_focus(node.input.widget.uuid)

    def list_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                node = interface_nodes.ValueNode.factory("list", None)
                self.place_node(node)
                self.set_widget_focus(node.input.widget.uuid)

    def options_handler(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
                for selected_nodes_uuid in selected_nodes_uuids:
                    node_object = dpg.get_item_user_data(selected_nodes_uuid)
                    node_object.toggle_show_hide_options()

    def M_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                self.show_minimap()

    def Q_handler(self):
        if self.control_or_command_down():
            self.quit()

    def C_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                self.clipboard = self.get_current_editor().copy_selection()
        else:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.comment_handler()

    def plus_handler(self):
        if self.control_or_command_down():
            self.space_out_selected()

    def minus_handler(self):
        if self.control_or_command_down():
            self.tighten_selected()

    def space_handler(self):
        pass

    def P_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                self.clipboard = self.get_current_editor().patchify_selection()

    def R_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                self.get_current_editor().reset_origin()

    def hide_selected(self):
        if self.get_current_editor() is not None:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node = dpg.get_item_user_data(selected_nodes_uuid)
                if node is not None:
                    node.set_visibility('hidden')

    def lock_position_for_selected(self):
        if self.get_current_editor() is not None:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node = dpg.get_item_user_data(selected_nodes_uuid)
                node.set_draggable(not node.draggable)
                node.set_visibility(node.visibility)

    def protect_selected(self):
        if self.get_current_editor() is not None:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node = dpg.get_item_user_data(selected_nodes_uuid)
                node.do_not_delete = not node.do_not_delete
                node.set_visibility(node.visibility)

    def options_for_selected(self):
        if self.get_current_editor() is not None:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node = dpg.get_item_user_data(selected_nodes_uuid)
                node.toggle_show_hide_options()

    def show_widget_only_for_selected(self):
        if self.get_current_editor() is not None:
            selected_nodes_uuids = dpg.get_selected_nodes(self.get_current_editor().uuid)
            for selected_nodes_uuid in selected_nodes_uuids:
                node = dpg.get_item_user_data(selected_nodes_uuid)
                # print(node)

                if node is not None:
                    node.set_visibility('widgets_only')

    def reveal_hidden(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().reveal_hidden()

    def set_presentation(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().remember_presentation()

    def toggle_presentation(self):
        if self.get_current_editor() is not None:
            editor = self.get_current_editor()
            if editor.presenting:
                editor.enter_edit_state()
                dpg.set_item_label(self.presentation_edit_menu_item, 'Enter Presentation Mode (E)')
            else:
                editor.enter_presentation_state()
                dpg.set_item_label(self.presentation_edit_menu_item, 'Enter Edit Mode (E)')

    def control_or_command_down(self):
        return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl) or dpg.is_key_down(dpg.mvKey_LWin) or dpg.is_key_down(dpg.mvKey_RWin)
        # return dpg.is_key_down(dpg.mvKey_ModCtrl) or dpg.is_key_down(dpg.mvKey_LWin) or dpg.is_key_down(dpg.mvKey_RWin)

    def X_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                if not self.get_current_editor().presenting:
                    self.clipboard = self.get_current_editor().cut_selection()

    def S_handler(self):
        if self.control_or_command_down():
            self.save_nodes()

    def O_handler(self):
        if self.control_or_command_down():
            self.load_nodes()
        else:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.options_handler()

    def K_handler(self):
        if self.get_current_editor() is not None and not self.get_current_editor().presenting:
            if self.control_or_command_down():
                self.connect_selected()

    def Y_handler(self):
        if self.get_current_editor() is not None and not self.get_current_editor().presenting:
            if self.control_or_command_down():
                self.align_distribute_selected()

    def N_handler(self):
        if self.control_or_command_down():
            self.add_node_editor()
        else:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.new_handler()

    def W_handler(self):
        if self.control_or_command_down():
            self.close_current_node_editor()

    def V_handler(self):
        if self.control_or_command_down():
            self.paste_selected()
        else:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.vector_handler()

    def D_handler(self):
        if self.control_or_command_down():
            if self.active_widget == -1:
                if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                    self.get_current_editor().duplicate_selection()

    def E_handler(self):
        if self.control_or_command_down():
            if self.get_current_editor() is not None:
                self.toggle_presentation()

    def duplicate_handler(self):
        if self.get_current_editor() is not None and not self.get_current_editor().presenting:
            if self.active_widget == -1:
                self.get_current_editor().duplicate_selection()

    def patchify_handler(self):
        if self.get_current_editor() is not None and not self.get_current_editor().presenting:
            self.get_current_editor().patchify_selection()

    def cut_selected(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.get_current_editor().cut_selection()

    def copy_selected(self):
        if self.active_widget == -1:
            if self.get_current_editor() is not None and not self.get_current_editor().presenting:
                self.get_current_editor().copy_selection()

    def mouse_down_handler(self):
        if self.control_or_command_down():
            self.toggle_presentation()
        else:
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
        if dpg.does_item_exist(widget_uuid):
            dpg.focus_item(widget_uuid)
            self.focussed_widget = widget_uuid

    # def mouse_drag_handler(self):
    #     print('dragged', self.active_widget)

    def key_handler(self, sender, app_data, user_data):
        from dpg_system.interface_nodes import KeyNode
        for node in KeyNode.node_list:
            node.key_down(app_data)

    def key_release_handler(self, sender, app_data, user_data):
        from dpg_system.interface_nodes import KeyNode
        for node in KeyNode.node_list:
            node.key_up(app_data)

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


    def new_handler(self, name=None):
        if self.get_current_editor() is not None and not self.get_current_editor().presenting:
            editor = self.get_current_editor()

            if self.active_widget == -1:
                node = PlaceholderNode.factory("New Node", None)
                self.place_node(node)
                # mouse_pos = dpg.get_mouse_pos(local=False)
                # editor_mouse_pos = editor.global_pos_to_editor_pos(mouse_pos)
                # node.create(editor.uuid, pos=editor_mouse_pos)
                # editor.add_node(node)
                if name is not None:
                    self.set_widget_focus(node.name_property.widget.uuid)
                    dpg.set_value(node.name_property.widget.uuid, name)
                    node.node_list = [name]
                    node.prompt_for_args()
                else:
                    self.set_widget_focus(node.name_property.widget.uuid)

    def update(self):
        pass

    def connect_link(self, links_container, index, link_index, node_editor_uuid):
        source_node = None
        dest_node = None
        link_container = links_container[link_index]
        new_link = link_container

        source_node_loaded_uuid = link_container['source_node']
        if source_node_loaded_uuid in self.created_nodes:
            source_node = self.created_nodes[source_node_loaded_uuid]
            new_link['source_node'] = source_node.uuid
        dest_node_loaded_uuid = link_container['dest_node']
        if dest_node_loaded_uuid in self.created_nodes:
            dest_node = self.created_nodes[dest_node_loaded_uuid]
            new_link['dest_node'] = dest_node.uuid
        if source_node is not None and dest_node is not None:
            source_output_name = ''
            source_output_index = link_container['source_output_index']
            if 'source_output_name' in link_container:
                source_output_name = link_container['source_output_name']
            dest_input_index = link_container['dest_input_index']
            dest_input_name = ''
            if 'dest_input_name' in link_container:
                dest_input_name = link_container['dest_input_name']
            if source_output_index < len(source_node.outputs):
                source_output = source_node.outputs[source_output_index]
                found_output = True
                if source_output_name != '':
                    found_output = False
                    if source_output.get_label() != source_output_name:
                        for index, output in enumerate(source_node.inputs):
                            if output.get_label() == source_output_name:
                                source_output_index = index
                                source_output = output
                                found_output = True
                                break
                        if not found_output:
                            #  check archive of names
                            for index, output in enumerate(source_node.inputs):
                                if source_output_name in output.name_archive:
                                    source_output_index = index
                                    source_output = output
                                    found_output = True
                                    break
                        if not found_output:
                            if len(source_node.outputs) == 1:
                                source_output_index = 0
                                source_output = source_node.outputs[0]
                                found_output = True
                    else:
                        found_output = True

                if dest_input_index < len(dest_node.inputs):
                    dest_input = dest_node.inputs[dest_input_index]
                    found_input = True
                    if dest_input_name != '':
                        found_input = False
                        if dest_input.get_label() != dest_input_name:
                            for index, input in enumerate(dest_node.inputs):
                                if input.get_label() == dest_input_name:
                                    dest_input_index = index
                                    dest_input = input
                                    found_input = True
                                    break
                            if not found_input:
                                for index, input in enumerate(dest_node.inputs):
                                    if dest_input_name in input.name_archive:
                                        dest_input_index = index
                                        dest_input = input
                                        found_input = True
                                        break
                            if not found_input:
                                if len(dest_node.inputs) == 1:
                                    dest_input_index = 0
                                    dest_input = dest_node.inputs[0]
                                    found_input = True
                        else:
                            found_input = True

                    if found_output and found_input:
                        source_output.add_child(dest_input, node_editor_uuid)
                    else:
                        if not found_output:
                            print('could not locate output', source_output_name, 'in', source_node)
                        if not found_input:
                            print('could not locate input', dest_input_name, 'in', dest_node)
        return new_link

    def load_from_file(self, path):
        self.loading = True
        self.loaded_patcher_nodes = []
        self.links_containers = {}
        self.created_nodes = {}
        main_editor = None
        if len(self.node_editors) == 0:
            self.add_node_editor()
        hold_current_editor = self.current_node_editor
        parent_tab = self.get_current_tab()
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
                    self.add_to_recent(patch_name, path)

                    patches_container = file_container['patches']
                    patch_count = len(patch_assign)
                    for index, patch_index in enumerate(patches_container):
                        nodes_container = patches_container[patch_index]
                        if len(patch_assign) > 0:
                            editor_index, editor = patch_assign[patch_count - index - 1]
                        else:
                            editor = None
                        if editor is not None:
                            self.current_node_editor = editor_index
                            editor.load_(nodes_container)
                            if editor.loaded_parent_node_uuid == -1:
                                if main_editor is None:
                                    main_editor = self.current_node_editor

                else:  # single patch
                    if self.fresh_patcher:
                        if len(patch_assign) > 0:
                            editor_index, editor = patch_assign[0]
                            self.current_node_editor = editor_index
                        else:
                            editor = self.get_current_editor()
                            if editor.num_nodes > 1:
                                self.add_node_editor()
                                # self.current_node_editor = len(self.node_editors) - 1
                        main_editor = self.current_node_editor
                    else:
                        if self.get_current_editor() is None:
                            self.add_node_editor()
                            main_editor = self.current_node_editor
                            # self.current_node_editor = len(self.node_editors) - 1
                    self.add_to_recent(patch_name, path)
                    self.select_tab(self.tabs[self.current_node_editor])
                    # dpg.set_value(self.tab_bar, self.tabs[self.current_node_editor])
                    self.get_current_editor().load_(file_container, path, patch_name)

                for node_editor_uuid in self.links_containers:
                    links_container = self.links_containers[node_editor_uuid]
                    for index, link_index in enumerate(links_container):
                        self.connect_link(links_container, index, link_index, node_editor_uuid)
                        # source_node = None
                        # dest_node = None
                        # link_container = links_container[link_index]
                        # source_node_loaded_uuid = link_container['source_node']
                        # if source_node_loaded_uuid in self.created_nodes:
                        #     source_node = self.created_nodes[source_node_loaded_uuid]
                        # dest_node_loaded_uuid = link_container['dest_node']
                        # if dest_node_loaded_uuid in self.created_nodes:
                        #     dest_node = self.created_nodes[dest_node_loaded_uuid]
                        # if source_node is not None and dest_node is not None:
                        #     source_output_index = link_container['source_output_index']
                        #     dest_input_index = link_container['dest_input_index']
                        #     if source_output_index < len(source_node.outputs):
                        #         source_output = source_node.outputs[source_output_index]
                        #         if dest_input_index < len(dest_node.inputs):
                        #             dest_input = dest_node.inputs[dest_input_index]
                        #             source_output.add_child(dest_input, node_editor_uuid)

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
            print('load failed while loading', self.currently_loading_patch_name, self.currently_loading_node_name)
            traceback.print_exception(exc_)

        if main_editor is not None:
            self.current_node_editor = main_editor
        self.select_editor_tab(self.current_node_editor)
        self.loading = False

    def load_nodes(self):
        self.load('', fresh_patcher=True)

    def load_nodes_in_patcher(self):
        fresh = False
        if len(self.patchers) == 0:
            fresh = True
        self.load('', fresh_patcher=fresh)

    def clear_recent(self):
        self.recent_files = {}
        with open(self.project_name + '_recent_patchers.json', 'w') as f:
            json.dump(self.recent_files, f, indent=4)
        self.update_recent_menu()

    def add_to_recent(self, name, path):
        if name in self.recent_files:
            self.recent_files[name] = path
        elif len(self.recent_files) >= 10:
            keys = list(self.recent_files.keys())
            del self.recent_files[keys[0]]
            self.recent_files[name] = path
        else:
            self.recent_files[name] = path
        with open(self.project_name + '_recent_patchers.json', 'w') as f:
            json.dump(self.recent_files, f, indent=4)
        self.update_recent_menu()

    def update_recent_menu(self):
        names = list(self.recent_files.keys())
        for i in range(10):
            if i < len(self.recent_files):
                dpg.configure_item(self.recent_menus[i], label=names[i], show=True)
            else:
                dpg.configure_item(self.recent_menus[i], show=False)

    def load(self, path='', fresh_patcher=True):
        if path != '':
            if self.get_current_editor() is None:
                self.add_node_editor()
            elif fresh_patcher:
                editor = self.get_current_editor()
                if editor.num_nodes > 1:
                    self.add_node_editor()
            if self.get_current_editor() is not None:
                self.get_current_editor().load(path)
                return
        self.active_widget = 1
        self.fresh_patcher = fresh_patcher
        with dpg.file_dialog(modal=True, default_path='patches', directory_selector=False, show=True, height=400, width=800, callback=load_patches_callback, cancel_callback=cancel_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")


    def load_example(self):
        self.active_widget = 1
        self.fresh_patcher = True
        with dpg.file_dialog(modal=True, default_path='examples', directory_selector=False, show=True, height=400, width=800, callback=load_patches_callback,
                             cancel_callback=cancel_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_as_nodes(self):
        self.save('patches')

    def save_internal(self, path):
        self.save_patch(path)
        if self.saving_to_lib:
            self.register_patcher(Node.app.patches_name)
            self.saving_to_lib = False

    def save_to_library(self):
        if not os.path.exists('dpg_system/patch_library'):
            os.makedirs('dpg_system/patch_library')
        if os.path.exists('dpg_system/patch_library'):
            self.saving_to_lib = True
            self.save('', default_directory='dpg_system/patch_library')
            self.patchers.append(self.patches_name)


    def save_nodes(self):
         if self.get_current_editor() is not None:
            if exists(self.get_current_editor().file_path):
                self.save_internal(self.get_current_editor().file_path)
            else:
                self.save_as_nodes()

    def save_as_patches(self):
        self.save_patches('')

    def connect_selected(self):
        self.get_current_editor().connect_selected()


    def paste_selected(self):
        if self.clipboard is not None:
            self.clear_remembered_ids()
            self.get_current_editor().paste(self.clipboard)
        else:
            print('clipboard is empty')

    def align_selected(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().align_selected()

    def space_out_selected(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().align_and_distribute_selected(align=-2, spread_factor=1.1)

    def tighten_selected(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().align_and_distribute_selected(align=-2, spread_factor=0.9)

    def align_distribute_selected(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().align_and_distribute_selected(0)

    def align_distribute_selected_top(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().align_and_distribute_selected(-1)

    def reset_node_editor_origin(self):
        if self.get_current_editor() is not None:
            self.get_current_editor().reset_origin()

    def show_style(self):
        dpg.show_style_editor()

    def show_demo(self):
        dpg.show_imgui_demo()

    def save_with_path(self, sender, data):
        if self.get_current_editor() is not None:
            filename = os.sep.join(data)
            for i in open(filename, "rt"):
                self.get_current_editor().save(filename)

    def save(self, path='', default_directory='patches'):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, width=800, callback=save_file_callback, cancel_callback=cancel_callback, default_path=default_directory, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def save_patches(self, path=''):
        self.active_widget = 1
        with dpg.file_dialog(directory_selector=False, show=True, height=400, width=800, callback=save_patches_callback, cancel_callback=cancel_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".json")

    def add_frame_task(self, dest):
        if dest not in self.frame_tasks:
            self.frame_tasks.append(dest)

    def register_node(self, label, factory, data=None):
        self.node_factory_container.add_node_factory(NodeFactory(label, factory, data))
        self.node_list.append(label)

    def select_tab(self, which_tab):
        dpg.set_value(self.tab_bar, which_tab)

    def select_editor_tab(self, which_editor):
        if 0 <= which_editor < len(self.tabs):
            self.select_tab(self.tabs[which_editor])
    def get_current_tab(self):
        return dpg.get_value(self.tab_bar)

    def set_current_tab_title(self, title):
        if len(self.tabs) > self.current_node_editor:
            dpg.configure_item(self.tabs[self.current_node_editor], label=title)

    def set_tab_title(self, tab_index, title):
        if len(self.tabs) > tab_index >= 0:
            dpg.configure_item(self.tabs[tab_index], label=title)

    def set_editor_tab_title(self, editor, name):
        for index, ed in enumerate(self.node_editors):
            if ed is editor:
                self.set_tab_title(index, name)

    def selected_tab(self):
        chosen_tab_uuid = dpg.get_value(self.tab_bar)
        chosen_tab_index = dpg.get_item_user_data(chosen_tab_uuid)
        self.current_node_editor = chosen_tab_index
        if self.get_current_editor() is not None:
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

    def clear_default_patch(self):
        if self.get_current_editor() is not None:
            patch = self.get_current_editor()
            if patch.file_path != '':
                default_patcher_dict = {}
                default_patcher_dict['path'] = ''
                with open(self.project_name + '_default_patcher.json', 'w') as f:
                    json.dump(default_patcher_dict, f, indent=4)

    def set_as_default_patch(self):
        if self.get_current_editor() is not None:
            patch = self.get_current_editor()
            if patch.file_path != '':
                default_patcher_path = patch.file_path
                default_patcher_dict = {}
                default_patcher_dict['path'] = default_patcher_path
                with open(self.project_name + '_default_patcher.json', 'w') as f:
                    json.dump(default_patcher_dict, f, indent=4)

    def open_default_patch(self):
        if os.path.exists(self.project_name + '_default_patcher.json'):
            with open(self.project_name + '_default_patcher.json', 'r') as f:
                default_patcher_dict = json.load(f)
                if 'path' in default_patcher_dict:
                    default_patcher_path = default_patcher_dict['path']
                    if default_patcher_path != '' and default_patcher_path is not None:
                        self.load_from_file(default_patcher_path)
                        return True
        return False

    def close_current_node_editor(self):
        # don't close node editor if it is a subpatcher
        if self.get_current_editor() is not None:
            patcher = self.get_current_editor()
            if patcher.parent_patcher is None:
                self.remove_node_editor(self.get_current_editor())

    def add_node_editor(self):
        editor_name = 'patch ' + str(self.new_patcher_index)
        self.new_patcher_index += 1
        with dpg.tab(label=editor_name, parent=self.tab_bar, user_data=len(self.tabs)) as tab:
            self.tabs.append(tab)
            panel_uuid = dpg.generate_uuid()
            with dpg.group(id=panel_uuid):
                new_editor = NodeEditor()
                if editor_name is not None:
                    new_editor.patch_name = editor_name
                self.node_editors.append(new_editor)
                self.node_editors[len(self.node_editors) - 1].create(panel_uuid)
                self.current_node_editor = len(self.node_editors) - 1
                self.select_tab(tab)
        return new_editor

    def start(self):
        dpg.set_viewport_title("Patchers")
        self.node_editors = [NodeEditor()]

        with dpg.window() as main_window:
            global dpg_app
            if opengl_active:
                self.window_context = glfw.get_current_context()
            self.main_window_id = main_window
            dpg.bind_item_theme(main_window, self.global_theme)
            dpg.add_spacer(height=14)
            with dpg.tab_bar(callback=self.selected_tab) as self.tab_bar:
                with dpg.tab(label='patch ' + str(self.new_patcher_index), user_data=len(self.tabs)) as tab:
                    self.new_patcher_index += 1
                    self.tabs.append(tab)
                    with dpg.group(id=self.center_panel):
                        self.node_editors[0].create(self.center_panel)
                        with dpg.handler_registry():
                            # dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                            dpg.add_key_press_handler(callback=self.key_handler)
                            dpg.add_key_release_handler(callback=self.key_release_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.up_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.down_handler)
                            dpg.add_key_press_handler(dpg.mvKey_I, callback=self.int_handler)
                            dpg.add_key_press_handler(dpg.mvKey_F, callback=self.float_handler)
                            dpg.add_key_press_handler(dpg.mvKey_T, callback=self.toggle_handler)
                            dpg.add_key_press_handler(dpg.mvKey_B, callback=self.button_handler)
                            dpg.add_key_press_handler(dpg.mvKey_M, callback=self.message_handler)
                            dpg.add_key_press_handler(dpg.mvKey_L, callback=self.list_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Q, callback=self.Q_handler)

                            dpg.add_key_press_handler(dpg.mvKey_E, callback=self.E_handler)
                            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.D_handler)
                            dpg.add_key_press_handler(dpg.mvKey_C, callback=self.C_handler)
                            dpg.add_key_press_handler(dpg.mvKey_V, callback=self.V_handler)
                            dpg.add_key_press_handler(dpg.mvKey_X, callback=self.X_handler)
                            dpg.add_key_press_handler(dpg.mvKey_W, callback=self.W_handler)
                            dpg.add_key_press_handler(dpg.mvKey_O, callback=self.O_handler)
                            dpg.add_key_press_handler(dpg.mvKey_S, callback=self.S_handler)
                            dpg.add_key_press_handler(dpg.mvKey_N, callback=self.N_handler)
                            dpg.add_key_press_handler(dpg.mvKey_K, callback=self.K_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Y, callback=self.Y_handler)
                            dpg.add_key_press_handler(dpg.mvKey_P, callback=self.P_handler)
                            dpg.add_key_press_handler(dpg.mvKey_R, callback=self.R_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Plus, callback=self.plus_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Minus, callback=self.minus_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=self.space_handler)

                            dpg.add_key_press_handler(dpg.mvKey_Back, callback=self.del_handler)
                            dpg.add_key_press_handler(dpg.mvKey_Return, callback=self.return_handler)
                            dpg.add_mouse_move_handler(callback=self.drag_create_nodes)
                            dpg.add_mouse_click_handler(callback=self.mouse_down_handler)

        dpg.set_primary_window(main_window, True)
        dpg.show_viewport()
        self.open_default_patch()

    def set_dpg_gl_context(self):
        if opengl_active:
            hold_context = glfw.get_current_context()
            glfw.make_context_current(self.window_context)
            return hold_context
        return None

    def restore_gl_context(self, held_context):
        if held_context:
            glfw.make_context_current(held_context)

    def run_loop(self):
        elapsed = 0
        do_gl = False
        do_osc_async = False
        if 'gl_nodes' in globals():
            do_gl = True
        if 'osc_nodes' in globals():
            do_osc_async = True

        while dpg.is_dearpygui_running():
            try:
                if not self.pausing:
                    hold_context = None
                    if not self.gl_on_separate_thread:
                        hold_context = self.set_dpg_gl_context()

                    now = time.time()
                    for node_editor in self.node_editors:
                        node_editor.reset_pins()
                    for task in self.frame_tasks:
                        if task.created:
                            task.frame_task()

                    jobs = dpg.get_callback_queue()  # retrieves and clears queue
                    dpg.run_callbacks(jobs)
                    self.frame_number += 1
                    self.frame_variable.set(self.frame_number)
                    self.frame_clock_conduit.transmit('bang')
                    self.frame_time_variable.set(elapsed)
                    if do_osc_async:
                        osc_nodes.OSCThreadingSource.osc_manager.relay_pending_messages()
                    dpg.render_dearpygui_frame()
                    then = time.time()
                    elapsed = then - now

                    #  openGL separate thread?
                    if not self.gl_on_separate_thread:
                        if opengl_active:
                            if do_gl:
                                gl_nodes.GLContextNode.maintenance_loop()
                            self.restore_gl_context(hold_context)
                            # glfw.make_context_current(hold_context)

                if self.do_exit:
                    _thread.interrupt_main()

            except Exception as exc_:
                print('run_loop exception:')
                traceback.print_exception(exc_)

