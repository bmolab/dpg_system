import dearpygui.dearpygui as dpg
import time
import numpy as np

from dpg_system.node import Node, InputNodeAttribute, OutputNodeAttribute, Variable
from dpg_system.node_editor import *
from os.path import exists
import json
import os

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
    if Node.app.return_pressed:
        Node.app.return_pressed = False
        item = dpg.get_item_user_data(data)
        # print('deactive', data, item)
        if item is not None:
            node = item.node
            if node is not None and node.label == 'New Node':
                node.execute()
    elif dpg.is_item_ok(data):
        item = dpg.get_item_user_data(data)
        if item is not None:
            node = item.node
            # print("about to deactivate", data, item, node)
            if node is not None:
                # print("deactivate", data)
                node.on_deactivate(item)
    Node.app.active_widget = -1
    Node.app.focussed_widget = -1


def widget_edited(source, data, user_data):
    if user_data is not None:
        user_data.on_edit()
        # print("edited+", user_data, data)
    else:
        if dpg.does_item_exist(data):
            item = dpg.get_item_user_data(data)
            # print('edited', data, item)
            if item is not None:
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
                # print("deactivated_after_edit", data, item)
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
        with dpg.font_registry():
            default_font = dpg.add_font("Inconsolata-g.otf", 24)

            dpg.bind_font(default_font)

        dpg.set_global_font_scale(0.5)

        self.viewport = dpg.create_viewport()
        # print(self.viewport)
        dpg.setup_dearpygui()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 4, 3, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 5, 2, category=dpg.mvThemeCat_Core)

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

        # self.register_node("metro", MetroNode.factory)
        # self.register_node("counter", CounterNode.factory)
        # self.register_node("gate", GateNode.factory)
        # self.register_node("switch", SwitchNode.factory)
        # self.register_node("unpack", UnpackNode.factory)
        # self.register_node("pack", PackNode.factory)
        # self.register_node("pak", PackNode.factory)
        # self.register_node("timer", TimerNode.factory)
        # self.register_node("elapsed", TimerNode.factory)
        # self.register_node("delay", DelayNode.factory)
        # self.register_node("select", SelectNode.factory)
        # self.register_node("decode", SelectNode.factory)
        # self.register_node("t", TriggerNode.factory)
        # self.register_node("combine", CombineNode.factory)
        # self.register_node("kombine", CombineNode.factory)
        # self.register_node("type", TypeNode.factory)
        # self.register_node("array", ArrayNode.factory)
        # self.register_node("string", StringNode.factory)
        # self.register_node("list", ListNode.factory)
        # self.register_node('prepend', PrependNode.factory)
        # self.register_node('append', AppendNode.factory)
        # self.register_node('coll', CollectionNode.factory)
        # self.register_node('repeat', RepeatNode.factory)
        # self.register_node('var', VariableNode.factory)

        # self.register_node('osc_source', OSCSourceNode.factory)
        # self.register_node('osc_receive', OSCReceiveNode.factory)
        # self.register_node('osc_target', OSCTargetNode.factory)
        # self.register_node('osc_send', OSCSendNode.factory)

        # self.register_node('gl_context', GLContextNode.factory)
        # self.register_node('gl_sphere', GLSphereNode.factory)
        # self.register_node('gl_cylinder', GLCylinderNode.factory)
        # self.register_node('gl_disk', GLDiskNode.factory)
        # self.register_node('gl_partial_disk', GLPartialDiskNode.factory)
        # self.register_node('gl_translate', GLTransformNode.factory)
        # self.register_node('gl_rotate', GLTransformNode.factory)
        # self.register_node('gl_scale', GLTransformNode.factory)
        # self.register_node('gl_material', GLMaterialNode.factory)
        # self.register_node('gl_align', GLAlignNode.factory)
        # self.register_node('gl_quaternion_rotate', GLQuaternionRotateNode.factory)

        # self.register_node("menu", MenuNode.factory)
        # self.register_node("toggle", ToggleNode.factory)
        # self.register_node("button", ButtonNode.factory)
        # self.register_node("b", ButtonNode.factory)
        # self.register_node("mouse", MouseNode.factory)
        # self.register_node("float", ValueNode.factory)
        # self.register_node("int", ValueNode.factory)
        # self.register_node("slider", ValueNode.factory)
        # self.register_node("message", ValueNode.factory)
        # self.register_node("knob", ValueNode.factory)
        # self.register_node("plot", PlotNode.factory)
        # self.register_node("heat_map", PlotNode.factory)
        # self.register_node("heat_scroll", PlotNode.factory)
        # self.register_node("Value Tool", ValueNode.factory)
        # self.register_node('print', PrintNode.factory)
        # self.register_node('load_action', LoadActionNode.factory)
        # self.register_node('color', ColorPickerNode.factory)
        # self.register_node('vector', VectorNode.factory)

        # self.register_node("+", ArithmeticNode.factory)
        # self.register_node("-", ArithmeticNode.factory)
        # self.register_node("!-", ArithmeticNode.factory)
        # self.register_node("*", ArithmeticNode.factory)
        # self.register_node("/", ArithmeticNode.factory)
        # self.register_node("!/", ArithmeticNode.factory)
        # self.register_node("min", ArithmeticNode.factory)
        # self.register_node("max", ArithmeticNode.factory)
        # self.register_node("mod", ArithmeticNode.factory)
        # self.register_node("%", ArithmeticNode.factory)
        # self.register_node("^", ArithmeticNode.factory)
        # self.register_node("pow", ArithmeticNode.factory)
        # self.register_node("sin", OpSingleTrigNode.factory)
        # self.register_node("cos", OpSingleTrigNode.factory)
        # self.register_node("asin", OpSingleTrigNode.factory)
        # self.register_node("acos", OpSingleTrigNode.factory)
        # self.register_node("tan", OpSingleTrigNode.factory)
        # self.register_node("atan", OpSingleTrigNode.factory)
        # self.register_node("log10", OpSingleNode.factory)
        # self.register_node("log2", OpSingleNode.factory)
        # self.register_node("exp", OpSingleNode.factory)
        # self.register_node("inverse", OpSingleNode.factory)
        # self.register_node("abs", OpSingleNode.factory)
        # self.register_node("sqrt", OpSingleNode.factory)
        # self.register_node("norm", OpSingleNode.factory)

        # self.register_node(">", ComparisonNode.factory)
        # self.register_node(">=", ComparisonNode.factory)
        # self.register_node("==", ComparisonNode.factory)
        # self.register_node("!=", ComparisonNode.factory)
        # self.register_node("<", ComparisonNode.factory)
        # self.register_node("<=", ComparisonNode.factory)

        # self.register_node('buffer', BufferNode.factory)
        # self.register_node('rolling_buffer', RollingBufferNode.factory)
        # self.register_node('flatten', FlattenMatrixNode.factory)
        # self.register_node('cwt', WaveletNode.factory)

        # self.register_node('gl_body', MoCapGLBody.factory)
        # self.register_node('take', MoCapTakeNode.factory)
        # self.register_node('body_to_joints', MoCapBody.factory)

        # self.register_node('quaternion_to_euler', QuaternionToEulerNode.factory)
        # self.register_node('quaternion_to_matrix', QuaternionToRotationMatrixNode.factory)
        # self.register_node('quaternion_distance', QuaternionDistanceNode.factory)

        # self.register_node("filter", FilterNode.factory)
        # self.register_node("smooth", FilterNode.factory)
        # self.register_node("diff_filter_bank", MultiDiffFilterNode.factory)
        # self.register_node("diff_filter", MultiDiffFilterNode.factory)
        # self.register_node("random", RandomNode.factory)
        # self.register_node("signal", SignalNode.factory)
        # self.register_node("togedge", TogEdgeNode.factory)
        # self.register_node("subsample", SubSampleNode.factory)
        # self.register_node("diff", DifferentiateNode.factory)
        # self.register_node('noise_gate', NoiseGateNode.factory)
        # self.register_node('trigger', ThresholdTriggerNode.factory)
        # self.register_node('hysteresis', ThresholdTriggerNode.factory)
        # self.register_node('sample_hold', SampleHoldNode.factory)

        # self.register_node('rephrase', RephraseNode.factory)

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

    def create_node_by_name_from_file(self, node_name, pos, args=None):
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

    def create_node_by_name(self, node_name, placeholder, args=None, pos=None):
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

    def create_node_from_model(self, model, pos, name=None, args=None):
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

    def up_handler(self):
        if self.hovered_item is not None:
            uuid = self.hovered_item.uuid
            if dpg.does_item_exist(uuid):
                if dpg.is_item_hovered(uuid):
                    self.hovered_item.increment()
        elif self.active_widget != -1:
            print('active widget up')
            if dpg.does_item_exist(self.active_widget):
                print('exists')
                widget = dpg.get_item_user_data(self.active_widget)
                if widget is not None:
                    print('widget', widget)
                    if widget.node is not None:
                        print('widget node', widget.node)
                        widget.node.increment_widget(widget)
                        if widget.callback is not None:
                            widget.callback()

    def down_handler(self):
        if self.hovered_item is not None:
            uuid = self.hovered_item.uuid
            if dpg.does_item_exist(uuid):
                if dpg.is_item_hovered(uuid):
                    self.hovered_item.decrement()
        elif self.active_widget != -1:
            print('active widget down')
            if dpg.does_item_exist(self.active_widget):
                print('exists')
                widget = dpg.get_item_user_data(self.active_widget)
                if widget is not None:
                    print('widget', widget)
                    if widget.node is not None:
                        print('widget node', widget.node)
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
        # with dpg.mutex():
        #     dpg.delete_item(self.left_panel, children_only=True)
        #     self.data_set_container.submit(self.left_panel)
        #     self.node_factory_container.submit(self.left_panel)
        #
        #     dpg.delete_item(self.right_panel, children_only=True)
        #     self.inspector_container.submit(self.right_panel)
        #     self.tool_container.submit(self.right_panel)
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
        print('before file_dialog')
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
        conf = dpg.get_item_configuration(self.tab_bar)
        print(conf)
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
            self.main_window_id = main_window
            dpg.bind_item_theme(main_window, self.global_theme)
            dpg.add_spacer(height=14)
            with dpg.tab_bar(callback=self.selected_tab) as self.tab_bar:
                with dpg.tab(label='node editor', user_data=len(self.tabs)) as tab:
                    self.tabs.append(tab)
                    with dpg.group(id=self.center_panel):
                        self.node_editors[0].submit(self.center_panel)
                        with dpg.handler_registry():
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
                        dpg.add_input_int(label='test')
                        dpg.add_input_int(label='test2')
                        dpg.add_input_int(label='test3')
                        new_editor = NodeEditor()
                        self.node_editors.append(new_editor)
                        self.node_editors[1].submit(self.side_panel)
                        dpg.add_input_int(label='test4')
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
                    task.frame_task()
                dpg.run_callbacks(jobs)
            except Exception as exc_:
                print('dpg calls', exc_)
            self.frame_number += 1
            self.frame_variable.set(self.frame_number)
            self.frame_time_variable.set(elapsed)
            dpg.render_dearpygui_frame()
            then = time.time()
            elapsed = then - now
            try:
                for p in GLContextNode.pending_contexts:
                    p.create()
                GLContextNode.pending_contexts = []
                for c in GLContextNode.context_list:
                    if c.ready:
                        if not glfw.window_should_close(c.context.window):
                            c.draw()
                deleted = []
                for c in GLContextNode.pending_deletes:
                    c.close()
                    deleted.append(c)
                for c in deleted:
                    GLContextNode.pending_deletes.remove(c)
            except Exception as exc_:
                print('gl_calls', exc_)
            # glfw.poll_events()
#            self.openGL_thread.

