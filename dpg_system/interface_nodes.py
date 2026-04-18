import dearpygui.dearpygui as dpg
import time
import platform
import numpy as np
import torch

from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
from dpg_system.matrix_nodes import RollingBuffer

def register_interface_nodes():
    Node.app.register_node("menu", MenuNode.factory)
    Node.app.register_node("toggle", ToggleNode.factory)
    Node.app.register_node("set_reset", ToggleNode.factory)
    Node.app.register_node("button", ButtonNode.factory)
    Node.app.register_node("b", ButtonNode.factory)
    Node.app.register_node("mouse", MouseNode.factory)
    Node.app.register_node("float", ValueNode.factory)
    Node.app.register_node("int", ValueNode.factory)
    Node.app.register_node("slider", ValueNode.factory)
    Node.app.register_node("message", ValueNode.factory)
    Node.app.register_node("text", ValueNode.factory)
    Node.app.register_node("string", ValueNode.factory)
    Node.app.register_node("list", ValueNode.factory)
    Node.app.register_node("knob", ValueNode.factory)

    # Node.app.register_node('param', ValueNode.factory)
    Node.app.register_node('param_slider', ValueNode.factory)
    Node.app.register_node('param_float', ValueNode.factory)
    Node.app.register_node('param_int', ValueNode.factory)
    Node.app.register_node('param_message', ValueNode.factory)
    Node.app.register_node('param_string', ValueNode.factory)
    Node.app.register_node('param_list', ValueNode.factory)
    Node.app.register_node('param_knob', ValueNode.factory)

    Node.app.register_node('print', PrintNode.factory)
    Node.app.register_node('load_action', LoadActionNode.factory)
    Node.app.register_node('load_bang', LoadActionNode.factory)
    Node.app.register_node('color', ColorPickerNode.factory)
    Node.app.register_node('color_cmy', CMYColorNode.factory)
    Node.app.register_node('vector', Vector2DNode.factory)

    Node.app.register_node('radio', RadioButtonsNode.factory)
    Node.app.register_node('radio_h', RadioButtonsNode.factory)
    Node.app.register_node('radio_v', RadioButtonsNode.factory)
    Node.app.register_node('presets', PresetsNode.factory)
    Node.app.register_node('snapshots', PresetsNode.factory)
    Node.app.register_node('states', PresetsNode.factory)
    Node.app.register_node('archive', PresetsNode.factory)
    Node.app.register_node('versions', PresetsNode.factory)
    Node.app.register_node('gain', GainNode.factory)
    Node.app.register_node('keys', KeyNode.factory)

    Node.app.register_node('table', TableNode.factory)
    Node.app.register_node('momentary_slider', MomentarySliderNode.factory)
    Node.app.register_node('momentary', MomentarySliderNode.factory)
    Node.app.register_node('momentary_slider_int', MomentarySliderNode.factory)
    Node.app.register_node('momentary_int', MomentarySliderNode.factory)
    Node.app.register_node('momentary_xy', XYPadNode.factory)
    Node.app.register_node('joy_stick', XYPadNode.factory)
    Node.app.register_node('envelope', EnvelopeNode.factory)


class ButtonNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ButtonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if label[:4] == 'osc_':
            args = args[2:]
        flash_duration = .100
        self.target_time = time.time() - flash_duration
        self.action_name = ''
        self.action = None

        if args is not None and len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == str:
                self.action_name = v

        self.input = self.add_input('', triggers_execution=True, widget_type='button', widget_width=14, callback=self.clicked_function)
        self.output = self.add_output('')

        self.bound_action = self.add_option('bind to', widget_type='text_input', width=120, default_value=self.action_name, callback=self.binding_changed)
        self.message = self.add_option('message', widget_type='text_input', default_value='bang', callback=self.message_changed)
        self.width = self.add_option('width', widget_type='input_int', default_value=14, min=14, max=None, callback=self.size_changed)
        self.height = self.add_option('height', widget_type='input_int', default_value=14, min=14, max=None, callback=self.size_changed)
        self.flash_duration = self.add_option('flash_duration', widget_type='drag_float', min=0, max=1.0, default_value=flash_duration)

    def size_changed(self):
        dpg.set_item_width(self.input.widget.uuid, self.width())
        dpg.set_item_height(self.input.widget.uuid, self.height())

    def binding_changed(self):
        action_name = self.bound_action()
        if action_name != '':
            a = Node.app.find_action(action_name)
            if a is not None:
                self.action_name = action_name
                self.action = a
                self.input.attach_to_action(a)
                if self.message() == 'bang':
                    size = dpg.get_text_size(self.action_name, font=dpg.get_item_font(self.input.widget.uuid))
                    if size is None:
                        size = [80, 14]
                    dpg.set_item_width(self.input.widget.uuid, int(size[0] * self.app.font_scale_variable()) + 12)
                    dpg.set_item_label(self.input.widget.uuid, self.action_name)
            else:
                self.input.attach_to_action(None)
        else:
            self.input.attach_to_action(None)

    def message_changed(self):
        new_name = self.message()

        if new_name != 'bang':
            dpg.set_item_label(self.input.widget.uuid, new_name)
            width = self.input.widget.get_label_width(minimum_width=14)
            dpg.set_item_width(self.input.widget.uuid, width)
            self.width.set(width)

    def clicked_function(self, input=None):
        self.target_time = time.time() + self.flash_duration()
        dpg.bind_item_theme(self.input.widget.uuid, Node.active_theme)
        self.add_frame_task()

    def custom_create(self, from_file):
        if self.action_name != '':
            self.binding_changed()
        width = self.input.widget.get_label_width(minimum_width=14)
        if width < 14:
            width = 14
        dpg.set_item_width(self.input.widget.uuid, width)

    def custom_cleanup(self):
        self.remove_frame_tasks()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            if dpg.does_item_exist(self.input.widget.uuid):
                dpg.bind_item_theme(self.input.widget.uuid, Node.inactive_theme)
            self.remove_frame_tasks()

    def execute(self):
        self.output.send(self.message())


class MenuNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MenuNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if label[:4] == 'osc_':
            ordered_args = self.ordered_args[2:]
        else:
            ordered_args = self.ordered_args
        self.choices = self.args_as_list(ordered_args)
        self.choice = self.add_input('##choice', widget_type='combo', default_value=self.choices[0], callback=self.set_choice)
        self.choice.widget.combo_items = self.choices
        self.font_size_option = self.add_option('font size', widget_type='combo', default_value='24',
                                                 callback=self.large_font_changed)
        self.font_size_option.widget.combo_items = ['24', '30', '36', '48']
        self.output = self.add_output('')

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.choice()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.choice.widget.set(preset['value'])
            self.execute()

    def large_font_changed(self):
        font_size = self.font_size_option()
        if font_size == '24':
            self.choice.set_font(self.app.font_24)
        elif font_size == '30':
            self.choice.set_font(self.app.font_30)
        elif font_size == '36':
            self.choice.set_font(self.app.font_36)
        elif font_size == '48':
            self.choice.set_font(self.app.font_48)
        adjusted_width = self.choice.widget.adjust_to_text_width()

    def set_choice_internal(self):
        input_choice = self.choice()
        t = type(input_choice)
        do_execute = True
        test_choice = None
        if t == list:
            if len(input_choice) == 1:
                test_choice = input_choice[0]
            else:
                if input_choice[0] == 'set':
                    test_choice = input_choice[1]
                    do_execute = False
                elif input_choice[0] == 'append':
                    for new_choice in input_choice[1:]:
                        if new_choice not in self.choices:
                            self.choices.append(new_choice)
                    dpg.configure_item(self.choice.widget.uuid, items=self.choices)
                    do_execute = False
                else:
                    self.choices = []
                    for new_choice in input_choice:
                        if new_choice not in self.choices:
                            self.choices.append(new_choice)
                    dpg.configure_item(self.choice.widget.uuid, items=self.choices)
                    do_execute = False
        elif t in [int, float, bool]:
            test_choice = str(input_choice)
            if test_choice not in self.choices:
                choice = int(input_choice)
                if choice < len(self.choices):
                    test_choice = self.choices[choice]
        elif t == str:
            test_choice = input_choice
        if test_choice is not None and test_choice in self.choices:
            self.choice.set(test_choice)
        return do_execute

    def set_choice(self):
        do_execute = True
        if self.choice.fresh_input:
            do_execute = self.set_choice_internal()
        if do_execute:
            self.execute()

    def execute(self):
        self.outputs[0].send(self.choice())


class MouseNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MouseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.mouse_pos = None
        self.streaming = False

        self.input = self.add_input('', triggers_execution=True, widget_type='checkbox', widget_width=40, callback=self.start_stop_streaming)
        self.output_x = self.add_output('x')
        self.output_y = self.add_output('y')

    def start_stop_streaming(self, input=None):
        if self.input():
            if not self.streaming:
                self.add_frame_task()
                self.streaming = True
        else:
            if self.streaming:
                self.remove_frame_tasks()
                self.streaming = False

    def frame_task(self):
        if self.input():
            self.mouse_pos = dpg.get_mouse_pos(local=False)
            self.execute()

    def execute(self):
        if self.mouse_pos is not None:
            self.output_y.send(self.mouse_pos[1])
            self.output_x.send(self.mouse_pos[0])


# presets can hold UI state, Nodes state, Patch state

class PresetsNode(Node):
    restoring_patch = False

    @staticmethod
    def factory(name, data, args=None):
        node = PresetsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.preset_count = 8
        self.buttons = []

        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [float, int]:
                self.preset_count = int(v)
        self.input = self.add_input('', triggers_execution=True)
        for i in range(self.preset_count):
            self.buttons.append(i + 1)

        self.radio_group = self.add_property(widget_type='radio_group', callback=self.preset_click)
        self.radio_group.widget.combo_items = self.buttons

        self.output = self.add_output('')
        self.radio_group.widget.horizontal = False

        remember_mode = 'ui'
        if label in ['snapshots', 'states']:
            remember_mode = 'nodes'
        if label in ['archive', 'versions']:
            remember_mode = 'patch'

        self.remember_mode = self.add_option('remember', widget_type='combo', default_value=remember_mode, callback=self.remember_mode_changed)
        self.remember_mode.widget.combo_items = ['ui', 'nodes', 'patch']
        self.presets = [None] * self.preset_count
        self.capturing_patch = False
        self.patch_preset_paste_pending = False
        self.created_nodes = None
        self.preset_clipboard = None

    def remember_mode_changed(self):
        self.presets = [None] * self.preset_count

    def preset_click(self):
        if PresetsNode.restoring_patch:
            return
        if dpg.is_key_down(dpg.mvKey_RShift) or dpg.is_key_down(dpg.mvKey_LShift):
            self.save_preset()
        else:
            self.load_preset()

    def save_preset(self):
        editor = self.my_editor
        remember_mode = self.remember_mode()
        current_preset_index = string_to_int(self.radio_group()) - 1
        if len(self.presets) >= current_preset_index + 1:
            if self.presets[current_preset_index] is None:
                self.presets[current_preset_index] = {}
            if remember_mode == 'patch':
                patch_container = {}
                editor = Node.app.get_current_editor()
                if editor is not None:
                    editor.containerize(patch_container, exclude_list=[self])
                    self.presets[current_preset_index] = patch_container
            else:
                kids = self.output.get_children()
                if len(kids) > 0:
                    for kid in kids:
                        node = kid.node
                        if node is not None:
                            key = str(node.uuid)
                            if remember_mode == 'nodes':
                                properties = {}
                                node.store_properties(properties)
                                if len(properties) > 0:
                                    self.presets[current_preset_index][key] = properties
                            elif remember_mode == 'ui':
                                ui_property = node.get_preset_state()
                                if len(ui_property) > 0:
                                    self.presets[current_preset_index][key] = ui_property
                else:
                    for node in editor._nodes:
                        key = str(node.uuid)
                        if remember_mode == 'nodes':
                            properties = {}
                            node.store_properties(properties)
                            if len(properties) > 0:
                                self.presets[current_preset_index][key] = properties
                        elif remember_mode == 'ui':
                            ui_property = node.get_preset_state()
                            if len(ui_property) > 0:
                                self.presets[current_preset_index][key] = ui_property

    def frame_task(self):
        if self.patch_preset_paste_pending:
            self.do_pending_archive_paste()
        self.remove_frame_tasks()

    def do_pending_archive_paste(self):
        self.patch_preset_paste_pending = False
        editor = Node.app.get_current_editor()
        current_preset_index = string_to_int(self.radio_group()) - 1
        editor.paste(self.presets[current_preset_index], drag=False, origin=True, clear_loaded_uuids=False)
        # on paste, the link ids in the preset will no longer reflect the node id's
        # so they must be updated
        self.created_nodes = self.app.created_nodes.copy()
        editor.paste(self.preset_clipboard, drag=False, origin=True, previously_created_nodes=self.created_nodes)
        editor.clear_loaded_uuids()

    def load_preset(self):
        editor = self.my_editor
        remember_mode = self.remember_mode()
        self.preset_clipboard = self.copy_to_clipboard()
        current_preset_index = string_to_int(self.radio_group()) - 1
        if len(self.presets) >= current_preset_index + 1:
            if self.presets[current_preset_index] is None:
                return
            if remember_mode == 'patch':
                PresetsNode.restoring_patch = True
                try:
                    editor = Node.app.get_current_editor()
                    if editor is not None:
                        editor.remove_all_nodes()
                        self.add_frame_task()
                        self.patch_preset_paste_pending = True
                except Exception as e:
                    print('error restoring patch:')
                    traceback.print_exception(e)

                    # self.app.resume()
                PresetsNode.restoring_patch = False
            else:
                kids = self.output.get_children()
                if len(kids) > 0:
                    for kid in kids:
                        node = kid.node
                        if node is not None:
                            key = str(node.uuid)
                            if key in self.presets[current_preset_index]:
                                if remember_mode == 'nodes' and node != self:
                                    node.restore_properties(self.presets[current_preset_index][key])
                                elif remember_mode == 'ui':
                                    node.set_preset_state(self.presets[current_preset_index][key])
                else:
                    for node in editor._nodes:
                        key = str(node.uuid)
                        if key in self.presets[current_preset_index]:
                            if remember_mode == 'nodes' and node != self:
                                node.restore_properties(self.presets[current_preset_index][key])
                            elif remember_mode == 'ui':
                                node.set_preset_state(self.presets[current_preset_index][key])

    def save_custom(self, container):
        # note this only works for save with the copy()
        # but it does not work for preset action with the copy()
        # problem is that the presets are actually empty when the preset is set
        container['presets'] = self.presets.copy()

    def load_custom(self, container):
        if 'presets' in container:
            self.presets = container['presets'].copy()

    def post_load_callback(self):
        editor = self.my_editor
        remember_mode = self.remember_mode()
        translation_table = {}
        if self.presets is not None:
            for preset in self.presets:
                if preset is not None:
                    if remember_mode == 'patch':
                        if 'nodes' in preset:
                            nodes_container = preset['nodes']
                            for index in nodes_container:
                                node_container = nodes_container[index]
                                if 'id' in node_container:
                                    node_preset_uuid_int = int(node_container['id'])
                                    if node_preset_uuid_int not in translation_table:
                                        for node in editor._nodes:
                                            if node.loaded_uuid == node_preset_uuid_int:
                                                translation_table[node_preset_uuid_int] = node.uuid
                    else:
                        for node_preset_uuid in preset:
                            node_preset_uuid_int = int(node_preset_uuid)
                            if node_preset_uuid_int not in translation_table:
                                for node in editor._nodes:
                                    if node.loaded_uuid == node_preset_uuid_int:
                                        translation_table[node_preset_uuid_int] = node.uuid
                                        break

            adjusted_presets = [None] * self.preset_count
            for index, preset in enumerate(self.presets):
                if preset is not None:
                    if remember_mode == 'patch':
                        if 'nodes' in preset:
                            adjusted_presets[index] = self.presets[index].copy()
                            adjusted_presets[index]['nodes'] = {}

                            nodes_container = preset['nodes']
                            for index_key in nodes_container:
                                node_container = nodes_container[index_key]
                                if 'id' in node_container:
                                    node_preset_uuid_int = int(node_container['id'])
                                    if node_preset_uuid_int in translation_table:
                                        new_uuid = translation_table[node_preset_uuid_int]
                                        node_container['id'] = new_uuid
                                        adjusted_presets[index]['nodes'][new_uuid] = node_container
                                    else:
                                        adjusted_presets[index]['nodes'][node_preset_uuid_int] = node_container
                    else:
                        adjusted_presets[index] = {}
                        for node_preset_uuid in preset:
                            node_preset_uuid_int = int(node_preset_uuid)
                            if node_preset_uuid_int in translation_table:
                                new_uuid = translation_table[node_preset_uuid_int]
                                adjusted_presets[index][str(new_uuid)] = preset[str(node_preset_uuid_int)]

            self.presets = adjusted_presets.copy()
        else:
            print('None presets')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            self.radio_group.widget.set(data)
            self.load_preset()


class TableNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TableNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.columns = 2
        self.rows = 2

        if len(args) > 1:
            self.rows = any_to_int(args[0])
            self.columns = any_to_int(args[1])
        kwargs = {'columns': self.columns, 'rows': self.rows}
        # print(kwargs)

        self.input = self.add_input('array in', widget_type='table', triggers_execution=True, **kwargs)
        self.set_input = self.add_input('set', callback=self.set)
        self.get_input = self.add_input('get', callback=self.get)
        self.output = self.add_output('out')


        self.source = [0.0] * (self.columns * self.rows)
        for i in range(self.rows):
            for j in range(self.columns):
                self.source[i * self.columns + j] = i * self.columns + j

    def set(self):
        incoming = self.set_input()
        if type(incoming) is list:
            if len(incoming) == 2:
                address = incoming[0]
                if type(address) is list and len(address) == 2:
                    row = any_to_int(address[0])
                    column = any_to_int(address[1])
                    value = incoming[1]
                    self.set_cell_widget_value(row, column, value)
            elif len(incoming) == 3:
                row = any_to_int(incoming[0])
                column = any_to_int(incoming[1])
                value = incoming[2]
                self.set_cell_widget_value(row, column, value)

    def get(self):
        incoming = self.get_input()
        if type(incoming) is list:
            if len(incoming) == 1:
                address = incoming[0]
                row = any_to_int(address[0])
                column = any_to_int(address[1])
                value = self.get_cell_widget_value(row, column)
                self.output.send(value)
            elif len(incoming) == 2:
                row = any_to_int(incoming[0])
                column = any_to_int(incoming[1])
                value = self.get_cell_widget_value(row, column)
                self.output.send(value)

    def custom_create(self, from_file):
        for column in range(self.columns):
            for row in range(self.rows):
                self.set_cell_widget_value(row, column, self.source[row * self.columns + column])

    def execute(self):
        incoming = self.input()
        handled = False
        t = type(incoming)
        if t is torch.Tensor:
            incoming = any_to_list(incoming.flatten())
            t = list
        if t is np.ndarray:
            incoming = any_to_list(incoming.ravel())
            t = list
        if t is list:
            if len(incoming) == self.columns:
                if len(incoming[0]) == self.rows:
                    handled = True
                    for row in range(self.rows):
                        for column in range(self.columns):
                            self.set_cell_widget_value(row, column, incoming[row][column])
            if not handled:
                if len(incoming) == self.columns * self.rows:
                    for row in range(self.rows):
                        for column in range(self.columns):
                            self.set_cell_widget_value(row, column, incoming[row * self.columns + column])

    def get_cell_tag(self, row, col):
        return f"cell_{row}_{col}"

    def get_cell_widget_value(self, row, col):
        target_tag = self.get_cell_tag(row, col)
        value = dpg.get_value(target_tag)
        return value

    def set_cell_widget_value(self, row, col, value):
        if row >= 0 and row < self.rows and col >= 0 and col < self.columns:
            target_tag = self.get_cell_tag(row, col)
            dpg.set_value(target_tag, any_to_string(value))


class RadioButtonsNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RadioButtonsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if label[:4] == 'osc_':
            args = args[2:]

        self.buttons = []
        if args is not None and len(args) > 0:
            for i in range(len(args)):
                v, t = decode_arg(args, i)
                self.buttons.append(v)

        self.radio_group = self.add_input(widget_type='radio_group', callback=self.execute)
        self.radio_group.widget.combo_items = self.buttons
        if label == 'radio_h':
            self.radio_group.widget.horizontal = True
        else:
            self.radio_group.widget.horizontal = False
        self.output = self.add_output("")

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.radio_group()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.radio_group.widget.set(preset['value'])
            self.execute()

    def call_execute(self, input=None):
        self.execute()

    def execute(self):
        self.output.send(self.radio_group())


class ToggleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ToggleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if label[:4] == 'osc_':
            ordered_args = self.ordered_args[2:]
        else:
            ordered_args = self.ordered_args
        variable_name = ''
        self.set_reset = False
        if ordered_args is not None and len(ordered_args) > 0:
            for i in range(len(ordered_args)):
                var_name, t = decode_arg(ordered_args, i)
                if t == str:
                    variable_name = var_name
        self.reset_input = None
        self.value = 0
        self.temp_block_output = False
        self.variable = None
        if self.label == 'set_reset':
            self.set_reset = True
            self.input = self.add_input('set', triggers_execution=True)
            self.reset_input = self.add_input('reset', triggers_execution=True)
        else:
            self.input = self.add_input('', triggers_execution=True, widget_type='checkbox', callback=self.call_execute)
        self.input.bang_repeats_previous = False
        self.output = self.add_output('')
        self.bound_variable = self.add_option('bind to', widget_type='text_input', width=120, default_value=variable_name, callback=self.binding_changed)

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.input()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.input.widget.set(preset['value'])
            self.execute()

    def binding_changed(self):
        variable_name = self.bound_variable()
        if self.variable is not None:
            self.variable.detach_client(self)
            self.variable = None
        if variable_name != '':
            v = Node.app.find_variable(variable_name)
            if v is None:
                default = False
                v = Node.app.add_variable(variable_name, default_value=default)
            if v:
                self.variable = v
                self.input.attach_to_variable(v)
                self.variable.attach_client(self)
                self.output.set_label(variable_name)
                self.variable_update()

    def custom_create(self, from_file):
        self.binding_changed()

    def variable_update(self):
        if self.variable is not None:
            data = self.variable.get_value()
            self.input.set(data, propagate=False)
        self.update(propagate=False)

    def update(self, propagate=True):
        value = dpg.get_value(self.value)
        if type(value) == str:
            value = value.split(' ')
            if len(value) == 1:
                value = value[0]
        value = any_to_int(value)
        if self.variable is not None and propagate:
            self.variable.set(value, from_client=self)
        self.outputs[0].send(value)

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def call_execute(self, input=None):
        self.execute()

    # def increment_widget(self, widget):
    #     widget.increment()
    #     self.execute()
    #
    # def decrement_widget(self, widget):
    #     widget.decrement()
    #     self.execute()

    def execute(self):
        if not self.set_reset:
            if self.input.fresh_input:
                received = self.input.get_received_data()     # so that we can catch 'bang' ?
                if type(received) == str and received == 'bang':
                    self.value = 1 - self.value
                    # self.value = not self.value
                    self.input.set(self.value)
                elif type(received) == list and len(received) > 1:
                    if type(received[0] == str):
                        if received[0] == 'set':
                            self.value = any_to_int(received[1])
                            if self.value != 0:
                                self.value = 1
                            self.input.set(self.value, propagate=False)
                            self.temp_block_output = True
                            if self.variable is not None:
                                self.variable.set(self.value, from_client=self)
                            return
                else:
                    self.value = any_to_int(received)
                    if self.value != 0:
                        self.value = 1
                    self.input.set(self.value)
            else:
                self.value = any_to_int(self.input())
                if self.value != 0:
                    self.value = 1
        else:
            if self.active_input == self.input:
                self.value = 1
            elif self.active_input == self.reset_input:
                self.value = 0
            self.output.set_label(str(self.value))
        if self.variable is not None:
            self.variable.set(self.value, from_client=self)
        if not self.temp_block_output:
            self.output.send(self.value)
        else:
            self.temp_block_output = False


class GainNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = GainNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        widget_type = 'slider_float'
        widget_width = 200
        self.value = dpg.generate_uuid()
        self.horizontal = True
        max = 1.0

        if self.ordered_args is not None:
            for i in range(len(self.ordered_args)):
                val, t = decode_arg(self.ordered_args, i)
                if t in [float, int]:
                    max = val

        self.input = self.add_input('', triggers_execution=True)
        self.gain = self.add_property('', widget_type=widget_type, width=widget_width, max=max)
        self.output = self.add_output('')
        self.max = self.add_option('max', widget_type='drag_float', callback=self.max_changed, default_value=max)

    def max_changed(self):
        self.gain.widget.set_limits(0.0, self.max())

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            t = type(data)
            if t is not str:
                self.output.send(data * self.gain())


class ValueNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        base_name = name.split('_')[-1]
        if base_name in ['float']:
            return FloatNode(name, data, args)
        elif base_name in ['int']:
            return IntNode(name, data, args)
        elif base_name in ['slider']:
            return SliderNode(name, data, args)
        elif base_name in ['knob']:
            return KnobNode(name, data, args)
        elif base_name in ['string', 'message', 'list']:
            return StringNode(name, data, args)
        elif base_name == 'text':
            return TextEditorNode(name, data, args)
        else:
            return StringNode(name, data, args)

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)

        self.param_name = None

        # --- Parsing Prefixes ---
        if label.startswith('osc_'):
            self.ordered_args = self.ordered_args[2:]
        elif label.startswith('param_'):
            self.param_name = self.ordered_args[0]
            if len(self.ordered_args) > 0:
                self.ordered_args = self.ordered_args[1:]

        # --- Common State ---
        self.value = dpg.generate_uuid()
        self.variable = None
        self.variable_name = ''
        self.input = None
        self.output = None
        self.width_option = None

        self.grow_mode = 'grow_to_fit'
        self.grow_option = None

        # Initialize specific UI components
        self.setup_specific_ui(self.ordered_args)

        # --- Common Options ---
        if self.ordered_args and len(self.ordered_args) > 0:
            for i in range(len(self.ordered_args)):
                var_name, t = decode_arg(self.ordered_args, i)
                if t == str and var_name != '+':
                    if getattr(self, 'widget_type', '') not in ['input_int', 'input_float']:
                        self.variable_name = var_name

        if self.output is None:
            out_label = self.variable_name if self.variable_name else 'out'
            self.output = self.add_output(out_label)

        self.variable_binding_property = self.add_option(
            'bind to', widget_type='text_input', width=120,
            default_value=self.variable_name, callback=self.binding_changed
        )

        # Widget Width
        default_width = getattr(self, 'widget_width', 100)
        self.width_option = self.add_option(
            'width', widget_type='drag_int', default_value=default_width,
            callback=self.options_changed
        )

        if 'knob' not in label:
            self.large_text_option = self.add_option(
                'font size', widget_type='combo', default_value='24',
                callback=self.large_font_changed
            )
            self.large_text_option.widget.combo_items = ['24', '30', '36', '48']

        if self.param_name is not None:
            self.param_name_option = self.add_option('parameter name', widget_type='text_input', default_value=self.param_name)

    # --- Button Handlers ---
    def increment_widget(self, widget):
        widget.increment()
        self.execute()

    def decrement_widget(self, widget):
        widget.decrement()
        self.execute()

    # --- Base Methods ---
    def setup_specific_ui(self, args):
        pass

    def cast_value(self, value):
        return value

    def large_font_changed(self):
        font_size = self.large_text_option()
        if font_size == '24':
            self.input.set_font(self.app.font_24)
            trigger_size = 14
        elif font_size == '30':
            self.input.set_font(self.app.font_30)
            trigger_size = 17
        elif font_size == '36':
            self.input.set_font(self.app.font_36)
            trigger_size = 20
        elif font_size == '48':
            self.input.set_font(self.app.font_48)
            trigger_size = 28

        adjusted_width = self.input.widget.adjust_to_text_width()
        self.width_option.widget.set(adjusted_width)

        trigger = self.input.widget.trigger_widget
        if trigger:
            dpg.set_item_width(trigger, trigger_size)

    def options_changed(self):
        width = self.width_option()
        dpg.set_item_width(self.input.widget.uuid, width)

    def binding_changed(self):
        binding = self.variable_binding_property()
        self.bind_to_variable(binding)

    def bind_to_variable(self, variable_name):
        if self.variable is not None:
            self.variable.detach_client(self)
            self.variable = None

        if variable_name != '':
            v = Node.app.find_variable(variable_name)
            if v is None:
                default = 0.0 if self.label in ['float', 'slider', 'knob'] else 0
                if self.label in ['string', 'text', 'message', 'list']: default = ''
                v = Node.app.add_variable(variable_name, default_value=default)

            if v:
                self.variable_name = variable_name
                self.variable = v
                self.input.attach_to_variable(v)
                self.variable.attach_client(self)
                self.output.set_label(self.variable_name)
                self.variable_update()

    def variable_update(self):
        if self.variable is not None:
            data = self.variable.get_value()
            self.input.set(data, propagate=False)
        self.update(propagate=False)

    def custom_create(self, from_file):
        if self.variable_name != '':
            self.bind_to_variable(self.variable_name)
        if hasattr(self, 'start_value') and self.start_value is not None:
            self.input.set(self.start_value)
        self.input.set_font(self.app.font_24)

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def get_preset_state(self):
        return {'value': self.input()}

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.input.widget.set(preset['value'])
            self.execute()

    def do_send(self, value):
        if self.param_name:
            output_list = [self.param_name_option(), value]
            self.output.send(output_list)
        else:
            self.outputs[0].send(value)

    def _parse_text_input(self, text_value):
        """
        Robust parsing for list/message nodes.
        Mimics string_to_list and space-splitting logic.
        """
        if 'string' in self.label or 'text' in self.label:
            return text_value

        if not text_value:
            return []
        # if isinstance(text_value, dict):

        output_data = any_to_list(text_value)

        if self.label != 'list':
            if len(output_data) == 1:
                return output_data[0]

        return output_data

    def update(self, propagate=True):
        raw_value = dpg.get_value(self.value)
        processed_value = raw_value

        # Parse text inputs for list/message nodes
        if isinstance(raw_value, str) and getattr(self, 'widget_type', '') in ['text_input', 'text_editor']:
            processed_value = self._parse_text_input(raw_value)

        if self.variable is not None and propagate:
            self.variable.set(processed_value, from_client=self)

        if getattr(self, 'widget_type', '') == 'text_input':
            self.input.widget.adjust_to_text_width(max=2048)
            self._handle_auto_grow()

        if getattr(self, 'power', None) is not None and self.power() != 1.0:
            try:
                processed_value = pow(float(processed_value), self.power())
            except:
                pass

        self.do_send(processed_value)

    def execute(self):
        output_data = None
        should_output = True

        # CASE A: Input from Pipe
        if self.inputs[0].fresh_input:
            in_data = self.inputs[0]()
            processed_data = in_data

            if isinstance(processed_data, list) and len(processed_data) == 1:
                processed_data = processed_data[0]

            if isinstance(processed_data, list) and len(processed_data) == 2 and processed_data[0] == 'set':
                processed_data = processed_data[1]
                should_output = False

            try:
                final_val = self.cast_value(processed_data)

                # Display conversion
                display_val = final_val
                if isinstance(final_val, list) and getattr(self, 'widget_type', '') == 'text_input':
                    display_val = any_to_string(final_val)

                self.input.widget.set(display_val, propagate=False)

                if self.variable is not None:
                    self.variable.set(final_val, from_client=self)

                if should_output:
                    output_data = final_val

            except (ValueError, TypeError):
                display_str = any_to_string(processed_data)
                self.input.widget.set(display_str, propagate=False)
                output_data = processed_data

        # CASE B: Input from GUI (Direct interaction)
        else:
            self.update()
            return

        # Post-processing
        if getattr(self, 'widget_type', '') == 'text_input':
            self._handle_auto_grow()

        if should_output and output_data is not None:
            if getattr(self, 'power', None) is not None and self.power() != 1.0:
                try:
                    output_data = pow(float(output_data), self.power())
                except:
                    pass

            self.do_send(output_data)

    def _handle_auto_grow(self):
        if self.grow_mode in ['grow_to_fit', 'grow_or_shrink_to_fit']:
            adjusted_width = self.input.widget.get_text_width()
            current_opt = self.width_option()
            if self.grow_mode == 'grow_to_fit':
                if adjusted_width > current_opt:
                    dpg.configure_item(self.input.widget.uuid, width=adjusted_width)
                    self.width_option.set(adjusted_width)
            else:
                dpg.configure_item(self.input.widget.uuid, width=adjusted_width)
                self.width_option.set(adjusted_width)


class NumericValueNode(ValueNode):
    def __init__(self, label, data, args):
        self.min = None
        self.max = None
        self.start_value = None
        self.format = '%.3f'
        self.min_property = None
        self.max_property = None
        self.speed_property = None
        self.format_property = None
        ValueNode.__init__(self, label, data, args)

    def create_numeric_options(self):
        if self.widget_type in ['drag_float', 'slider_float', 'input_float', 'knob_float',
                                'drag_int', 'slider_int', 'input_int']:
            w_type = 'drag_int' if 'int' in self.widget_type else 'drag_float'
            self.min_property = self.add_option('min', widget_type=w_type, default_value=self.min,
                                                callback=self.options_changed)
            self.max_property = self.add_option('max', widget_type=w_type, default_value=self.max,
                                                callback=self.options_changed)
            if self.widget_type in ['drag_float', 'drag_int', 'input_float', 'input_int']:
                default_value = 1
                if self.widget_type == 'drag_float':
                    default_value = 0.01
                elif self.widget_type == 'input_float':
                    default_value = 0.1

                self.speed_property = self.add_option('speed_property', widget_type=w_type, default_value=default_value, callback=self.options_changed)
            self.format_property = self.add_option('format', widget_type='text_input', default_value=self.format,
                                                   callback=self.options_changed)

    def options_changed(self):
        ValueNode.options_changed(self)
        if self.min_property and self.max_property:
            self.min = self.min_property()
            self.max = self.max_property()

            current_min = self.min if self.min is not None else 0
            current_max = self.max if self.max is not None else 0

            if current_max > current_min:
                self.input.widget.set_limits(current_min, current_max)
            else:
                # Pass large limits to unbind
                if 'int' in self.widget_type:
                    self.input.widget.set_limits(-2000000000, 2000000000)
                else:
                    self.input.widget.set_limits(-1e15, 1e15)
        if self.speed_property:
            speed = self.speed_property()
            self.input.widget.set_speed(speed)
        if self.format_property:
            self.format = self.format_property()
            self.input.widget.set_format(self.format)


class FloatNode(NumericValueNode):
    def setup_specific_ui(self, args):
        self.format = '%.3f'
        self.widget_type = 'drag_float'
        self.widget_width = 60

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t in [float, int]:
                self.start_value = val
            elif t == str and val == '+':
                self.widget_type = 'input_float'

        self.input = self.add_input('', triggers_execution=True, widget_type=self.widget_type,
                                          widget_uuid=self.value, widget_width=self.widget_width, trigger_button=True)
        if self.param_name is not None:
            self.output = self.add_output(self.param_name + ' out')
        else:
            self.output = self.add_float_output('float out')
        self.create_numeric_options()

    def cast_value(self, value):
        return any_to_float(value)


class IntNode(NumericValueNode):
    def setup_specific_ui(self, args):
        self.format = '%d'
        self.widget_type = 'drag_int'
        self.widget_width = 60

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t in [float, int]:
                self.max = int(val)
            elif t == str and val == '+':
                self.widget_type = 'input_int'

        kwargs = {}
        if self.max is not None:
            kwargs['max'] = self.max
        if self.min is not None:
            kwargs['min'] = self.min

        self.input = self.add_input('', triggers_execution=True, widget_type=self.widget_type,
                                        widget_uuid=self.value, widget_width=self.widget_width, trigger_button=True,
                                        **kwargs)
        if self.param_name is not None:
            self.output = self.add_output(self.param_name + ' out')
        else:
            self.output = self.add_int_output('int out')
        self.create_numeric_options()

    def cast_value(self, value):
        return any_to_int(value)


class SliderNode(NumericValueNode):
    def setup_specific_ui(self, args):
        self.widget_type = 'slider_float'
        self.widget_width = 100
        is_int = False

        if args:
            for i in range(len(args)):
                val, t = decode_arg(args, i)
                if t == float:
                    self.widget_type = 'slider_float'
                    self.max = val
                    self.format = '%.3f'
                elif t == int:
                    self.widget_type = 'slider_int'
                    self.max = val
                    self.format = '%d'
                    is_int = True

        if self.max is None:
            self.max = 100 if is_int else 1.0

        if is_int:
            self.input = self.add_input('', triggers_execution=True, widget_type=self.widget_type,
                                            widget_uuid=self.value, widget_width=self.widget_width,
                                            trigger_button=True, max=self.max)
            if self.param_name is not None:
                self.output = self.add_output(self.param_name + ' out')
            else:
                self.output = self.add_int_output('int out')
        else:
            self.input = self.add_input('', triggers_execution=True, widget_type=self.widget_type,
                                              widget_uuid=self.value, widget_width=self.widget_width,
                                              trigger_button=True, max=self.max)
            if self.param_name is not None:
                self.output = self.add_output(self.param_name + ' out')
            else:
                self.output = self.add_float_output('float out')

        self.create_numeric_options()
        if not is_int:
            self.power = self.add_option('power', widget_type='drag_float', default_value=1.0)

    def cast_value(self, value):
        if 'int' in self.widget_type:
            return any_to_int(value)
        return any_to_float(value)


class KnobNode(NumericValueNode):
    def setup_specific_ui(self, args):
        self.widget_type = 'knob_float'
        self.widget_width = 100
        value_type = float

        if args:
            for i in range(len(args)):
                val, t = decode_arg(args, i)
                if t in [float, int]:
                    self.max = val
                    value_type = t
                    break

        if self.max is None:
            self.max = 1.0 if value_type is float else 100

        if value_type is float:
            self.format = '%.3f'
            self.input = self.add_input('', triggers_execution=True, widget_type='knob_float',
                                              widget_uuid=self.value, widget_width=self.widget_width,
                                              trigger_button=True, max=self.max)
            if self.param_name is not None:
                self.output = self.add_output(self.param_name + ' out')
            else:
                self.output = self.add_float_output('float out')
        else:
            self.format = '%d'
            self.input = self.add_input('', triggers_execution=True, widget_type='knob_float',
                                            widget_uuid=self.value, widget_width=self.widget_width,
                                            trigger_button=True, max=self.max)
            if self.param_name is not None:
                self.output = self.add_output(self.param_name + ' out')
            else:
                self.output = self.add_int_output('int out')

        self.create_numeric_options()

    def cast_value(self, value):
        if self.format == '%d':
            return any_to_int(value)
        return any_to_float(value)


class StringNode(ValueNode):
    def setup_specific_ui(self, args):
        self.widget_type = 'text_input'
        self.widget_width = 100
        self.input = self.add_input('###text in', triggers_execution=True, widget_type=self.widget_type,
                                           widget_uuid=self.value, widget_width=self.widget_width,
                                           trigger_button=True)

        if self.param_name is not None:
            self.output = self.add_output(self.param_name + ' out')
        else:
            if 'list' in self.label:
                self.output = self.add_list_output('list out')
            elif 'message' in self.label:
                self.output = self.add_output('message out')
            else:
                self.output = self.add_string_output('string out')

        self.grow_option = self.add_option('adapt_width', widget_type='combo', width=150,
                                           default_value='grow_to_fit', callback=self.options_changed)
        self.grow_option.widget.combo_items = ['grow_to_fit', 'grow_or_shrink_to_fit', 'fixed_width']

    def options_changed(self):
        super().options_changed()
        if self.grow_option:
            self.grow_mode = self.grow_option()

    def cast_value(self, value):
        if 'string' in self.label:
            return any_to_string(value)
        elif 'list' in self.label or 'message' in self.label:
            return any_to_list(value)
        return str(value)


class TextEditorNode(StringNode):
    def setup_specific_ui(self, args):
        self.widget_type = 'text_editor'
        self.widget_width = 400
        self.input = self.add_string_input('###text in', triggers_execution=True, widget_type=self.widget_type,
                                           widget_uuid=self.value, widget_width=self.widget_width,
                                           trigger_button=True)
        self.input.set_strip_returns(False)
        if self.param_name is not None:
            self.output = self.add_output(self.param_name + ' out')
        else:
            self.output = self.add_string_output('string out')

        self.height_option = self.add_option('height', widget_type='drag_int', default_value=200,
                                             callback=self.options_changed)

    def custom_create(self, from_file):
        super().custom_create(from_file)
        dpg.set_item_height(self.input.widget.uuid, 200)

    def options_changed(self):
        super().options_changed()
        if self.height_option:
            dpg.set_item_height(self.input.widget.uuid, self.height_option())


class ValueNode_o(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ValueNode_o(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if label[:4] == 'osc_':
            ordered_args = self.ordered_args[2:]
        elif label[:3] == 'param_':
            ordered_args = self.ordered_args[1:]
        else:
            ordered_args = self.ordered_args
        widget_type = 'drag_float'
        widget_width = 100
        self.value = dpg.generate_uuid()
        self.horizontal = True
        self.min = None
        self.max = None
        self.format = '%.3f'
        self.variable = None
        self.variable_name = ''
        self.min_property = None
        self.max_property = None
        self.start_value = None
        self.height_option = None
        self.format_property = None
        self.grow_option = None
        self.grow_mode = 'grow_to_fit'
        self.input = None
        output_name = 'out'

        self.output = None

        if label in ['float', 'osc_float', 'param_float']:
            self.format = '%.3f'
            widget_type = 'drag_float'
            for i in range(len(ordered_args)):
                val, t = decode_arg(ordered_args, i)
                if t in [float, int]:
                    self.start_value = val
                elif t == str:
                    if val == '+':
                        widget_type = 'input_float'
            self.input = self.add_float_input('', triggers_execution=True, widget_type=widget_type,
                                            widget_uuid=self.value, widget_width=widget_width, trigger_button=True)
            self.output = self.add_float_output('float out')

        elif label in ['int', 'osc_int', 'param_int']:
            self.format = '%d'
            widget_type = 'drag_int'
            for i in range(len(ordered_args)):
                val, t = decode_arg(ordered_args, i)
                if t in [float, int]:
                    self.max = val
                elif t == str:
                    if val == '+':
                        widget_type = 'input_int'
            if self.max is None:
                self.input = self.add_int_input('', triggers_execution=True, widget_type=widget_type,
                                            widget_uuid=self.value, widget_width=widget_width, trigger_button=True)
            else:
                if self.min is None:
                    self.input = self.add_int_input('', triggers_execution=True, widget_type=widget_type,
                                                widget_uuid=self.value, widget_width=widget_width, max=self.max, trigger_button=True)
                else:
                    self.input = self.add_int_input('', triggers_execution=True, widget_type=widget_type,
                                                widget_uuid=self.value, widget_width=widget_width, max=self.max, min=self.min, trigger_button=True)

            self.output = self.add_int_output('int out')

        elif label in ['slider', 'osc_slider', 'param_slider']:
            widget_type = 'slider_float'
            if ordered_args is not None:
                for i in range(len(ordered_args)):
                    val, t = decode_arg(ordered_args, i)
                    if t == float:
                        widget_type = 'slider_float'
                        self.max = val
                        self.format = '%.3f'
                    elif t == int:
                        widget_type = 'slider_int'
                        self.max = val
                        self.format = '%d'

            if widget_type == 'slider_float':
                if self.max is None:
                    self.max = 1.0
                self.input = self.add_float_input('', triggers_execution=True, widget_type=widget_type,
                                            widget_uuid=self.value, widget_width=widget_width, trigger_button=True,
                                            max=self.max)
                self.output = self.add_float_output('float out')
            else:
                if self.max is None:
                    self.max = 100
                self.input = self.add_int_input('', triggers_execution=True, widget_type=widget_type,
                                                  widget_uuid=self.value, widget_width=widget_width,
                                                  trigger_button=True,
                                                  max=self.max)
                self.output = self.add_int_output('int out')

        elif label in ['knob', 'osc_knob', 'param_knob']:
            widget_type = 'knob_float'
            value_type = float
            if ordered_args is not None:
                for i in range(len(ordered_args)):
                    val, t = decode_arg(ordered_args, i)
                    if t in [float, int]:
                        self.max = val
                        value_type = t
                        break
            if self.max is None:
                if value_type is float:
                    self.max = 1.0
                else:
                    self.max = 100

            if value_type is float:
                self.input = self.add_float_input('', triggers_execution=True, widget_type=widget_type,
                                            widget_uuid=self.value, widget_width=widget_width,
                                            trigger_button=True,
                                            max=self.max)
                self.output = self.add_float_output('float out')
                self.format = '%.3f'
            else:
                self.input = self.add_int_input('', triggers_execution=True, widget_type=widget_type,
                                                  widget_uuid=self.value, widget_width=widget_width,
                                                  trigger_button=True,
                                                  max=self.max)
                self.output = self.add_int_output('int out')
                self.format = '%d'

        elif label in ['string', 'osc_string', 'param_string']:
            widget_type = 'text_input'
            self.input = self.add_string_input('###text in', triggers_execution=True, widget_type=widget_type,
                                        widget_uuid=self.value, widget_width=widget_width,
                                        trigger_button=True)
            self.output = self.add_string_output('string out')

        elif label in ['message', 'osc_message', 'param_message']:
            widget_type = 'text_input'
            self.input = self.add_input('###text in', triggers_execution=True, widget_type=widget_type,
                                              widget_uuid=self.value, widget_width=widget_width,
                                              trigger_button=True)
            self.output = self.add_output('message out')

        elif label in ['list', 'osc_list', 'param_list']:
            widget_type = 'text_input'
            self.input = self.add_input('###text in', triggers_execution=True, widget_type=widget_type,
                                              widget_uuid=self.value, widget_width=widget_width,
                                              trigger_button=True)
            self.output = self.add_list_output('list out')
        elif label in ['text', 'param_text']:
            widget_type = 'text_editor'
            self.input = self.add_string_input('###text in', triggers_execution=True, widget_type=widget_type,
                                        widget_uuid=self.value, widget_width=400,
                                        trigger_button=True)
            self.input.set_strip_returns(False)
            self.output = self.add_string_output('string out')
        if ordered_args is not None and len(ordered_args) > 0:
            for i in range(len(ordered_args)):
                var_name, t = decode_arg(ordered_args, i)
                if t == str:
                    if widget_type not in ['input_int', 'input_float'] or var_name != '+':
                        self.variable_name = var_name
        if self.input is None:
            if self.max is None:
                self.input = self.add_input('', triggers_execution=True, widget_type=widget_type, widget_uuid=self.value, widget_width=widget_width, trigger_button=True)
            else:
                self.input = self.add_input('', triggers_execution=True, widget_type=widget_type, widget_uuid=self.value, widget_width=widget_width, trigger_button=True, max=self.max)
            # print(self.input)

        if self.output is None:
            if self.variable_name != '':
                self.output = self.add_output(self.variable_name)
            else:
                self.output = self.add_output('out')

        self.variable_binding_property = self.add_option('bind to', widget_type='text_input', width=120, default_value=self.variable_name, callback=self.binding_changed)

        if widget_type in ['drag_float', 'slider_float', 'input_float', 'knob_float']:
            self.min_property = self.add_option('min', widget_type='drag_float', default_value=self.min, callback=self.options_changed)
            self.max_property = self.add_option('max', widget_type='drag_float', default_value=self.max, callback=self.options_changed)

        if widget_type in ['drag_int', 'slider_int', "input_int"]:
            self.min_property = self.add_option('min', widget_type='drag_int', default_value=self.min,
                                                callback=self.options_changed)
            self.max_property = self.add_option('max', widget_type='drag_int', default_value=self.max,
                                                callback=self.options_changed)

        if widget_type == 'slider_float':
            self.power = self.add_option('power', widget_type='drag_float', default_value=1.0)

        self.width_option = self.add_option('width', widget_type='drag_int', default_value=widget_width, callback=self.options_changed)
        if widget_type == 'text_input':
            self.grow_option = self.add_option('adapt_width', widget_type='combo', width=150, default_value='grow_to_fit', callback=self.options_changed)
            self.grow_option.widget.combo_items = ['grow_to_fit', 'grow_or_shrink_to_fit', 'fixed_width']
        if widget_type == 'text_editor':
            self.height_option = self.add_option('height', widget_type='drag_int', default_value=200,
                                                callback=self.options_changed)

        if widget_type in ['drag_float', 'slider_float', 'drag_int', 'slider_int', 'knob_float', 'input_int', 'input_float']:
            self.format_property = self.add_option('format', widget_type='text_input', default_value=self.format, callback=self.options_changed)
        if widget_type != 'knob':
            self.large_text_option = self.add_option('font_size', widget_type='combo', default_value='24', callback=self.large_font_changed)
            self.large_text_option.widget.combo_items = ['24', '30', '36', '48']

    def large_font_changed(self):
        trigger_size = 14
        font_size = self.large_text_option()
        if font_size == '24':
            self.input.set_font(self.app.font_24)
            trigger_size = 14
        elif font_size == '30':
            self.input.set_font(self.app.font_30)
            trigger_size = 17
        elif font_size == '36':
            self.input.set_font(self.app.font_36)
            trigger_size = 20
        elif font_size == '48':
            self.input.set_font(self.app.font_48)
            trigger_size = 28

        adjusted_width = self.input.widget.adjust_to_text_width()
        self.width_option.widget.set(adjusted_width)

        if self.input.widget.trigger_widget is not None:
            dpg.set_item_width(self.input.widget.trigger_widget, trigger_size)

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.input()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.input.widget.set(preset['value'])
            self.execute()

    def binding_changed(self):
        binding = self.variable_binding_property()
        self.bind_to_variable(binding)

    def bind_to_variable(self, variable_name):
        # change name
        if self.variable is not None:
            self.variable.detach_client(self)
            self.variable = None
        if variable_name != '':
            v = Node.app.find_variable(variable_name)
            if v is None:
                default = 0.0
                if self.input.widget.widget in ['drag_float', 'slider_float', "knob_float"]:
                    default = 0.0
                elif self.input.widget.widget in ['drag_int', 'slider_int', 'input_int']:
                    default = 0
                elif self.input.widget.widget in ['combo', 'text_input', 'radio_group', 'text_editor']:
                    default = ''
                v = Node.app.add_variable(variable_name, default_value=default)
            if v:
                self.variable_name = variable_name
                self.variable = v
                self.input.attach_to_variable(v)
                self.variable.attach_client(self)
                self.output.set_label(self.variable_name)
                self.variable_update()

    def custom_create(self, from_file):
        if self.variable_name != '':
            self.bind_to_variable(self.variable_name)
        if self.start_value is not None:
            self.input.set(self.start_value)
        self.input.set_font(self.app.font_24)
        if self.input.widget.widget == 'text_editor':
            dpg.set_item_height(self.input.widget.uuid, 200)

    def options_changed(self):
        if self.min_property is not None and self.max_property is not None:
            self.min = self.min_property()
            self.max = self.max_property()
            self.input.widget.set_limits(self.min, self.max)

        if self.format_property is not None:
            self.format = self.format_property()
            self.input.widget.set_format(self.format)

        width = self.width_option()
        dpg.set_item_width(self.input.widget.uuid, width)

        if self.height_option is not None:
            dpg.set_item_height(self.input.widget.uuid, self.height_option())

        if self.grow_option is not None:
            self.grow_mode = self.grow_option()
        # height = self.height_option()
        # dpg.set_item_height(self.input.widget.uuid, height)

    def value_changed(self, force=True):
        pass

    def increment_widget(self, widget):
        widget.increment()
        self.execute()

    def decrement_widget(self, widget):
        widget.decrement()
        self.execute()

    def variable_update(self):
        if self.variable is not None:
            data = self.variable.get_value()
            self.input.set(data, propagate=False)
        self.update(propagate=False)

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def execute(self):
        value = None
        output = True
        display_data = None
        output_data = None
        if self.inputs[0].fresh_input:
            in_data = self.inputs[0]()
            t = type(in_data)
            if t == str:
                if self.label != 'string':
                    in_data, _, types = string_to_hybrid_list(in_data)
                    t = list
                else:
                    display_data = in_data
                    self.input.widget.set(display_data, propagate=False)
                    output_data = display_data
            if t == list:
                if len(in_data) == 1:
                    in_data = in_data[0]
                    t = type(in_data)
                    if is_number(in_data):
                        if self.input.widget.widget in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
                            display_data = in_data
                            display_data = any_to_float(display_data)
                            self.input.widget.set(display_data, propagate=False)
                            output_data = display_data
                        if self.input.widget.widget in ['drag_int', 'input_int', 'slider_int']:
                            display_data = in_data
                            display_data = any_to_int(display_data)
                            self.input.widget.set(display_data, propagate=False)
                            output_data = display_data
                elif len(in_data) > 0:
                    if self.input.widget.widget in ['drag_float', 'drag_int', 'input_float', 'input_int', 'slider_float', 'slider_int', 'knob_float']:
                        if not is_number(in_data[0]):
                            if type(in_data[0]) == str:
                                if in_data[0] == 'set':
                                    if len(in_data) == 2 and is_number(in_data[1]):
                                        display_value = in_data[1]
                                        if self.input.widget.widget in ['drag_float', 'input_float', 'slider_float',
                                                                        'knob_float']:
                                            display_data = any_to_float(display_data)
                                        else:
                                            display_data = any_to_int(display_data)
                                        self.input.widget.set(display_value, propagate=False)
                                        output = False
                        elif len(in_data) > 0:
                            display_data = in_data[0]
                            if is_number(display_data):
                                if self.input.widget.widget in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
                                    display_data = any_to_float(display_data)
                                    self.input.widget.set(display_data, propagate=False)
                                    output_data = display_data
                                if self.input.widget.widget in ['drag_int', 'input_int', 'slider_int']:
                                    display_data = any_to_int(display_data)
                                    self.input.widget.set(display_data, propagate=False)
                                    output_data = display_data
                    else:
                        display_data = in_data
                        output_data = display_data
            if t in [float, int]:
                display_data = in_data
                output_data = display_data
            elif t == bool:
                if in_data:
                    display_data = 1
                else:
                    display_data = 0
                output_data = display_data
            elif t in [np.float32]:
                display_data = float(in_data)
                output_data = display_data
            elif t in [np.int64]:
                display_data = int(in_data)
                output_data = display_data
            elif t == np.ndarray:
                if in_data.dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64]:
                    if self.input.widget.widget in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
                        display_data = in_data
                        display_data = any_to_float(display_data)
                        self.input.widget.set(display_data, propagate=False)
                        output_data = display_data
                    elif self.input.widget.widget in ['drag_int', 'input_int', 'slider_int']:
                        display_data = in_data
                        display_data = any_to_int(display_data)
                        self.input.widget.set(display_data, propagate=False)
                        output_data = display_data
                    elif self.input.widget.widget in ['text_input', 'text_editor']:
                        display_data = any_to_string(in_data)
                        self.input.widget.set(display_data, propagate=False)
                        if self.label == 'string':
                            output_data = display_data
                        elif self.label in ['list', 'message']:
                            output_data = any_to_list(in_data)
            else:
                if self.input.widget.widget in ['text_input', 'text_editor']:
                    if t == np.ndarray:
                        display_data = in_data.tolist()
                        output_data = in_data
                    elif Node.app.torch_available and t == torch.Tensor:
                        display_data = in_data.tolist()
                        output_data = in_data
                    elif t == list:
                        if self.label == 'string':
                            display_data = any_to_string(in_data)
                            output_data = display_data
                            self.data = output_data
                        else:
                            if len(in_data) > 0 and type(in_data[0]) == list:
                                display_data = in_data
                                output_data = display_data
                            else:
                                display_data = any_to_string(in_data)
                                output_data = in_data
                    else:
                        display_data = any_to_string(in_data)
                        output_data = in_data
            if self.variable is not None and output_data is not None:
                self.variable.set(output_data, from_client=self)  # !!!!!
        else:
            output_data = dpg.get_value(self.value)
            self.input.set(output_data, propagate=False)
            if type(output_data) == str:
                if self.output.output_type is not str:
                    if len(output_data) > 0:
                        is_list = False
                        if output_data[0] == '[':
                            try:
                                output_data = string_to_list(output_data)
                                is_list = True
                            except:
                                pass
                        if not is_list:
                            output_data = output_data.split(' ')
                            if len(output_data) == 1:
                                output_value = output_data[0]
                                output_data = output_value
                if self.input.widget.widget in ['drag_float', 'drag_int', 'input_float', 'input_int', 'slider_float', 'slider_int', 'knob_float']:
                    if not is_number(output_data):
                        return

            if self.variable is not None:
                self.variable.set(output_data, from_client=self)
        if self.input.widget.widget == 'text_input':
            adjusted_width = self.input.widget.get_text_width()
            if self.grow_mode == 'grow_to_fit':
                if adjusted_width > self.width_option():
                    dpg.configure_item(self.input.widget.uuid, width=adjusted_width)
                    self.width_option.set(adjusted_width)
            elif self.grow_mode == 'grow_or_shrink_to_fit':
                dpg.configure_item(self.input.widget.uuid, width=adjusted_width)
                self.width_option.set(adjusted_width)
        if output and output_data is not None:
            if self.input.widget.widget == 'slider_float':
                if self.power() != 1.0:
                    output_data = pow(output_data, self.power())
            elif self.label == 'message':
                if type(output_data) == list:
                    if len(output_data) == 1:
                        output_data = output_data[0]
            self.do_send(output_data)

    def update(self, propagate=True):
        value = dpg.get_value(self.value)
        if type(value) == str:
            if self.input.widget.widget == 'text_input':
                value = value.split(' ')
                if len(value) == 1:
                    value = value[0]
        if self.variable is not None and propagate:
            self.variable.set(value, from_client=self)

        if self.input.widget.widget == 'text_input':
            self.input.widget.adjust_to_text_width(max=2048)
        if self.input.widget.widget == 'slider_float':
            if self.power() != 1.0:
                value = pow(value, self.power())
        self.do_send(value)

    def do_send(self, value):
        self.outputs[0].send(value)


# class VectorNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = VectorNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#
#         self.max_component_count = 64
#         if len(args) > 0:
#             self.max_component_count = any_to_int(args[0])
#         self.format = '%.3f'
#
#         self.current_component_count = self.arg_as_int(default_value=4)
#
#         self.input = self.add_input('in', triggers_execution=True)
#         self.input.bang_repeats_previous = False
#         self.zero_input = self.add_input('zero', widget_type='button', callback=self.zero)
#         self.vector_format_input = self.add_input('###vector format', widget_type='combo', default_value='numpy', callback=self.vector_format_changed)
#         if Node.app.torch_available:
#             self.vector_format_input.widget.combo_items = ['numpy', 'torch', 'list']
#         else:
#             self.vector_format_input.widget.combo_items = ['numpy', 'list']
#         self.output_vector = None
#         self.component_properties = []
#         for i in range(self.max_component_count):
#             cp = self.add_input('##' + str(i), widget_type='drag_float', callback=self.component_changed)
#             self.component_properties.append(cp)
#
#         self.output = self.add_output('out')
#
#         self.component_count_property = self.add_option('component count', widget_type='drag_int', default_value=self.current_component_count, callback=self.component_count_changed)
#         self.format_option = self.add_option(label='number format', widget_type='text_input', default_value=self.format, callback=self.change_format)
#         self.output_vector = np.zeros(self.current_component_count)
#
#         self.first_component_input_index = -1
#
#     def vector_format_changed(self):
#         t = type(self.output_vector)
#         vf = self.vector_format_input()
#
#         if t == np.ndarray:
#             if vf == 'torch':
#                 self.output_vector = torch.from_numpy(self.output_vector)
#             elif vf == 'list':
#                 self.output_vector = self.output_vector.tolist()
#         elif t == torch.Tensor:
#             if vf == 'numpy':
#                 self.output_vector = torch.numpy(self.output_vector)
#             elif vf == 'list':
#                 self.output_vector = self.output_vector.tolist()
#         elif t == list:
#             if vf == 'numpy':
#                 self.output_vector = np.array(self.output_vector)
#             elif vf == 'torch':
#                 self.output_vector = torch.tensor(self.output_vector)
#
#     def zero(self):
#         if self.vector_format_input() == 'numpy':
#             self.output_vector = np.zeros(self.current_component_count)
#         elif self.vector_format_input() == 'torch':
#             self.output_vector = torch.zeros(self.current_component_count)
#         else:
#             self.output_vector = [0.0] * self.current_component_count
#         self.execute()
#
#     def get_preset_state(self):
#         preset = {}
#         values = []
#         for i in range(self.current_component_count):
#             values.append(self.component_properties[i]())
#         preset['values'] = values
#         return preset
#
#     def set_preset_state(self, preset):
#         if 'values' in preset:
#             values = preset['values']
#             count = len(values)
#             if count != self.current_component_count:
#                 self.component_count_property.set(count)
#                 self.component_count_changed()
#             for i in range(self.current_component_count):
#                 self.component_properties[i].widget.set(values[i])
#             self.execute()
#
#     def custom_create(self, from_file):
#         for i in range(self.max_component_count):
#             if i < self.current_component_count:
#                 dpg.show_item(self.component_properties[i].uuid)
#             else:
#                 dpg.hide_item(self.component_properties[i].uuid)
#         self.first_component_input_index = self.component_properties[0].input_index
#
#     def component_count_changed(self):
#         self.current_component_count = self.component_count_property()
#         if self.current_component_count > self.max_component_count:
#             self.current_component_count = self.max_component_count
#             self.component_count_property.set(self.current_component_count)
#         for i in range(self.max_component_count):
#             if i < self.current_component_count:
#                 dpg.show_item(self.component_properties[i].uuid)
#             else:
#                 dpg.hide_item(self.component_properties[i].uuid)
#
#     def component_changed(self):
#         self.execute()
#
#     def change_format(self):
#         self.format = self.format_option()
#         for i in range(self.max_component_count):
#             dpg.configure_item(self.component_properties[i].widget.uuid, format=self.format)
#
#     def execute(self):
#         if self.input.fresh_input:
#             value = self.input()
#             t = type(value)
#             if t == str:
#                 if value == 'bang':
#                     output_array = np.ndarray(self.current_component_count)
#                     for i in range(self.current_component_count):
#                         output_array[i] = self.component_properties[i]()
#                     self.output.set_value(output_array)
#                 else:
#                     if self.vector_format_input() == 'list':
#                         value = string_to_list(value)
#                         t = list
#                     elif self.vector_format_input() == 'numpy':
#                         value = string_to_array(value)
#                         t = np.ndarray
#                     elif self.vector_format_input() == 'torch':
#                         value = string_to_tensor(value)
#                         t = torch.tensor
#             if t == list:
#                 value = any_to_numerical_list(value)
#                 if self.vector_format_input() == 'list':
#                     self.output_vector = value.copy()
#                 elif self.vector_format_input() == 'numpy':
#                     self.output_vector = np.array(value)
#                 elif self.vector_format_input() == 'torch':
#                     self.output_vector = torch.tensor(value)
#
#             elif t in [float, int, np.double, np.int64]:
#                 if self.vector_format_input() == 'list':
#                     self.output_vector = [value]
#                 elif self.vector_format_input() == 'numpy':
#                     self.output_vector = np.array([value])
#                 elif self.vector_format_input() == 'torch':
#                     self.output_vector = torch.tensor([value])
#
#             elif t == np.ndarray:
#                 if self.vector_format_input() == 'list':
#                     self.output_vector = value.tolist()
#                 elif self.vector_format_input() == 'numpy':
#                     self.output_vector = value.copy()
#                 elif self.vector_format_input() == 'torch':
#                     self.output_vector = torch.from_numpy(value)
#
#             elif t == torch.Tensor:
#                 if self.vector_format_input() == 'list':
#                     self.output_vector = value.tolist()
#                 elif self.vector_format_input() == 'numpy':
#                     self.output_vector = value.numpy()
#                 elif self.vector_format_input() == 'torch':
#                     self.output_vector = value.clone()
#
#             if type(self.output_vector) == np.ndarray:
#                 if self.current_component_count != self.output_vector.size:
#                     self.component_count_property.set(self.output_vector.size)
#             elif type(self.output_vector) == torch.Tensor:
#                 if self.current_component_count != self.output_vector.numel():
#                     self.component_count_property.set(self.output_vector.numel())
#             elif type(self.output_vector) == list:
#                 if self.current_component_count != len(self.output_vector):
#                     self.component_count_property.set(len(self.output_vector))
#             self.current_component_count = self.component_count_property()
#
#             if self.current_component_count > self.max_component_count:
#                 self.current_component_count = self.max_component_count
#             for i in range(self.max_component_count):
#                 if i < self.current_component_count:
#                     dpg.show_item(self.component_properties[i].uuid)
#                     self.component_properties[i].set(any_to_float(self.output_vector[i]))
#                 else:
#                     dpg.hide_item(self.component_properties[i].uuid)
#                 self.output.set_value(self.output_vector)
#         else:
#             did_set = False
#             if self.active_input is not None:
#                 which = self.active_input.input_index - self.first_component_input_index
#                 if which >= 0:
#                     if which < self.current_component_count:
#                         self.output_vector[which] = self.component_properties[which]()
#                         did_set = True
#             # elif self.vector_format_input() == 'torch':
#             #     self.output_vector[which] = self.component_properties[which]()
#             # else:
#             #     self.output_vector[which] = self.component_properties[which]()
#                 self.output.set_value(self.output_vector)
#             if not did_set:
#                 for i in range(self.current_component_count):
#                     self.component_properties[i].set(any_to_float(self.output_vector[i]))
#                 self.output.set_value(self.output_vector)
#         self.output.send()

class Vector2DNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = Vector2DNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.max_component_count = 64
        dim1 = 4
        dim2 = 1
        if len(args) > 0:
            dim1 = int(args[0])
        if len(args) > 1:
            dim2 = int(args[1])

        self.format = '%.3f'

        self.current_dims = [dim1, dim2]

        self.input = self.add_input('in', triggers_execution=True, trigger_button=True, trigger_callback=self.send)
        self.input.bang_repeats_previous = False
        self.output_vector = None
        self.component_properties = []
        if dim2 > 8:
            dim2 = 8
        self.component_widget_width = 45
        kwargs = {'columns': dim2}
        for i in range(self.max_component_count):
            cp = self.add_input('[' + str(i) + ']', widget_type='drag_float_n', widget_width=self.component_widget_width, callback=self.component_changed, **kwargs)
            cp.name_archive.append('row ' + str(i))
            self.component_properties.append(cp)

        self.zero_input = self.add_input('zero', widget_type='button', callback=self.zero)
        self.vector_format_input = self.add_input('output type', widget_type='combo', default_value='numpy', callback=self.vector_format_changed)
        if Node.app.torch_available:
            self.vector_format_input.widget.combo_items = ['numpy', 'torch', 'list']
        else:
            self.vector_format_input.widget.combo_items = ['numpy', 'list']
        self.output = self.add_output('out')

        self.component_count_property = self.add_option('component count', widget_type='drag_int', default_value=self.current_dims[0], callback=self.component_count_changed)
        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value=self.format, callback=self.change_format)
        self.all_inputs_trigger_option = self.add_option('all inputs trigger', widget_type='checkbox', default_value=True)
        self.width_option = self.add_option('width', widget_type='drag_int', default_value=self.component_widget_width, callback=self.width_changed)
        self.output_vector = np.zeros(self.current_dims)

        self.first_component_input_index = -1

    def vector_format_changed(self):
        t = type(self.output_vector)
        vf = self.vector_format_input()

        if t == np.ndarray:
            if vf == 'torch':
                self.output_vector = torch.from_numpy(self.output_vector)
            elif vf == 'list':
                self.output_vector = self.output_vector.tolist()
        elif t == torch.Tensor:
            if vf == 'numpy':
                self.output_vector = self.output_vector.numpy()
            elif vf == 'list':
                self.output_vector = self.output_vector.tolist()
        elif t == list:
            if vf == 'numpy':
                self.output_vector = np.array(self.output_vector)
            elif vf == 'torch':
                self.output_vector = torch.tensor(self.output_vector)

    def zero(self):
        not_zeroed = True
        if self.vector_format_input() == 'numpy':
            if self.current_dims[0] == self.output_vector.shape[0]:
                if self.current_dims[1] == 1 and len(self.output_vector.shape) == 1:
                    self.output_vector = np.zeros(self.current_dims[0])
                    not_zeroed = False
            if not_zeroed:
                self.output_vector = np.zeros(self.current_dims)

        elif self.vector_format_input() == 'torch':
            if self.current_dims[0] == self.output_vector.shape[0]:
                if self.current_dims[1] == 1 and len(self.output_vector.shape) == 1:
                    self.output_vector = torch.zeros(self.current_dims[0])
                    not_zeroed = False
            if not_zeroed:
                self.output_vector = torch.zeros(self.current_dims)
        else:
            if self.current_dims[0] == len(self.output_vector):
                if self.current_dims[1] == 1 and not isinstance(self.output_vector[0], list):
                    self.output_vector = [0.0] * self.current_dims[0]
                    not_zeroed = False
            if not_zeroed:
                self.output_vector = [[0.0] * self.current_dims[0]] * self.current_dims[1]
        self.execute()

    def get_preset_state(self):
        preset = {}
        values = []
        for i in range(self.current_dims[0]):
            values.append(list(self.output_vector[i]))
        preset['values'] = values
        return preset

    def set_preset_state(self, preset):
        if 'values' in preset:
            values = preset['values']
            self.input._data = values
            self.input.fresh_input = True
            self.execute()

    def custom_create(self, from_file):
        for i in range(self.max_component_count):
            if i < self.current_dims[0]:
                dpg.show_item(self.component_properties[i].uuid)
                for uuid in self.component_properties[i].widget.uuids:
                    dpg.show_item(uuid)
            else:
                dpg.hide_item(self.component_properties[i].uuid)
                for uuid in self.component_properties[i].widget.uuids:
                    dpg.hide_item(uuid)
        self.first_component_input_index = self.component_properties[0].input_index

    def component_count_changed(self):
        self.current_dims[0] = self.component_count_property()
        for i in range(self.max_component_count):
            if i < self.current_dims[0]:
                dpg.show_item(self.component_properties[i].uuid)
                for uuid in self.component_properties[i].widget.uuids:
                    dpg.show_item(uuid)
            else:
                dpg.hide_item(self.component_properties[i].uuid)
                for uuid in self.component_properties[i].widget.uuids:
                    dpg.hide_item(uuid)
        # if type(self.output_vector) == np.ndarray:
        #     if tuple(self.current_dims) != self.output_vector.shape:
        #         self.component_count_property.set(self.output_vector.shape[0])
        # elif type(self.output_vector) == torch.Tensor:
        #     if self.current_dims != self.output_vector.shape:
        #         self.component_count_property.set(self.output_vector.shape[0])
        # elif type(self.output_vector) == list:
        #     if self.current_dims != len(self.output_vector):
        #         self.component_count_property.set(len(self.output_vector))

    def component_changed(self):
        if self.first_component_input_index != -1:
            input = self.active_input()
            self.active_input.widget.set(any_to_list(input))
            if self.all_inputs_trigger_option():
                self.execute()

    def change_format(self):
        self.format = self.format_option()
        for i in range(self.max_component_count):
            for uuid in self.component_properties[i].widget.uuids:
                dpg.configure_item(uuid, format=self.format)

    def width_changed(self):
        width = self.width_option()
        for i in range(self.max_component_count):
            for uuid in self.component_properties[i].widget.uuids:
                dpg.configure_item(uuid, width=width)

    def send(self):
        output_array = np.ndarray(self.current_dims)
        for i in range(self.current_dims[0]):
            for j in range(self.current_dims[1]):
                output_array[i] = self.component_properties[i]()
        self.output.send(output_array)

    def execute(self):
        if self.input.fresh_input:
            value = self.input()
            t = type(value)
            if t == str:
                if value == 'bang':
                    output_array = np.ndarray(self.current_dims)
                    for i in range(self.current_dims[0]):
                        for j in range(self.current_dims[1]):
                            output_array[i] = self.component_properties[i]()
                    self.output.send(output_array)
                    return
                else:
                    if self.vector_format_input() == 'list':
                        value = string_to_list(value)
                        t = list
                    elif self.vector_format_input() == 'numpy':
                        value = string_to_array(value)
                        t = np.ndarray
                    elif self.vector_format_input() == 'torch':
                        value = string_to_tensor(value)
                        t = torch.tensor
            if t == list:
                dim1 = len(value)
                dim2 = 1
                if type(value[0]) is list:
                    dim2 = len(value[0])
                new_dims = [dim1, dim2]
                if new_dims != self.current_dims:
                    self.current_dims = new_dims
                value = any_to_numerical_list(value)
                if self.vector_format_input() == 'list':
                    self.output_vector = value.copy()
                elif self.vector_format_input() == 'numpy':
                    self.output_vector = np.array(value)
                elif self.vector_format_input() == 'torch':
                    self.output_vector = torch.tensor(value)

            elif t in [float, int, np.double, np.int64]:
                if self.vector_format_input() == 'list':
                    self.output_vector = [value]
                elif self.vector_format_input() == 'numpy':
                    self.output_vector = np.array([value])
                elif self.vector_format_input() == 'torch':
                    self.output_vector = torch.tensor([value])
                self.current_dims = [1, 1]

            elif t == np.ndarray:
                self.current_dims = list(value.shape)
                if self.vector_format_input() == 'list':
                    self.output_vector = value.tolist()
                elif self.vector_format_input() == 'numpy':
                    self.output_vector = value.copy()
                elif self.vector_format_input() == 'torch':
                    self.output_vector = torch.from_numpy(value)

            elif t == torch.Tensor:
                self.current_dims = list(value.shape)
                if self.vector_format_input() == 'list':
                    self.output_vector = value.tolist()
                elif self.vector_format_input() == 'numpy':
                    self.output_vector = value.numpy()
                elif self.vector_format_input() == 'torch':
                    self.output_vector = value.clone()

            if type(self.output_vector) == np.ndarray:
                if tuple(self.current_dims) != self.output_vector.shape or self.component_count_property() != self.current_dims[0]:
                    self.component_count_property.set(self.output_vector.shape[0])
            elif type(self.output_vector) == torch.Tensor:
                if self.current_dims != self.output_vector.shape or self.component_count_property() != self.current_dims[0]:
                    self.component_count_property.set(self.output_vector.shape[0])
            elif type(self.output_vector) == list:
                if self.current_dims != len(self.output_vector) or self.component_count_property() != self.current_dims[0]:
                    self.component_count_property.set(len(self.output_vector))
            # self.current_component_count = self.component_count_property()

            if self.current_dims[0] > self.max_component_count:
                self.current_dims[0] = self.max_component_count
            for i in range(self.max_component_count):
                if i < self.current_dims[0]:
                    dpg.show_item(self.component_properties[i].uuid)
                    for uuid in self.component_properties[i].widget.uuids:
                        dpg.show_item(uuid)
                    self.component_properties[i].set(any_to_list(self.output_vector[i]))
                else:
                    dpg.hide_item(self.component_properties[i].uuid)
                    for uuid in self.component_properties[i].widget.uuids:
                        dpg.hide_item(uuid)
                self.output.set_value(self.output_vector)
        else:
            did_set = False
            if self.active_input is not None:
                which = self.active_input.input_index - self.first_component_input_index
                if which >= 0:
                    if which < self.current_dims[0]:
                        if self.vector_format_input() == 'torch':
                            self.output_vector[which] = torch.tensor(self.component_properties[which]())
                        elif self.vector_format_input() == 'numpy':
                            self.output_vector[which] = np.array(self.component_properties[which]())
                        else:
                            self.output_vector[which] = self.component_properties[which]()
                        did_set = True
            # elif self.vector_format_input() == 'torch':
            #     self.output_vector[which] = self.component_properties[which]()
            # else:
            #     self.output_vector[which] = self.component_properties[which]()
                self.output.set_value(self.output_vector)
            if not did_set:
                for i in range(self.current_dims[0]):
                    self.component_properties[i].set(any_to_list(self.output_vector[i]))
                self.output.set_value(self.output_vector)
        self.output.send()


class PrintNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PrintNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.identifier = ''
        if len(args) > 0:
            self.identifier = ' '.join(args)
        self.precision = 3
        self.format_string = '{:.3f}'
        if self.identifier != '':
            self.input = self.add_input(self.identifier, triggers_execution=True)
        else:
            self.input = self.add_input('in', triggers_execution=True)
        self.input.bang_repeats_previous = False
        self.identifier_option = self.add_option('identifier', widget_type='text_input', default_value=self.identifier, callback=self.identifier_changed)
        self.precision = self.add_option(label='precision', widget_type='drag_int', default_value=self.precision, min=0, max=32, callback=self.change_format)
        self.end = self.add_option(label='end', widget_type='text_input', default_value='\n', callback=self.end_changed)

    def end_changed(self):
        end = self.end()
        if end is None:
            end = '\n'
        print('end len', len(end))
        for i in range(len(end)):
            print(end[i])
        if end in ['\\n', '\n']:
            print('setting \\n')
            self.end.set('\n')
        self.end = self.add_option(label='end', widget_type='text_input', default_value='\\n')

    def identifier_changed(self):
        self.identifier = any_to_string(self.identifier_option())
        self.input.set_label(self.identifier)

    def change_format(self):
        precision = self.precision()
        if precision < 0:
            precision = 0
            self.precision.set(precision)
        self.format_string = '{:.' + str(precision) + 'f}'

    def print_list(self, in_list):
        print('[', end='')
        n = len(in_list)
        end = ' '
        for i, d in enumerate(in_list):
            if i == n - 1:
                end = ''
            tt = type(d)
            if tt in [int, np.int64, bool, np.bool_, str]:
                print(d, end=end)
            elif tt in [float, np.double]:
                print(self.format_string.format(d), end=end)
            elif tt == list:
                self.print_list(d)
            elif tt == np.ndarray:
                np.set_printoptions(precision=self.precision())
                print(d)
            elif self.app.torch_available and tt == torch.Tensor:
                torch.set_printoptions(precision=self.precision())
                print(d)
        print(']', end=end)

    def execute(self):
        data = self.input()
        t = type(data)
        end = self.end()
        if end == '\\n':
            end = '\n'
        if self.identifier != '':
            print(self.identifier, end=': ')
        if t in [int, np.int64, bool, np.bool_, str]:
            print(data, end=end)
        elif t in [float, np.double]:
            print(self.format_string.format(data), end=end)
        elif t is list:
            self.print_list(data)
            print('', end=end)
        elif t is np.ndarray:
            np.set_printoptions(precision=self.precision())
            print(data)
        elif t is dict:
            print(data)
        elif self.app.torch_available and t is torch.Tensor:
            torch.set_printoptions(precision=self.precision())
            print(data, end=end)


class LoadActionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = LoadActionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.first_time = True
        load_bang = False
        if self.label == 'load_bang':
            load_bang = True

        if len(args) == 1 and args[0] == 'bang':
            load_bang = True

        if len(args) > 0 and not load_bang:
            self.message = []
            for arg in args:
                self.message.append(arg)
                message_string = ' '.join(self.message)
        else:
            self.message = 'bang'
            message_string = 'bang'

        self.input = self.add_input('trigger', widget_type='button', triggers_execution=True)
        if not load_bang:
            self.load_action = self.add_property(label='##loadActionString', widget_type='text_input', default_value=message_string, callback=self.action_changed)
        self.output = self.add_output("out")

    def action_changed(self):
        action = self.load_action()
        if action == 'bang':
            self.message = 'bang'
        else:
            self.message = action.split(' ')

    def frame_task(self):
        if self.first_time:
            self.first_time = False
            self.first_time = False
            self.remove_frame_tasks()
            self.output.send(self.message)

    def custom_create(self, from_file):
        self.add_frame_task()

    def execute(self):
        self.output.send(self.message)


class ColorPickerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ColorPickerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.wheel = True
        self.alpha = True
        self.has_inputs = False

        self.input = self.add_input('##color', triggers_execution=True, widget_type='color_picker', widget_width=128, callback=self.color_changed)
        self.output = self.add_output('')
        self.hue_wheel_option = self.add_option('hue_wheel', widget_type='checkbox', default_value=self.wheel, callback=self.hue_wheel_changed)
        self.alpha_option = self.add_option('alpha', widget_type='checkbox', default_value=self.alpha, callback=self.alpha_changed)
        self.inputs_option = self.add_option('inputs', widget_type='checkbox', default_value=self.has_inputs, callback=self.inputs_changed)

    def inputs_changed(self):
        has_inputs = self.inputs_option()
        if has_inputs != self.has_inputs:
            if has_inputs:
                dpg.configure_item(self.input.widget.uuid, no_inputs=False)
            else:
                dpg.configure_item(self.input.widget.uuid, no_inputs=True)
            self.has_inputs = has_inputs

    def hue_wheel_changed(self):
        wheel = self.hue_wheel_option()
        if wheel != self.wheel:
            if wheel:
                dpg.configure_item(self.input.widget.uuid, picker_mode=dpg.mvColorPicker_wheel)
            else:
                dpg.configure_item(self.input.widget.uuid, picker_mode=dpg.mvColorPicker_bar)
            self.wheel = wheel

    def alpha_changed(self):
        alpha = self.alpha_option()
        if alpha != self.alpha:
            if alpha:
                dpg.configure_item(self.input.widget.uuid, no_alpha=False)
                dpg.configure_item(self.input.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf)
            else:
                dpg.configure_item(self.input.widget.uuid, no_alpha=True)
                dpg.configure_item(self.input.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)
            self.alpha = alpha

    def color_changed(self):
        self.execute()

    def get_preset_state(self):
        preset = {}
        preset['color'] = list(self.input())
        return preset

    def set_preset_state(self, preset):
        if 'color' in preset:
            color_val = preset['color']
            self.input.widget.set(tuple(color_val))
            self.execute()

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            values = any_to_array(self.input())
            values *= 256.0
            self.input.widget.set(values)
        else:
            values = any_to_array(self.input())
        data = values / 256
        self.output.send(data)

    # def post_creation_callback(self):
    #     print(self.input())


class CMYColorNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CMYColorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.cyan_val = 0
        self.magenta_val = 0
        self.yellow_val = 0
        self._updating = False  # guard against feedback loops

        self.picker_tag = dpg.generate_uuid()

        self.input = self.add_input('cmy in', triggers_execution=True)
        self.cyan_input = self.add_input('cyan', widget_type='slider_int', widget_width=120,
                                         min=0, max=100, default_value=self.cyan_val,
                                         callback=self.slider_changed)
        self.magenta_input = self.add_input('magenta', widget_type='slider_int', widget_width=120,
                                            min=0, max=100, default_value=self.magenta_val,
                                            callback=self.slider_changed)
        self.yellow_input = self.add_input('yellow', widget_type='slider_int', widget_width=120,
                                           min=0, max=100, default_value=self.yellow_val,
                                           callback=self.slider_changed)

        # Color picker display
        self.picker_display = self.add_display('')
        self.picker_display.submit_callback = self.submit_picker

        self.cmy_output = self.add_output('cmy')

    def submit_picker(self):
        rgb = self._cmy_to_rgb_float()
        dpg.add_color_picker(label='##cmy_picker', tag=self.picker_tag,
                             width=128, default_value=(rgb[0], rgb[1], rgb[2], 1.0),
                             picker_mode=dpg.mvColorPicker_wheel,
                             no_alpha=True, no_side_preview=False,
                             no_inputs=True, no_small_preview=False,
                             display_type=dpg.mvColorEdit_float,
                             callback=self._picker_changed)

    def _cmy_to_rgb_float(self):
        """Convert CMY (0-100) to RGB (0.0-1.0) for the picker."""
        r = (1.0 - self.cyan_val / 100.0) * 255
        g = (1.0 - self.magenta_val / 100.0) * 255
        b = (1.0 - self.yellow_val / 100.0) * 255
        return (r, g, b, 255.0)

    def _update_picker(self):
        """Update the color picker to reflect current CMY values."""
        if dpg.does_item_exist(self.picker_tag):
            rgb = self._cmy_to_rgb_float()
            dpg.set_value(self.picker_tag, rgb)

    def _update_sliders(self):
        """Update the CMY sliders to reflect current CMY values."""
        self.cyan_input.widget.set(self.cyan_val)
        self.magenta_input.widget.set(self.magenta_val)
        self.yellow_input.widget.set(self.yellow_val)

    def _picker_changed(self, sender, app_data):
        """Called when the color picker is interacted with directly."""
        if self._updating:
            return
        self._updating = True
        # app_data is (R, G, B, A) in 0.0-1.0 range
        r, g, b = app_data[0], app_data[1], app_data[2]
        self.cyan_val = int(round((1.0 - r) * 100.0))
        self.magenta_val = int(round((1.0 - g) * 100.0))
        self.yellow_val = int(round((1.0 - b) * 100.0))
        self._update_sliders()
        self._send_outputs()
        self._updating = False

    def slider_changed(self):
        """Called when any CMY slider is adjusted."""
        if self._updating:
            return
        self._updating = True
        self.cyan_val = self.cyan_input()
        self.magenta_val = self.magenta_input()
        self.yellow_val = self.yellow_input()
        self._update_picker()
        self._send_outputs()
        self._updating = False

    def _send_outputs(self):
        self.cmy_output.send(np.array([self.cyan_val, self.magenta_val, self.yellow_val], dtype=float))

    def get_preset_state(self):
        return {'cmy': [self.cyan_val, self.magenta_val, self.yellow_val]}

    def set_preset_state(self, preset):
        if 'cmy' in preset:
            vals = preset['cmy']
            if len(vals) >= 3:
                self.cyan_val = any_to_int(vals[0])
                self.magenta_val = any_to_int(vals[1])
                self.yellow_val = any_to_int(vals[2])
                self._update_sliders()
                self._update_picker()
                self._send_outputs()

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            values = any_to_list(data)
            if len(values) >= 3:
                self.cyan_val = any_to_int(values[0])
                self.magenta_val = any_to_int(values[1])
                self.yellow_val = any_to_int(values[2])
                self._update_sliders()
        else:
            self.cyan_val = self.cyan_input()
            self.magenta_val = self.magenta_input()
            self.yellow_val = self.yellow_input()
        self._update_picker()
        self._send_outputs()



class KeyNode(Node):
    node_list = []
    inited = False
    map = {}
    reverse_map = {}

    @staticmethod
    def factory(name, data, args=None):
        node = KeyNode(name, data, args)
        return node
#
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if not KeyNode.inited:
            self.init()
        KeyNode.node_list.append(self)
        self.shift_key_pressed = False
        self.meta_key_pressed = False
        self.control_key_pressed = False
        self.alt_key_pressed = False

        self.key_list = {}
        self.output_dict = {}
        self.print_keys_button = self.add_input('list keys', widget_type='button', callback=self.list_keys)
        for arg in args:
            if arg in KeyNode.map:
                self.key_list[arg] = KeyNode.map[arg]
                out = self.add_output(arg)
                self.output_dict[arg] = out
        self.reverse_shifted_key_list = {}
        self.reverse_key_list = {}

        for key in self.key_list:
            k = self.key_list[key]
            if k[1]:
                self.reverse_shifted_key_list[k[0]] = key
            else:
                self.reverse_key_list[k[0]] = key

        if len(args) == 0:
            self.output = self.add_output('character out')
            self.code_output = self.add_output('key code out')
        self.shift_key_output = self.add_output('shift')
        self.control_key_output = self.add_output('control')
        self.meta_key_output = self.add_output('command / window')
        self.alt_key_output = self.add_output('alt / option')
        self.shift_key_changed = False
        self.control_key_changed = False
        self.meta_key_changed = False
        self.alt_key_changed = False

        self.shifted_keys = {}
        self.shifted_keys['1'] = '!'
        self.shifted_keys['2'] = '@'
        self.shifted_keys['3'] = '#'
        self.shifted_keys['4'] = '$'
        self.shifted_keys['5'] = '%'
        self.shifted_keys['6'] = '^'
        self.shifted_keys['7'] = '&'
        self.shifted_keys['8'] = '*'
        self.shifted_keys['9'] = '('
        self.shifted_keys['0'] = ')'
        self.shifted_keys['`'] = '~'
        self.shifted_keys['-'] = '_'
        self.shifted_keys['='] = '+'
        self.shifted_keys['['] = '{'
        self.shifted_keys[']'] = '}'
        self.shifted_keys['\\'] = '|'
        self.shifted_keys[';'] = ':'
        self.shifted_keys["'"] = '"'
        self.shifted_keys[','] = '<'
        self.shifted_keys['.'] = '>'
        self.shifted_keys["/"] = '?'

        self.unshifted_keys = {}
        self.unshifted_keys['!'] = '1'
        self.unshifted_keys['@'] = '2'
        self.unshifted_keys['#'] = '3'
        self.unshifted_keys['$'] = '4'
        self.unshifted_keys['%'] = '5'
        self.unshifted_keys['^'] = '6'
        self.unshifted_keys['&'] = '7'
        self.unshifted_keys['*'] = '8'
        self.unshifted_keys['('] = '9'
        self.unshifted_keys[')'] = '0'
        self.unshifted_keys['~'] = '`'
        self.unshifted_keys['_'] = '-'
        self.unshifted_keys['+'] = '='
        self.unshifted_keys['{'] = '['
        self.unshifted_keys['}'] = ']'
        self.unshifted_keys['|'] = '\\'
        self.unshifted_keys[':'] = ';'
        self.unshifted_keys['"'] = "'"
        self.unshifted_keys['<'] = ','
        self.unshifted_keys['>'] = '.'
        self.unshifted_keys['?'] = '/'

    def key_up(self, key_code):
        key_name = ''

        if key_code in self.reverse_map:
            key_name = self.reverse_map[key_code]
        if key_name in ['right_shift', 'left_shift']:
            if self.shift_key_pressed:
                self.shift_key_pressed = False
                self.shift_key_changed = True
        elif key_name in ['right_meta', 'left_meta']:
            if self.meta_key_pressed:
                self.meta_key_pressed = False
                self.meta_key_changed = True
        elif key_name in ['right_control', 'left_control']:
            if self.control_key_pressed:
                self.control_key_pressed = False
                self.control_key_changed = True
        elif key_name in ['right_alt', 'left_alt']:
            if self.alt_key_pressed:
                self.alt_key_pressed = False
                self.alt_key_changed = True

        if self.shift_key_changed:
            self.shift_key_changed = False
            self.shift_key_output.send(self.shift_key_pressed)
        if self.alt_key_changed:
            self.alt_key_changed = False
            self.alt_key_output.send(self.alt_key_pressed)
        if self.control_key_changed:
            self.control_key_changed = False
            self.control_key_output.send(self.control_key_pressed)
        if self.meta_key_changed:
            self.meta_key_changed = False
            self.meta_key_output.send(self.meta_key_pressed)

    def key_down(self, key_code):
        key_name = ''
        key_ascii = -1
        if key_code in self.reverse_map:
            key_name = self.reverse_map[key_code]
        if len(key_name) == 1:
            key_ascii = ord(key_name)
        if key_name in ['right_shift', 'left_shift']:
            if not self.shift_key_pressed:
                self.shift_key_pressed = True
                self.shift_key_changed = True
        elif key_name in ['right_meta', 'left_meta']:
            if not self.meta_key_pressed:
                self.meta_key_pressed = True
                self.meta_key_changed = True
        elif key_name in ['right_control', 'left_control']:
            if not self.control_key_pressed:
                self.control_key_pressed = True
                self.control_key_changed = True
        elif key_name in ['right_alt', 'left_alt']:
            if not self.alt_key_pressed:
                self.alt_key_pressed = True
                self.alt_key_changed = True

        if key_ascii != -1:
            character = chr(key_ascii)
            if self.shift_key_pressed:
                if key_ascii in self.reverse_shifted_key_list:
                    character = self.reverse_shifted_key_list[key_ascii]
                if ord('A') <= key_ascii <= ord('Z'):
                    character = character.upper()
                else:
                    if character in self.shifted_keys:
                        character = self.shifted_keys[character]
            else:
                if key_ascii in self.reverse_key_list:
                    character = self.reverse_key_list[key_ascii]
                if ord('A') <= key_ascii <= ord('Z'):
                    character = character.lower()
                if character in self.unshifted_keys:
                    character = self.unshifted_keys[character]

            if len(self.key_list) > 0:
                if character in self.key_list:
                    if character in self.output_dict:
                        self.output_dict[character].send('bang')
        if len(self.key_list) == 0:
            if self.output is not None:
                if key_ascii != -1:
                    self.code_output.send(key_ascii)
                    if key_ascii < 256 and character.isprintable():
                        self.output.send(character)

        if self.shift_key_changed:
            self.shift_key_changed = False
            self.shift_key_output.send(self.shift_key_pressed)
        if self.alt_key_changed:
            self.alt_key_changed = False
            self.alt_key_output.send(self.alt_key_pressed)
        if self.control_key_changed:
            self.control_key_changed = False
            self.control_key_output.send(self.control_key_pressed)
        if self.meta_key_changed:
            self.meta_key_changed = False
            self.meta_key_output.send(self.meta_key_pressed)

    def list_keys(self):
        keys = list(KeyNode.map.keys())
        counter = 0
        for key in keys:
            print(key, KeyNode.map[key], end=' ')
            counter += 1
            if counter % 10 == 0:
                print()

    def custom_cleanup(self):
        if self in KeyNode.node_list:
            KeyNode.node_list.remove(self)

    def init(self):
        KeyNode.inited = True

        KeyNode.map['0'] = [dpg.mvKey_0, False]
        KeyNode.map['1'] = [dpg.mvKey_1, False]
        KeyNode.map['2'] = [dpg.mvKey_2, False]
        KeyNode.map['3'] = [dpg.mvKey_3, False]
        KeyNode.map['4'] = [dpg.mvKey_4, False]
        KeyNode.map['5'] = [dpg.mvKey_5, False]
        KeyNode.map['6'] = [dpg.mvKey_6, False]
        KeyNode.map['7'] = [dpg.mvKey_7, False]
        KeyNode.map['8'] = [dpg.mvKey_8, False]
        KeyNode.map['9'] = [dpg.mvKey_9, False]

        KeyNode.map[')'] = [dpg.mvKey_0, True]
        KeyNode.map['!'] = [dpg.mvKey_1, True]
        KeyNode.map['@'] = [dpg.mvKey_2, True]
        KeyNode.map['#'] = [dpg.mvKey_3, True]
        KeyNode.map['$'] = [dpg.mvKey_4, True]
        KeyNode.map['%'] = [dpg.mvKey_5, True]
        KeyNode.map['^'] = [dpg.mvKey_6, True]
        KeyNode.map['&'] = [dpg.mvKey_7, True]
        KeyNode.map['*'] = [dpg.mvKey_8, True]
        KeyNode.map['('] = [dpg.mvKey_9, True]

        KeyNode.map['numpad_0'] = [dpg.mvKey_NumPad0, False]
        KeyNode.map['numpad_1'] = [dpg.mvKey_NumPad1, False]
        KeyNode.map['numpad_2'] = [dpg.mvKey_NumPad2, False]
        KeyNode.map['numpad_3'] = [dpg.mvKey_NumPad3, False]
        KeyNode.map['numpad_4'] = [dpg.mvKey_NumPad4, False]
        KeyNode.map['numpad_5'] = [dpg.mvKey_NumPad5, False]
        KeyNode.map['numpad_6'] = [dpg.mvKey_NumPad6, False]
        KeyNode.map['numpad_7'] = [dpg.mvKey_NumPad7, False]
        KeyNode.map['numpad_8'] = [dpg.mvKey_NumPad8, False]
        KeyNode.map['numpad_9'] = [dpg.mvKey_NumPad9, False]
        KeyNode.map['numpad_/'] = [dpg.mvKey_Divide, False]
        KeyNode.map['numpad_*'] = [dpg.mvKey_Multiply, False]
        KeyNode.map['numpad_+'] = [dpg.mvKey_Add, False]
        KeyNode.map['numpad_-'] = [dpg.mvKey_Subtract, False]
        KeyNode.map['numpad_.'] = [dpg.mvKey_Decimal, False]

        KeyNode.map['`'] = [dpg.mvKey_Tilde, False]
        KeyNode.map['~'] = [dpg.mvKey_Tilde, True]
        KeyNode.map['\\'] = [dpg.mvKey_Backslash, False]
        KeyNode.map['|'] = [dpg.mvKey_Backslash, True]
        KeyNode.map['clear'] = [dpg.mvKey_Clear, False]
        KeyNode.map[':'] = [dpg.mvKey_Colon, False]
        KeyNode.map[';'] = [dpg.mvKey_Colon, True]
        KeyNode.map[','] = [dpg.mvKey_Comma, False]
        KeyNode.map['<'] = [dpg.mvKey_Comma, True]
        KeyNode.map['delete'] = [dpg.mvKey_Delete, False]
        KeyNode.map['down'] = [dpg.mvKey_Down, False]
        KeyNode.map['end'] = [dpg.mvKey_End, False]
        KeyNode.map['escape'] = [dpg.mvKey_Escape, False]
        KeyNode.map['F1'] = [dpg.mvKey_F1, False]
        KeyNode.map['F10'] = [dpg.mvKey_F10, False]
        KeyNode.map['F11'] = [dpg.mvKey_F11, False]
        KeyNode.map['F12'] = [dpg.mvKey_F12, False]
        KeyNode.map['F13'] = [dpg.mvKey_F13, False]
        KeyNode.map['F14'] = [dpg.mvKey_F14, False]
        KeyNode.map['F15'] = [dpg.mvKey_F15, False]
        KeyNode.map['F2'] = [dpg.mvKey_F2, False]
        KeyNode.map['F3'] = [dpg.mvKey_F3, False]
        KeyNode.map['F4'] = [dpg.mvKey_F4, False]
        KeyNode.map['F5'] = [dpg.mvKey_F5, False]
        KeyNode.map['F6'] = [dpg.mvKey_F6, False]
        KeyNode.map['F7'] = [dpg.mvKey_F7, False]
        KeyNode.map['F8'] = [dpg.mvKey_F8, False]
        KeyNode.map['F9'] = [dpg.mvKey_F9, False]
        KeyNode.map['help'] = [dpg.mvKey_Help, False]
        KeyNode.map['home'] = [dpg.mvKey_Home, False]
        KeyNode.map['insert'] = [dpg.mvKey_Insert, False]
        KeyNode.map['left_control'] = [dpg.mvKey_LControl, False]
        KeyNode.map['left'] = [dpg.mvKey_Left, False]
        KeyNode.map['['] = [dpg.mvKey_Open_Brace, False]
        KeyNode.map['{'] = [dpg.mvKey_Open_Brace, True]
        KeyNode.map['left_meta'] = [dpg.mvKey_LWin, False]
        KeyNode.map['left_shift'] = [dpg.mvKey_LShift, False]
        KeyNode.map['-'] = [dpg.mvKey_Minus, False]
        KeyNode.map['_'] = [dpg.mvKey_Minus, True]
        KeyNode.map['num_lock'] = [dpg.mvKey_NumLock, False]
        KeyNode.map['pause'] = [dpg.mvKey_Pause, False]
        KeyNode.map['.'] = [dpg.mvKey_Period, False]
        KeyNode.map['+'] = [dpg.mvKey_Plus, True]
        KeyNode.map['='] = [dpg.mvKey_Plus, False]
        KeyNode.map['print'] = [dpg.mvKey_Print, False]
        KeyNode.map["'"] = [dpg.mvKey_Quote, False]
        KeyNode.map['"'] = [dpg.mvKey_Quote, True]
        if platform.system() != 'Darwin':
            KeyNode.map['left_alt'] = [dpg.mvKey_LAlt, False]
            KeyNode.map['right_alt'] = [dpg.mvKey_RAlt, False]
        KeyNode.map['right_control'] = [dpg.mvKey_RControl, False]
        KeyNode.map['return'] = [dpg.mvKey_Return, False]
        KeyNode.map['right'] = [dpg.mvKey_Right, False]
        KeyNode.map[']'] = [dpg.mvKey_Close_Brace, False]
        KeyNode.map['}'] = [dpg.mvKey_Close_Brace, True]
        KeyNode.map['right_meta'] = [dpg.mvKey_RWin, False]
        KeyNode.map['right_shift'] = [dpg.mvKey_RShift, False]
        KeyNode.map['scroll_lock'] = [dpg.mvKey_ScrollLock, False]
        KeyNode.map['/'] = [dpg.mvKey_Slash, False]
        KeyNode.map['?'] = [dpg.mvKey_Slash, True]
        KeyNode.map['space'] = [dpg.mvKey_Spacebar, False]
        KeyNode.map['tab'] = [dpg.mvKey_Tab, False]
        KeyNode.map['up'] = [dpg.mvKey_Up, False]
        KeyNode.map['a'] = [dpg.mvKey_A, False]
        KeyNode.map['b'] = [dpg.mvKey_B, False]
        KeyNode.map['c'] = [dpg.mvKey_C, False]
        KeyNode.map['d'] = [dpg.mvKey_D, False]
        KeyNode.map['e'] = [dpg.mvKey_E, False]
        KeyNode.map['f'] = [dpg.mvKey_F, False]
        KeyNode.map['g'] = [dpg.mvKey_G, False]
        KeyNode.map['h'] = [dpg.mvKey_H, False]
        KeyNode.map['i'] = [dpg.mvKey_I, False]
        KeyNode.map['j'] = [dpg.mvKey_J, False]
        KeyNode.map['k'] = [dpg.mvKey_K, False]
        KeyNode.map['l'] = [dpg.mvKey_L, False]
        KeyNode.map['m'] = [dpg.mvKey_M, False]
        KeyNode.map['n'] = [dpg.mvKey_N, False]
        KeyNode.map['o'] = [dpg.mvKey_O, False]
        KeyNode.map['p'] = [dpg.mvKey_P, False]
        KeyNode.map['q'] = [dpg.mvKey_Q, False]
        KeyNode.map['r'] = [dpg.mvKey_R, False]
        KeyNode.map['s'] = [dpg.mvKey_S, False]
        KeyNode.map['t'] = [dpg.mvKey_T, False]
        KeyNode.map['u'] = [dpg.mvKey_U, False]
        KeyNode.map['v'] = [dpg.mvKey_V, False]
        KeyNode.map['w'] = [dpg.mvKey_W, False]
        KeyNode.map['x'] = [dpg.mvKey_X, False]
        KeyNode.map['y'] = [dpg.mvKey_Y, False]
        KeyNode.map['z'] = [dpg.mvKey_Z, False]
        KeyNode.map['A'] = [dpg.mvKey_A, True]
        KeyNode.map['B'] = [dpg.mvKey_B, True]
        KeyNode.map['C'] = [dpg.mvKey_C, True]
        KeyNode.map['D'] = [dpg.mvKey_D, True]
        KeyNode.map['E'] = [dpg.mvKey_E, True]
        KeyNode.map['F'] = [dpg.mvKey_F, True]
        KeyNode.map['G'] = [dpg.mvKey_G, True]
        KeyNode.map['H'] = [dpg.mvKey_H, True]
        KeyNode.map['I'] = [dpg.mvKey_I, True]
        KeyNode.map['J'] = [dpg.mvKey_J, True]
        KeyNode.map['K'] = [dpg.mvKey_K, True]
        KeyNode.map['L'] = [dpg.mvKey_L, True]
        KeyNode.map['M'] = [dpg.mvKey_M, True]
        KeyNode.map['N'] = [dpg.mvKey_N, True]
        KeyNode.map['O'] = [dpg.mvKey_O, True]
        KeyNode.map['P'] = [dpg.mvKey_P, True]
        KeyNode.map['Q'] = [dpg.mvKey_Q, True]
        KeyNode.map['R'] = [dpg.mvKey_R, True]
        KeyNode.map['S'] = [dpg.mvKey_S, True]
        KeyNode.map['T'] = [dpg.mvKey_T, True]
        KeyNode.map['U'] = [dpg.mvKey_U, True]
        KeyNode.map['V'] = [dpg.mvKey_V, True]
        KeyNode.map['W'] = [dpg.mvKey_W, True]
        KeyNode.map['X'] = [dpg.mvKey_X, True]
        KeyNode.map['Y'] = [dpg.mvKey_Y, True]
        KeyNode.map['Z'] = [dpg.mvKey_Z, True]
        KeyNode.map['media_play_pause'] = [dpg.mvKey_Media_Play_Pause, False]
        KeyNode.map['media_stop'] = [dpg.mvKey_Media_Stop, False]
        KeyNode.map['media_next_track'] = [dpg.mvKey_Media_Next_Track, False]
        KeyNode.map['media_previous_track'] = [dpg.mvKey_Media_Prev_Track, False]
        KeyNode.map['volume_up'] = [dpg.mvKey_Volume_Up, False]
        KeyNode.map['volume_down'] = [dpg.mvKey_Volume_Down, False]
        KeyNode.map['volume_mute'] = [dpg.mvKey_Volume_Mute, False]

        for k in KeyNode.map:
            v = KeyNode.map[k][0]
            KeyNode.reverse_map[v] = k


class ParamValueNode(ValueNode):
    @staticmethod
    def factory(name, data, args=None):
        def factory(name, data, args=None):
            base_name = name.split('_')[-1]
            if base_name in ['float']:
                return FloatNode(name, data, args)
            elif base_name in ['int']:
                return IntNode(name, data, args)
            elif base_name in ['slider']:
                return SliderNode(name, data, args)
            elif base_name in ['knob']:
                return KnobNode(name, data, args)
            elif base_name in ['string', 'message', 'list']:
                return StringNode(name, data, args)
            elif base_name == 'text':
                return TextEditorNode(name, data, args)
            else:
                return StringNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        param_name = ''
        if len(args) > 0:
            param_name = args[0]

        super().__init__(label, data, args)
        self.unparsed_args = args
        self.param_output = self.add_output(param_name)
        self.param_name = self.add_option('parameter name', widget_type='text_input', default_value=param_name, callback=self.param_name_changed)

    def param_name_changed(self):
        self.param_name.set_label(self.param_name())

    def custom_create(self, from_file):
        dpg.hide_item(self.output.uuid)

    def do_send(self, value):
        output_list = [self.param_name(), value]
        self.param_output.send(output_list)


class MomentarySliderNode(Node):
    """A slider (or set of sliders) that auto-returns to center (0) when released.

    Usage:
        momentary_slider          - one float slider, range -1.0 to 1.0
        momentary_slider 3        - three float sliders, range -1.0 to 1.0
        momentary_slider 20       - one int slider, range -20 to 20
        momentary_slider pan tilt - two named float sliders
    """

    @staticmethod
    def factory(name, data, args=None):
        node = MomentarySliderNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.slider_inputs = []
        self.slider_outputs = []
        self.slider_active = []
        self.slider_initiated = []
        self.is_int = False
        self.slider_width = 120

        # Detect int mode from node name
        if '_int' in label:
            self.is_int = True

        # Parse args to determine number and names of sliders
        slider_names = []
        self.range_val = 20 if self.is_int else 1.0

        if args is not None and len(args) > 0:
            for arg in args:
                val, t = decode_arg([arg], 0)
                if t == int:
                    # If it's a small int (1-10), treat as slider count
                    if 1 <= val <= 10:
                        slider_names = [str(i) for i in range(val)]
                    else:
                        # Larger int: treat as range for a single int slider
                        self.range_val = val
                        self.is_int = True
                elif t == float:
                    self.range_val = val
                elif t == str:
                    slider_names.append(val)

        if len(slider_names) == 0:
            slider_names = ['value']

        # Create slider inputs and outputs
        widget_type = 'slider_int' if self.is_int else 'slider_float'
        min_val = -self.range_val
        max_val = self.range_val
        default_val = 0 if self.is_int else 0.0

        for i, name in enumerate(slider_names):
            slider_input = self.add_input(
                name,
                widget_type=widget_type,
                widget_width=self.slider_width,
                default_value=default_val,
                min=min_val,
                max=max_val,
                callback=self._make_slider_callback(i)
            )
            self.slider_inputs.append(slider_input)
            self.slider_active.append(False)
            self.slider_initiated.append(False)

        # Outputs in reverse order (bottom-up convention)
        for name in slider_names:
            out = self.add_output(name + ' out')
            self.slider_outputs.append(out)

        # Options
        self.range_option = self.add_option(
            'range', widget_type='drag_float', default_value=self.range_val,
            callback=self._range_changed
        )
        self.width_option = self.add_option(
            'width', widget_type='drag_int', default_value=self.slider_width,
            callback=self._width_changed
        )

    def _make_slider_callback(self, index):
        def callback():
            self._slider_moved(index)
        return callback

    def _slider_moved(self, index):
        val = self.slider_inputs[index]()
        if self.slider_active[index] or val != (0 if self.is_int else 0.0):
            self.slider_initiated[index] = True

    def _range_changed(self):
        self.range_val = self.range_option()
        min_val = -self.range_val
        max_val = self.range_val
        for slider_input in self.slider_inputs:
            slider_input.widget.set_limits(min_val, max_val)

    def _width_changed(self):
        width = self.width_option()
        for slider_input in self.slider_inputs:
            dpg.set_item_width(slider_input.widget.uuid, width)

    def custom_create(self, from_file):
        self.add_frame_task()

    def frame_task(self):
        try:
            for i, slider_input in enumerate(self.slider_inputs):
                # Check if slider was just released
                if dpg.is_item_deactivated(slider_input.widget.uuid) and self.slider_active[i]:
                    self._slider_release(i)
                    self.slider_initiated[i] = False

                # While slider is being dragged, send current value
                if self.slider_initiated[i]:
                    val = slider_input()
                    if self.is_int:
                        val = int(val)
                    self.slider_active[i] = (val != (0 if self.is_int else 0.0))
                    self.slider_outputs[i].send(val)
        except Exception:
            pass

    def _slider_release(self, index):
        self.slider_initiated[index] = False
        self.slider_active[index] = False
        reset_val = 0 if self.is_int else 0.0
        self.slider_inputs[index].set(reset_val)
        self.slider_outputs[index].send(reset_val)


class XYPadNode(Node):
    """A 2D XY pad. Can be momentary (auto-returns to center) or persistent.

    Uses a plot with a visual drag point marker. Click and drag to set position.

    Usage:
        xy_pad                - persistent, range -1.0 to 1.0
        xy_pad 20             - persistent, range -20 to 20
        momentary_xy          - momentary (snaps back to 0,0), range -1.0 to 1.0
        momentary_xy 0.5      - momentary, range -0.5 to 0.5
    """

    @staticmethod
    def factory(name, data, args=None):
        node = XYPadNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.range_val = 1.0
        self.pad_size = 150
        self.marker_size = 8
        self.dragging = False
        self.last_x = 0.0
        self.last_y = 0.0

        # Detect momentary mode from node name
        self.is_momentary = 'momentary' in label

        # Parse args
        if args is not None and len(args) > 0:
            val, t = decode_arg(args, 0)
            if t in [float, int]:
                self.range_val = float(val)

        # UUIDs for plot components
        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.scatter_tag = dpg.generate_uuid()
        self.scatter_theme_tag = dpg.generate_uuid()
        self.marker_size_style_tag = dpg.generate_uuid()
        self.crosshair_h_tag = dpg.generate_uuid()
        self.crosshair_v_tag = dpg.generate_uuid()

        # Display for the plot
        self.plot_display = self.add_display('')
        self.plot_display.submit_callback = self.submit_display

        # Outputs
        self.x_output = self.add_output('x out')
        self.y_output = self.add_output('y out')

        # Options
        self.momentary_option = self.add_option(
            'momentary', widget_type='checkbox', default_value=self.is_momentary
        )
        self.range_option = self.add_option(
            'range', widget_type='drag_float', default_value=self.range_val,
            callback=self._range_changed
        )
        self.size_option = self.add_option(
            'size', widget_type='drag_int', default_value=self.pad_size,
            callback=self._size_changed
        )
        self.marker_size_option = self.add_option(
            'marker size', widget_type='drag_int', default_value=self.marker_size,
            callback=self._marker_size_changed
        )

    def submit_display(self):
        with dpg.plot(
            label='', tag=self.plot_tag,
            height=self.pad_size, width=self.pad_size,
            no_title=True, no_menus=True,
            no_mouse_pos=True
        ):
            dpg.add_plot_axis(
                dpg.mvXAxis, label='', tag=self.x_axis_tag,
                no_tick_labels=True, no_gridlines=False
            )
            dpg.add_plot_axis(
                dpg.mvYAxis, label='', tag=self.y_axis_tag,
                no_tick_labels=True, no_gridlines=False
            )
            dpg.set_axis_limits(self.x_axis_tag, -self.range_val, self.range_val)
            dpg.set_axis_limits(self.y_axis_tag, -self.range_val, self.range_val)

            # Crosshair lines at origin
            dpg.add_inf_line_series(
                x=[0.0], parent=self.y_axis_tag, tag=self.crosshair_v_tag
            )
            dpg.add_hline_series(
                x=[0.0], parent=self.y_axis_tag, tag=self.crosshair_h_tag
            )

            # Visual marker for current position
            with dpg.theme(tag=self.scatter_theme_tag):
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, self.marker_size, tag=self.marker_size_style_tag, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 0, 255), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 0, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_scatter_series([0.0], [0.0], tag=self.scatter_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.scatter_tag, self.scatter_theme_tag)

    def custom_create(self, from_file):
        self.add_frame_task()

    def frame_task(self):
        try:
            mouse_down = dpg.is_mouse_button_down(0)
            is_hovered = dpg.is_item_hovered(self.plot_tag)

            if mouse_down:
                if is_hovered and not self.dragging:
                    # Mouse just pressed on our plot — start dragging
                    self.dragging = True

                if self.dragging:
                    # Read mouse position in plot data coordinates
                    plot_pos = dpg.get_plot_mouse_pos()
                    if plot_pos is not None:
                        x = max(-self.range_val, min(self.range_val, plot_pos[0]))
                        y = max(-self.range_val, min(self.range_val, plot_pos[1]))
                        # Move the visual marker
                        dpg.set_value(self.scatter_tag, [[x], [y]])
                        self.last_x = x
                        self.last_y = y
                        self.y_output.send(y)
                        self.x_output.send(x)
            else:
                if self.dragging:
                    self.dragging = False
                    if self.momentary_option():
                        # Momentary: snap back to center
                        dpg.set_value(self.scatter_tag, [[0.0], [0.0]])
                        self.last_x = 0.0
                        self.last_y = 0.0
                        self.y_output.send(0.0)
                        self.x_output.send(0.0)
                    else:
                        # Persistent: send final position once
                        self.y_output.send(self.last_y)
                        self.x_output.send(self.last_x)
        except Exception:
            pass

    def _range_changed(self):
        self.range_val = self.range_option()
        dpg.set_axis_limits(self.x_axis_tag, -self.range_val, self.range_val)
        dpg.set_axis_limits(self.y_axis_tag, -self.range_val, self.range_val)

    def _size_changed(self):
        size = self.size_option()
        self.pad_size = size
        dpg.set_item_width(self.plot_tag, size)
        dpg.set_item_height(self.plot_tag, size)

    def _marker_size_changed(self):
        self.marker_size = self.marker_size_option()
        dpg.configure_item(self.marker_size_style_tag, x=self.marker_size)


class EnvelopeNode(Node):
    """A breakpoint function / envelope editor with draggable control points.

    Points are connected by lines. Drag points to reposition.
    Right-click: add point (if far from existing) or remove point (if near one).
    Shift+right-click drag near a point: adjust curvature (up=convex, down=concave).
    Send an x value to interpolate the envelope.

    Usage:
        envelope          - x range 0-1, y range 0-1
        envelope 10       - x range 0-10, y range 0-1
        envelope 10 5     - x range 0-10, y range 0-5
    """

    SAMPLES_PER_CURVE = 32

    @staticmethod
    def factory(name, data, args=None):
        return EnvelopeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.plot_width = 300
        self.plot_height = 150
        self.remove_threshold = 0.08

        # Parse args
        if args and len(args) > 0:
            val, t = decode_arg(args, 0)
            if t in [float, int]:
                self.x_max = float(val)
        if args and len(args) > 1:
            val, t = decode_arg(args, 1)
            if t in [float, int]:
                self.y_max = float(val)

        # Point data: {'tag': uuid, 'x': float, 'y': float, 'curve': float}
        # 'curve' controls the outgoing segment shape:
        #   0 = linear, positive = convex (bows up), negative = concave (bows down)
        self.points = []
        self.right_was_down = False
        self.left_was_down = False

        # Ramp / trigger state
        self.ramp_active = False
        self.ramp_start_time = 0.0
        self.ramp_time = self.x_max  # default ramp duration = x range

        # Curvature drag state
        self.curving = False
        self.curving_point_idx = -1
        self.curve_drag_start_screen_y = 0.0
        self.curve_drag_start_val = 0.0

        # UUIDs
        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.line_tag = dpg.generate_uuid()
        self.playhead_tag = dpg.generate_uuid()

        # Inputs
        self.sample_input = self.add_input('x', triggers_execution=True)
        self.trigger_input = self.add_input('trigger', widget_type='button',
                                            triggers_execution=True,
                                            callback=self._start_ramp)
        self.ramp_time_input = self.add_input('duration', widget_type='drag_float',
                                              default_value=self.ramp_time,
                                              triggers_execution=True)

        # Display for the plot
        self.plot_display = self.add_display('')
        self.plot_display.submit_callback = self.submit_display

        # Outputs
        self.points_output = self.add_output('points out')
        self.value_output = self.add_output('value out')

        # Options
        self.x_max_option = self.add_option(
            'x max', widget_type='drag_float', default_value=self.x_max,
            callback=self._range_changed
        )
        self.y_min_option = self.add_option(
            'y min', widget_type='drag_float', default_value=self.y_min,
            callback=self._range_changed
        )
        self.y_max_option = self.add_option(
            'y max', widget_type='drag_float', default_value=self.y_max,
            callback=self._range_changed
        )
        self.width_option = self.add_option(
            'width', widget_type='drag_int', default_value=self.plot_width,
            callback=self._size_changed
        )
        self.height_option = self.add_option(
            'height', widget_type='drag_int', default_value=self.plot_height,
            callback=self._size_changed
        )

    def submit_display(self):
        with dpg.theme() as self.line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (80, 140, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)

        with dpg.plot(
            label='', tag=self.plot_tag,
            height=self.plot_height, width=self.plot_width,
            no_title=True, no_menus=True, no_box_select=True,
            no_mouse_pos=True
        ):
            dpg.add_plot_axis(
                dpg.mvXAxis, label='', tag=self.x_axis_tag,
                no_tick_labels=True
            )
            dpg.add_plot_axis(
                dpg.mvYAxis, label='', tag=self.y_axis_tag,
                no_tick_labels=True
            )
            dpg.set_axis_limits(self.x_axis_tag, 0, self.x_max)
            dpg.set_axis_limits(self.y_axis_tag, self.y_min, self.y_max)

            # Line series connecting points
            dpg.add_line_series([], [], parent=self.y_axis_tag, tag=self.line_tag)
            dpg.bind_item_theme(self.line_tag, self.line_theme)

            # Playhead vertical line (hidden until ramp active)
            dpg.add_inf_line_series(x=[], parent=self.y_axis_tag, tag=self.playhead_tag)

        # Default points: triangle
        self._create_point(0.0, self.y_min)
        self._create_point(self.x_max * 0.5, self.y_max)
        self._create_point(self.x_max, self.y_min)
        self._update_line()

    def _create_point(self, x, y, curve=0.0):
        tag = dpg.generate_uuid()
        color = self._curve_color(curve)
        dpg.add_drag_point(
            tag=tag, default_value=(x, y),
            color=color,
            parent=self.plot_tag
        )
        self.points.append({'tag': tag, 'x': x, 'y': y, 'curve': curve})
        return tag

    def _remove_point_at_index(self, index):
        if len(self.points) <= 2:
            return
        p = self.points.pop(index)
        if dpg.does_item_exist(p['tag']):
            dpg.delete_item(p['tag'])

    @staticmethod
    def _curve_color(curve_val):
        """Interpolate point color: yellow (linear) → green (curved)."""
        blend = min(1.0, abs(curve_val) / 2.0)
        r = int(255 * (1 - blend))
        g = 255
        b = int(100 * blend)
        return (r, g, b, 255)

    def _update_point_color(self, index):
        p = self.points[index]
        color = self._curve_color(p['curve'])
        if dpg.does_item_exist(p['tag']):
            dpg.configure_item(p['tag'], color=color)

    def _find_nearest_point(self, mx, my):
        """Find index and normalized distance of nearest point to (mx, my)."""
        min_dist = float('inf')
        min_idx = -1
        x_range = max(self.x_max, 0.001)
        y_range = max(self.y_max - self.y_min, 0.001)
        for i, p in enumerate(self.points):
            dx = (p['x'] - mx) / x_range
            dy = (p['y'] - my) / y_range
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx, min_dist

    def _find_segment_at_x(self, mx):
        """Find the index (in self.points) of the start point of the segment containing mx."""
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        for i in range(len(sorted_pts) - 1):
            if sorted_pts[i]['x'] <= mx <= sorted_pts[i + 1]['x']:
                # Return index in self.points (not sorted index)
                return self.points.index(sorted_pts[i])
        return -1

    @staticmethod
    def _ease(t, curve_val):
        """Attempt to apply curvature using Schlick bias function.

        f(t) = t / (t + (1 - t) * k)

        This function always has nonzero curvature at t=0 (no flat start).
        curve_val = 0: k=1, linear
        curve_val > 0: k<1, convex (f(t) > t, bows above straight line)
        curve_val < 0: k>1, concave (f(t) < t, bows below straight line)
        """
        if abs(curve_val) < 0.001:
            return t
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        # Map curve_val to bias parameter k
        if curve_val > 0:
            k = 1.0 / (1.0 + curve_val)
        else:
            k = 1.0 - curve_val
        return t / (t + (1.0 - t) * k)

    def _generate_line_data(self):
        """Generate x/y arrays for the line series, with curved segments densely sampled."""
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        n = len(sorted_pts)
        if n == 0:
            return [], []
        if n == 1:
            return [sorted_pts[0]['x']], [sorted_pts[0]['y']]

        x_data = []
        y_data = []

        for i in range(n - 1):
            p_cur = sorted_pts[i]
            p_next = sorted_pts[i + 1]
            curve_val = p_cur.get('curve', 0.0)

            if abs(curve_val) > 0.001:
                # Curved segment: generate dense samples
                y0 = p_cur['y']
                y1 = p_next['y']
                x0 = p_cur['x']
                x1 = p_next['x']
                # Flip curve direction for descending segments so that
                # drag-up always bows the curve upward visually
                effective_curve = curve_val if y1 >= y0 else -curve_val
                for s in range(self.SAMPLES_PER_CURVE):
                    t = s / float(self.SAMPLES_PER_CURVE)
                    eased = self._ease(t, effective_curve)
                    x_data.append(x0 + t * (x1 - x0))
                    y_data.append(y0 + eased * (y1 - y0))
            else:
                # Linear: just the start point
                x_data.append(p_cur['x'])
                y_data.append(p_cur['y'])

        # Add the final point
        x_data.append(sorted_pts[-1]['x'])
        y_data.append(sorted_pts[-1]['y'])

        return x_data, y_data

    def _update_line(self):
        x_data, y_data = self._generate_line_data()
        dpg.set_value(self.line_tag, [x_data, y_data])

    def _send_points(self):
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        point_list = [[p['x'], p['y'], p.get('curve', 0.0)] for p in sorted_pts]
        self.points_output.send(point_list)

    def _interpolate_at(self, x_val):
        """Interpolate the envelope at x_val, respecting per-segment curvature."""
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        n = len(sorted_pts)
        if n == 0:
            return 0.0
        if x_val <= sorted_pts[0]['x']:
            return sorted_pts[0]['y']
        if x_val >= sorted_pts[-1]['x']:
            return sorted_pts[-1]['y']

        for i in range(n - 1):
            if sorted_pts[i]['x'] <= x_val <= sorted_pts[i + 1]['x']:
                dx = sorted_pts[i + 1]['x'] - sorted_pts[i]['x']
                if dx == 0:
                    return sorted_pts[i]['y']
                t = (x_val - sorted_pts[i]['x']) / dx
                curve_val = sorted_pts[i].get('curve', 0.0)
                # Flip curve direction for descending segments
                effective_curve = curve_val if sorted_pts[i + 1]['y'] >= sorted_pts[i]['y'] else -curve_val
                eased = self._ease(t, effective_curve)
                return sorted_pts[i]['y'] + eased * (sorted_pts[i + 1]['y'] - sorted_pts[i]['y'])

        return sorted_pts[-1]['y']

    def _start_ramp(self):
        """Start ramp playback (called by button or trigger input)."""
        self.ramp_active = True
        self.ramp_start_time = time.time()

    def execute(self):
        if self.trigger_input.fresh_input:
            self._start_ramp()
        if self.ramp_time_input.fresh_input:
            self.ramp_time = max(0.001, any_to_float(self.ramp_time_input()))
        if self.sample_input.fresh_input and not self.ramp_active:
            x_val = any_to_float(self.sample_input())
            y_val = self._interpolate_at(x_val)
            self.value_output.send(y_val)

    def custom_create(self, from_file):
        self.add_frame_task()

    def frame_task(self):
        try:
            # --- Ramp playback ---
            if self.ramp_active:
                elapsed = time.time() - self.ramp_start_time
                ramp_t = max(0.001, any_to_float(self.ramp_time_input()))
                # Map elapsed time to envelope x position
                x_pos = (elapsed / ramp_t) * self.x_max
                if elapsed >= ramp_t:
                    # Ramp finished
                    y_val = self._interpolate_at(self.x_max)
                    self.value_output.send(y_val)
                    self.ramp_active = False
                    dpg.set_value(self.playhead_tag, [[]])
                else:
                    y_val = self._interpolate_at(x_pos)
                    self.value_output.send(y_val)
                    dpg.set_value(self.playhead_tag, [[x_pos]])

            shift_held = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            left_down = dpg.is_mouse_button_down(0)
            right_down = dpg.is_mouse_button_down(1)

            # --- Curvature drag (Shift+left-click held) ---
            if self.curving:
                if left_down and shift_held:
                    # Use screen mouse (pixels) so drag isn't clipped at plot edge
                    screen_pos = dpg.get_mouse_pos()
                    # Screen Y is inverted: drag up = decrease in screen Y = positive curve
                    delta_px = self.curve_drag_start_screen_y - screen_pos[1]
                    # ~18 pixels of drag per unit of curve_val
                    new_curve = self.curve_drag_start_val + delta_px / 18.0
                    new_curve = max(-16.0, min(16.0, new_curve))
                    idx = self.curving_point_idx
                    if 0 <= idx < len(self.points):
                        self.points[idx]['curve'] = new_curve
                        self._update_point_color(idx)
                        self._update_line()
                else:
                    # Released: finalize
                    self.curving = False
                    self._send_points()
                self.left_was_down = left_down
                self.right_was_down = right_down
                return

            # --- Shift+left-click: start curvature drag ---
            if left_down and not self.left_was_down and shift_held:
                if dpg.is_item_hovered(self.plot_tag):
                    plot_pos = dpg.get_plot_mouse_pos()
                    if plot_pos:
                        seg_idx = self._find_segment_at_x(plot_pos[0])
                        if seg_idx >= 0:
                            self.curving = True
                            self.curving_point_idx = seg_idx
                            self.curve_drag_start_screen_y = dpg.get_mouse_pos()[1]
                            self.curve_drag_start_val = self.points[seg_idx].get('curve', 0.0)
                            self.left_was_down = left_down
                            self.right_was_down = right_down
                            return

            # --- Poll drag point positions for changes ---
            changed = False
            for p in self.points:
                if dpg.does_item_exist(p['tag']):
                    pos = dpg.get_value(p['tag'])
                    x = max(0, min(self.x_max, pos[0]))
                    y = max(self.y_min, min(self.y_max, pos[1]))
                    if abs(x - pos[0]) > 1e-6 or abs(y - pos[1]) > 1e-6:
                        dpg.set_value(p['tag'], [x, y])
                    if abs(x - p['x']) > 1e-6 or abs(y - p['y']) > 1e-6:
                        p['x'] = x
                        p['y'] = y
                        changed = True

            if changed:
                self._update_line()
                self._send_points()

            # --- Right-click: add or remove point ---
            if right_down and not self.right_was_down:
                if dpg.is_item_hovered(self.plot_tag):
                    plot_pos = dpg.get_plot_mouse_pos()
                    if plot_pos:
                        nearest_idx, nearest_dist = self._find_nearest_point(
                            plot_pos[0], plot_pos[1]
                        )
                        if nearest_dist < self.remove_threshold and len(self.points) > 2:
                            self._remove_point_at_index(nearest_idx)
                        else:
                            x = max(0, min(self.x_max, plot_pos[0]))
                            y = max(self.y_min, min(self.y_max, plot_pos[1]))
                            self._create_point(x, y)
                        self._update_line()
                        self._send_points()

            self.left_was_down = left_down
            self.right_was_down = right_down

        except Exception:
            pass

    def save_custom(self, container):
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        container['envelope_points'] = [
            [p['x'], p['y'], p.get('curve', 0.0)] for p in sorted_pts
        ]

    def load_custom(self, container):
        if 'envelope_points' in container:
            for p in self.points:
                if dpg.does_item_exist(p['tag']):
                    dpg.delete_item(p['tag'])
            self.points.clear()
            for pt_data in container['envelope_points']:
                x, y = pt_data[0], pt_data[1]
                curve = float(pt_data[2]) if len(pt_data) > 2 else 0.0
                self._create_point(x, y, curve=curve)
            self._update_line()

    def _range_changed(self):
        self.x_max = self.x_max_option()
        self.y_min = self.y_min_option()
        self.y_max = self.y_max_option()
        dpg.set_axis_limits(self.x_axis_tag, 0, self.x_max)
        dpg.set_axis_limits(self.y_axis_tag, self.y_min, self.y_max)

    def _size_changed(self):
        dpg.set_item_width(self.plot_tag, self.width_option())
        dpg.set_item_height(self.plot_tag, self.height_option())

