import dearpygui.dearpygui as dpg
import time
from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
from dpg_system.matrix_nodes import RollingBuffer

def register_interface_nodes():
    Node.app.register_node("menu", MenuNode.factory)
    Node.app.register_node("toggle", ToggleNode.factory)
    Node.app.register_node("button", ButtonNode.factory)
    Node.app.register_node("b", ButtonNode.factory)
    Node.app.register_node("mouse", MouseNode.factory)
    Node.app.register_node("float", ValueNode.factory)
    Node.app.register_node("int", ValueNode.factory)
    Node.app.register_node("slider", ValueNode.factory)
    Node.app.register_node("message", ValueNode.factory)
    Node.app.register_node("knob", ValueNode.factory)
    Node.app.register_node("plot", PlotNode.factory)
    Node.app.register_node("heat_map", PlotNode.factory)
    Node.app.register_node("heat_scroll", PlotNode.factory)
    Node.app.register_node("profile", PlotNode.factory)
    Node.app.register_node("Value Tool", ValueNode.factory)
    Node.app.register_node('print', PrintNode.factory)
    Node.app.register_node('load_action', LoadActionNode.factory)
    Node.app.register_node('color', ColorPickerNode.factory)
    Node.app.register_node('vector', VectorNode.factory)
    Node.app.register_node('radio', RadioButtonsNode.factory)
    Node.app.register_node('radio_h', RadioButtonsNode.factory)
    Node.app.register_node('radio_v', RadioButtonsNode.factory)
    Node.app.register_node('presets', PresetsNode.factory)



class ButtonNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ButtonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.target_time = 0
        self.flash_duration = .100
        self.action_name = ''
        self.action = None

        if args is not None and len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == str:
                self.action_name = v

        self.input = self.add_input('', triggers_execution=True, widget_type='button', widget_width=14, callback=self.clicked_function)
        self.output = self.add_output("")

        self.action_binding_property = self.add_option('bind to', widget_type='text_input', width=120, default_value=self.action_name, callback=self.binding_changed)
        self.message_option = self.add_option('message', widget_type='text_input', default_value='bang', callback=self.message_changed)
        self.width_option = self.add_option('width', widget_type='input_int', default_value='14', callback=self.size_changed)
        self.height_option = self.add_option('height', widget_type='input_int', default_value='14', callback=self.size_changed)
        self.flash_duration_option = self.add_option('flash duration', widget_type='drag_float', min=0, max=1.0, default_value=self.flash_duration)

        with dpg.theme() as self.active_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.inactive_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)

    def size_changed(self):
        width = self.width_option.get_widget_value()
        height = self.height_option.get_widget_value()
        dpg.set_item_width(self.input.widget.uuid, width)
        dpg.set_item_height(self.input.widget.uuid, height)

    def binding_changed(self):
        print('binding_changed')
        binding = self.action_binding_property.get_widget_value()
        self.bind_to_action(binding)

    def bind_to_action(self, action_name):
        if action_name != '':
            a = Node.app.find_action(action_name)
            if a is not None:
                self.action_name = action_name
                self.action = a
                self.input.attach_to_action(a)
                if self.message_option.get_widget_value() == 'bang':
                    size = dpg.get_text_size(self.action_name, font=dpg.get_item_font(self.input.widget.uuid))
                    if size is None:
                        size = [80, 14]
                    dpg.set_item_width(self.input.widget.uuid, int(size[0]) + 12)
                    dpg.set_item_label(self.input.widget.uuid, self.action_name)
            else:
                self.input.attach_to_action(None)
        else:
            self.input.attach_to_action(None)

    def message_changed(self):
        new_name = self.message_option.get_widget_value()
        print(new_name)
        if new_name != 'bang':
            size = dpg.get_text_size(new_name, font=dpg.get_item_font(self.input.widget.uuid))
            if size is None:
                size = [80, 14]
            dpg.set_item_width(self.input.widget.uuid, int(size[0]) + 12)
            dpg.set_item_label(self.input.widget.uuid, new_name)

    def clicked_function(self, input=None):
        self.flash_duration = self.flash_duration_option.get_widget_value()
        self.target_time = time.time() + self.flash_duration
        dpg.bind_item_theme(self.input.widget.uuid, self.active_theme)
        self.add_frame_task()

    def custom_setup(self, from_file):
        if self.action_name != '':
            self.binding_changed()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            dpg.bind_item_theme(self.input.widget.uuid, self.inactive_theme)
            self.remove_frame_tasks()

    def execute(self):
        # if self.action is not None:
        #     print('execute', self.action)
        #     self.action()
        self.output.send(self.message_option.get_widget_value())


class MenuNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MenuNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.choice = ''
        self.choices = self.args_as_list()
        self.choice_input = self.add_input('##choice', widget_type='combo', default_value=self.choice, callback=self.set_choice)
        self.choice_input.widget.combo_items = self.choices

        self.output = self.add_output("")

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.choice_input.get_widget_value()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.choice_input.widget.set(preset['value'])
            self.execute()

    def set_choice(self, input=None):
        do_execute = True
        if self.choice_input.fresh_input:
            self.choice_input.fresh_input = False
            input_choice = self.choice_input._data
            t = type(input_choice)
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
                        dpg.configure_item(self.choice_input.widget.uuid, items=self.choices)
                        do_execute = False
                    else:
                        self.choices = []
                        for new_choice in input_choice:
                            if new_choice not in self.choices:
                                self.choices.append(new_choice)
                        dpg.configure_item(self.choice_input.widget.uuid, items=self.choices)
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
                self.choice_input.set(test_choice)
        if do_execute:
            self.execute()

    def execute(self):
        self.outputs[0].send(self.choice_input.get_widget_value())


class MouseNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MouseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.mouse_pos = None
        self.streaming = False

        self.input = self.add_input("", triggers_execution=True, widget_type='checkbox', widget_width=40, callback=self.start_stop_streaming)
        self.output_x = self.add_output("x")
        self.output_y = self.add_output("y")

    def start_stop_streaming(self, input=None):
        if self.input.get_widget_value():
            if not self.streaming:
                self.add_frame_task()
                self.streaming = True
        else:
            if self.streaming:
                self.remove_frame_tasks()
                self.streaming = False

    def frame_task(self):
        if self.input.get_widget_value():
            self.mouse_pos = dpg.get_mouse_pos(local=False)
            self.execute()

    def execute(self):
        if self.mouse_pos is not None:
            self.output_y.set_value(self.mouse_pos[1])
            self.output_x.set_value(self.mouse_pos[0])
        self.send_all()


class PresetsNode(Node):
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
        # if label == 'radio_h':
        #     self.radio_group.widget.horizontal = True
        # else:
        self.radio_group.widget.horizontal = False
        self.presets = [None] * self.preset_count

    def preset_click(self):
        if dpg.is_key_down(dpg.mvKey_Shift):
            self.save_preset()
        else:
            self.load_preset()

    def save_preset(self):
        editor = self.my_editor
        current_preset_index = string_to_int(self.radio_group.get_widget_value()) - 1
        if len(self.presets) > current_preset_index + 1:
            if self.presets[current_preset_index] is None:
                self.presets[current_preset_index] = {}
            kids = self.output.get_children()
            if len(kids) > 0:
                for kid in kids:
                    node = kid.node
                    if node is not None:
                        self.presets[current_preset_index][node.uuid] = node.get_preset_state()
            else:
                for node in editor._nodes:
                    self.presets[current_preset_index][node.uuid] = node.get_preset_state()

    def load_preset(self):
        editor = self.my_editor
        current_preset_index = string_to_int(self.radio_group.get_widget_value()) - 1
        if len(self.presets) > current_preset_index + 1:
            if self.presets[current_preset_index] is None:
                return
            kids = self.output.get_children()
            if len(kids) > 0:
                for kid in kids:
                    node = kid.node
                    if node is not None:
                        node.set_preset_state(self.presets[current_preset_index][node.uuid])
            else:
                for node in editor._nodes:
                    if node.uuid in self.presets[current_preset_index]:
                        node.set_preset_state(self.presets[current_preset_index][node.uuid])

    def save_custom_setup(self, container):
        container['presets'] = self.presets

    def load_custom_setup(self, container):
        if 'presets' in container:
            self.presets = container['presets']

    def post_load_callback(self):
        editor = self.my_editor
        translation_table = {}
        for preset in self.presets:
            if preset is not None:
                for node_preset_uuid in preset:
                    node_preset_uuid_int = int(node_preset_uuid)
                    if node_preset_uuid_int not in translation_table:
                        for node in editor._nodes:
                            if node.loaded_uuid == node_preset_uuid_int:
                                translation_table[node_preset_uuid_int] = node.uuid
        adjusted_presets = [None] * self.preset_count
        for index, preset in enumerate(self.presets):
            if preset is not None:
                adjusted_presets[index] = {}
                for node_preset_uuid in preset:
                    node_preset_uuid_int = int(node_preset_uuid)
                    if node_preset_uuid_int in translation_table:
                        new_uuid = translation_table[node_preset_uuid_int]
                        adjusted_presets[index][new_uuid] = preset[node_preset_uuid]
        self.presets = adjusted_presets.copy()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            self.radio_group.widget.set(data)
            self.load_preset()

class RadioButtonsNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RadioButtonsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.buttons = []
        if args is not None and len(args) > 0:
            for i in range(len(args)):
                v, t = decode_arg(args, i)
                self.buttons.append(v)
        self.radio_group = self.add_property(widget_type='radio_group', callback=self.execute)
        self.radio_group.widget.combo_items = self.buttons
        if label == 'radio_h':
            self.radio_group.widget.horizontal = True
        else:
            self.radio_group.widget.horizontal = False
        self.output = self.add_output("")

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.radio_group.get_widget_value()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.radio_group.widget.set(preset['value'])
            self.execute()

    def call_execute(self, input=None):
        self.execute()

    def execute(self):
        self.output.send(self.radio_group.get_widget_value())


class ToggleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ToggleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.value = False
        self.variable = None
        self.variable_name = ''
        self.input = self.add_input("", triggers_execution=True, widget_type='checkbox', widget_width=40, callback=self.call_execute)
        self.output = self.add_output("")
        self.variable_binding_property = self.add_option('bind to', widget_type='text_input', width=120, default_value=self.variable_name, callback=self.binding_changed)
        if self.ordered_args is not None and len(self.ordered_args) > 0:
            for i in range(len(self.ordered_args)):
                var_name, t = decode_arg(self.ordered_args, i)
                if t == str:
                    self.variable_name = var_name

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.input.get_widget_value()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.input.widget.set(preset['value'])
            self.execute()

    def binding_changed(self):
        binding = self.variable_binding_property.get_widget_value()
        self.bind_to_variable(binding)

    def bind_to_variable(self, variable_name):
        # change name
        if self.variable is not None:
            self.variable.detach_client(self)
            self.variable = None
        if variable_name != '':
            v = Node.app.find_variable(variable_name)
            if v is None:
                default = False
                v = Node.app.add_variable(variable_name, default_value=default)
            if v:
                self.variable_name = variable_name
                self.variable = v
                self.input.attach_to_variable(v)
                self.variable.attach_client(self)
                self.output.set_label(self.variable_name)
                self.variable_update()

    def custom_setup(self, from_file):
        if self.variable_name != '':
            self.bind_to_variable(self.variable_name)

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
        if self.variable is not None and propagate:
            self.variable.set(value, from_client=self)
        self.outputs[0].send(value)

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def call_execute(self, input=None):
        self.execute()

    def execute(self):
        if self.input.fresh_input:
            received = self.input._data     # so that we can catch 'bang' ?
            if type(received) == str:
                if received == 'bang':
                    self.value = not self.value
                    self.input.set(self.value)
        else:
            self.value = self.input.get_widget_value()
        if self.variable is not None:
            self.variable.set(self.value, from_client=self)
        self.output.send(self.value)


class ValueNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ValueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

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
        self.format_property = None

        if label == 'float':
            widget_type = 'drag_float'
            for i in range(len(self.ordered_args)):
                val, t = decode_arg(self.ordered_args, i)
                if t in [float, int]:
                    self.start_value = val
                elif t == str:
                    if val == '+':
                        widget_type = 'input_float'
        elif label == 'int':
            widget_type = 'drag_int'
            for i in range(len(self.ordered_args)):
                val, t = decode_arg(self.ordered_args, i)
                if t in [float, int]:
                    self.start_value = val
                elif t == str:
                    if val == '+':
                        widget_type = 'input_int'
        elif label == 'slider':
            widget_type = 'slider_float'
            if self.ordered_args is not None:
                for i in range(len(self.ordered_args)):
                    val, t = decode_arg(self.ordered_args, i)
                    if t == float:
                        widget_type = 'slider_float'
                        self.max = val
                    elif t == int:
                        widget_type = 'slider_int'
                        self.max = val
            if self.max is None:
                self.max = 1.0
        elif label == 'knob':
            widget_type = 'knob_float'
            if self.ordered_args is not None:
                for i in range(len(self.ordered_args)):
                    val, t = decode_arg(self.ordered_args, i)
                    if t in [float, int]:
                        self.max = val
            if self.max is None:
                self.max = 100
        elif label == 'string' or label == 'message':
            widget_type = 'text_input'

        if self.ordered_args is not None and len(self.ordered_args) > 0:
            for i in range(len(self.ordered_args)):
                var_name, t = decode_arg(self.ordered_args, i)
                if t == str:
                    if widget_type not in ['input_int', 'input_float'] or var_name != '+':
                        self.variable_name = var_name

        if self.max is None:
            self.input = self.add_input("", triggers_execution=True, widget_type=widget_type, widget_uuid=self.value, widget_width=widget_width, trigger_button=True)
        else:
            self.input = self.add_input("", triggers_execution=True, widget_type=widget_type, widget_uuid=self.value, widget_width=widget_width, trigger_button=True, max=self.max)

        if self.variable_name != '':
            self.output = self.add_output(self.variable_name)
        else:
            self.output = self.add_output('out')

        self.variable_binding_property = self.add_option('bind to', widget_type='text_input', width=120, default_value=self.variable_name, callback=self.binding_changed)

        if widget_type in ['drag_float', 'slider_float', "knob_float"]:
            self.min_property = self.add_option('min', widget_type='drag_float', default_value=self.min, callback=self.options_changed)
            self.max_property = self.add_option('max', widget_type='drag_float', default_value=self.max, callback=self.options_changed)

        self.width_option = self.add_option('width', widget_type='drag_int', default_value=widget_width, callback=self.options_changed)
        # self.height_option = self.add_option('height', widget_type='drag_int', default_value=14, callback=self.options_changed)
        if widget_type in ['drag_float', 'slider_float', 'drag_int', 'knob_int', 'input_int']:
            self.format_property = self.add_option('format', widget_type='text_input', default_value=self.format, callback=self.options_changed)

    def get_preset_state(self):
        preset = {}
        preset['value'] = self.input.get_widget_value()
        return preset

    def set_preset_state(self, preset):
        if 'value' in preset:
            self.input.widget.set(preset['value'])
            self.execute()

    def binding_changed(self):
        binding = self.variable_binding_property.get_widget_value()
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
                elif self.input.widget.widget in ['drag_int', 'slider_int', "knob_int", 'input_int']:
                    default = 0
                elif self.input.widget.widget in ['combo', 'text_input', 'radio_group']:
                    default = ''
                v = Node.app.add_variable(variable_name, default_value=default)
            if v:
                self.variable_name = variable_name
                self.variable = v
                self.input.attach_to_variable(v)
                self.variable.attach_client(self)
                self.output.set_label(self.variable_name)
                self.variable_update()

    def custom_setup(self, from_file):
        if self.variable_name != '':
            self.bind_to_variable(self.variable_name)
        if self.start_value is not None:
            self.input.set(self.start_value)

    def options_changed(self):
        if self.min_property is not None and self.max_property is not None:
            self.min = self.min_property.get_widget_value()
            self.max = self.max_property.get_widget_value()
            self.input.widget.set_limits(self.min, self.max)

        if self.format_property is not None:
            self.format = self.format_property.get_widget_value()
            self.input.widget.set_format(self.format)

        width = self.width_option.get_widget_value()
        dpg.set_item_width(self.input.widget.uuid, width)
        # height = self.height_option.get_widget_value()
        # dpg.set_item_height(self.input.widget.uuid, height)

    def value_changed(self):
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
        if self.inputs[0].fresh_input:
            in_data = self.inputs[0].get_data()
            t = type(in_data)
            if t == str:
                value = in_data.split(' ')
            elif t == list:
                value = in_data
            elif t in [float, int, bool]:
                value = in_data
            else:
                if self.input.widget.widget == 'text_input':
                    value = str(in_data)
            if self.variable is not None:
                self.variable.set(value, from_client=self)
        else:
            value = dpg.get_value(self.value)
            if type(value) == str:
                value = value.split(' ')
                if len(value) == 1:
                    value = value[0]
            if self.variable is not None:
                self.variable.set(value, from_client=self)
        if self.input.widget.widget == 'text_input':
            size = dpg.get_text_size(self.input.get_widget_value(), font=dpg.get_item_font(self.input.widget.uuid))
            if size is None:
                size = [80, 14]
            width = size[0]
            if width > 2048:
                width = 2048
            if width > dpg.get_item_width(self.input.widget.uuid):
                dpg.set_item_width(self.input.widget.uuid, width)
        self.outputs[0].send(value)

    def update(self, propagate=True):
        value = dpg.get_value(self.value)
        if type(value) == str:
            value = value.split(' ')
            if len(value) == 1:
                value = value[0]
        if self.variable is not None and propagate:
            self.variable.set(value, from_client=self)

        if self.input.widget.widget == 'text_input':
            size = dpg.get_text_size(self.input.get_widget_value(), font=dpg.get_item_font(self.input.widget.uuid))
            if size is None:
                size = [80, 14]
            width = size[0]
            if width > 1024:
                width = 1024
            if width > dpg.get_item_width(self.input.widget.uuid):
                dpg.set_item_width(self.input.widget.uuid, width)
        self.outputs[0].send(value)


class VectorNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = VectorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.max_component_count = 32
        self.format = '%.3f'

        self.current_component_count = self.arg_as_int(default_value=4)

        self.input = self.add_input("in", triggers_execution=True)

        self.component_properties = []
        for i in range(self.max_component_count):
            cp = self.add_property('##' + str(i), widget_type='drag_float', callback=self.component_changed)
            self.component_properties.append(cp)

        self.output = self.add_output('out')

        self.component_count_property = self.add_option('component count', widget_type='drag_int', default_value=self.current_component_count, callback=self.component_count_changed)
        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value=self.format, callback=self.change_format)

    def get_preset_state(self):
        preset = {}
        values = []
        for i in range(self.current_component_count):
            values.append(self.component_properties[i].get_widget_value())
        preset['values'] = values
        return preset

    def set_preset_state(self, preset):
        if 'values' in preset:
            values = preset['values']
            count = len(values)
            if count != self.current_component_count:
                self.component_count_property.set(count)
                self.component_count_changed()
            for i in range(self.current_component_count):
                self.component_properties[i].widget.set(values[i])
            self.execute()

    def custom_setup(self, from_file):
        for i in range(self.max_component_count):
            if i < self.current_component_count:
                dpg.show_item(self.component_properties[i].uuid)
            else:
                dpg.hide_item(self.component_properties[i].uuid)

    def component_count_changed(self):
        self.current_component_count = self.component_count_property.get_widget_value()
        if self.current_component_count > self.max_component_count:
            self.current_component_count = self.max_component_count
            self.component_count_property.set(self.current_component_count)
        for i in range(self.max_component_count):
            if i < self.current_component_count:
                dpg.show_item(self.component_properties[i].uuid)
            else:
                dpg.hide_item(self.component_properties[i].uuid)

    def component_changed(self):
        self.execute()

    def change_format(self):
        self.format = self.format_option.get_widget_value()
        for i in range(self.max_component_count):
            dpg.configure_item(self.component_properties[i].widget.uuid, format=self.format)

    def execute(self):
        if self.input.fresh_input:
            value = self.input.get_received_data()
            t = type(value)
            if t == list:
                value = np.array(value)
                t = np.ndarray
            elif t in [float, int, np.double, np.int64]:
                self.current_component_count = 1
                value = np.array([value])
                t = np.ndarray
            if t == np.ndarray:
                if self.current_component_count != value.size:
                    self.component_count_property.set(value.size)
                self.current_component_count = value.size
                ar = value.reshape((value.size))
                if self.current_component_count > self.max_component_count:
                    self.current_component_count = self.max_component_count
                for i in range(self.max_component_count):
                    if i < self.current_component_count:
                        dpg.show_item(self.component_properties[i].uuid)
                        self.component_properties[i].set(any_to_float(ar[i]))
                    else:
                        dpg.hide_item(self.component_properties[i].uuid)
                self.output.set_value(value)
        else:
            output_array = np.ndarray((self.current_component_count))
            for i in range(self.current_component_count):
                output_array[i] = self.component_properties[i].get_widget_value()
            self.output.set_value(output_array)
        self.output.send()


class PrintNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PrintNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.precision = 3
        self.format_string = '{:.3f}'
        self.input = self.add_input('in', triggers_execution=True)

        self.precision_option = self.add_option(label='precision', widget_type='drag_int', default_value=self.precision, min=0, max=32, callback=self.change_format)

    def change_format(self):
        self.precision = self.precision_option.get_widget_value()
        if self.precision < 0:
            self.precision = 0
            self.precision_option.set(0)
        self.format_string = '{:.' + str(self.precision) + 'f}'

    def print_list(self, list):
        print('[', end='')
        n = len(list)
        end = ' '
        for i, d in enumerate(list):
            if i == n - 1:
                end = ''
            tt = type(d)
            if tt in [int, np.int64, bool, np.bool_, str]:
                print(d, end=end)
            elif tt in [float, np.double]:
                print(self.format_string.format(d), end=end)
            elif tt == list:
                self.print_list(d)
        print(']')

    def execute(self):
        data = self.input.get_received_data()
        t = type(data)
        if t in [int, np.int64, bool, np.bool_, str]:
            print(data)
        elif t in [float, np.double]:
            print(self.format_string.format(data))
        elif t == list:
            self.print_list(data)
        elif t == np.ndarray:
            np.set_printoptions(precision=self.precision)
            print(data)


class LoadActionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = LoadActionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.first_time = True

        self.message_list = []
        if len(args) > 0:
            for arg in args:
                self.message_list.append(arg)
        self.message_string = ' '.join(self.message_list)

        self.input = self.add_input('trigger', triggers_execution=True)
        self.load_action_property = self.add_property(label='##loadActionString', widget_type='text_input', default_value=self.message_string)
        self.output = self.add_output("out")
        self.add_frame_task()

    def frame_task(self):
        if self.first_time:
            self.first_time = False
            self.remove_frame_tasks()
            self.output.send(self.message_list)

    def execute(self):
        self.output.send(self.message_list)


class PlotNode(Node):
    mousing_plot = None

    @staticmethod
    def factory(name, data, args=None):
        node = PlotNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.heat_map_style = 6
        self.heat_scroll_style = 5
        self.plot_style = 0
        self.xy_plot_style = 1

        self.sample_count = 200
        self.min_y = -1.0
        self.max_y = 1.0
        default_color_map = 'viridis'
        self.style = -1
        if label == 'plot':
            self.style_type = 'line'
            self.style = 0
            self.update_style = 'input is stream of samples'
            default_color_map = 'deep'
        elif label == 'heat_scroll':
            self.style = self.heat_scroll_style
            self.style_type = label
            self.update_style = 'input is multi-channel sample'
        elif label == 'heat_map':
            self.style = 6
            self.style_type = label
            self.update_style = 'buffer holds one sample of input'
        elif label == 'profile':
            self.style_type = 'bar'
            self.sample_count = 16
            self.min_y = 0
            self.max_y = 1.0
            self.style = 0
            self.update_style = 'input is stream of samples'
            default_color_map = 'deep'

        self.width = 300
        self.height = 128
        self.min_x = 0
        self.max_x = self.sample_count
        self.format = ''
        self.rows = 1
        self.elapser = 0

        self.hovered = False

        self.range = 1.0
        self.offset = 0.0
        self.x_axis = dpg.generate_uuid()
        self.y_axis = dpg.generate_uuid()
        self.plot_data_tag = dpg.generate_uuid()
        self.plot_tag = dpg.generate_uuid()

        self.x_data = np.linspace(0, self.sample_count, self.sample_count)
        self.roll_along_x = False

        self.input = self.add_input("y", triggers_execution=True)

        self.input_x = None
        if self.style == 1:
            self.input_x = self.add_input("x")

        self.output = self.add_output('')
        self.plot_display = self.add_display('')
        self.plot_display.submit_callback = self.submit_display

        self.style_property = self.add_option('style', widget_type='combo', default_value=self.style_type, callback=self.change_style_property)
        self.style_property.widget.combo_items = ['line', 'scatter', 'stair', 'stem', 'bar', 'heat_map', 'heat_scroll']

        self.heat_map_colour_property = self.add_option('color', widget_type='combo', default_value=default_color_map, callback=self.change_colormap)
        self.heat_map_colour_property.widget.combo_items = ['deep', 'dark', 'pastel', 'paired', 'viridis', 'plasma', 'hot', 'cool', 'pink', 'jet', 'twilight', 'red-blue', 'brown-bluegreen', 'pink-yellowgreen', 'spectral', 'greys']

        self.sample_count_option = self.add_option(label='sample count', widget_type='drag_int', default_value=self.sample_count, max=3840, callback=self.change_sample_count)
        self.width_option = self.add_option(label='width', widget_type='drag_int', default_value=self.width, max=3840, callback=self.change_size)
        self.height_option = self.add_option(label='height', widget_type='drag_int', default_value=self.height, max=3840, callback=self.change_size)

        self.min_x_option = self.add_option(label='min x', widget_type='drag_float', default_value=self.min_x, max=3840, callback=self.change_range)
        self.min_x_option.widget.speed = .01

        self.max_x_option = self.add_option(label='max x', widget_type='drag_float', default_value=self.max_x, max=3840, callback=self.change_range)
        self.max_x_option.widget.speed = .01

        self.min_y_option = self.add_option(label='min y', widget_type='drag_float', default_value=self.min_y, max=3840, callback=self.change_range)
        self.min_y_option.widget.speed = .01

        self.max_y_option = self.add_option(label='max y', widget_type='drag_float', default_value=self.max_y, max=3840, callback=self.change_range)
        self.max_y_option.widget.speed = .01

        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value='', callback=self.change_format)

        self.continuous_output = self.add_option(label='continuous output', widget_type='checkbox', default_value=False)

        self.lock = threading.Lock()
        self.plotter = None
        self.was_drawing = False
        self.add_frame_task()
        self.last_pos = [0, 0]
        self.hold_format = self.format

    def submit_display(self):
        with dpg.plot(label='', tag=self.plot_tag, height=self.height, width=self.width, no_title=True) as self.plotter:
            if self.style in [self.heat_map_style, self.heat_scroll_style]:

                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis, no_tick_labels=True)

    def custom_setup(self, from_file):
        self.reallocate_buffer()

        if self.style in [self.plot_style, self.xy_plot_style]:
            if self.style_type == 'bar':
                self.min_y = 0.0
            else:
                self.min_y = -1.0

            self.min_y_option.set(self.min_y)
            self.max_y = 1.0
            self.max_y_option.set(self.max_y)
            dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)
            buffer = self.y_data.get_buffer(block=True)
            if buffer is not None:
                dpg.add_line_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag)
            self.y_data.release_buffer()
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Deep)
        elif self.style == self.heat_scroll_style:
            self.min_y = 0.0
            self.min_y_option.set(0.0)
            self.max_y = 1.0
            self.max_y_option.set(1.0)
            dpg.set_axis_limits(self.y_axis, 0, 1)
            dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
            buffer = self.y_data.get_buffer(block=True)
            if buffer is not None:
                dpg.add_heat_series(x=buffer, rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                tag=self.plot_data_tag, format=self.format, scale_min=0, scale_max=1)
                self.y_data.release_buffer()
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            self.change_range()
        elif self.style == self.heat_map_style:
            self.min_y = 0.0
            self.min_y_option.set(0.0)
            self.max_y = 1.0
            self.max_y_option.set(1.0)
            self.format = '%.3f'
            dpg.set_axis_limits(self.y_axis, 0, 1)
            dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
            buffer = self.y_data.get_buffer(block=True)
            if buffer is not None:
                dpg.add_heat_series(x=buffer.ravel(), rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                    tag=self.plot_data_tag, format=self.format, scale_min=self.min_y, scale_max=self.max_y)
                self.y_data.release_buffer()
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            self.change_range()
            self.format_option.set(self.format)
        self.change_style_property()

    def save_custom_setup(self, container):
        self.lock.acquire(blocking=True)
        if self.label == 'profile':
            container['data'] = self.y_data.get_buffer().tolist()
            self.y_data.release_buffer()
        self.lock.release()

    def load_custom_setup(self, container):
        self.lock.acquire(blocking=True)
        if 'data' in container:
            data = np.array(container['data'])
            if len(data.shape) == 1:
                self.y_data.update(data)
            elif len(data.shape) == 2 and data.shape[0] == 1:
                self.y_data.update(data[0])
            buffer = self.y_data.get_buffer()
            dpg.set_value(self.plot_data_tag, [self.x_data, buffer.ravel()])
            self.y_data.release_buffer()
        self.lock.release()

    def change_colormap(self):
        colormap = self.heat_map_colour_property.get_widget_value()
        if colormap == 'deep':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Deep)
        elif colormap == 'dark':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Dark)
        elif colormap == 'pastel':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Pastel)
        elif colormap == 'paired':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Paired)
        elif colormap == 'viridis':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
        elif colormap == 'plasma':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Plasma)
        elif colormap == 'hot':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Hot)
        elif colormap == 'cool':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Cool)
        elif colormap == 'pink':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Pink)
        elif colormap == 'jet':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Jet)
        elif colormap == 'twilight':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Twilight)
        elif colormap == 'red-blue':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_RdBu)
        elif colormap == 'brown-bluegreen':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_BrBG)
        elif colormap == 'pink-yellowgreen':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_PiYG)
        elif colormap == 'spectral':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Spectral)
        elif colormap == 'greys':
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Greys)

    def frame_task(self):
        if PlotNode.mousing_plot == self.plotter or PlotNode.mousing_plot is None:
            x = 0
            y = 0
            ref_pos = [-1, -1]
            if self.was_drawing:
                if not dpg.is_mouse_button_down(0):
                    self.was_drawing = False
                    self.output.send(self.y_data.get_buffer()[0])
                    self.y_data.release_buffer()
                    PlotNode.mousing_plot = None
                else:
                    editor = self.app.get_current_editor()
                    if editor is not None:
                        node_padding = editor.node_scalers[dpg.mvNodeStyleVar_NodePadding]
                        window_padding = self.app.window_padding
                        plot_padding = 10
                        mouse = dpg.get_mouse_pos(local=True)
                        pos_x = dpg.get_item_pos(self.plotter)[0] + plot_padding + node_padding[0] + window_padding[0]
                        pos_y = dpg.get_item_pos(self.plotter)[1] + plot_padding + node_padding[1] + window_padding[1] + 4  # 4 is from unknown source

                        size = dpg.get_item_rect_size(self.plotter)
                        size[0] -= (2 * plot_padding)
                        size[1] -= (2 * plot_padding)
                        x_scale = self.sample_count / size[0]
                        y_scale = self.range / size[1]

                        off_x = mouse[0] - pos_x
                        off_y = mouse[1] - pos_y
                        unit_x = off_x * x_scale
                        unit_y = off_y * y_scale
                        unit_y = self.max_y - unit_y
                        if unit_x < 0:
                            unit_x = 0
                        elif unit_x >= self.sample_count:
                            unit_x = self.sample_count - 1
                        if unit_y < self.min_y:
                            unit_y = self.min_y
                        elif unit_y > self.max_y:
                            unit_y = self.max_y
                        x = unit_x
                        y = unit_y
                        ref_pos = [x, y]
                        x = int(x)

            if dpg.is_item_hovered(self.plotter):
                if dpg.is_mouse_button_down(0):
                    if self.hovered and not self.was_drawing:
                        PlotNode.mousing_plot = self.plotter
                        self.was_drawing = True
                else:
                    self.hovered = True
                    self.last_pos = [-1, -1]
                    if self.was_drawing:
                        self.was_drawing = False
                        PlotNode.mousing_plot = None

            else:
                self.hovered = False
                if not dpg.is_mouse_button_down(0):
                    self.was_drawing = False
                    PlotNode.mousing_plot = None

            if self.was_drawing:
                if self.last_pos[0] != -1:
                    last_y = self.last_pos[1]
                    last_x = int(round(self.last_pos[0]))
                    change_x = x - last_x
                    change_y = y - last_y
                    if change_x > 0:
                        for i in range(last_x, x):
                            interpolated_y = ((i - last_x) / change_x) * change_y + last_y
                            self.y_data.set_value(i, interpolated_y)
                    else:
                        for i in range(x, last_x):
                            interpolated_y = ((i - x) / change_x) * change_y + last_y
                            self.y_data.set_value(i, interpolated_y)
                if ref_pos[0] != -1:
                    self.last_pos = ref_pos
                    self.y_data.set_value(x, y)
                    self.y_data.set_write_pos(0)
                    self.update_plot()
                    self.was_drawing = True
                    if self.continuous_output.get_widget_value():
                        self.output.send(self.y_data.get_buffer()[0])
                        self.y_data.release_buffer()

    def value_dragged(self):
        if not dpg.is_mouse_button_down(0):
            return
        self.value_changed()

    def buffer_changed(self, buffer):
        self.sample_count_option.set(self.sample_count)
        self.min_x_option.set(0)
        self.max_x_option.set(self.sample_count)
        dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
        if self.style != 0:
            if dpg.does_item_exist(self.plot_data_tag):
                dpg.configure_item(self.plot_data_tag, rows=self.y_data.breadth, cols=self.y_data.sample_count)
        self.change_range()

    def change_sample_count_no_lock(self):
        self.sample_count_option.set(self.sample_count)
        self.min_x_option.set(0)
        self.max_x_option.set(self.sample_count)
        dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
        del self.y_data
        self.reallocate_buffer()
        del self.x_data
        self.x_data = np.linspace(0, self.sample_count, self.sample_count)
        dpg.configure_item(self.plot_data_tag, rows=self.y_data.breadth, cols=self.y_data.sample_count)
        self.change_range()

    def reallocate_buffer(self):
        self.y_data = RollingBuffer((self.sample_count, self.rows), roll_along_x=False)
        self.y_data.owner = self
        self.y_data.buffer_changed_callback = self.buffer_changed
        self.y_data.set_update_style(self.update_style)

    def change_style_property(self):
        self.style_type = self.style_property.get_widget_value()
        if self.style_type == 'heat_map':
            self.style = self.heat_map_style
            self.update_style = 'buffer holds one sample of input'
        elif self.style_type == 'heat_scroll':
            self.style = self.heat_scroll_style
            self.update_style = 'input is multi-channel sample'
        else:
            self.style = self.plot_style
            self.heat_map_colour_property.set('deep')
            self.update_style = 'input is stream of samples'
        self.y_data.set_update_style(self.update_style)
        if self.style_type != 'heat_map' and self.sample_count_option.get_widget_value() == 1:
            self.sample_count_option.set(200)
        self.change_sample_count()

    def change_sample_count(self):
        self.lock.acquire(blocking=True)
        self.sample_count = self.sample_count_option.get_widget_value()
        if self.sample_count < 1:
            self.sample_count = 1
            self.sample_count_option.set(self.sample_count)
        del self.x_data
        del self.y_data
        self.x_data = np.linspace(0, self.sample_count, self.sample_count)
        self.reallocate_buffer()

        self.min_x_option.set(0)
        self.max_x_option.set(self.sample_count)

        dpg.delete_item(self.plot_data_tag)
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            if self.style == self.plot_style:
                if self.style_type == 'line':
                    dpg.add_line_series(self.x_data, buffer, parent=self.y_axis, tag=self.plot_data_tag)
                elif self.style_type == 'scatter':
                    dpg.add_scatter_series(self.x_data, buffer, parent=self.y_axis, tag=self.plot_data_tag)
                elif self.style_type == 'stair':
                    dpg.add_stair_series(self.x_data, buffer, parent=self.y_axis, tag=self.plot_data_tag)
                elif self.style_type == 'stem':
                    dpg.add_stem_series(self.x_data, buffer, parent=self.y_axis, tag=self.plot_data_tag)
                elif self.style_type == 'bar':
                    dpg.add_bar_series(self.x_data, buffer, parent=self.y_axis, tag=self.plot_data_tag)
                self.change_colormap()
            else:
                dpg.set_axis_limits(self.y_axis, 0, 1)
                dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
                dpg.add_heat_series(x=buffer, rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                    tag=self.plot_data_tag, format=self.format, scale_min=self.min_y, scale_max=self.max_y)
                self.change_colormap()
            self.y_data.release_buffer()
        self.change_range()
        self.lock.release()

    def change_range(self):
        if self.style in [self.heat_map_style, self.heat_scroll_style]:
            self.max_y = self.max_y_option.get_widget_value()
            self.min_y = self.min_y_option.get_widget_value()
            self.range = self.max_y - self.min_y
            self.offset = - self.min_y
            dpg.set_axis_limits(self.x_axis, self.min_x_option.get_widget_value() / self.sample_count, self.max_x_option.get_widget_value() / self.sample_count)
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        else:
            self.max_y = self.max_y_option.get_widget_value()
            self.min_y = self.min_y_option.get_widget_value()
            self.range = self.max_y - self.min_y
            self.offset = - self.min_y
            dpg.set_axis_limits(self.y_axis, self.min_y_option.get_widget_value(), self.max_y_option.get_widget_value())
            dpg.set_axis_limits(self.x_axis, self.min_x_option.get_widget_value(), self.max_x_option.get_widget_value())

    def change_size(self):
        dpg.set_item_width(self.plot_tag, self.width_option.get_widget_value())
        dpg.set_item_height(self.plot_tag, self.height_option.get_widget_value())

    def change_format(self):
        self.format = self.format_option.get_widget_value()
        if self.style in [self.heat_scroll_style, self.heat_map_style]:
            dpg.configure_item(self.plot_data_tag, format=self.format)

    def get_preset_state(self):
        preset = {}
        data = self.y_data.get_buffer()[0]
        preset['data'] = data.tolist()
        self.y_data.release_buffer()
        return preset

    def set_preset_state(self, preset):
        if 'data' in preset:
            data = preset['data']
            data = np.array(data, dtype=np.float)
            self.y_data.set_write_pos(0)
            self.y_data.update(data)
            self.execute()

    def execute(self):
        self.lock.acquire(blocking=True)
        if self.input.fresh_input:   # standard plot
            data = self.input.get_received_data()
            t = type(data)
            if t == str:
                if data == 'dump':
                    self.output.send(self.y_data.get_buffer()[0])
                    self.y_data.release_buffer()
            if self.style == self.plot_style:
                if t in [float, np.double, int, np.int64, bool, np.bool_]:
                    ii = any_to_array(float(data))
                    self.y_data.update(ii)
                elif t == list:
                    length = len(data)
                    if length > self.sample_count:
                        length = self.sample_count
                    ii = list_to_array(data)
                    self.y_data.update(ii)
                elif t == np.ndarray:
                    if len(data.shape) == 1:
                        self.y_data.update(data)
            elif self.style == self.heat_scroll_style:  # heat_scroll ... input might be list or array
                if t not in [list, np.ndarray]:
                    ii = any_to_array(data)
                    ii = (ii + self.offset) / self.range
                    self.y_data.update(ii)
                elif t == list:
                    rows = len(data)
                    if rows != self.rows:
                        self.rows = rows
                        self.sample_count_option.set(self.sample_count)
                    ii = list_to_array(data).reshape((rows, 1))
                    ii = (ii + self.offset) / self.range
                    self.y_data.update(ii)
                elif t == np.ndarray:
                    rows = data.size
                    if rows != self.rows:
                        self.rows = rows
                    ii = (data.reshape((rows, 1)) + self.offset) / self.range
                    self.y_data.update(ii)

            elif self.style == self.heat_map_style:  # heat map
                if t not in [list, np.ndarray]:
                    rows = 1
                    sample_count = 1
                    if rows != self.rows or sample_count != self.sample_count:
                        self.rows = rows
                        self.sample_count = sample_count
                    self.y_data.update(np.array([data]))

                elif t == list:
                    h_data, _, _ = list_to_hybrid_list(data)
                    rows = len(h_data)
                    if rows != self.rows or self.sample_count != 1:
                        self.rows = rows
                        self.sample_count = 1
                    ii = list_to_array(h_data).reshape((rows, self.sample_count))
                    self.y_data.update(ii)

                elif t == np.ndarray:
                    dims = len(data.shape)
                    sample_count = 1
                    rows = data.shape[0]
                    if dims > 1:
                        sample_count = data.shape[1]
                    if rows != self.rows or self.sample_count != sample_count:
                        self.rows = rows
                        self.sample_count = sample_count
                    self.y_data.update(data)

        if self.style == self.xy_plot_style:
            if self.input_x.fresh_input:
                self.x_data[1:] = self.x_data[0:-1]
                self.x_data[0] = self.input_x.get_received_data()
        if self.style == self.plot_style:
            buffer = self.y_data.get_buffer()
            if buffer is not None:
                dpg.set_value(self.plot_data_tag, [self.x_data, buffer.ravel()])
                self.y_data.release_buffer()
        elif self.style in [self.heat_scroll_style, self.heat_map_style]:
            buffer = self.y_data.get_buffer()
            forced_format = False
            if len(buffer.shape) == 1:
                if self.width / self.sample_count < 40:
                    forced_format = True
                    if len(self.format) > 0:
                        self.hold_format = self.format
                        self.format_option.set('')
                        self.change_format()
            else:
                if self.width / self.sample_count < 40:
                    forced_format = True
                    if len(self.format) > 0:
                        self.hold_format = self.format
                        self.format_option.set('')
                        self.change_format()

                elif len(buffer.shape) > 1 and (self.height / buffer.shape[0]) < 16:
                    forced_format = True
                    if len(self.format) > 0:
                        self.hold_format = self.format
                        self.format_option.set('')
                        self.change_format()

            if not forced_format and self.hold_format != self.format:
                self.format = self.hold_format
                self.format_option.set(self.hold_format)
                self.change_format()

            if buffer is not None:
                dpg.set_value(self.plot_data_tag, [buffer.ravel(), self.x_data])
                self.y_data.release_buffer()

        self.lock.release()
        self.send_all()

    def update_plot(self):
        if self.style == self.xy_plot_style:
            if self.input_x.fresh_input:
                self.x_data[1:] = self.x_data[0:-1]
                self.x_data[0] = self.input_x.get_received_data()
        if self.style == self.plot_style:
            buffer = self.y_data.get_buffer()
            if buffer is not None:
                dpg.set_value(self.plot_data_tag, [self.x_data, buffer.ravel()])
                self.y_data.release_buffer()
        elif self.style in [self.heat_scroll_style, self.heat_map_style]:
            buffer = self.y_data.get_buffer()
            if buffer is not None:
                dpg.set_value(self.plot_data_tag, [buffer.ravel(), self.x_data])
                self.y_data.release_buffer()


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

        self.input = self.add_input("", triggers_execution=True, widget_type='color_picker', widget_width=128, callback=self.color_changed)
        self.output = self.add_output("")
        self.hue_wheel_option = self.add_option('hue_wheel', widget_type='checkbox', default_value=self.wheel, callback=self.hue_wheel_changed)
        self.alpha_option = self.add_option('alpha', widget_type='checkbox', default_value=self.alpha, callback=self.alpha_changed)
        self.inputs_option = self.add_option('inputs', widget_type='checkbox', default_value=self.has_inputs, callback=self.inputs_changed)

    def inputs_changed(self):
        has_inputs = self.inputs_option.get_widget_value()
        if has_inputs != self.has_inputs:
            if has_inputs:
                dpg.configure_item(self.input.widget.uuid, no_inputs=False)
            else:
                dpg.configure_item(self.input.widget.uuid, no_inputs=True)
            self.has_inputs = has_inputs

    def hue_wheel_changed(self):
        wheel = self.hue_wheel_option.get_widget_value()
        if wheel != self.wheel:
            if wheel:
                dpg.configure_item(self.input.widget.uuid, picker_mode=dpg.mvColorPicker_wheel)
            else:
                dpg.configure_item(self.input.widget.uuid, picker_mode=dpg.mvColorPicker_bar)
            self.wheel = wheel

    def alpha_changed(self):
        alpha = self.alpha_option.get_widget_value()
        if alpha != self.alpha:
            if alpha:
                dpg.configure_item(self.input.widget.uuid, no_alpha=False)
                dpg.configure_item(self.input.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf)
            else:
                data = self.input.get_widget_value()
                if data is not None:
                    data[3] = 255
                    self.input.widget.set(data)
                dpg.configure_item(self.input.widget.uuid, no_alpha=True)
                dpg.configure_item(self.input.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)
            self.alpha = alpha

    def color_changed(self, input=None):
        self.execute()

    def get_preset_state(self):
        preset = {}
        preset['color'] = list(self.input.get_widget_value())
        return preset

    def set_preset_state(self, preset):
        if 'color' in preset:
            color_val = preset['color']
            self.input.widget.set(tuple(color_val))
            self.execute()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            self.input.widget.set(tuple(data))
        else:
            data = list(self.input.get_widget_value())
        self.output.send(data)


