import dearpygui.dearpygui as dpg
import math
import time
import platform
import traceback
import numpy as np
import torch

from dpg_system.node import Node, SaveDialog, LoadDialog
import threading
from dpg_system.conversion_utils import *
from dpg_system.matrix_nodes import RollingBuffer

def _log_frame_error_once(node):
    if not getattr(node, '_frame_error_logged', False):
        traceback.print_exc()
        node._frame_error_logged = True


def register_interface_nodes():
    Node.app.register_node("menu", MenuNode.factory)
    Node.app.register_node("toggle", ToggleNode.factory)
    Node.app.register_node("set_reset", ToggleNode.factory)
    Node.app.register_node("button", ButtonNode.factory)
    Node.app.register_node("b", ButtonNode.factory)
    Node.app.register_node("pan_view", PanViewNode.factory)
    Node.app.register_node("home_view", HomeViewNode.factory)
    Node.app.register_node("mouse", MouseNode.factory)
    Node.app.register_node("float", ValueNode.factory)
    Node.app.register_node("int", ValueNode.factory)
    Node.app.register_node("slider", ValueNode.factory)
    Node.app.register_node("message", ValueNode.factory)
    Node.app.register_node("text", ValueNode.factory)
    # NOT 'display' -- that collides with patch_library/display.json, and
    # PlaceholderNameNode.execute() would run both the node and patcher
    # branches, using the placeholder after it had already been removed.
    Node.app.register_node("text_display", ValueNode.factory)
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

    Node.app.register_node('slider_bank', SliderBankNode.factory)
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
    Node.app.register_node('shape_seq', ShapeSequencerNode.factory)
    Node.app.register_node('shape_sequencer', ShapeSequencerNode.factory)
    Node.app.register_node('function_sequencer', ShapeSequencerNode.factory)


_view_button_chromeless_theme = None


def _get_view_button_chromeless_theme():
    global _view_button_chromeless_theme
    if _view_button_chromeless_theme is None:
        with dpg.theme() as _view_button_chromeless_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
    return _view_button_chromeless_theme


class _HideTitleBarMixin:
    """Adds a `hide_title_bar` checkbox option that renders the node chromeless
    (transparent background/outline, no label) like ClosePatchNode."""

    def _add_hide_title_bar_option(self, default_value=True):
        self.hide_title_bar = self.add_option(
            'hide_title_bar', widget_type='checkbox',
            default_value=default_value, callback=self._apply_title_bar_visibility)

    def _apply_title_bar_visibility(self):
        if self.hide_title_bar():
            dpg.bind_item_theme(self.uuid, _get_view_button_chromeless_theme())
            dpg.configure_item(self.uuid, label='')
        else:
            # Re-run visibility to restore the right base theme (global / locked / do_not_delete).
            self.set_visibility(getattr(self, 'visibility', 'show_all'))

    def set_custom_visibility(self):
        if self.hide_title_bar():
            dpg.configure_item(self.uuid, label='')
            dpg.bind_item_theme(self.uuid, _get_view_button_chromeless_theme())


class _ViewButtonNodeMixin(_HideTitleBarMixin):
    """Adds the chromeless option plus a `title` text option that drives the
    button's label/width, and 28px button height, for view-control nodes."""

    def _add_view_button_options(self, default_title=''):
        self.title = self.add_option('title', widget_type='text_input',
                                     default_value=default_title, callback=self.title_changed)
        self._add_hide_title_bar_option(default_value=True)

    def title_changed(self):
        new_title = self.title()
        dpg.set_item_label(self.input.widget.uuid, new_title)
        if new_title == '':
            dpg.set_item_width(self.input.widget.uuid, 14)
        else:
            width = self.input.widget.get_label_width(minimum_width=14)
            dpg.set_item_width(self.input.widget.uuid, width)
        dpg.set_item_height(self.input.widget.uuid, 28)

    def _apply_view_button_styling(self):
        dpg.set_item_height(self.input.widget.uuid, 28)
        if self.title() != '':
            self.title_changed()
        self._apply_title_bar_visibility()


class ButtonNode(_HideTitleBarMixin, Node):
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
        self.color = self.add_option('color', widget_type='color_picker', default_value=[0.0, 0.0, 0.0, 0.0], callback=self.color_changed)
        self._add_hide_title_bar_option(default_value=False)
        self._color_theme = None

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

    def _rebuild_color_theme(self):
        c = self.color()
        if c is None or len(c) < 4 or c[3] <= 0:
            self._color_theme = None
            return
        r, g, b, a = c[0], c[1], c[2], c[3]
        base = (int(r), int(g), int(b), int(a))
        hov = (min(base[0] + 30, 255), min(base[1] + 30, 255), min(base[2] + 30, 255), base[3])
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, base, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, base, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hov, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        self._color_theme = theme

    def color_changed(self):
        self._rebuild_color_theme()
        if time.time() >= self.target_time and dpg.does_item_exist(self.input.widget.uuid):
            dpg.bind_item_theme(
                self.input.widget.uuid,
                self._color_theme if self._color_theme is not None else Node.inactive_theme,
            )

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
        self._apply_title_bar_visibility()
        self.color_changed()

    def custom_cleanup(self):
        self.remove_frame_tasks()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            if dpg.does_item_exist(self.input.widget.uuid):
                dpg.bind_item_theme(
                    self.input.widget.uuid,
                    self._color_theme if self._color_theme is not None else Node.inactive_theme,
                )
            self.remove_frame_tasks()

    def execute(self):
        self.output.send(self.message())


class PanViewNode(_ViewButtonNodeMixin, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PanViewNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        default_h = 0
        default_v = 0
        if args is not None:
            if len(args) > 0:
                v, t = decode_arg(args, 0)
                if t in (int, float):
                    default_h = int(v)
            if len(args) > 1:
                v, t = decode_arg(args, 1)
                if t in (int, float):
                    default_v = int(v)

        self.input = self.add_input('', triggers_execution=True, widget_type='button', widget_width=14, callback=self.clicked_function)
        self._add_view_button_options(default_title='pan')
        self.h_offset = self.add_option('h_offset', widget_type='input_int', default_value=default_h)
        self.v_offset = self.add_option('v_offset', widget_type='input_int', default_value=default_v)

    def custom_create(self, from_file):
        self._apply_view_button_styling()

    def clicked_function(self, input=None):
        self.do_pan()

    def execute(self):
        self.do_pan()

    def do_pan(self):
        editor = Node.app.get_current_editor()
        if editor is None or getattr(editor, 'presenting', False):
            return
        h = self.h_offset()
        v = self.v_offset()
        if h == 0 and v == 0:
            return
        editor.pan_nodes(-h, -v)


class HomeViewNode(_ViewButtonNodeMixin, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = HomeViewNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('', triggers_execution=True, widget_type='button', widget_width=14, callback=self.clicked_function)
        self._add_view_button_options(default_title='home')

    def custom_create(self, from_file):
        self._apply_view_button_styling()

    def clicked_function(self, input=None):
        # Defer to the next frame so dpg has settled all click-time layout
        # (focus follow, widget activation) before we read screen positions.
        # Calling home_nodes() directly here behaves differently from the
        # 'h' key handler — it can alternate between two views.
        self._home_pending = True
        self.add_frame_task()

    def execute(self):
        self._home_pending = True
        self.add_frame_task()

    def frame_task(self):
        if getattr(self, '_home_pending', False):
            self._home_pending = False
            self.do_home()
        self.remove_frame_tasks()

    def do_home(self):
        editor = Node.app.get_current_editor()
        if editor is None or getattr(editor, 'presenting', False):
            return
        editor.home_nodes()


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
        self.choices = self.args_as_list(ordered_args) or ['']
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
            pos = dpg.get_mouse_pos(local=False)
            if pos is not None:
                self.mouse_pos = pos
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
                except Exception:
                    print('error restoring patch:')
                    traceback.print_exc()
                finally:
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
            if len(incoming) == 1 and isinstance(incoming[0], (list, tuple)) and len(incoming[0]) == 2:
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
            if len(incoming) == self.columns and incoming and isinstance(incoming[0], (list, tuple)):
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
            if v is not None:
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
                    if isinstance(received[0], str):
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
            if isinstance(data, (int, float, np.ndarray, torch.Tensor)) and not isinstance(data, bool):
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
        elif base_name == 'display':
            return TextDisplayNode(name, data, args)
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

        if self.input is not None and self.input.widget is not None:
            self.input.widget.wants_resize_handle = True

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

        # Widget Width (DPG's knob_float is fixed-size, so no width option for knobs)
        default_width = getattr(self, 'widget_width', 100)
        if 'knob' not in label:
            self.width_option = self.add_option(
                'width', widget_type='drag_int', default_value=default_width,
                callback=self.options_changed
            )
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
        if self.width_option is not None:
            self.width_option.widget.set(adjusted_width)

        trigger = self.input.widget.trigger_widget
        if trigger:
            dpg.set_item_width(trigger, trigger_size)

    def options_changed(self):
        # A knob has no width option -- DPG's knob_float is a fixed size, so
        # ValueNode deliberately skips creating one. Without this guard the
        # callback raises while a saved knob's options are being restored, and
        # the node is dropped from the patch with only a console message.
        if self.width_option is None:
            return
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

            if v is not None:
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
        self.install_resize_handle()

    def install_resize_handle(self):
        self.add_resize_handle(self.input.widget, axis='x', width_option=self.width_option)

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
            except (TypeError, ValueError):
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
                except (TypeError, ValueError):
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

    def install_resize_handle(self):
        pass


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
            self.output.set_strip_returns(False)

        self.height_option = self.add_option('height', widget_type='drag_int', default_value=200,
                                             callback=self.options_changed)
        self.wrap_option = self.add_option('wrap', widget_type='checkbox', default_value=True,
                                           callback=self.options_changed)

    def custom_create(self, from_file):
        dpg.set_item_height(self.input.widget.uuid, self.height_option())
        super().custom_create(from_file)
        self.input.widget.rewrap()

    def install_resize_handle(self):
        self.resize_handle = self.add_resize_handle(
            self.input.widget, axis='xy',
            width_option=self.width_option, height_option=self.height_option,
            on_resize=self.on_resized
        )

    def on_resized(self, new_w, new_h):
        # Hard-wrapped text does not re-flow on its own.
        self.input.widget.rewrap()

    def options_changed(self):
        super().options_changed()
        if self.height_option:
            h = self.height_option()
            dpg.set_item_height(self.input.widget.uuid, h)
            if getattr(self, 'resize_handle', None) and dpg.does_item_exist(self.resize_handle.uuid):
                dpg.set_item_height(self.resize_handle.uuid, h)
        if getattr(self, 'wrap_option', None):
            self.input.widget.set_wrap_enabled(self.wrap_option())


class TextDisplayNode(TextEditorNode):
    """Read-only wrapped text view. Keeps the tail of long input and scrolls
    to the bottom as new text arrives."""

    def setup_specific_ui(self, args):
        self.widget_type = 'text_display'
        self.widget_width = 400
        self.input = self.add_string_input('###text in', triggers_execution=True, widget_type=self.widget_type,
                                           widget_uuid=self.value, widget_width=self.widget_width)
        self.input.set_strip_returns(False)
        if self.param_name is not None:
            self.output = self.add_output(self.param_name + ' out')
        else:
            self.output = self.add_string_output('string out')
            self.output.set_strip_returns(False)

        self.height_option = self.add_option('height', widget_type='drag_int', default_value=200,
                                             callback=self.options_changed)
        self.max_lines_option = self.add_option('max_lines', widget_type='drag_int', default_value=500,
                                                callback=self.options_changed)
        self.autoscroll_option = self.add_option('autoscroll', widget_type='checkbox', default_value=True,
                                                 callback=self.options_changed)
        self.add_option('copy to clipboard', widget_type='button', callback=self.copy_text)
        # wrap is native here (ImGui wraps add_text), so there is no wrap option.
        self.wrap_option = None

    def custom_create(self, from_file):
        super().custom_create(from_file)
        self.add_frame_task()

    def frame_task(self):
        # Scroll pinning has to happen on the main thread; incoming text may
        # arrive on another one.
        self.input.widget.service_scroll()

    def on_resized(self, new_w, new_h):
        self.input.widget.refit()

    def copy_text(self):
        self.input.widget.copy_to_clipboard()

    def options_changed(self):
        super().options_changed()
        if getattr(self, 'max_lines_option', None):
            self.input.widget.set_max_lines(any_to_int(self.max_lines_option()))
        if getattr(self, 'autoscroll_option', None):
            self.input.widget.autoscroll = any_to_bool(self.autoscroll_option())
        self.input.widget.refit()


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
            dim1 = any_to_int(args[0])
        if len(args) > 1:
            dim2 = any_to_int(args[1])

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
            cp.name_archive.append(str(i))
            self.component_properties.append(cp)

        self.zero_input = self.add_input('zero', widget_type='button', callback=self.zero)
        self.vector_format_input = self.add_input('output type', widget_type='combo', default_value='numpy', callback=self.vector_format_changed)
        self.vector_format_input.name_archive.append('vector format')
        if Node.app.torch_available:
            self.vector_format_input.widget.combo_items = ['numpy', 'torch', 'list']
        else:
            self.vector_format_input.widget.combo_items = ['numpy', 'list']
        self.output = self.add_output('out')

        self.component_count_property = self.add_option('component count', widget_type='drag_int', default_value=self.current_dims[0], callback=self.component_count_changed)
        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value=self.format, callback=self.change_format)
        self.all_inputs_trigger_option = self.add_option('all inputs trigger', widget_type='checkbox', default_value=True)
        self.width_option = self.add_option('width', widget_type='drag_int', default_value=self.component_widget_width, callback=self.width_changed)
        self.save_option = self.add_option('save', widget_type='button', callback=self._save_values)
        self.load_option = self.add_option('load', widget_type='button', callback=self._load_values)
        if self.current_dims[1] == 1:
            self.output_vector = np.zeros(self.current_dims[0])
        else:
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

    def _save_values(self):
        SaveDialog(self, callback=self._save_file_callback, extensions=['.npy'],
                   default_filename='vector2d.npy')

    def _save_file_callback(self, save_path):
        if not save_path:
            return
        if not save_path.endswith('.npy'):
            save_path += '.npy'
        try:
            dim1 = self.current_dims[0]
            values = [any_to_list(self.component_properties[i]()) for i in range(dim1)]
            payload = {'values': values, 'dims': list(self.current_dims)}
            np.save(save_path, payload)
            print(f'Vector2DNode: saved values to {save_path}')
        except Exception as e:
            print(f'Vector2DNode: error saving values: {e}')

    def _load_values(self):
        LoadDialog(self, callback=self._load_file_callback, extensions=['.npy'])

    def _load_file_callback(self, load_path):
        if not load_path:
            return
        try:
            data = np.load(load_path, allow_pickle=True).item()
            values = data.get('values', [])
            if not values:
                print(f'Vector2DNode: no values found in {load_path}')
                return
            if len(values) != self.current_dims[0]:
                new_count = min(len(values), self.max_component_count)
                self.component_count_property.set(new_count)
                self.component_count_changed()
            for i in range(min(len(values), self.current_dims[0])):
                self.component_properties[i].widget.set(any_to_list(values[i]))
            self.execute()
            print(f'Vector2DNode: loaded values from {load_path}')
        except Exception as e:
            print(f'Vector2DNode: error loading values: {e}')

    def get_preset_state(self):
        preset = {}
        values = []
        for i in range(self.current_dims[0]):
            values.append(any_to_list(self.output_vector[i]))
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

    def _collect_component_array(self):
        # Each component widget (drag_float_n) returns its value as a list of
        # length dim2 -- even when dim2 == 1. Stacking the rows yields a
        # (dim1, dim2) array; assigning these lists element-wise into a 1-D
        # array raises "setting an array element with a sequence" on numpy >= 1.25.
        dim1 = self.current_dims[0]
        dim2 = self.current_dims[1] if len(self.current_dims) > 1 else 1
        rows = [any_to_list(self.component_properties[i]()) for i in range(dim1)]
        values = np.array(rows, dtype=float)
        if dim2 == 1:
            values = values.reshape(dim1)
        return values

    def send(self):
        output_array = self._collect_component_array()
        self.output.send(output_array)

    def load_custom(self, container):
        values = self._collect_component_array()
        vf = self.vector_format_input()
        if vf == 'torch':
            self.output_vector = torch.from_numpy(values)
        elif vf == 'list':
            self.output_vector = values.tolist()
        else:
            self.output_vector = values
        self.output.set_value(self.output_vector)

    def execute(self):
        if self.input.fresh_input:
            value = self.input()
            t = type(value)
            if t == str:
                if value == 'bang':
                    self.output.send(self._collect_component_array())
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
                if value.ndim == 2 and value.shape[1] == 1:
                    value = value.reshape(-1)
                self.current_dims = list(value.shape)
                if len(self.current_dims) < 2:
                    self.current_dims.append(1)
                if self.vector_format_input() == 'list':
                    self.output_vector = value.tolist()
                elif self.vector_format_input() == 'numpy':
                    self.output_vector = value.copy()
                elif self.vector_format_input() == 'torch':
                    self.output_vector = torch.from_numpy(value)

            elif t == torch.Tensor:
                if value.dim() == 2 and value.shape[1] == 1:
                    value = value.reshape(-1)
                self.current_dims = list(value.shape)
                if len(self.current_dims) < 2:
                    self.current_dims.append(1)
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
        if end is None or end == '\\n':
            self.end.set('\n')

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
            values = any_to_array(self.input()).astype(float) * 256.0
            self.input.widget.set(values)
        else:
            values = any_to_array(self.input()).astype(float)
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
        character = ''
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
            _log_frame_error_once(self)

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
        self.pad_width = 150
        self.pad_height = 150
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
        self.width_option = self.add_option(
            'width', widget_type='drag_int', default_value=self.pad_width,
            callback=self._size_changed
        )
        self.height_option = self.add_option(
            'height', widget_type='drag_int', default_value=self.pad_height,
            callback=self._size_changed
        )
        self.marker_size_option = self.add_option(
            'marker size', widget_type='drag_int', default_value=self.marker_size,
            callback=self._marker_size_changed
        )

    def submit_display(self):
        with dpg.plot(
            label='', tag=self.plot_tag,
            height=self.pad_height, width=self.pad_width,
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
            self._build_scatter_theme()
            dpg.add_scatter_series([0.0], [0.0], tag=self.scatter_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.scatter_tag, self.scatter_theme_tag)
        self._install_resize_handle()

    def _install_resize_handle(self):
        from dpg_system.node import ResizeHandle, _get_resize_handle_theme
        btn_uuid = dpg.add_button(parent=self.plot_display.uuid, label='', width=self.pad_width, height=4)
        handle = ResizeHandle(
            btn_uuid, self.plot_tag, axis='xy',
            width_option=self.width_option, height_option=self.height_option,
            sync_width=True, sync_height=False
        )
        dpg.set_item_user_data(btn_uuid, handle)
        dpg.bind_item_handler_registry(btn_uuid, "resize handle handler")
        dpg.bind_item_theme(btn_uuid, _get_resize_handle_theme())
        self.resize_handle = handle

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
            _log_frame_error_once(self)

    def _range_changed(self):
        self.range_val = self.range_option()
        dpg.set_axis_limits(self.x_axis_tag, -self.range_val, self.range_val)
        dpg.set_axis_limits(self.y_axis_tag, -self.range_val, self.range_val)

    def _size_changed(self):
        self.pad_width = self.width_option()
        self.pad_height = self.height_option()
        dpg.set_item_width(self.plot_tag, self.pad_width)
        dpg.set_item_height(self.plot_tag, self.pad_height)
        rh = getattr(self, 'resize_handle', None)
        if rh is not None and dpg.does_item_exist(rh.uuid):
            dpg.set_item_width(rh.uuid, self.pad_width)

    def _build_scatter_theme(self):
        if dpg.does_item_exist(self.scatter_theme_tag):
            dpg.delete_item(self.scatter_theme_tag)
        with dpg.theme(tag=self.scatter_theme_tag):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, self.marker_size, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 0, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 0, 255), category=dpg.mvThemeCat_Plots)

    def _marker_size_changed(self):
        self.marker_size = self.marker_size_option()
        self._build_scatter_theme()
        if dpg.does_item_exist(self.scatter_tag):
            dpg.bind_item_theme(self.scatter_tag, self.scatter_theme_tag)


def breakpoint_ease(t, curve_val):
    """Segment shape, as the Schlick bias function.

    f(t) = t / (t + (1 - t) * k)

    Curvature is nonzero at t=0, so a segment never starts flat.
      curve_val = 0: k=1, linear
      curve_val > 0: k<1, convex (f(t) > t, bows above the straight line)
      curve_val < 0: k>1, concave (f(t) < t, bows below it)
    """
    if abs(curve_val) < 0.001:
        return t
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if curve_val > 0:
        k = 1.0 / (1.0 + curve_val)
    else:
        k = 1.0 - curve_val
    return t / (t + (1.0 - t) * k)


# Slack around a breakpoint plot's range, in pixels. A control point sitting
# exactly on a boundary would otherwise straddle the edge of the plot rect, and
# the half that gets clipped is the half you have to click -- which made corner
# points nearly impossible to pick up.
BREAKPOINT_GRAB_PADDING = 9


def breakpoint_axis_limits(x_max, y_min, y_max, plot_width, plot_height,
                           pixels=BREAKPOINT_GRAB_PADDING):
    """(x_low, x_high, y_low, y_high) for a plot framed with grab slack.

    The slack is derived from the plot's pixel size, so it stays the same few
    pixels whatever the ranges are and however the plot is resized.
    """
    span_x = abs(x_max) or 1.0
    span_y = abs(y_max - y_min) or 1.0
    pad_x = span_x * pixels / max(plot_width, 1)
    pad_y = span_y * pixels / max(plot_height, 1)
    return -pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def breakpoint_point_color(curve_val):
    """Control point color: yellow (linear) -> green (curved)."""
    blend = min(1.0, abs(curve_val) / 2.0)
    r = int(255 * (1 - blend))
    g = 255
    b = int(100 * blend)
    return (r, g, b, 255)


def breakpoint_value_at(points, x_val):
    """Read a breakpoint function at x_val, respecting per-segment curvature.

    `points` is a list of {'x', 'y', 'curve'} dicts in any order; outside the
    span the end values are held, which is what makes a function usable as a
    lookup for an input that can wander past its range.
    """
    sorted_pts = sorted(points, key=lambda p: p['x'])
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
            # Flip curve direction for descending segments, so dragging up
            # always bows the curve upward on screen
            effective_curve = (curve_val if sorted_pts[i + 1]['y'] >= sorted_pts[i]['y']
                               else -curve_val)
            eased = breakpoint_ease(t, effective_curve)
            return sorted_pts[i]['y'] + eased * (sorted_pts[i + 1]['y'] - sorted_pts[i]['y'])

    return sorted_pts[-1]['y']


def breakpoint_line_data(points, samples_per_curve=32):
    """x/y arrays for drawing a breakpoint function, curved segments sampled."""
    sorted_pts = sorted(points, key=lambda p: p['x'])
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
            y0 = p_cur['y']
            y1 = p_next['y']
            x0 = p_cur['x']
            x1 = p_next['x']
            effective_curve = curve_val if y1 >= y0 else -curve_val
            for s in range(samples_per_curve):
                t = s / float(samples_per_curve)
                eased = breakpoint_ease(t, effective_curve)
                x_data.append(x0 + t * (x1 - x0))
                y_data.append(y0 + eased * (y1 - y0))
        else:
            # Linear: just the start point
            x_data.append(p_cur['x'])
            y_data.append(p_cur['y'])

    x_data.append(sorted_pts[-1]['x'])
    y_data.append(sorted_pts[-1]['y'])

    return x_data, y_data


class BreakpointEditor:
    """An editable breakpoint curve on its own plot.

    Owns the plot, the control points and the mouse handling; the host node
    owns where the plot sits and what the curve means. The gestures are the
    envelope node's: drag a point to move it, right-click to add one or, near
    an existing point, to remove it, shift + left-drag a segment to bend it.

    A host builds one in __init__, calls submit() from a display's
    submit_callback, poll() from its frame task, and hears about edits through
    on_change.
    """

    SAMPLES_PER_CURVE = 32
    REMOVE_THRESHOLD = 0.08

    def __init__(self, x_max=1.0, y_min=0.0, y_max=1.0, width=200, height=100,
                 on_change=None, on_resize=None, line_color=(80, 140, 255),
                 name='curve'):
        self.name = name        # only used to name the node in diagnostics
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.width = int(width)
        self.height = int(height)
        self.on_change = on_change
        self.on_resize = on_resize
        self.line_color = line_color

        self.points = self.straight_line()
        self.point_tags = []
        self.ready = False

        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.line_tag = dpg.generate_uuid()
        self.resize_handle = None

        self.left_was_down = False
        self.right_was_down = False
        self.curving = False
        self.curving_index = -1
        self.curve_drag_start_screen_y = 0.0
        self.curve_drag_start_val = 0.0

    def straight_line(self):
        """The identity: an untouched editor passes its input through."""
        return [{'x': 0.0, 'y': self.y_min, 'curve': 0.0},
                {'x': self.x_max, 'y': self.y_max, 'curve': 0.0}]

    # -- construction --------------------------------------------------------

    def submit(self, display_uuid, width_option=None, height_option=None):
        """Build the plot. Call inside the host display's submit_callback."""
        with dpg.theme() as self.line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, self.line_color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                    category=dpg.mvThemeCat_Plots)

        with dpg.plot(label='', tag=self.plot_tag,
                      height=self.height, width=self.width,
                      no_title=True, no_menus=True, no_box_select=True,
                      no_mouse_pos=True):
            dpg.add_plot_axis(dpg.mvXAxis, label='', tag=self.x_axis_tag,
                              no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label='', tag=self.y_axis_tag,
                              no_tick_labels=True)
            dpg.add_line_series([], [], parent=self.y_axis_tag, tag=self.line_tag)
            dpg.bind_item_theme(self.line_tag, self.line_theme)

        self.ready = True
        self.apply_axis_limits()
        self.rebuild_points()
        if width_option is not None or height_option is not None:
            self.install_resize_handle(display_uuid, width_option, height_option)

    def install_resize_handle(self, display_uuid, width_option, height_option):
        from dpg_system.node import ResizeHandle, _get_resize_handle_theme
        btn_uuid = dpg.add_button(parent=display_uuid, label='',
                                  width=self.width, height=4)
        handle = ResizeHandle(
            btn_uuid, self.plot_tag, axis='xy',
            width_option=width_option, height_option=height_option,
            sync_width=True, sync_height=False,
            on_resize=self.handle_resized
        )
        dpg.set_item_user_data(btn_uuid, handle)
        dpg.bind_item_handler_registry(btn_uuid, "resize handle handler")
        dpg.bind_item_theme(btn_uuid, _get_resize_handle_theme())
        self.resize_handle = handle

    def handle_resized(self, new_w, new_h):
        # The handle sizes the plot itself but does not fire the size options'
        # callbacks, and the grab slack is measured in pixels.
        self.width = int(new_w)
        self.height = int(new_h)
        self.apply_axis_limits()
        if self.on_resize is not None:
            self.on_resize(self.width, self.height)

    def set_size(self, width, height):
        self.width = int(width)
        self.height = int(height)
        if not self.ready:
            return
        dpg.set_item_width(self.plot_tag, self.width)
        dpg.set_item_height(self.plot_tag, self.height)
        if self.resize_handle is not None and dpg.does_item_exist(self.resize_handle.uuid):
            dpg.set_item_width(self.resize_handle.uuid, self.width)
        self.apply_axis_limits()

    def set_ranges(self, x_max=None, y_min=None, y_max=None, notify=True):
        """Change the axes, pulling any out-of-range points back inside."""
        if x_max is not None:
            self.x_max = float(x_max)
        if y_min is not None:
            self.y_min = float(y_min)
        if y_max is not None:
            self.y_max = float(y_max)
        moved = False
        for point in self.points:
            x = max(0.0, min(self.x_max, point['x']))
            y = max(self.y_min, min(self.y_max, point['y']))
            if x != point['x'] or y != point['y']:
                point['x'], point['y'] = x, y
                moved = True
        self.apply_axis_limits()
        self.rebuild_points()
        if moved and notify:
            self.changed()

    def apply_axis_limits(self):
        if not self.ready:
            return
        x_low, x_high, y_low, y_high = breakpoint_axis_limits(
            self.x_max, self.y_min, self.y_max, self.width, self.height)
        dpg.set_axis_limits(self.x_axis_tag, x_low, x_high)
        dpg.set_axis_limits(self.y_axis_tag, y_low, y_high)

    # -- the curve -----------------------------------------------------------

    def set_points(self, points, notify=True):
        """Replace the curve. Accepts dicts or [x, y, curve] sequences."""
        parsed = []
        for entry in points or ():
            if isinstance(entry, dict):
                parsed.append({'x': any_to_float(entry.get('x', 0.0)),
                               'y': any_to_float(entry.get('y', 0.0)),
                               'curve': any_to_float(entry.get('curve', 0.0))})
                continue
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            parsed.append({'x': any_to_float(entry[0]),
                           'y': any_to_float(entry[1]),
                           'curve': any_to_float(entry[2]) if len(entry) > 2 else 0.0})
        if len(parsed) < 2:
            return False
        self.points = sorted(parsed, key=lambda p: p['x'])
        self.rebuild_points()
        if notify:
            self.changed()
        return True

    def get_points(self):
        return [[p['x'], p['y'], p.get('curve', 0.0)]
                for p in sorted(self.points, key=lambda p: p['x'])]

    def value_at(self, x):
        return breakpoint_value_at(self.points, x)

    def line_data(self, samples_per_curve=None):
        return breakpoint_line_data(
            self.points,
            samples_per_curve or self.SAMPLES_PER_CURVE)

    def changed(self):
        self.update_line()
        if self.on_change is not None:
            self.on_change()

    def update_line(self):
        if not self.ready:
            return
        x_data, y_data = self.line_data()
        dpg.set_value(self.line_tag, [x_data, y_data])

    # -- control points ------------------------------------------------------

    def rebuild_points(self):
        if not self.ready:
            return
        for tag in self.point_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.point_tags = [self.create_point_widget(p) for p in self.points]
        self.update_line()

    def create_point_widget(self, point):
        tag = dpg.generate_uuid()
        dpg.add_drag_point(tag=tag, default_value=(point['x'], point['y']),
                           color=breakpoint_point_color(point.get('curve', 0.0)),
                           parent=self.plot_tag)
        return tag

    def update_point_color(self, index):
        if index >= len(self.point_tags):
            return
        tag = self.point_tags[index]
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, color=breakpoint_point_color(
                self.points[index].get('curve', 0.0)))

    def add_point(self, x, y, curve=0.0):
        point = {'x': max(0.0, min(self.x_max, x)),
                 'y': max(self.y_min, min(self.y_max, y)),
                 'curve': curve}
        self.points.append(point)
        self.point_tags.append(self.create_point_widget(point))
        return point

    def remove_point(self, index):
        if len(self.points) <= 2 or not 0 <= index < len(self.points):
            return False
        self.points.pop(index)
        tag = self.point_tags.pop(index)
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        return True

    def x_bounds(self, index):
        """How far a point may travel in x: up to its neighbours, not past.

        A breakpoint function has to be single-valued in x, and letting points
        cross also makes their indices meaningless -- move point 3 past point 4
        and the next message addressed to 'point 3' lands on a different point,
        so a run of them piles several points onto one spot. Neighbours may
        meet (a vertical step is a legitimate shape) but never swap.
        """
        if not 0 <= index < len(self.points):
            return 0.0, self.x_max
        order = sorted(range(len(self.points)),
                       key=lambda i: self.points[i]['x'])
        place = order.index(index)
        low = self.points[order[place - 1]]['x'] if place > 0 else 0.0
        high = (self.points[order[place + 1]]['x']
                if place < len(order) - 1 else self.x_max)
        return max(0.0, low), min(self.x_max, high)

    def point_at(self, index):
        """Internal index of the index-th point in x order, or None.

        The stored order is whatever editing left behind -- points are not
        re-sorted as they are dragged, since the drag widgets are matched to
        them by position in the list. Anything addressing points from outside
        means the nth from the left, so it comes through here.
        """
        order = sorted(range(len(self.points)), key=lambda i: self.points[i]['x'])
        if 0 <= index < len(order):
            return order[index]
        return None

    def move_point(self, index, x=None, y=None, curve=None, notify=True):
        """Set the nth point (in x order). Only the values given are changed."""
        target = self.point_at(index)
        if target is None:
            return False
        point = self.points[target]
        if x is not None:
            low, high = self.x_bounds(target)
            point['x'] = max(low, min(high, any_to_float(x)))
        if y is not None:
            point['y'] = max(self.y_min, min(self.y_max, any_to_float(y)))
        if curve is not None:
            point['curve'] = max(-16.0, min(16.0, any_to_float(curve)))
            self.update_point_color(target)
        if target < len(self.point_tags) and dpg.does_item_exist(self.point_tags[target]):
            dpg.set_value(self.point_tags[target], [point['x'], point['y']])
        if notify:
            self.changed()
        return True

    # -- messages ------------------------------------------------------------

    MESSAGES = ('point', 'curve', 'add', 'remove', 'line')

    def handle_message(self, message, message_data):
        """The editing gestures as messages, so a curve can be driven by patch.

            point <n> <x> <y> [curve]   move the nth point from the left
            curve <n> <amount>          bend the segment leaving the nth point
            add <x> <y> [curve]         add a point
            remove <n>                  remove the nth point
            line                        back to a straight line

        Point values are clamped to the ranges, so a sequence driving this
        cannot push the curve outside itself. Anything that cannot be applied
        at all -- an index past the end, a missing argument -- is reported
        rather than passed over in silence, since a message that does nothing
        looks exactly like a message that never arrived.

        Indices count from 0, as everywhere else in the patch.
        """
        if message == 'line':
            self.set_points(self.straight_line())
            return True

        numbers = [any_to_float(value) for value in message_data]

        if message == 'point':
            if len(numbers) < 3:
                return self._reject(message, message_data,
                                    'needs an index, an x and a y')
            return self._move_or_reject(
                message, message_data, int(numbers[0]), numbers[1], numbers[2],
                numbers[3] if len(numbers) > 3 else None)

        if message == 'curve':
            if len(numbers) < 2:
                return self._reject(message, message_data,
                                    'needs an index and an amount')
            return self._move_or_reject(message, message_data, int(numbers[0]),
                                        curve=numbers[1])

        if message == 'add':
            if len(numbers) < 2:
                return self._reject(message, message_data, 'needs an x and a y')
            self.add_point(numbers[0], numbers[1],
                           numbers[2] if len(numbers) > 2 else 0.0)
            self.changed()
            return True

        if message == 'remove':
            if not numbers:
                return self._reject(message, message_data, 'needs an index')
            target = self.point_at(int(numbers[0]))
            if target is None:
                return self._reject(message, message_data,
                                    self._index_hint(int(numbers[0])))
            if not self.remove_point(target):
                return self._reject(message, message_data,
                                    'a curve cannot have fewer than 2 points')
            self.changed()
            return True

        return False

    def _move_or_reject(self, message, message_data, index, x=None, y=None,
                        curve=None):
        if self.move_point(index, x, y, curve):
            return True
        return self._reject(message, message_data, self._index_hint(index))

    def _index_hint(self, index):
        count = len(self.points)
        return ('there is no point ' + str(index) + ' -- the curve has '
                + str(count) + ' points, so indices run 0..' + str(count - 1))

    def _reject(self, message, message_data, reason):
        text = ' '.join([str(message)] + [str(value) for value in message_data])
        print(self.name + ": '" + text + "' ignored -- " + reason)
        return False

    def nearest_point(self, mx, my):
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

    def segment_at_x(self, mx):
        """Index of the point that starts the segment containing mx."""
        sorted_pts = sorted(self.points, key=lambda p: p['x'])
        for i in range(len(sorted_pts) - 1):
            if sorted_pts[i]['x'] <= mx <= sorted_pts[i + 1]['x']:
                return self.points.index(sorted_pts[i])
        return -1

    def hovered(self):
        return self.ready and dpg.is_item_hovered(self.plot_tag)

    def interacting(self):
        """True while the user is working on the plot."""
        return self.curving or (self.left_was_down and self.hovered())

    # -- interaction ---------------------------------------------------------

    def poll(self):
        """Run the mouse gestures. Call once a frame from the host."""
        if not self.ready:
            return
        shift_held = (dpg.is_key_down(dpg.mvKey_LShift)
                      or dpg.is_key_down(dpg.mvKey_RShift))
        left_down = dpg.is_mouse_button_down(0)
        right_down = dpg.is_mouse_button_down(1)

        # --- Curvature drag (shift + left-click held) ---
        if self.curving:
            if left_down and shift_held:
                # Screen pixels rather than plot coordinates, so the drag is
                # not clipped at the edge of the plot
                screen_pos = dpg.get_mouse_pos()
                delta_px = self.curve_drag_start_screen_y - screen_pos[1]
                new_curve = self.curve_drag_start_val + delta_px / 18.0
                new_curve = max(-16.0, min(16.0, new_curve))
                index = self.curving_index
                if 0 <= index < len(self.points):
                    self.points[index]['curve'] = new_curve
                    self.update_point_color(index)
                    self.changed()
            else:
                self.curving = False
            self.left_was_down = left_down
            self.right_was_down = right_down
            return

        if left_down and not self.left_was_down and shift_held and self.hovered():
            plot_pos = dpg.get_plot_mouse_pos()
            if plot_pos:
                index = self.segment_at_x(plot_pos[0])
                if index >= 0:
                    self.curving = True
                    self.curving_index = index
                    self.curve_drag_start_screen_y = dpg.get_mouse_pos()[1]
                    self.curve_drag_start_val = self.points[index].get('curve', 0.0)
                    self.left_was_down = left_down
                    self.right_was_down = right_down
                    return

        # --- Poll the drag points ---
        changed = False
        for index, tag in enumerate(self.point_tags):
            if index >= len(self.points) or not dpg.does_item_exist(tag):
                continue
            pos = dpg.get_value(tag)
            # Dragged points stop at their neighbours for the same reason
            # messaged ones do -- see x_bounds.
            low, high = self.x_bounds(index)
            x = max(low, min(high, pos[0]))
            y = max(self.y_min, min(self.y_max, pos[1]))
            if abs(x - pos[0]) > 1e-6 or abs(y - pos[1]) > 1e-6:
                dpg.set_value(tag, [x, y])
            point = self.points[index]
            if abs(x - point['x']) > 1e-6 or abs(y - point['y']) > 1e-6:
                point['x'] = x
                point['y'] = y
                changed = True

        # --- Right-click: add or remove a point ---
        if right_down and not self.right_was_down and self.hovered():
            plot_pos = dpg.get_plot_mouse_pos()
            if plot_pos:
                nearest_idx, nearest_dist = self.nearest_point(plot_pos[0],
                                                               plot_pos[1])
                if nearest_dist < self.REMOVE_THRESHOLD and len(self.points) > 2:
                    self.remove_point(nearest_idx)
                else:
                    x = max(0.0, min(self.x_max, plot_pos[0]))
                    y = max(self.y_min, min(self.y_max, plot_pos[1]))
                    self.add_point(x, y)
                changed = True

        if changed:
            self.changed()

        self.left_was_down = left_down
        self.right_was_down = right_down

    def set_visible(self, visible):
        """Show or hide the whole editor, plot and handle together.

        For a host that owns more than one editor and shows one at a time.
        A hidden plot cannot be hovered, so poll() goes quiet on its own.
        """
        if not self.ready:
            return
        dpg.configure_item(self.plot_tag, show=visible)
        if self.resize_handle is not None \
                and dpg.does_item_exist(self.resize_handle.uuid):
            dpg.configure_item(self.resize_handle.uuid, show=visible)


class BarEditor:
    """One bar per slot: values set by index rather than drawn through.

    BreakpointEditor is the right instrument for a shape and the wrong one for
    a specific set of values, because a curve through slot 9 has to pass over
    8 and 10 on its way. Here each bar is one slot and nothing reaches its
    neighbours, which is what picking out individual harmonics needs.

    Left-drag sets the bar under the cursor and paints across the ones you
    sweep over; right-drag clears them. Sweeping fills in the bars the mouse
    skipped between frames, so a fast gesture does not come out as a comb.

    The store is `capacity` long while only the first `count` are drawn: a host
    whose visible range changes -- additive~'s partial count -- can open and
    close it without the hidden bars being lost.

    A host builds one in __init__, calls submit() from a display's
    submit_callback, poll() from its frame task, and hears about edits through
    on_change.
    """

    def __init__(self, count=16, capacity=512, y_min=0.0, y_max=1.0,
                 width=220, height=96, on_change=None,
                 bar_color=(240, 170, 80), name='bars'):
        self.name = name        # only used to name the node in diagnostics
        self.capacity = max(1, int(capacity))
        self.count = max(1, min(self.capacity, int(count)))
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.width = int(width)
        self.height = int(height)
        self.on_change = on_change
        self.bar_color = bar_color

        self.data = np.zeros(self.capacity, dtype=np.float64)
        self.ready = False

        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.bar_tag = dpg.generate_uuid()
        self.resize_handle = None

        self.painting = False
        self.last_index = -1

    # -- construction --------------------------------------------------------

    def submit(self, display_uuid, width_option=None, height_option=None):
        """Build the plot. Call inside the host display's submit_callback."""
        with dpg.theme() as self.bar_theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, self.bar_color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, self.bar_color,
                                    category=dpg.mvThemeCat_Plots)

        with dpg.plot(label='', tag=self.plot_tag,
                      height=self.height, width=self.width,
                      no_title=True, no_menus=True, no_box_select=True,
                      no_mouse_pos=True):
            # The x labels stay on here, unlike the curve editors: the whole
            # point of bars is knowing which one is the ninth.
            dpg.add_plot_axis(dpg.mvXAxis, label='', tag=self.x_axis_tag)
            dpg.add_plot_axis(dpg.mvYAxis, label='', tag=self.y_axis_tag,
                              no_tick_labels=True)
            dpg.add_bar_series([], [], weight=0.7, parent=self.y_axis_tag,
                               tag=self.bar_tag)
            dpg.bind_item_theme(self.bar_tag, self.bar_theme)

        self.ready = True
        self.apply_axis_limits()
        self.update_bars()
        if width_option is not None or height_option is not None:
            self.install_resize_handle(display_uuid, width_option,
                                       height_option)

    def install_resize_handle(self, display_uuid, width_option, height_option):
        from dpg_system.node import ResizeHandle, _get_resize_handle_theme
        btn_uuid = dpg.add_button(parent=display_uuid, label='',
                                  width=self.width, height=4)
        handle = ResizeHandle(
            btn_uuid, self.plot_tag, axis='xy',
            width_option=width_option, height_option=height_option,
            sync_width=True, sync_height=False,
            on_resize=self.handle_resized
        )
        dpg.set_item_user_data(btn_uuid, handle)
        dpg.bind_item_handler_registry(btn_uuid, "resize handle handler")
        dpg.bind_item_theme(btn_uuid, _get_resize_handle_theme())
        self.resize_handle = handle

    def handle_resized(self, new_w, new_h):
        self.width = int(new_w)
        self.height = int(new_h)

    # -- geometry ------------------------------------------------------------

    def apply_axis_limits(self):
        if not self.ready:
            return
        # Bars are centred on their own index, so the axis runs half a bar
        # past each end and the first and last are drawn whole.
        dpg.set_axis_limits(self.x_axis_tag, 0.5, self.count + 0.5)
        dpg.set_axis_limits(self.y_axis_tag, self.y_min, self.y_max)

    def set_size(self, width, height):
        self.width = int(width)
        self.height = int(height)
        if not self.ready:
            return
        dpg.set_item_width(self.plot_tag, self.width)
        dpg.set_item_height(self.plot_tag, self.height)
        if self.resize_handle is not None \
                and dpg.does_item_exist(self.resize_handle.uuid):
            dpg.set_item_width(self.resize_handle.uuid, self.width)

    def set_count(self, count):
        """How many bars are drawn. The rest keep their values, unseen."""
        count = max(1, min(self.capacity, int(count)))
        if count == self.count:
            return
        self.count = count
        self.apply_axis_limits()
        self.update_bars()

    def set_ranges(self, y_min=None, y_max=None):
        if y_min is not None:
            self.y_min = float(y_min)
        if y_max is not None:
            self.y_max = float(y_max)
        np.clip(self.data, self.y_min, self.y_max, out=self.data)
        self.apply_axis_limits()
        self.update_bars()

    def set_visible(self, visible):
        if not self.ready:
            return
        dpg.configure_item(self.plot_tag, show=visible)
        if self.resize_handle is not None \
                and dpg.does_item_exist(self.resize_handle.uuid):
            dpg.configure_item(self.resize_handle.uuid, show=visible)

    # -- values --------------------------------------------------------------

    def set_values(self, values, notify=True):
        """Replace the store. Shorter input fills from the bottom, rest zero."""
        incoming = np.asarray(values, dtype=np.float64).reshape(-1)
        self.data[:] = 0.0
        size = min(incoming.size, self.capacity)
        if size:
            self.data[:size] = np.clip(incoming[:size], self.y_min, self.y_max)
        self.update_bars()
        if notify:
            self.changed()

    def get_values(self):
        """The whole store, one value per slot."""
        return self.data.copy()

    def get_visible(self):
        return self.data[:self.count].copy()

    def clear(self, notify=True):
        self.data[:] = 0.0
        self.update_bars()
        if notify:
            self.changed()

    def update_bars(self):
        if not self.ready:
            return
        x = np.arange(1, self.count + 1, dtype=np.float64)
        dpg.set_value(self.bar_tag, [x, self.data[:self.count].copy()])

    def changed(self):
        if self.on_change is not None:
            self.on_change()

    # -- interaction ---------------------------------------------------------

    def hovered(self):
        return self.ready and dpg.is_item_hovered(self.plot_tag)

    def interacting(self):
        return self.painting

    def paint(self, index, value):
        """Set one bar, filling in any the mouse jumped over since last frame."""
        index = max(0, min(self.count - 1, int(index)))
        value = max(self.y_min, min(self.y_max, float(value)))
        previous = self.last_index
        if previous < 0 or previous == index:
            self.data[index] = value
        else:
            # A sweep between frames crosses several bars. Ramping across them
            # rather than setting them all to the latest value keeps a diagonal
            # drag reading as the line it was drawn as.
            step = 1 if index > previous else -1
            span = abs(index - previous)
            start = self.data[previous]
            for offset in range(1, span + 1):
                position = previous + offset * step
                blend = offset / span
                self.data[position] = start + (value - start) * blend
        self.last_index = index
        self.update_bars()
        self.changed()

    def poll(self):
        """Run the mouse gestures. Call once a frame from the host."""
        if not self.ready:
            return
        left_down = dpg.is_mouse_button_down(0)
        right_down = dpg.is_mouse_button_down(1)

        if not (left_down or right_down):
            self.painting = False
            self.last_index = -1
            return

        if not self.painting:
            if not self.hovered():
                return
            self.painting = True
            self.last_index = -1

        position = dpg.get_plot_mouse_pos()
        if not position:
            return
        index = int(round(position[0])) - 1
        if index < 0 or index >= self.count:
            # Off the end of the bars: stop bridging, so coming back in does
            # not draw a ramp across everything in between.
            self.last_index = -1
            return
        self.paint(index, 0.0 if right_down else position[1])


def mode_point_color(decay):
    """Mode point color: cool (dies fast) -> warm (rings long).

    Decay multiples live on a ratio scale -- 0.5 and 2.0 are the same step
    either side of 1 -- so the blend runs on log2, two octaves each way.
    """
    value = max(0.05, min(4.0, float(decay)))
    blend = (math.log2(value) + 2.0) / 4.0
    blend = max(0.0, min(1.0, blend))
    return (int(90 + blend * 165), int(120 + blend * 50),
            int(255 - blend * 175), 255)


class ModeEditor:
    """An editable mode table on a plot: one stem per resonant mode.

    Where BreakpointEditor edits a shape and BarEditor a row of slots, this
    edits a *set*: each mode is a stem standing at its frequency ratio, as
    tall as its weight, colored by how long it rings -- cool for a mode that
    dies fast, warm for one that lasts. There is no curve through them
    because there is nothing between modes; that is what makes an object
    sound like a thing rather than a filter.

    The gestures are the breakpoint editor's, re-meant: drag a stem to tune
    it (x) and weight it (y), shift + drag vertically to set how long it
    rings, right-click empty space to add a mode, right-click a stem to
    remove it. Stems may pass each other freely -- modes have no order to
    preserve -- and messages address the nth from the left at the moment
    they arrive.

    The x axis grows and shrinks to fit the table when one is loaded, and
    keeps its tick labels: where a stem stands is the whole reading.

    A host builds one in __init__, calls submit() from a display's
    submit_callback, poll() from its frame task, and hears about edits
    through on_change.
    """

    DECAY_MIN, DECAY_MAX = 0.05, 4.0
    RATIO_MIN = 0.02
    REMOVE_THRESHOLD = 0.08

    def __init__(self, x_max=8.0, width=220, height=96, on_change=None,
                 on_resize=None, stem_color=(240, 170, 80), name='modes'):
        self.name = name        # only used to name the node in diagnostics
        self.x_max = float(x_max)
        self.width = int(width)
        self.height = int(height)
        self.on_change = on_change
        self.on_resize = on_resize
        self.stem_color = stem_color

        self.modes = [{'ratio': 1.0, 'weight': 1.0, 'decay': 1.0}]
        self.point_tags = []
        self.ready = False

        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.stem_tag = dpg.generate_uuid()
        self.resize_handle = None

        self.left_was_down = False
        self.right_was_down = False
        self.decay_dragging = False
        self.decay_index = -1
        self.decay_drag_start_screen_y = 0.0
        self.decay_drag_start_val = 1.0

    # -- construction --------------------------------------------------------

    def submit(self, display_uuid, width_option=None, height_option=None):
        """Build the plot. Call inside the host display's submit_callback."""
        with dpg.theme() as self.stem_theme:
            with dpg.theme_component(dpg.mvStemSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, self.stem_color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, self.stem_color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline,
                                    self.stem_color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                    category=dpg.mvThemeCat_Plots)

        with dpg.plot(label='', tag=self.plot_tag,
                      height=self.height, width=self.width,
                      no_title=True, no_menus=True, no_box_select=True,
                      no_mouse_pos=True):
            # Tick labels stay on: where a stem stands is the reading.
            dpg.add_plot_axis(dpg.mvXAxis, label='', tag=self.x_axis_tag)
            dpg.add_plot_axis(dpg.mvYAxis, label='', tag=self.y_axis_tag,
                              no_tick_labels=True)
            dpg.add_stem_series([], [], parent=self.y_axis_tag,
                                tag=self.stem_tag)
            dpg.bind_item_theme(self.stem_tag, self.stem_theme)

        self.ready = True
        self.apply_axis_limits()
        self.rebuild_points()
        if width_option is not None or height_option is not None:
            self.install_resize_handle(display_uuid, width_option,
                                       height_option)

    def install_resize_handle(self, display_uuid, width_option, height_option):
        from dpg_system.node import ResizeHandle, _get_resize_handle_theme
        btn_uuid = dpg.add_button(parent=display_uuid, label='',
                                  width=self.width, height=4)
        handle = ResizeHandle(
            btn_uuid, self.plot_tag, axis='xy',
            width_option=width_option, height_option=height_option,
            sync_width=True, sync_height=False,
            on_resize=self.handle_resized
        )
        dpg.set_item_user_data(btn_uuid, handle)
        dpg.bind_item_handler_registry(btn_uuid, "resize handle handler")
        dpg.bind_item_theme(btn_uuid, _get_resize_handle_theme())
        self.resize_handle = handle

    def handle_resized(self, new_w, new_h):
        self.width = int(new_w)
        self.height = int(new_h)
        self.apply_axis_limits()
        if self.on_resize is not None:
            self.on_resize(self.width, self.height)

    def set_size(self, width, height):
        self.width = int(width)
        self.height = int(height)
        if not self.ready:
            return
        dpg.set_item_width(self.plot_tag, self.width)
        dpg.set_item_height(self.plot_tag, self.height)
        if self.resize_handle is not None \
                and dpg.does_item_exist(self.resize_handle.uuid):
            dpg.set_item_width(self.resize_handle.uuid, self.width)
        self.apply_axis_limits()

    def apply_axis_limits(self):
        if not self.ready:
            return
        x_low, x_high, y_low, y_high = breakpoint_axis_limits(
            self.x_max, 0.0, 1.0, self.width, self.height)
        dpg.set_axis_limits(self.x_axis_tag, x_low, x_high)
        dpg.set_axis_limits(self.y_axis_tag, y_low, y_high)

    def fit_range(self):
        """Grow or shrink the x axis to frame the table it holds."""
        top = max(mode['ratio'] for mode in self.modes)
        wanted = max(2.0, float(math.ceil(top * 1.1)))
        if wanted != self.x_max:
            self.x_max = wanted
            self.apply_axis_limits()

    # -- the table -----------------------------------------------------------

    def set_modes(self, table, notify=True):
        """Replace the table. Accepts dicts or (ratio, weight, decay) rows."""
        parsed = []
        for entry in table or ():
            if isinstance(entry, dict):
                parsed.append({
                    'ratio': any_to_float(entry.get('ratio', 1.0)),
                    'weight': any_to_float(entry.get('weight', 1.0)),
                    'decay': any_to_float(entry.get('decay', 1.0))})
                continue
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            parsed.append({
                'ratio': any_to_float(entry[0]),
                'weight': any_to_float(entry[1]),
                'decay': (any_to_float(entry[2])
                          if len(entry) > 2 else 1.0)})
        if not parsed:
            return False
        for mode in parsed:
            mode['ratio'] = max(ModeEditor.RATIO_MIN, mode['ratio'])
            mode['weight'] = max(0.0, min(1.0, mode['weight']))
            mode['decay'] = max(ModeEditor.DECAY_MIN,
                                min(ModeEditor.DECAY_MAX, mode['decay']))
        self.modes = sorted(parsed, key=lambda m: m['ratio'])
        self.fit_range()
        self.rebuild_points()
        if notify:
            self.changed()
        return True

    def get_modes(self):
        return [[m['ratio'], m['weight'], m['decay']]
                for m in sorted(self.modes, key=lambda m: m['ratio'])]

    def changed(self):
        self.update_stems()
        if self.on_change is not None:
            self.on_change()

    def update_stems(self):
        if not self.ready:
            return
        xs = [m['ratio'] for m in self.modes]
        ys = [m['weight'] for m in self.modes]
        dpg.set_value(self.stem_tag, [xs, ys])

    # -- the points ----------------------------------------------------------

    def rebuild_points(self):
        if not self.ready:
            # No widgets to make yet, but the bookkeeping keeps step so
            # edits arriving before submit() line up with their modes.
            self.point_tags = [None] * len(self.modes)
            return
        for tag in self.point_tags:
            if tag is not None and dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.point_tags = [self.create_point_widget(m) for m in self.modes]
        self.update_stems()

    def create_point_widget(self, mode):
        # Before the plot exists there is nothing to hang a widget on;
        # rebuild_points() makes the real ones when submit() runs.
        if not self.ready:
            return None
        tag = dpg.generate_uuid()
        dpg.add_drag_point(tag=tag,
                           default_value=(mode['ratio'], mode['weight']),
                           color=mode_point_color(mode['decay']),
                           parent=self.plot_tag)
        return tag

    def update_point_color(self, index):
        if index >= len(self.point_tags):
            return
        tag = self.point_tags[index]
        if tag is not None and dpg.does_item_exist(tag):
            dpg.configure_item(tag, color=mode_point_color(
                self.modes[index]['decay']))

    def add_mode(self, ratio, weight, decay=1.0):
        mode = {'ratio': max(ModeEditor.RATIO_MIN, min(self.x_max, ratio)),
                'weight': max(0.0, min(1.0, weight)),
                'decay': max(ModeEditor.DECAY_MIN,
                             min(ModeEditor.DECAY_MAX, decay))}
        self.modes.append(mode)
        self.point_tags.append(self.create_point_widget(mode))
        return mode

    def remove_mode(self, index):
        if len(self.modes) <= 1 or not 0 <= index < len(self.modes):
            return False
        self.modes.pop(index)
        tag = self.point_tags.pop(index)
        if tag is not None and dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        return True

    def mode_at(self, index):
        """Internal index of the index-th mode in ratio order, or None.

        Stored order is whatever editing left behind, for the same reason as
        the breakpoint editor: drag widgets are matched to modes by list
        position. Messages mean the nth from the left, so they come through
        here.
        """
        order = sorted(range(len(self.modes)),
                       key=lambda i: self.modes[i]['ratio'])
        if 0 <= index < len(order):
            return order[index]
        return None

    def nearest_mode(self, mx, my):
        min_dist = float('inf')
        min_idx = -1
        x_range = max(self.x_max, 0.001)
        for i, mode in enumerate(self.modes):
            dx = (mode['ratio'] - mx) / x_range
            dy = mode['weight'] - my
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx, min_dist

    # -- messages ------------------------------------------------------------

    MESSAGES = ('mode', 'add', 'remove')

    def handle_message(self, message, message_data):
        """The editing gestures as messages, so a table can be driven by patch.

            mode <n> <ratio> <weight> [decay]   set the nth mode from the left
            add <ratio> <weight> [decay]        add a mode
            remove <n>                          remove the nth mode

        Values are clamped to the editor's ranges. Anything that cannot be
        applied is reported rather than passed over in silence.

        Indices count from 0, as everywhere else in the patch.
        """
        numbers = [any_to_float(value) for value in message_data]

        if message == 'mode':
            if len(numbers) < 3:
                return self._reject(message, message_data,
                                    'needs an index, a ratio and a weight')
            target = self.mode_at(int(numbers[0]))
            if target is None:
                return self._reject(message, message_data,
                                    self._index_hint(int(numbers[0])))
            mode = self.modes[target]
            mode['ratio'] = max(ModeEditor.RATIO_MIN,
                                min(self.x_max, numbers[1]))
            mode['weight'] = max(0.0, min(1.0, numbers[2]))
            if len(numbers) > 3:
                mode['decay'] = max(ModeEditor.DECAY_MIN,
                                    min(ModeEditor.DECAY_MAX, numbers[3]))
                self.update_point_color(target)
            if target < len(self.point_tags) \
                    and self.point_tags[target] is not None \
                    and dpg.does_item_exist(self.point_tags[target]):
                dpg.set_value(self.point_tags[target],
                              [mode['ratio'], mode['weight']])
            self.changed()
            return True

        if message == 'add':
            if len(numbers) < 2:
                return self._reject(message, message_data,
                                    'needs a ratio and a weight')
            self.add_mode(numbers[0], numbers[1],
                          numbers[2] if len(numbers) > 2 else 1.0)
            self.fit_range()
            self.changed()
            return True

        if message == 'remove':
            if not numbers:
                return self._reject(message, message_data, 'needs an index')
            target = self.mode_at(int(numbers[0]))
            if target is None:
                return self._reject(message, message_data,
                                    self._index_hint(int(numbers[0])))
            if not self.remove_mode(target):
                return self._reject(message, message_data,
                                    'a table cannot have fewer than 1 mode')
            self.changed()
            return True

        return False

    def _index_hint(self, index):
        count = len(self.modes)
        return ('there is no mode ' + str(index) + ' -- the table has '
                + str(count) + ' modes, so indices run 0..' + str(count - 1))

    def _reject(self, message, message_data, reason):
        text = ' '.join([str(message)] + [str(value) for value in message_data])
        print(self.name + ": '" + text + "' ignored -- " + reason)
        return False

    # -- interaction ---------------------------------------------------------

    def hovered(self):
        return self.ready and dpg.is_item_hovered(self.plot_tag)

    def interacting(self):
        return self.decay_dragging or (self.left_was_down and self.hovered())

    def _begin_decay_drag(self, index):
        self.decay_dragging = True
        self.decay_index = index
        self.decay_drag_start_screen_y = dpg.get_mouse_pos()[1]
        self.decay_drag_start_val = self.modes[index]['decay']

    def _hold_stem(self, index):
        """Pin a drag point to its mode, undoing any capture-drag motion."""
        if 0 <= index < len(self.point_tags):
            tag = self.point_tags[index]
            if tag is not None and dpg.does_item_exist(tag):
                mode = self.modes[index]
                dpg.set_value(tag, [mode['ratio'], mode['weight']])

    def poll(self):
        """Run the mouse gestures. Call once a frame from the host."""
        if not self.ready:
            return
        shift_held = (dpg.is_key_down(dpg.mvKey_LShift)
                      or dpg.is_key_down(dpg.mvKey_RShift))
        left_down = dpg.is_mouse_button_down(0)
        right_down = dpg.is_mouse_button_down(1)

        # --- Decay drag (shift + left-drag) ---
        if self.decay_dragging:
            index = self.decay_index
            if left_down and shift_held and 0 <= index < len(self.modes):
                screen_pos = dpg.get_mouse_pos()
                delta_px = self.decay_drag_start_screen_y - screen_pos[1]
                # Pixels to octaves of ring time: 40 px doubles or halves.
                value = self.decay_drag_start_val * (2.0 ** (delta_px / 40.0))
                value = max(ModeEditor.DECAY_MIN,
                            min(ModeEditor.DECAY_MAX, value))
                self.modes[index]['decay'] = value
                self.update_point_color(index)
                self.changed()
            else:
                self.decay_dragging = False
            # Every frame, not just at the end: if the press landed on the
            # drag point it is dragging too, and left free it would wander
            # off and retune the stem the gesture was aimed at.
            self._hold_stem(index)
            self.left_was_down = left_down
            self.right_was_down = right_down
            return

        # Shift + press on the plot: adjust the nearest mode's ring time.
        # No distance gate, matching the segment-bend gesture it is copied
        # from -- on a plot this small the nearest stem is the meant one.
        if left_down and not self.left_was_down and shift_held \
                and self.hovered():
            plot_pos = dpg.get_plot_mouse_pos()
            if plot_pos:
                index, _ = self.nearest_mode(plot_pos[0], plot_pos[1])
                if index >= 0:
                    self._begin_decay_drag(index)
                    self._hold_stem(index)
                    self.left_was_down = left_down
                    self.right_was_down = right_down
                    return

        # --- Poll the drag points ---
        changed = False
        for index, tag in enumerate(self.point_tags):
            if index >= len(self.modes) or tag is None \
                    or not dpg.does_item_exist(tag):
                continue
            pos = dpg.get_value(tag)
            mode = self.modes[index]
            moved = (abs(pos[0] - mode['ratio']) > 1e-6
                     or abs(pos[1] - mode['weight']) > 1e-6)
            if moved and shift_held and left_down:
                # The press landed on the point itself, where the plot may
                # not report hover, so the shift path above never saw it.
                # The point's own motion is the tell: same gesture, caught
                # the other way round.
                self._begin_decay_drag(index)
                self._hold_stem(index)
                self.left_was_down = left_down
                self.right_was_down = right_down
                return
            x = max(ModeEditor.RATIO_MIN, min(self.x_max, pos[0]))
            y = max(0.0, min(1.0, pos[1]))
            if abs(x - pos[0]) > 1e-6 or abs(y - pos[1]) > 1e-6:
                dpg.set_value(tag, [x, y])
            if abs(x - mode['ratio']) > 1e-6 or abs(y - mode['weight']) > 1e-6:
                mode['ratio'] = x
                mode['weight'] = y
                changed = True

        # --- Right-click: add or remove a mode ---
        if right_down and not self.right_was_down and self.hovered():
            plot_pos = dpg.get_plot_mouse_pos()
            if plot_pos:
                index, dist = self.nearest_mode(plot_pos[0], plot_pos[1])
                if dist < ModeEditor.REMOVE_THRESHOLD and len(self.modes) > 1:
                    self.remove_mode(index)
                else:
                    self.add_mode(max(ModeEditor.RATIO_MIN,
                                      min(self.x_max, plot_pos[0])),
                                  max(0.0, min(1.0, plot_pos[1])))
                changed = True

        if changed:
            self.changed()

        self.left_was_down = left_down
        self.right_was_down = right_down


class EnvelopeNode(Node):
    """A breakpoint function / envelope editor with draggable control points.

    Drag a point to move it, right-click to add one or, near an existing point,
    to remove it, shift + left-drag a segment to bend it. Send an x value to
    read the curve there; bang 'trigger' to sweep across it over 'duration',
    sending the value as the playhead goes.

    The editing is BreakpointEditor, shared with shaper~ and shape_seq, so the
    three behave identically under the mouse.

    Usage:
        envelope          - x range 0-1, y range 0-1
        envelope 10       - x range 0-10, y range 0-1
        envelope 10 5     - x range 0-10, y range 0-5
    """

    @staticmethod
    def factory(name, data, args=None):
        return EnvelopeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        x_max, y_min, y_max = 1.0, 0.0, 1.0
        self.plot_width = 300
        self.plot_height = 150

        # Parse args
        if args and len(args) > 0:
            val, t = decode_arg(args, 0)
            if t in [float, int]:
                x_max = float(val)
        if args and len(args) > 1:
            val, t = decode_arg(args, 1)
            if t in [float, int]:
                y_max = float(val)

        self.editor = BreakpointEditor(x_max=x_max, y_min=y_min, y_max=y_max,
                                       width=self.plot_width,
                                       height=self.plot_height,
                                       on_change=self._send_points,
                                       name=label)
        # The classic default shape: a triangle across the range.
        self.editor.set_points([[0.0, y_min, 0.0],
                                [x_max * 0.5, y_max, 0.0],
                                [x_max, y_min, 0.0]], notify=False)
        # 'point 1 0.5 0.8' and friends -- see BreakpointEditor.handle_message.
        for name in BreakpointEditor.MESSAGES:
            self.message_handlers[name] = self._curve_message

        # Ramp / trigger state
        self.ramp_active = False
        self.ramp_start_time = 0.0
        self.ramp_time = x_max      # default ramp duration = x range
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
            'x max', widget_type='drag_float', default_value=self.editor.x_max,
            callback=self._range_changed
        )
        self.y_min_option = self.add_option(
            'y min', widget_type='drag_float', default_value=self.editor.y_min,
            callback=self._range_changed
        )
        self.y_max_option = self.add_option(
            'y max', widget_type='drag_float', default_value=self.editor.y_max,
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

    # The editor owns the curve and the ranges; these keep the old names
    # working for anything that reads them.
    @property
    def points(self):
        return self.editor.points

    @property
    def x_max(self):
        return self.editor.x_max

    @property
    def y_min(self):
        return self.editor.y_min

    @property
    def y_max(self):
        return self.editor.y_max

    @property
    def plot_tag(self):
        return self.editor.plot_tag

    def submit_display(self):
        self.editor.submit(self.plot_display.uuid,
                           width_option=self.width_option,
                           height_option=self.height_option)
        # Playhead vertical line, on the editor's plot (empty until a ramp runs)
        dpg.add_inf_line_series(x=[], parent=self.editor.y_axis_tag,
                                tag=self.playhead_tag)

    def _curve_message(self, message='', message_data=[]):
        self.editor.handle_message(message, message_data)

    def _send_points(self):
        self.points_output.send(self.editor.get_points())

    def _interpolate_at(self, x_val):
        """Interpolate the envelope at x_val, respecting per-segment curvature."""
        return self.editor.value_at(x_val)

    def _start_ramp(self):
        """Start ramp playback (called by button or trigger input)."""
        self.ramp_active = True
        self.ramp_start_time = time.time()

    def execute(self):
        if self.trigger_input.fresh_input:
            self.trigger_input.get_received_data()
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

            self.editor.poll()

        except Exception:
            _log_frame_error_once(self)

    def save_custom(self, container):
        container['envelope_points'] = self.editor.get_points()

    def load_custom(self, container):
        if 'envelope_points' in container:
            self.editor.set_points(container['envelope_points'], notify=False)

    def _range_changed(self):
        self.editor.set_ranges(x_max=any_to_float(self.x_max_option()),
                               y_min=any_to_float(self.y_min_option()),
                               y_max=any_to_float(self.y_max_option()))

    def _size_changed(self):
        self.plot_width = any_to_int(self.width_option())
        self.plot_height = any_to_int(self.height_option())
        self.editor.set_size(self.plot_width, self.plot_height)


class ShapeSequencerNode(Node):
    """A step sequencer whose steps are functions rather than values.

    Every step holds its own breakpoint function, edited the way the envelope
    node's is -- drag the points, right-click to add or remove one, shift +
    left-drag a segment to curve it. A beat advances to the next step, samples
    the 'x' inlet, reads that step's function at x, and sends the result out.

    A plain value sequencer is the degenerate case where every function is
    flat. The interesting case is one continuous input -- effort data, a fader,
    an lfo~ through snapshot~ -- being re-interpreted step by step: this beat
    passes it through, the next inverts it, the next compresses it into the top
    of the range, the next ignores it and holds a constant.

    Beat it from clock~'s bang outlet, from metro, or by clicking the button.
    Nothing is sent between beats: the value is a sample taken at the beat, and
    it stands until the next one.

    The upper plot carries the whole sequence at once -- the step being edited
    in blue with its control points, the step last played in green, the others
    ghosted, and a marker where the input was last sampled. 'follow play'
    walks the editor along with the sequence, and suspends itself while you are
    dragging so it cannot pull the shape out from under the mouse.

    The lower plot is the profile: one bar per step, each the value that step
    would output at the input's current value, with the playing step's bar
    highlighted. Where the upper plot shows what one step does to any input,
    this shows what every step does to this input -- the shape of the phrase
    you are about to hear. It follows the x inlet continuously rather than only
    on beats, so moving the input redraws the whole profile. Click a bar to
    edit that step; since that is a deliberate choice of step, it also switches
    'follow play' off so the next beat cannot pull the editor away.

    Usage:
        shape_seq             - 8 steps, x range 0-1, y range 0-1
        shape_seq 16          - 16 steps
        shape_seq 16 10       - 16 steps, x range 0-10
        shape_seq 16 10 5     - ... and y range 0-5
    """

    SAMPLES_PER_CURVE = 32
    MAX_STEPS = 64
    DIRECTIONS = ('forward', 'reverse', 'ping pong', 'random')

    @staticmethod
    def factory(name, data, args=None):
        return ShapeSequencerNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.step_count = 8
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.plot_width = 200
        self.plot_height = 100
        self.remove_threshold = 0.08

        # Parse args
        if args and len(args) > 0:
            val, t = decode_arg(args, 0)
            if t in [float, int]:
                self.step_count = max(1, min(self.MAX_STEPS, int(val)))
        if args and len(args) > 1:
            val, t = decode_arg(args, 1)
            if t in [float, int]:
                self.x_max = float(val)
        if args and len(args) > 2:
            val, t = decode_arg(args, 2)
            if t in [float, int]:
                self.y_max = float(val)

        # One list of {'x', 'y', 'curve'} dicts per step. The edited step is
        # held by the editor -- it owns that curve's line, its control points
        # and the mouse -- while every other step exists here only as data and
        # a ghost line the node draws on the editor's plot.
        self.shapes = [self._default_shape() for _ in range(self.step_count)]
        self.editor = BreakpointEditor(x_max=self.x_max, y_min=self.y_min,
                                       y_max=self.y_max,
                                       width=self.plot_width,
                                       height=self.plot_height,
                                       on_change=self._edited,
                                       on_resize=self._editor_resized,
                                       name=label)
        self.editor.set_points(self.shapes[0], notify=False)
        # 'point 1 0.5 0.8' and friends, acting on whichever step is being
        # edited -- 'edit step <n>' selects it, being an option label. See
        # BreakpointEditor.handle_message.
        for name in BreakpointEditor.MESSAGES:
            self.message_handlers[name] = self._curve_message

        self.play_index = -1        # -1 = nothing played yet; first beat plays 0
        self.edit_index = 0
        self.ping_pong_sign = 1
        self.clipboard = None
        self.display_ready = False
        self.profile_ready = False

        # Mouse state for the profile plot; the editor keeps its own.
        self.right_was_down = False
        self.left_was_down = False

        # UUIDs
        self.marker_tag = dpg.generate_uuid()
        self.series_tags = []

        self.profile_tag = dpg.generate_uuid()
        self.profile_x_axis_tag = dpg.generate_uuid()
        self.profile_y_axis_tag = dpg.generate_uuid()
        self.profile_bars_tag = dpg.generate_uuid()
        self.profile_playing_tag = dpg.generate_uuid()
        self.profile_height = 80
        self.profile_x = None       # x the bars were last drawn for

        # Inputs. The beat and reset buttons carry a callback but do not
        # trigger execution: an incoming bang runs the widget's callback too,
        # so triggering as well would advance the sequence twice per beat.
        self.beat_input = self.add_input('beat', widget_type='button',
                                         callback=self.beat)
        self.x_input = self.add_input('x', widget_type='drag_float',
                                      default_value=0.0)
        if self.x_input.widget is not None:
            self.x_input.widget.speed = 0.01
        self.reset_input = self.add_input('reset', widget_type='button',
                                          callback=self.reset_sequence)

        self.step_display = self.add_property('step', widget_type='label',
                                              default_value='-')
        self.steps_property = self.add_property(
            'steps', widget_type='drag_int', default_value=self.step_count,
            min=1, max=self.MAX_STEPS, callback=self._steps_changed
        )
        self.direction_property = self.add_property(
            'direction', widget_type='combo', default_value='forward',
            width=110
        )
        self.direction_property.widget.combo_items = list(self.DIRECTIONS)
        self.follow_property = self.add_property(
            'follow play', widget_type='checkbox', default_value=True
        )

        self.plot_display = self.add_display('')
        self.plot_display.submit_callback = self.submit_display

        self.profile_display = self.add_display('')
        self.profile_display.submit_callback = self.submit_profile

        # Outputs
        self.value_output = self.add_output('value out')
        self.step_output = self.add_output('step out')
        self.cycle_output = self.add_output('cycle')

        # Options
        self.edit_step_option = self.add_option(
            'edit step', widget_type='drag_int', default_value=0,
            min=0, max=self.MAX_STEPS - 1, callback=self._edit_step_changed
        )
        self.ghost_option = self.add_option(
            'show other steps', widget_type='checkbox', default_value=True,
            callback=self._refresh_series
        )
        # Wide enough for the longest of the three labels, so they read as a
        # set and none of them is clipped.
        self.copy_option = self.add_option(
            'copy shape', widget_type='button', width=130,
            callback=self._copy_shape
        )
        self.paste_option = self.add_option(
            'paste shape', widget_type='button', width=130,
            callback=self._paste_shape
        )
        self.copy_all_option = self.add_option(
            'copy to all steps', widget_type='button', width=130,
            callback=self._copy_to_all
        )
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
        self.show_profile_option = self.add_option(
            'show profile', widget_type='checkbox', default_value=True,
            callback=self._profile_visibility_changed
        )
        self.profile_height_option = self.add_option(
            'profile height', widget_type='drag_int',
            default_value=self.profile_height, min=20, max=400,
            callback=self._size_changed
        )

    def _default_shape(self):
        """Identity: an unedited step passes its input through unchanged."""
        return [{'x': 0.0, 'y': self.y_min, 'curve': 0.0},
                {'x': self.x_max, 'y': self.y_max, 'curve': 0.0}]

    @property
    def plot_tag(self):
        return self.editor.plot_tag

    # -- display ------------------------------------------------------------

    def submit_display(self):
        self.play_line_theme = self._line_theme((90, 230, 120), 2.5)
        self.ghost_line_theme = self._line_theme((130, 130, 140, 110), 1.0)

        with dpg.theme() as self.marker_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker,
                                    dpg.mvPlotMarker_Circle,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 5,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 70, 70, 255),
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 70, 70, 255),
                                    category=dpg.mvThemeCat_Plots)

        # The editor builds the plot; the ghost lines and the sample marker are
        # the node's own series, drawn on it alongside the edited curve.
        self.editor.submit(self.plot_display.uuid,
                           width_option=self.width_option,
                           height_option=self.height_option)
        dpg.add_scatter_series([], [], parent=self.editor.y_axis_tag,
                               tag=self.marker_tag)
        dpg.bind_item_theme(self.marker_tag, self.marker_theme)

        self.display_ready = True
        self._rebuild_series()
        self._update_step_display()

    @staticmethod
    def _line_theme(color, weight):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                    category=dpg.mvThemeCat_Plots)
        return theme

    @staticmethod
    def _bar_theme(color):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, color,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                    category=dpg.mvThemeCat_Plots)
        return theme

    def submit_profile(self):
        """The profile: what every step would output at the current input.

        The shapes plot answers 'what does this step do to its input'; this one
        answers 'what does the sequence do to the input it has right now',
        which is the shape of the phrase you are about to hear. It is a reading
        of all the steps at one x, not a history of what was played -- change
        the input and the whole profile moves.
        """
        self.profile_bar_theme = self._bar_theme((110, 150, 210, 200))
        self.profile_playing_theme = self._bar_theme((90, 230, 120, 255))

        with dpg.plot(
            label='', tag=self.profile_tag,
            height=self.profile_height, width=self.plot_width,
            no_title=True, no_menus=True, no_box_select=True,
            no_mouse_pos=True
        ):
            dpg.add_plot_axis(dpg.mvXAxis, label='', tag=self.profile_x_axis_tag,
                              no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label='', tag=self.profile_y_axis_tag,
                              no_tick_labels=True)

            # The playing step is a second series of one bar drawn over the
            # first, since a bar series carries a single color for all its bars.
            dpg.add_bar_series([], [], weight=0.7,
                               parent=self.profile_y_axis_tag,
                               tag=self.profile_bars_tag)
            dpg.bind_item_theme(self.profile_bars_tag, self.profile_bar_theme)
            dpg.add_bar_series([], [], weight=0.7,
                               parent=self.profile_y_axis_tag,
                               tag=self.profile_playing_tag)
            dpg.bind_item_theme(self.profile_playing_tag, self.profile_playing_theme)

        # Anything that reads an option is left to custom_create: option
        # widgets are created after displays, and a widget's value is None
        # until its own create() installs the default.
        self.profile_ready = True
        self._profile_axes()

    def _profile_axes(self):
        if not self.profile_ready:
            return
        dpg.set_axis_limits(self.profile_x_axis_tag, -0.6, self.step_count - 0.4)
        # Bars grow from zero, so the axis has to reach it even when the
        # shapes themselves live entirely above or below.
        low = min(self.y_min, 0.0)
        high = max(self.y_max, 0.0)
        if high <= low:
            high = low + 1.0
        dpg.set_axis_limits(self.profile_y_axis_tag, low, high)

    def _refresh_profile(self, force=False):
        """Read every step at the current input value and redraw the bars."""
        if not self.profile_ready or not any_to_bool(self.show_profile_option()):
            return
        x_val = any_to_float(self.x_input.get_widget_value())
        if not force and self.profile_x is not None and x_val == self.profile_x:
            return
        self.profile_x = x_val

        values = [breakpoint_value_at(shape, x_val) for shape in self.shapes]
        dpg.set_value(self.profile_bars_tag,
                      [list(range(len(values))), values])
        if 0 <= self.play_index < len(values):
            dpg.set_value(self.profile_playing_tag,
                          [[self.play_index], [values[self.play_index]]])
        else:
            dpg.set_value(self.profile_playing_tag, [[], []])

    def _profile_clicked(self):
        """Click a bar to edit that step. Returns True if a bar was hit.

        Picking a step by hand is a statement that you want to stay on it, so
        this switches 'follow play' off rather than letting the next beat drag
        the editor away again. Tick the option back on to resume following.
        """
        if not self.profile_ready or not dpg.is_item_shown(self.profile_tag):
            return False
        if not dpg.is_item_hovered(self.profile_tag):
            return False
        plot_pos = dpg.get_plot_mouse_pos()
        if not plot_pos:
            return False
        index = int(round(plot_pos[0]))
        if index < 0 or index >= self.step_count:
            return False
        if any_to_bool(self.follow_property()):
            self.follow_property.widget.set(False)
        self._set_edit_index(index)
        return True

    def _profile_visibility_changed(self):
        if not self.profile_ready:
            return
        if any_to_bool(self.show_profile_option()):
            dpg.show_item(self.profile_tag)
            self._refresh_profile(force=True)
        else:
            dpg.hide_item(self.profile_tag)

    def _editor_resized(self, new_w, new_h):
        """Drag the handle and the profile keeps the width, not the height.

        The profile cannot be an extra target of the editor's handle: that
        handle resizes on both axes, and the profile is deliberately shorter
        than the shapes plot.
        """
        self.plot_width = int(new_w)
        self.plot_height = int(new_h)
        if dpg.does_item_exist(self.profile_tag):
            dpg.set_item_width(self.profile_tag, self.plot_width)

    def _curve_message(self, message='', message_data=[]):
        self.editor.handle_message(message, message_data)

    def _edited(self):
        """The editor moved: adopt its points as the edited step's shape."""
        self.shapes[self.edit_index] = [dict(p) for p in self.editor.points]
        self._refresh_profile(force=True)

    def _rebuild_series(self):
        """A ghost line per step. Only called when the step count changes.

        The edited step is not among them -- the editor draws that one -- but
        it keeps a slot so the list stays index-aligned with the shapes.
        """
        if not self.display_ready:
            return
        for tag in self.series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.series_tags = []
        for _ in range(self.step_count):
            tag = dpg.generate_uuid()
            dpg.add_line_series([], [], parent=self.editor.y_axis_tag, tag=tag)
            self.series_tags.append(tag)
        self._refresh_series()

    def _refresh_series(self):
        """Redraw the ghost lines and rebind their themes."""
        self._refresh_profile(force=True)
        if not self.display_ready:
            return
        show_others = any_to_bool(self.ghost_option())
        for index, tag in enumerate(self.series_tags):
            if not dpg.does_item_exist(tag) or index >= len(self.shapes):
                continue
            # The edited step's line belongs to the editor, so this one stays
            # empty. With the others off, nothing here is drawn at all --
            # and nothing may depend on the played step, since beats take the
            # themes-only path and would leave a stale line behind.
            if show_others and index != self.edit_index:
                x_data, y_data = breakpoint_line_data(self.shapes[index],
                                                      self.SAMPLES_PER_CURVE)
            else:
                x_data, y_data = [], []
            dpg.set_value(tag, [x_data, y_data])
            dpg.bind_item_theme(tag, self._theme_for(index))

    def _refresh_themes(self):
        """Rebind line themes without regenerating the curves."""
        # Which bar is highlighted follows the played step, so the profile
        # comes along on the light path as well as the heavy one.
        self._refresh_profile(force=True)
        if not self.display_ready:
            return
        for index, tag in enumerate(self.series_tags):
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, self._theme_for(index))

    def _theme_for(self, index):
        if index == self.play_index:
            return self.play_line_theme
        return self.ghost_line_theme

    def _update_step_display(self):
        if self.play_index < 0:
            text = '- / ' + str(self.step_count)
        else:
            text = str(self.play_index + 1) + ' / ' + str(self.step_count)
        self.step_display.set(text)

    # -- sequencing ----------------------------------------------------------

    def beat(self):
        """Advance one step, sample the input, send it through that shape."""
        if self.step_count <= 0:
            return
        wrapped = self._advance()
        x_val = any_to_float(self.x_input())
        y_val = breakpoint_value_at(self.shapes[self.play_index], x_val)

        self._show_sample(x_val, y_val)

        if wrapped:
            self.cycle_output.send('bang')
        self.step_output.send(self.play_index)
        self.value_output.send(y_val)

    def _advance(self):
        """Move play_index one step. Returns True if the cycle started over."""
        n = self.step_count
        direction = any_to_string(self.direction_property())

        if self.play_index < 0:
            self.play_index = n - 1 if direction == 'reverse' else 0
            self.ping_pong_sign = 1
            return False

        wrapped = False
        if direction == 'reverse':
            self.play_index -= 1
            if self.play_index < 0:
                self.play_index = n - 1
                wrapped = True
        elif direction == 'ping pong':
            self.play_index += self.ping_pong_sign
            if self.play_index >= n:
                # Turn around one short of the end so the end step is not
                # played twice running; a 1-step sequence just stays put.
                self.play_index = max(0, n - 2)
                self.ping_pong_sign = -1
            elif self.play_index < 0:
                self.play_index = min(n - 1, 1)
                self.ping_pong_sign = 1
                wrapped = True
        elif direction == 'random':
            self.play_index = int(np.random.randint(n))
        else:
            self.play_index += 1
            if self.play_index >= n:
                self.play_index = 0
                wrapped = True
        return wrapped

    def _show_sample(self, x_val, y_val):
        if not self.display_ready:
            return
        if dpg.does_item_exist(self.marker_tag):
            dpg.set_value(self.marker_tag, [[x_val], [y_val]])
        self._update_step_display()

        if any_to_bool(self.follow_property()) and not self._interacting():
            if self.play_index != self.edit_index:
                self._set_edit_index(self.play_index)
        self._refresh_themes()
        self._refresh_profile(force=True)

    def _interacting(self):
        """True while the user is working on the plot, so following backs off."""
        return self.editor.interacting()

    def reset_sequence(self):
        """Back to the top: the next beat plays the first step again."""
        self.play_index = -1
        self.ping_pong_sign = 1
        self._update_step_display()
        self._refresh_themes()

    # -- options -------------------------------------------------------------

    def _set_edit_index(self, index):
        index = max(0, min(self.step_count - 1, int(index)))
        if index == self.edit_index and self.editor.point_tags:
            return
        self.edit_index = index
        self._sync_edit_option()
        # Hand the editor the new step's curve. It keeps its own copies, so
        # _edited() is what writes any change back into self.shapes.
        self.editor.set_points(self.shapes[index], notify=False)
        # Which ghost lines are drawn depends on the edited step, so the
        # curves have to be regenerated, not just re-themed.
        self._refresh_series()

    def _sync_edit_option(self):
        """Keep the option widget honest when the edited step moves by itself."""
        if any_to_int(self.edit_step_option()) != self.edit_index:
            self.edit_step_option.widget.set(self.edit_index)

    def _edit_step_changed(self):
        self._set_edit_index(any_to_int(self.edit_step_option()))

    def _steps_changed(self):
        count = max(1, min(self.MAX_STEPS, any_to_int(self.steps_property())))
        if count == len(self.shapes):
            return
        if count < len(self.shapes):
            self.shapes = self.shapes[:count]
        else:
            self.shapes.extend(self._default_shape()
                               for _ in range(count - len(self.shapes)))
        self.step_count = count
        if self.play_index >= count:
            self.play_index = -1
        self.edit_index = min(self.edit_index, count - 1)
        self._sync_edit_option()
        self.editor.set_points(self.shapes[self.edit_index], notify=False)
        self._profile_axes()
        self._rebuild_series()
        self._update_step_display()

    def _copy_shape(self):
        self.clipboard = self._shape_as_lists(self.shapes[self.edit_index])

    def _paste_shape(self):
        if self.clipboard is None:
            return
        self.shapes[self.edit_index] = self._shape_from_lists(self.clipboard)
        self.editor.set_points(self.shapes[self.edit_index], notify=False)
        self._refresh_series()

    def _copy_to_all(self):
        """Author one shape, then give it to every step as a starting point."""
        source = self._shape_as_lists(self.shapes[self.edit_index])
        self.shapes = [self._shape_from_lists(source)
                       for _ in range(self.step_count)]
        self.editor.set_points(self.shapes[self.edit_index], notify=False)
        self._refresh_series()

    @staticmethod
    def _shape_as_lists(shape):
        return [[p['x'], p['y'], p.get('curve', 0.0)]
                for p in sorted(shape, key=lambda p: p['x'])]

    @staticmethod
    def _shape_from_lists(data):
        shape = []
        for entry in data:
            shape.append({'x': float(entry[0]), 'y': float(entry[1]),
                          'curve': float(entry[2]) if len(entry) > 2 else 0.0})
        return shape

    def _range_changed(self):
        self.x_max = any_to_float(self.x_max_option())
        self.y_min = any_to_float(self.y_min_option())
        self.y_max = any_to_float(self.y_max_option())
        # The editor clamps the edited step; the others are pulled into range
        # here so the whole sequence stays inside the axes.
        self.editor.set_ranges(x_max=self.x_max, y_min=self.y_min,
                               y_max=self.y_max, notify=False)
        for index, shape in enumerate(self.shapes):
            if index == self.edit_index:
                continue
            for point in shape:
                point['x'] = max(0.0, min(self.x_max, point['x']))
                point['y'] = max(self.y_min, min(self.y_max, point['y']))
        self.shapes[self.edit_index] = [dict(p) for p in self.editor.points]
        self._profile_axes()
        self._refresh_series()

    def _size_changed(self):
        self.plot_width = any_to_int(self.width_option())
        self.plot_height = any_to_int(self.height_option())
        self.profile_height = any_to_int(self.profile_height_option())
        self.editor.set_size(self.plot_width, self.plot_height)
        # The profile sits under the shapes and reads as one panel with them,
        # so it takes its width from the same handle.
        if dpg.does_item_exist(self.profile_tag):
            dpg.set_item_width(self.profile_tag, self.plot_width)
            dpg.set_item_height(self.profile_tag, self.profile_height)

    # -- editing -------------------------------------------------------------

    def custom_create(self, from_file):
        # First point at which the options hold their real values, so this is
        # where every drawing decision that depends on one gets made.
        self._profile_visibility_changed()
        self._refresh_series()
        self.add_frame_task()

    def frame_task(self):
        try:
            # The profile reads the x inlet continuously, not only on beats, so
            # you can see the whole sequence respond while you move the input.
            # It redraws only when x has actually moved.
            self._refresh_profile()

            left_down = dpg.is_mouse_button_down(0)

            # --- Click a profile bar: edit that step ---
            # Checked before the editor is given the frame, so a click meant
            # for a bar cannot also land on the shapes plot.
            if left_down and not self.left_was_down and self._profile_clicked():
                self.left_was_down = left_down
                return
            self.left_was_down = left_down

            self.editor.poll()

        except Exception:
            _log_frame_error_once(self)

    # -- persistence ---------------------------------------------------------

    def save_custom(self, container):
        container['shape_sequence'] = [self._shape_as_lists(shape)
                                       for shape in self.shapes]

    def load_custom(self, container):
        if 'shape_sequence' not in container:
            return
        data = container['shape_sequence']
        if not data:
            return
        # The saved sequence is the authority on how many steps there are; the
        # steps option may already have rebuilt them at its restored value.
        self.shapes = [self._shape_from_lists(entry) for entry in data]
        self.step_count = len(self.shapes)
        if self.steps_property() != self.step_count:
            self.steps_property.widget.set(self.step_count)
        self.edit_index = min(self.edit_index, self.step_count - 1)
        self._sync_edit_option()
        self.editor.set_points(self.shapes[self.edit_index], notify=False)
        self.play_index = -1
        self._profile_axes()
        self._rebuild_series()
        self._update_step_display()



class SliderBankNode(Node):
    """A named bank of sliders, each of which sends a message when moved.

    slider_bank 6                      six sliders, named 1..6
    slider_bank root spine left_arm    one per name

    Every slider has a name (editable in the options, shown as its label) and
    the bank has one message template, e.g. 'weight {name} {value}'.  Moving a
    slider sends the template filled in for that slider, as a list, so a node
    downstream receives it exactly as if the message had been typed.  With
    the default template '{name} {value}' each slider is its own message.

    Subclasses fix the names and template for a particular use -- see
    ragdoll_blend_ui -- and keep this node's options for everything else.

    Messages accepted at the 'in' input:
        set <name|index> <value>   move a slider (and send its message)
        send                       send every slider's message, in order
    A plain list of numbers sets the sliders in order.
    """
    default_names = None
    default_template = '{name} {value}'
    default_min = 0.0
    default_max = 1.0
    default_value = 0.0

    @staticmethod
    def factory(name, data, args=None):
        return SliderBankNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        names = []
        count = None
        for i in range(len(self.ordered_args)):
            val, t = decode_arg(self.ordered_args, i)
            if t == int and count is None and not names:
                count = int(val)
            elif t == str:
                names.append(str(val))
        if not names:
            if count is not None:
                names = [str(i + 1) for i in range(max(count, 1))]
            elif self.default_names:
                names = list(self.default_names)
            else:
                names = [str(i + 1) for i in range(4)]
        self.names = names
        self.count = len(names)

        self.control_input = self.add_input('in', triggers_execution=True)
        self.sliders = []
        for i, name in enumerate(names):
            slider = self.add_input(name, widget_type='slider_float', widget_width=140,
                                    default_value=float(self.default_value),
                                    min=float(self.default_min), max=float(self.default_max),
                                    callback=(lambda i=i: self.slider_changed(i)))
            self.sliders.append(slider)
        self.output = self.add_output('messages')

        self.template_option = self.add_option('message', widget_type='text_input', width=200,
                                               default_value=self.default_template)
        self.min_option = self.add_option('min', widget_type='drag_float',
                                          default_value=float(self.default_min),
                                          callback=self.limits_changed)
        self.max_option = self.add_option('max', widget_type='drag_float',
                                          default_value=float(self.default_max),
                                          callback=self.limits_changed)
        self.name_options = []
        for i, name in enumerate(names):
            self.name_options.append(self.add_option('name %d' % (i + 1), widget_type='text_input',
                                                     width=140, default_value=name,
                                                     callback=self.names_changed))
        self.message_handlers['set'] = self._set_message
        self.message_handlers['send'] = self._send_message

    # -- options --------------------------------------------------------------

    def names_changed(self):
        for i, opt in enumerate(self.name_options):
            name = str(opt()).strip()
            if name and name != self.names[i]:
                self.names[i] = name
                self.sliders[i].widget.set_label(name)

    def limits_changed(self):
        lo = float(self.min_option()); hi = float(self.max_option())
        if hi <= lo:
            hi = lo + 1e-6
        for slider in self.sliders:
            slider.widget.set_limits(lo, hi)

    # -- messages out ---------------------------------------------------------

    def message_for(self, i):
        value = float(self.sliders[i]())
        out = []
        for token in str(self.template_option()).split():
            tok = token.replace('{name}', self.names[i]).replace('{index}', str(i))
            if '{value}' in tok:
                if tok == '{value}':
                    out.append(value)
                    continue
                tok = tok.replace('{value}', repr(value))
            try:
                out.append(int(tok) if tok.lstrip('-').isdigit() else float(tok))
            except ValueError:
                out.append(tok)
        return out

    def slider_changed(self, i):
        # restored values fire this during patch load; nothing downstream is
        # ready to hear from us yet
        if getattr(self, 'in_loading_process', False):
            return
        self.output.send(self.message_for(i))

    def send_all(self):
        for i in range(self.count):
            self.output.send(self.message_for(i))

    # -- messages in ----------------------------------------------------------

    def _index_of(self, token):
        tok = str(token).strip()
        if tok in self.names:
            return self.names.index(tok)
        if tok.lstrip('-').isdigit():
            i = int(tok)
            if 0 <= i < self.count:
                return i
        return None

    def _set_message(self, message='', args=None):
        args = list(args or [])
        if len(args) < 2:
            return
        i = self._index_of(args[0])
        if i is None:
            print(f'{self.label}: no slider named {args[0]!r}; names are {self.names}')
            return
        self.sliders[i].set(any_to_float(args[1]))
        self.output.send(self.message_for(i))

    def _send_message(self, message='', args=None):
        self.send_all()

    def execute(self):
        if self.control_input.fresh_input:
            data = self.control_input()
            if isinstance(data, str) and data == 'bang':
                self.send_all()
                return
            values = any_to_array(data) if not isinstance(data, str) else None
            if values is not None:
                values = np.asarray(values, dtype=float).reshape(-1)
                for i in range(min(self.count, values.size)):
                    self.sliders[i].set(float(values[i]))
                    self.output.send(self.message_for(i))
