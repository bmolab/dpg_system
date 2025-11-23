# import dearpygui.dearpygui as dpg
import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.conversion_utils import *
import json
from typing import List, Any, Callable, Union, Tuple, Optional, Dict, Set, Type, TypeVar, cast
from fuzzywuzzy import fuzz
import sys
import os


class NodeOutput:
    _pin_active_theme = None
    _pin_active_string_theme = None
    _pin_theme_created = False
    _pin_active_array_theme = None
    _pin_active_tensor_theme = None
    _pin_active_list_theme = None
    _pin_active_bang_theme = None

    def __init__(self, label: str = "output", node=None, pos=None):
        if not self._pin_theme_created:
            self.create_pin_themes()
        self.uuid = -1
        self._label = label
        self.label_uuid = None
        self._children = []  # output attributes
        self.links = []
        self.pos = pos
        self.node = node
        self.output_always = True
        self.new_output = False
        self.output_type = None
        self.name_archive = []
        self.sent_type = None
        self.sent_bang = False

    def get_label(self) -> str:
        return self._label

    def set_label(self, name: str) -> None:
        self._label = name
        if self.label_uuid is None:
            self.label_uuid = dpg.add_text(self._label)
        else:
            dpg.set_value(self.label_uuid, self._label)

    def create_pin_themes(self) -> None:
        #   could add other colours?
        with dpg.theme() as self._pin_active_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (153, 212, 255), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_array_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 255, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_tensor_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 0, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_list_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 0, 255), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_string_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 128, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_bang_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 255, 0), category=dpg.mvThemeCat_Nodes)

    def set_visibility(self, visibility_state: str = 'show_all') -> None:
        if visibility_state == 'show_all':
            if self.node.do_not_delete:
                print('node output protected')
                dpg.bind_item_theme(self.uuid, theme=Node.app.do_not_delete_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=Node.app.global_theme)
        elif visibility_state == 'widgets_only':
            dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)
        else:
            dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)

    def add_child(self, child: 'NodeInput', parent: int) -> None:
        link_uuid = dpg.add_node_link(self.uuid, child.uuid, parent=parent)
        # print('added link', link_uuid, self.uuid, child.uuid, self.node.label)
        child.set_parent(self)
        self._children.append(child)
        self.links.append(link_uuid)
        dpg.set_item_user_data(link_uuid, [self, child])

    def remove_links(self) -> None:
        for child in self._children:
            child.remove_parent(self)
        for link in self.links:
            if dpg.does_item_exist(link):
                dpg.delete_item(link)
        self.links = []
        self._children = []

    def remove_link(self, link: int, child: 'NodeInput') -> None:
        # print('remove_link')
        if link in self.links:
            if dpg.does_item_exist(link):
                self.links.remove(link)
        if child in self._children:
            self._children.remove(child)
        child.remove_parent(self)
        dpg.delete_item(link)

    def remove_child(self, child: 'NodeInput') -> None:
        for kid in self._children:
            if kid == child:
                self._children.remove(kid)
                break

    def set_value(self, data: Any) -> Any:
        from dpg_system.basic_nodes import print_info
        if data is not None:
            self.new_output = True
            for child in self._children:
                if Node.app.trace:
                    print(Node.app.trace_indent, end='')
                    print('node \'' + self.node.label + '\' output ', end='')
                    if self.get_label() != '':
                        print('\'' + self.get_label() + '\' -> ', end='')
                    type_string, value_string = print_info(input_=data)
                    print(type_string, end='')
                    if len(value_string) > 0:
                        print(':' + value_string, end='')
                    print(' -> node \'' + child.node.label + '\' input', end='')
                    if child.get_label() != '':
                        print(' \'' +  child.get_label() + '\'')
                    else:
                        print()
                    child.receive_data(data)
                else:
                    child.receive_data(data)

            self.sent_type = type(data)
            if self.sent_type == str and data == 'bang':
                self.sent_bang = True
            else:
                self.sent_bang = False

        return data

    def send(self, data: Optional[Any] = None, no_trigger=False) -> None:  # called every time
        if data is None:
            self.send_internal(no_trigger)
        elif self.set_value(data) is not None:
            self.send_internal(no_trigger)
        self.new_output = False

    def send_internal(self, no_trigger=False) -> None:
        if self.output_always or self.new_output:
            if self.node.visibility == 'show_all':
                try:
                    if Node.app.color_code_pins:
                        t = self.sent_type

                        if t is np.ndarray:
                            dpg.bind_item_theme(self.uuid, self._pin_active_array_theme)
                        elif t is torch.Tensor:
                            dpg.bind_item_theme(self.uuid, self._pin_active_tensor_theme)
                        elif t is list:
                            dpg.bind_item_theme(self.uuid, self._pin_active_list_theme)
                        elif t is str:
                            if self.sent_bang:
                                dpg.bind_item_theme(self.uuid, self._pin_active_bang_theme)
                            else:
                                dpg.bind_item_theme(self.uuid, self._pin_active_string_theme)
                        else:
                            dpg.bind_item_theme(self.uuid, self._pin_active_theme)
                    else:
                        dpg.bind_item_theme(self.uuid, self._pin_active_theme)
                    Node.app.get_current_editor().add_active_pin(self.uuid)
                except Exception as e:
                    pass
            if not no_trigger:
                for child in self._children: # we want to be able to step debug...dialog
                    self.send_to_one_child(child)

    def send_to_one_child(self, child: 'NodeInput') -> None:
        child.node.active_input = child
        if Node.app.trace:
            if child.triggers_execution or child.callback is not None:
                print(Node.app.trace_indent, end='')
                if child.triggers_execution:
                    print('node \'' + child.node.label + '\' execute')
                else:
                    print('node', child.node.label, 'input', child.get_label(), 'callback')
        child.trigger()
        child.node.active_input = None

    # would be called to make a step. App must record the node, output and which_child for next step
    # if returns -1, then step has to know how to move to next bit of code that causes output.
    # maybe just suspend the main thread? and continue on step to next output point
    def step_output(self, which_child) -> int:
        if which_child < len(self._children):
            child = self._children[which_child]
            self.send_to_one_child(child)
            which_child += 1
            if which_child >= len(self._children):
                return -1
            return which_child
        else:
            return -1

    def create(self, parent: int) -> None:
        if self.pos is not None:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self, pos=self.pos) as self.uuid:
                self.label_uuid = dpg.add_text(self._label)
        else:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self) as self.uuid:
                self.label_uuid = dpg.add_text(self._label)

    def get_children(self) -> List['NodeInput']:
        return self._children

    def save(self, output_container: Dict[str, Any]) -> None:
        output_container['name'] = self._label
        output_container['id'] = self.uuid
        children = {}
        for index, child in enumerate(self._children):
            children[index] = child.uuid
        output_container['children'] = children


class NodeIntOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = int

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            int_data = any_to_int(data)
            super().send(int_data)
        else:
            super().send()

    def set_value(self, data: Any) -> int:
        int_data = any_to_int(data)
        super().set_value(int_data)
        return int_data


class NodeFloatOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = float

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            float_data = any_to_float(data)
            super().send(float_data)
        else:
            super().send()

    def set_value(self, data: Any) -> float:
        float_data = any_to_float(data)
        super().set_value(float_data)
        return float_data


class NodeBoolOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = bool

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            bool_data = any_to_bool(data)
            super().send(bool_data)
        else:
            super().send()

    def set_value(self, data: Any) -> bool:
        bool_data = any_to_bool(data)
        super().set_value(bool_data)
        return bool_data


class NodeListOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = list

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            list_data = any_to_list(data)
            # if len(list_data) == 1 and type(data) is str:
            #     super().send(data)
            # else:
            super().send(list_data)
        else:
            super().send()

    def set_value(self, data: Any) -> Union[str, List[Any]]:
        list_data = any_to_list(data)
        if len(list_data) == 1 and type(data) is str:
            super().set_value(data)
            return data
        else:
            super().set_value(list_data)
            return list_data

class NodeStringOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = str

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            string_data = any_to_string(data)
            super().send(string_data)
        else:
            super().send()

    def set_value(self, data: Any) -> str:
        string_data = any_to_string(data)
        super().set_value(string_data)
        return string_data


class NodeArrayOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = np.ndarray

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None:
            array_data = any_to_array(data)
            super().send(array_data)
        else:
            super().send()

    def set_value(self, data: Any) -> np.ndarray:
        array_value = any_to_array(data)
        super().set_value(array_value)
        return array_value


class NodeTensorOutput(NodeOutput):
    def __init__(self, label: str = "output", node=None, pos=None):
        super().__init__(label, node, pos)
        self.output_type = torch.Tensor

    def send(self, data: Optional[Any] = None) -> None:
        if data is not None and torch_available:
            tensor_data = any_to_tensor(data)
            super().send(tensor_data)
        else:
            super().send()
    def set_value(self, data: Any) -> Optional[torch.Tensor]:
        if torch_available:
            tensor_value = any_to_tensor(data)
            super().set_value(tensor_value)
            return tensor_value


class NodeDisplay:
    def __init__(self, label: str = "", uuid=None, node=None, width=80):
        self.uuid = dpg.generate_uuid()
        self.callback = None
        self.user_data = None
        self.node = node
        self.submit_callback = None

    def create(self, parent: int) -> None:
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Static, user_data=self, id=self.uuid)
        with self.node_attribute:
            if self.submit_callback is not None:
                self.submit_callback()
            if self.callback is not None:
                dpg.set_item_callback(self.widget.uuid, self.callback)
                dpg.set_item_user_data(self.widget.uuid, self.user_data)

    def set_visibility(self, visibility_state: str = 'show_all') -> None:
        if visibility_state == 'show_all':
            if self.node.do_not_delete:
                dpg.bind_item_theme(self.uuid, theme=Node.app.do_not_delete_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=Node.app.global_theme)
        elif visibility_state == 'widgets_only':
            dpg.bind_item_theme(self.uuid, theme=Node.app.widget_only_theme)
        else:
            dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)

    def load(self, property_container: Dict[str, Any]) -> None:
        pass

    def save(self, property_container: Dict[str, Any]) -> None:
        pass

    def add_callback(self, callback: Callable, user_data: Optional[Any] = None) -> None:
        if user_data is None:
            self.user_data = self
        else:
            self.user_data = user_data
        self.callback = callback

    def get_widget_value(self) -> Any:
        return 0

    def set(self, data: Any) -> None:
        pass


class NodeProperty:
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None, **kwargs):
        self.label = label
        if uuid == None:
            self.uuid = dpg.generate_uuid()
        self.widget = PropertyWidget(label, uuid, node=node, widget_type=widget_type, width=width, triggers_execution=triggers_execution, trigger_button=trigger_button, default_value=default_value, min=min, max=max, **kwargs)
        self.callback = None
        self.user_data = None
        self.node_attribute = None
        self.variable = None
        self.action = None
        self.node = node

    def create(self, parent):
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Static, user_data=self, id=self.uuid)
        with self.node_attribute:
            if self.callback is not None:
                self.widget.callback = self.callback
                self.widget.user_data = self.user_data
            self.widget.create()

    # def set_node(self, node):
    #     self.node = node
    #     if self.widget:
    #         self.widget.node = node
    def set_default_value(self, data):
        self.widget.set_default_value(data)

    def set_visibility(self, visibility_state='show_all'):
        if visibility_state == 'show_all':
            if self.node.do_not_delete:
                dpg.bind_item_theme(self.uuid, theme=Node.app.do_not_delete_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=Node.app.global_theme)
        elif visibility_state == 'widgets_only':
            dpg.bind_item_theme(self.uuid, theme=Node.app.widget_only_theme)
        else:
            dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)

        if self.widget is not None:
            self.widget.set_visibility(visibility_state)

    def attach_to_variable(self, variable):
        self.variable = variable
        self.variable.property = self
        self.widget.attach_to_variable(variable)

    def attach_to_action(self, action):
        self.action = action
        self.widget.attach_to_action(action)

    def load(self, property_container):
        self.widget.load(property_container)

    def save(self, property_container):
        self.widget.save(property_container)

    def add_callback(self, callback, user_data=None):
        if user_data is None:
            self.user_data = self
        else:
            self.user_data = user_data
        self.callback = callback

    def __call__(self):
        return self.widget.value

    def get_widget_value(self):
        return self.widget.value

    def set_label(self, new_name):
        self.label = new_name
        if self.widget:
            self.widget.set_label(new_name)

    def set_and_callback(self, data, propagate=True):
        self.set(data, propagate)
        if self.callback is not None:
            self.callback()

    def value_changed(self, uuid, force=False):
        pass

    def set(self, data, propagate=True):
        if type(data) == list:
            if len(data) == 1:
                data = data[0]
            else:
                if self.widget.widget in ['text_input', 'combo', 'radio_group', 'text_editor']:
                    data = any_to_string(data)
                else:
                    data = data[0]
        self.widget.set(data, propagate)
        if self.variable and propagate:
            self.variable.set_value(data)  # will propagate to all instances


class BasePropertyWidget:
    def __init__(self, label: str, uuid=None, node=None, widget_type=None,
                 width=80, triggers_execution=False, trigger_button=False,
                 default_value=None, **kwargs):

        self._label = label
        self.uuid = uuid if uuid is not None else dpg.generate_uuid()
        self.uuids = [self.uuid]  # List for handling vectors/multi-item widgets
        self.node = node
        self.widget = widget_type  # Stored for save/load identification
        self.widget_width = width

        # Execution & Trigger Logic
        self.triggers_execution = triggers_execution
        self.widget_has_trigger = trigger_button
        self.trigger_widget = None

        # State
        self.default_value = default_value
        self.value = None
        self.user_data = self
        self.tag = self.uuid
        self.input = None  # assigned externally usually

        # Callbacks
        self.callback = None
        self.trigger_callback = None
        self.variable = None  # Attached variable
        self.action = None  # Attached action

        # Theme
        self.active_theme = getattr(node, 'active_theme_base', None) if node else None

    def create(self) -> None:
        """Template Method: Defines the skeleton of widget creation."""
        # 1. Determine layout
        horizontal = self.widget_has_trigger or self._force_horizontal()

        # 2. Initialize Value
        self._init_default_value()

        # 3. Draw
        with dpg.group(horizontal=horizontal):
            self._draw_widget()
            self._setup_interaction()
            self._create_trigger_button()

    def _draw_widget(self):
        """Subclasses must implement the specific DPG draw call."""
        raise NotImplementedError

    def _force_horizontal(self):
        """Hook for subclasses (like vectors) to force horizontal layout."""
        return False

    def _init_default_value(self):
        """Subclasses can normalize default_value type here."""
        if self.default_value is None:
            self.default_value = self._get_zero_value()
        self.value = self.default_value

    def _get_zero_value(self):
        """Fallback value if None provided."""
        return None

    def _setup_interaction(self):
        """Binds callbacks to the generated UUIDs."""
        if self.widget in ['spacer', 'label', 'table']:
            return

        # Specific widgets use 'clickable_changed', others 'value_changed'
        clickable = ['checkbox', 'radio_group', 'button', 'combo', 'color_picker']

        handler = self.clickable_changed if self.widget in clickable else \
            lambda s, a, u: self.value_changed(a)

        for uid in self.uuids:
            dpg.set_item_user_data(uid, user_data=self)
            dpg.set_item_callback(uid, callback=handler)

    def _create_trigger_button(self):
        if not self.widget_has_trigger:
            return

        cb = self.trigger_callback if self.trigger_callback else self.trigger_value
        self.trigger_widget = dpg.add_button(label='', width=14, callback=cb)
        if self.active_theme:
            dpg.bind_item_theme(self.trigger_widget, self.active_theme)

    # --- Core Logic (Visibility, Themes, Execution) ---

    def trigger_value(self) -> None:
        if self.node:
            self.node.active_input = self.input
            self.node.execute()
            self.node.active_input = None

    def set_visibility(self, visibility_state: str = 'show_all') -> None:
        # Determine themes based on state
        item_theme = None
        trigger_theme = None
        enable_item = True
        enable_trigger = True

        if visibility_state == 'show_all':
            if self.node and self.node.do_not_delete:
                item_theme = self.node.app.do_not_delete_theme
            else:
                item_theme = self.node.app.global_theme
            trigger_theme = self.active_theme
        elif visibility_state == 'widgets_only':
            item_theme = self.node.app.widget_only_theme
            # Custom theme for trigger in widgets_only mode
            # Note: In a real app, creating a theme every frame/call is bad for performance.
            # Ideally this theme is cached, but keeping logic identical to source:
            with dpg.theme() as t_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
            trigger_theme = t_theme
        else:  # Hidden
            item_theme = self.node.app.invisible_theme
            trigger_theme = self.node.app.invisible_theme
            enable_item = False
            enable_trigger = False

        # Apply to main widget(s)
        for uid in self.uuids:
            dpg.bind_item_theme(uid, item_theme)
            if self.widget != 'label':
                if enable_item:
                    dpg.enable_item(uid)
                else:
                    dpg.disable_item(uid)

        # Apply to trigger
        if self.trigger_widget:
            dpg.bind_item_theme(self.trigger_widget, trigger_theme)
            if enable_trigger:
                dpg.enable_item(self.trigger_widget)
            else:
                dpg.disable_item(self.trigger_widget)
        elif self.widget == 'button':
            if visibility_state != 'hidden':
                dpg.bind_item_theme(self.uuid, self.active_theme)

    def set_active_theme(self, theme):
        self.active_theme = theme
        if self.trigger_widget:
            dpg.bind_item_theme(self.trigger_widget, self.active_theme)
            dpg.enable_item(self.trigger_widget)
        elif self.widget == 'button':
            dpg.bind_item_theme(self.uuid, self.active_theme)

    def set_height(self, height):
        if self.input and hasattr(self.input.widget, 'uuid'):
            dpg.set_item_height(self.input.widget.uuid, height)

    # --- Data Flow & Callbacks ---

    def _update_value_from_dpg(self):
        if len(self.uuids) > 1:
            self.value = [dpg.get_value(uid) for uid in self.uuids]
        else:
            self.value = dpg.get_value(self.uuid)

    def value_changed(self, uuid: int = -1, force: bool = False) -> None:
        if not dpg.is_mouse_button_down(0) and not force:
            return

        hold_active_input = self.node.active_input if self.node else None

        self._update_value_from_dpg()
        self._propagate_changes(hold_active_input)

    def clickable_changed(self) -> None:
        self._update_value_from_dpg()
        # Clickables don't check mouse down
        self._propagate_changes(None)

    def _propagate_changes(self, hold_active_input):
        if self.variable:
            self.variable.set_value(self.value)
        if self.action:
            self.action()

        if self.callback:
            if self.node: self.node.active_input = self.input
            self.callback()
            if self.node: self.node.active_input = hold_active_input

        if self.triggers_execution and self.node and not self.node.in_loading_process:
            self.node.active_input = self.input
            self.node.execute()
            self.node.active_input = hold_active_input

    def set(self, data: Any, propagate: bool = True) -> None:
        """Abstract set. Subclasses implement _convert_and_set."""
        self._convert_and_set(data)

        if self.variable and propagate:
            self.variable.set_value(self.value)
        if self.action and propagate:
            self.action()

    def _convert_and_set(self, data):
        """Override in subclasses."""
        pass

    # --- Public API Utils ---

    def set_label(self, name: str) -> None:
        self._label = name
        dpg.set_item_label(self.uuid, name)

    def set_limits(self, min_: Union[int, float], max_: Union[int, float]) -> None:
        # Base implementation, overridden by scalars
        pass

    def set_format(self, format: str) -> None:
        pass

    def attach_to_variable(self, variable) -> None:
        self.variable = variable

    def attach_to_action(self, action) -> None:
        self.action = action

    def increment(self) -> None:
        pass

    def decrement(self) -> None:
        pass

    def get_text_width(self, pad: int = 12, minimum_width: int = 100) -> float:
        ttt = any_to_string(self.value)
        return self._calculate_width(ttt, pad, minimum_width)

    def get_label_width(self, pad: int = 12, minimum_width: int = 100) -> float:
        label = dpg.get_item_label(self.uuid)
        return self._calculate_width(label, pad, minimum_width)

    def _calculate_width(self, text, pad, minimum_width):
        font_id = dpg.get_item_font(self.uuid)
        size = dpg.get_text_size(text, font=font_id)
        width = minimum_width
        if size is not None:
            width = size[0] + pad
        font_scale = 1.0
        if font_id is not None and self.node:
            font_scale = self.node.app.font_scale_variable()
        if width < minimum_width / font_scale:
            width = minimum_width / font_scale
        return width * font_scale

    def adjust_to_text_width(self, max: int = 0) -> float:
        width = self.get_text_width()
        if width is not None:
            dpg.configure_item(self.uuid, width=width)
        return width

    def set_font(self, font: Any) -> None:
        dpg.bind_item_font(self.uuid, font)

    # --- Load / Save ---

    def load(self, widget_container: Dict[str, Any]) -> None:
        if 'value' in widget_container:
            self.set(widget_container['value'])

    def save(self, widget_container: Dict[str, Any]) -> None:
        property_label = self._label.strip('#')
        widget_container['name'] = property_label

        # Get value directly from DPG to ensure sync
        if len(self.uuids) > 1:
            value = [dpg.get_value(u) for u in self.uuids]
        else:
            value = dpg.get_value(self.uuid)

        widget_container['value'] = value

        value_type = type(value).__name__
        if value_type == 'str':
            widget_container['value_type'] = 'string'
        else:
            widget_container['value_type'] = value_type

    # --- Helpers ---
    def get_as_float(self, data):
        return any_to_float(data)

    def get_as_bool(self, data):
        return any_to_bool(data)

    def get_as_string(self, data):
        return any_to_string(data)

    def get_as_list(self, data):
        return any_to_list(data)

    def get_as_int(self, data):
        return any_to_int(data)

    def get_as_array(self, data):
        return any_to_array(data)

    def set_default_value(self, data):
        self.default_value = data  # Generic fallback


# --- 2. Intermediate Category Classes ---

class ScalarWidget(BasePropertyWidget):
    """Base for Float and Int widgets handling clamping and limits."""

    def __init__(self, *args, min=None, max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min
        self.max = max
        # Defaults for step/speed are handled in concrete classes or kwargs

    def set_limits(self, min_: Union[int, float], max_: Union[int, float]) -> None:
        self.min = min_
        self.max = max_
        dpg.configure_item(self.uuid, min_value=self.min, max_value=self.max)

    def _clamp(self, val):
        if val is None: return val
        if self.min is not None and val < self.min: return self.min
        if self.max is not None and val > self.max: return self.max
        return val

    def _get_limits(self, default_min, default_max):
        # Logic from original create() to determine dpg args
        mn = self.min if self.min is not None else default_min
        mx = self.max if self.max is not None else default_max
        return mn, mx

    def set_format(self, format: str) -> None:
        dpg.configure_item(self.uuid, format=format)

    def set_speed(self, speed: float) -> None:
        dpg.configure_item(self.uuid, speed=speed)



class NumericInteractionWidget(ScalarWidget):
    """Adds increment/decrement logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = 1
        self.step = 1

    def increment(self) -> None:
        val = dpg.get_value(self.uuid)
        self.set(val + self.speed)

    def decrement(self) -> None:
        val = dpg.get_value(self.uuid)
        self.set(val - self.speed)


# --- 3. Specific Implementations ---

class FloatWidget(NumericInteractionWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Original logic: drag_float defaults speed 0.01, others 1
        if self.widget and self.widget.startswith('drag_float'):
            self.speed = 0.01
        if self.widget == 'input_float':
            self.step = 0.1

    def _get_zero_value(self):
        return 0.0

    def set_default_value(self, data):
        self.default_value = any_to_float(data)

    def _convert_and_set(self, data):
        if is_number(data):
            val = any_to_float(data)
            if val is not None:
                val = self._clamp(val)
            dpg.set_value(self.uuid, val)
            self.value = val
        elif isinstance(data, bool):
            val = 1.0 if data else 0.0
            dpg.set_value(self.uuid, val)
            self.value = val


class DragFloat(FloatWidget):
    def _draw_widget(self):
        mn, mx = self._get_limits(-math.inf, math.inf)
        dpg.add_drag_float(width=self.widget_width, clamped=True, label=self._label,
                           tag=self.uuid, max_value=mx, min_value=mn, user_data=self.node,
                           default_value=self.default_value, speed=self.speed)


class SliderFloat(FloatWidget):
    def _draw_widget(self):
        mn, mx = self._get_limits(0.0, 100.0)
        dpg.add_slider_float(label=self._label, width=self.widget_width, tag=self.uuid,
                             user_data=self.node, default_value=self.default_value,
                             min_value=mn, max_value=mx)


class KnobFloat(FloatWidget):
    def _draw_widget(self):
        if self.min is None: self.min = 0
        if self.max is None: self.max = 1.0
        dpg.add_knob_float(label=self._label, width=self.widget_width, tag=self.uuid,
                           user_data=self.node, default_value=self.default_value,
                           min_value=self.min, max_value=self.max)

    def set_format(self, format: str) -> None:
        pass


class InputFloat(FloatWidget):
    def _draw_widget(self):
        if self.min is None: self.min = sys.float_info.min
        if self.max is None: self.max = sys.float_info.max
        dpg.add_input_float(label=self._label, width=self.widget_width, tag=self.uuid,
                            user_data=self.node, default_value=self.default_value,
                            step=self.step, min_value=self.min, max_value=self.max)


class IntWidget(NumericInteractionWidget):
    def _get_zero_value(self):
        return 0

    def set_default_value(self, data):
        self.default_value = any_to_int(data)

    def _convert_and_set(self, data):
        if is_number(data):
            val = any_to_int(data)
            if val is not None:
                val = self._clamp(val)
            dpg.set_value(self.uuid, val)
            self.value = val
        elif isinstance(data, bool):
            val = 1 if data else 0
            dpg.set_value(self.uuid, val)
            self.value = val


class DragInt(IntWidget):
    def _draw_widget(self):
        mn, mx = self._get_limits(-math.inf, math.inf)
        dpg.add_drag_int(label=self._label, width=self.widget_width, tag=self.uuid,
                         max_value=mx, min_value=mn, user_data=self.node,
                         default_value=self.default_value)


class SliderInt(IntWidget):
    def _draw_widget(self):
        # Original logic set default limits to 0-100 AND updated self.min/max
        if self.min is None: self.min = 0
        if self.max is None: self.max = 100
        dpg.add_slider_int(label=self._label, width=self.widget_width, tag=self.uuid,
                           user_data=self.node, default_value=self.default_value,
                           min_value=self.min, max_value=self.max)


class InputInt(IntWidget):
    def _draw_widget(self):
        if self.min is None: self.min = 0
        if self.max is None: self.max = 2 ** 31 - 1
        dpg.add_input_int(label=self._label, width=self.widget_width, tag=self.uuid,
                          user_data=self.node, default_value=self.default_value,
                          step=self.step, min_value=self.min, max_value=self.max)

    def set_format(self, format: str) -> None:
        pass

class CheckboxWidget(BasePropertyWidget):
    def _get_zero_value(self):
        return False

    def _draw_widget(self):
        dpg.add_checkbox(label=self._label, tag=self.uuid,
                         default_value=self.default_value, user_data=self)

    def set_default_value(self, data):
        self.default_value = any_to_bool(data)

    def _convert_and_set(self, data):
        if data == 'bang':
            val = not self.value
        else:
            val = any_to_bool(data)
        dpg.set_value(self.uuid, val)
        self.value = val

    def increment(self):  # Toggle
        val = not dpg.get_value(self.uuid)
        self.set(val)

    def decrement(self):  # Toggle
        self.increment()


class StringWidget(BasePropertyWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Combo items stored here
        self.combo_items = []

    def _get_zero_value(self):
        return ""

    def set_default_value(self, data):
        self.default_value = any_to_string(data)

    def _convert_and_set(self, data, strip=True):
        # Specialized set handling
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            val = str(data)
        elif isinstance(data, str) and data == '\n':
            val = data
        else:
            val = any_to_string(data, strip_returns=strip)
        dpg.set_value(self.uuid, val)
        self.value = val


class TextInput(StringWidget):
    def _draw_widget(self):
        dpg.add_input_text(label=self._label, width=self.widget_width, tag=self.uuid,
                           user_data=self.node, default_value=self.default_value, on_enter=True)

    def _convert_and_set(self, data):
        super()._convert_and_set(data, strip=True)


class TextEditor(StringWidget):
    def _draw_widget(self):
        dpg.add_input_text(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node,
                           default_value=self.default_value, on_enter=False, multiline=True)

    def _convert_and_set(self, data):
        super()._convert_and_set(data, strip=False)


class SelectorWidget(StringWidget):
    """Base for Combos and Radio Groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizontal_layout = kwargs.get('horizontal', False)

    def _convert_and_set(self, data):
        val = any_to_string(data)
        dpg.set_value(self.uuid, val)
        self.value = val

    def increment(self):
        val = dpg.get_value(self.uuid)
        try:
            idx = self.combo_items.index(val)
            if idx + 1 < len(self.combo_items):
                self.set(self.combo_items[idx + 1])
        except ValueError:
            pass

    def decrement(self):
        val = dpg.get_value(self.uuid)
        try:
            idx = self.combo_items.index(val)
            if idx - 1 >= 0:
                self.set(self.combo_items[idx - 1])
        except ValueError:
            pass


class Combo(SelectorWidget):
    def _draw_widget(self):
        dpg.add_combo(self.combo_items, label=self._label, width=self.widget_width,
                      tag=self.uuid, user_data=self.node, default_value=self.default_value)


class RadioGroup(SelectorWidget):
    def _draw_widget(self):
        dpg.add_radio_button(self.combo_items, label=self._label, tag=self.uuid,
                             user_data=self.node, horizontal=self.horizontal_layout)


class ListBox(SelectorWidget):
    def _draw_widget(self):
        dpg.add_listbox(label=self._label, width=self.widget_width, tag=self.uuid,
                        user_data=self.node, num_items=8)


class ColorPicker(BasePropertyWidget):
    def _get_zero_value(self):
        return (0, 0, 0, 255)

    def set_default_value(self, data):
        self.default_value = tuple(any_to_list(data))

    def _draw_widget(self):
        dpg.add_color_picker(label='color', width=self.widget_width, display_type=dpg.mvColorEdit_float,
                             tag=self.uuid, picker_mode=dpg.mvColorPicker_wheel, no_side_preview=False,
                             no_alpha=False, alpha_bar=True, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf,
                             user_data=self.node, no_inputs=True, default_value=self.default_value)

    def _convert_and_set(self, data):
        if not isinstance(data, tuple):
            val = tuple(any_to_array(data))
        else:
            val = data
        dpg.set_value(self.uuid, val)
        self.value = val


class Button(BasePropertyWidget):
    def _draw_widget(self):
        btn = dpg.add_button(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node)
        if self.active_theme:
            dpg.bind_item_theme(btn, self.active_theme)


class Label(BasePropertyWidget):
    def _draw_widget(self):
        dpg.add_text(self._label, tag=self.uuid)

    def _convert_and_set(self, data):
        label_string = any_to_string(data)
        dpg.set_value(self.uuid, label_string)


class Spacer(BasePropertyWidget):
    def _draw_widget(self):
        dpg.add_spacer(label='', height=13)


class TableWidget(BasePropertyWidget):
    def __init__(self, *args, rows=1, columns=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.columns = columns

    def _draw_widget(self):
        with dpg.table(tag="table", header_row=False, width=300):
            for i in range(self.columns):
                dpg.add_table_column()
            for i in range(self.rows):
                with dpg.table_row():
                    for j in range(self.columns):
                        dpg.add_text('0', tag=f"cell_{i}_{j}")

    def set_format(self, format: str) -> None:
        dpg.configure_item(self.uuid, format=format)


class DragFloatN(ScalarWidget):
    def __init__(self, *args, columns=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = columns
        # Generate extra UUIDs
        for _ in range(self.columns - 1):
            self.uuids.append(dpg.generate_uuid())
        self.speed = 0.01

    def _force_horizontal(self):
        return True

    def _get_zero_value(self):
        return [0.0] * self.columns

    def _draw_widget(self):
        mn, mx = self._get_limits(-math.inf, math.inf)
        # Default value comes as list from init
        for i in range(self.columns):
            val = self.default_value[0] if self.default_value else 0.0
            dpg.add_drag_float(width=self.widget_width, clamped=True, tag=self.uuids[i],
                               max_value=mx, min_value=mn, user_data=self.node,
                               default_value=val, speed=self.speed)

    def _convert_and_set(self, data):
        if isinstance(data, list):
            if len(data) == 1 and is_number(data[0]):
                self._apply_val_to_all(any_to_float(data[0]))
                self.value = data
            elif len(data) == self.columns:
                vals = []
                for index, datum in enumerate(data):
                    if is_number(datum):
                        val = self._clamp(any_to_float(datum))
                        dpg.set_value(self.uuids[index], val)
                        vals.append(val)
                self.value = vals
        elif is_number(data):
            val = any_to_float(data)
            self._apply_val_to_all(val)
            self.value = data

    def _apply_val_to_all(self, val):
        clamped = self._clamp(val)
        for uuid in self.uuids:
            dpg.set_value(uuid, clamped)

    def set_format(self, format: str) -> None:
        for uuid in self.uuids:
            dpg.configure_item(uuid, format=format)



# --- 4. The Factory ---

class WidgetFactory:
    """
    Replaces the monolithic __init__ dispatch.
    Usage: widget = WidgetFactory.create('drag_float', label="My Float", node=self, ...)
    """
    _REGISTRY = {
        'drag_float': DragFloat,
        'slider_float': SliderFloat,
        'knob_float': KnobFloat,
        'input_float': InputFloat,
        'drag_int': DragInt,
        'slider_int': SliderInt,
        'input_int': InputInt,
        'checkbox': CheckboxWidget,
        'text_input': TextInput,
        'text_editor': TextEditor,
        'combo': Combo,
        'radio_group': RadioGroup,
        'list_box': ListBox,
        'color_picker': ColorPicker,
        'button': Button,
        'label': Label,
        'spacer': Spacer,
        'table': TableWidget,
        'drag_float_n': DragFloatN
    }

    @staticmethod
    def create(widget_type, label, **kwargs) -> BasePropertyWidget:
        # Handle specific logic for kwargs that was in original init
        # e.g., converting rows/columns from kwargs

        widget_class = WidgetFactory._REGISTRY.get(widget_type)
        if widget_class is None:
            # Fallback or error handling. Returning base to avoid crash, though it won't draw much.
            return BasePropertyWidget(label, widget_type=widget_type, **kwargs)

        return widget_class(label, widget_type=widget_type, **kwargs)


# --- Backwards Compatibility Wrapper (Optional) ---
# If you don't want to change the code instantiation in Node class, use this:

def PropertyWidget(label: str = "", uuid=None, node=None, widget_type=None, **kwargs):
    return WidgetFactory.create(widget_type, label, uuid=uuid, node=node, **kwargs)


class NodeInput:
    _pin_active_theme = None
    _pin_active_string_theme = None
    _pin_theme_created = False
    _pin_active_array_theme = None
    _pin_active_tensor_theme = None
    _pin_active_list_theme = None
    _pin_active_bang_theme = None
    _pin_theme_created = False

    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None, **kwargs):
        if not self._pin_theme_created:
            self.create_pin_themes()
        self._label = label
        if uuid is None:
            self.uuid = dpg.generate_uuid()
        else:
            self.uuid = uuid
        self.label_uuid = None
        self._parents = []  # input attribute
        self._data = 0
        if default_value is not None:
            self._data = default_value
        self.executor = False
        self.triggers_execution = triggers_execution
        self.node = node
        self.input = None
        self.input_index = -1
        self.fresh_input = False
        self.node_attribute = None
        self.received_bang = False
        self.received_type = None

        self.bang_repeats_previous = True
        if widget_type == 'checkbox':
            self.bang_repeats_previous = False

        self.widget = None
        self.widget_has_trigger = trigger_button
        self.trigger_widget = None
        if widget_type:
            self.widget = PropertyWidget(label, uuid=widget_uuid, node=node, widget_type=widget_type, width=widget_width, triggers_execution=triggers_execution, trigger_button=trigger_button, default_value=default_value, min=min, max=max, **kwargs)
            self.widget.input = self
        self.callback = None
        self.trigger_callback = None
        self.user_data = None
        self.variable = None
        self.action = None
        self.accepted_types = None
        self.type_mask = 0
        self.name_archive = []

    def get_label(self) -> str:
        return self._label

    def set_label(self, name: str) -> None:
        self._label = name
        if self.widget is None:
            dpg.set_value(self.label_uuid, self._label)
        else:
            dpg.set_item_label(self.widget.uuid, self._label)

    def set_input(self, widget_input: 'NodeInput') -> None:
        self.input = widget_input
        if self.widget is not None:
            self.widget.input = self.input

    def show(self) -> None:
        dpg.show_item(self.uuid)

    def hide(self) -> None:
        dpg.hide_item(self.uuid)

    def set_visibility(self, visibility_state: str = 'show_all') -> None:
        if visibility_state == 'show_all':
            if self.node.do_not_delete:
                dpg.bind_item_theme(self.uuid, theme=Node.app.do_not_delete_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=Node.app.global_theme)
        elif visibility_state == 'widgets_only':
            if self.widget is None:
                dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=Node.app.widget_only_theme)
        else:
            dpg.bind_item_theme(self.uuid, theme=Node.app.invisible_theme)
        if self.widget is not None:
            self.widget.set_visibility(visibility_state)

    def create_pin_themes(self) -> None:
        self._pin_inactive_theme = dpg.theme()
        with dpg.theme() as self._pin_active_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (153, 212, 255), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_array_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 255, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_tensor_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 0, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_list_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 0, 255), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_string_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 128, 0), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_bang_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 255, 0), category=dpg.mvThemeCat_Nodes)

    def create(self, parent: int) -> None:
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Input, user_data=self, id=self.uuid)

        with self.node_attribute:
            if self.widget is None:
                with dpg.group(horizontal=self.widget_has_trigger):
                    if self.widget_has_trigger:
                        if self.trigger_callback is not None:
                            self.trigger_widget = dpg.add_button(label='', width=14, callback=self.trigger_callback)
                        else:
                            self.trigger_widget = dpg.add_button(label='', width=14, callback=self.trigger)
                        with dpg.theme() as item_theme:
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
                        dpg.bind_item_theme(self.trigger_widget, item_theme)
                    self.label_uuid = dpg.add_text(self._label)

            else:
                if self.callback:
                    self.widget.callback = self.callback
                if self.widget_has_trigger and self.trigger_callback is not None:
                    self.widget.trigger_callback = self.trigger_callback
                self.widget.user_data = self.user_data
                self.widget.create()



    def set_default_value(self, data: Any) -> None:
        if self.widget is not None:
            self.widget.set_default_value(data)

    def add_callback(self, callback: Callable, user_data: Optional[Any] = None) -> None:
        self.callback = callback
        self.user_data = user_data

    def add_trigger_callback(self, callback: Callable, user_data: Optional[Any] = None) -> None:
        self.trigger_callback = callback
        self.user_data = user_data
        if self.trigger_widget is not None:
            self.trigger_widget.add_callback(callback, user_data)

    def __call__(self) -> Any:
        if self.fresh_input:
            self.fresh_input = False
            return self._data
        self.fresh_input = False
        if self.widget:
            return self.get_widget_value()
        return self._data

    def get_received_data(self) -> Any:
        self.fresh_input = False
        return self._data

    def get_data(self) -> Any:
        self.fresh_input = False
        if self.widget:
            return self.get_widget_value()
        return self._data

    def attach_to_variable(self, variable: 'Variable') -> None:
        self.variable = variable
        self.variable.property = self
        self.widget.attach_to_variable(variable)

    def attach_to_action(self, action: 'Action') -> None:
        self.action = action
        self.widget.attach_to_action(action)

    def delete_parents(self) -> None:
        for p in self._parents:  # output linking to this
            p.remove_child(self)
            for l in p.links:
                if dpg.does_item_exist(l):
                    d = dpg.get_item_user_data(l)
                    input_ = d[1]
                    if input_.uuid == self.uuid:
                        dpg.delete_item(l)

        self._parents = []

    def remove_parent(self, parent: 'NodeOutput') -> None:
        if parent in self._parents:
            self._parents.remove(parent)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        # if Node.app.trace:
        #     Node.app.increment_trace_indent()
        #     print(Node.app.trace_indent, end='')
        #     print('->', self.node.label + ':[' + self.get_label() + ']')
        if data is not None:
            if orig_type is None:
                self.received_type = type(data)
            if not self.node.check_for_messages(data):
                self.node.active_input = self
                if type(data) == list and len(data) == 1 and type(data[0]) == str and data[0] == 'bang':
                    data = data[0]
                if type(data) == str and data == 'bang':
                    self.received_bang = True
                    if self.bang_repeats_previous:
                        if self.widget:
                            data = self.get_widget_value()
                        else:
                            data = self._data

                self._data = data
                self.fresh_input = True
                if self.node.visibility == 'show_all':
                    try:
                        if Node.app.color_code_pins:
                            if self.received_type is list:
                                dpg.bind_item_theme(self.uuid, self._pin_active_list_theme)
                            elif self.received_type is np.ndarray:
                                dpg.bind_item_theme(self.uuid, self._pin_active_array_theme)
                            elif self.received_type is torch.Tensor:
                                dpg.bind_item_theme(self.uuid, self._pin_active_tensor_theme)
                            elif self.received_type is str:
                                if self.received_bang:
                                    dpg.bind_item_theme(self.uuid, self._pin_active_bang_theme)
                                else:
                                    dpg.bind_item_theme(self.uuid, self._pin_active_string_theme)
                            else:
                                dpg.bind_item_theme(self.uuid, self._pin_active_theme)
                        else:
                            dpg.bind_item_theme(self.uuid, self._pin_active_theme)
                        Node.app.get_current_editor().add_active_pin(self.uuid)
                    except Exception as e:
                        pass
                if self.accepted_types:
                    if self.type_mask == 0:
                        self.type_mask = create_type_mask_from_list(self.accepted_types)
                    data = conform_to_type_mask(data, self.type_mask)
                if self.widget:
                    self.widget.set(data)
                if self.callback:
                    self.callback()
                self.received_bang = False
                self.node.active_input = None
        else:
            self._data = data
            self.fresh_input = True
        self.received_bang = False
        # if Node.app.trace:
        #     Node.app.decrement_trace_indent()

    def conform_to_accepted_types(self, data: Any) -> Any:
        if self.accepted_types:
            t = type(data)
            if t in self.accepted_types:
                return data

        return data

    def trigger(self) -> None:
        if self.triggers_execution and not self.node.message_handled:
            self.node.active_input = self
            if Node.app.trace:
                Node.app.increment_trace_indent()
                # print(Node.app.trace_indent, end='')
                # print('>> ' + self.node.label + ':[' + self.get_label() + ']')
            self.node.execute()
            if Node.app.trace:
                Node.app.decrement_trace_indent()
            self.node.active_input = None
        else:
            self.node.message_handled = False

    def set_parent(self, parent: 'NodeOutput') -> None:
        if parent not in self._parents:
            self._parents.append(parent)

    def get_parents(self) -> List['NodeOutput']:
        return self._parents

    def save(self, input_container: Dict[str, Any]) -> bool:
        if self.widget:
            self.widget.save(input_container)
            return True
        return False

    def load(self, input_container: Dict[str, Any]) -> None:
        if self.widget:
            self.widget.load(input_container)

    def get_widget_value(self) -> Any:
        if self.widget:
            return self.widget.value

    def set(self, data: Any, propagate: bool = True) -> None:
        if type(data) == list:
            if len(data) == 1:
                data = data[0]
            else:
                if self.widget is not None and self.widget.widget in ['text_input', 'combo', 'radio_group', 'text_editor']:
                    data = any_to_string(data)
                elif self.widget is not None and self.widget.widget == 'drag_float_n':
                    pass
                else:
                    data = data[0]
        self._data = data
        if self.widget:
            self.widget.set(data, False)
        if self.variable and propagate:
            self.variable.set_value(data)

    def set_font(self, font: Any) -> None:
        dpg.bind_item_font(self.uuid, font)
        if self.widget:
            self.widget.set_font(font)

    @property
    def data(self):
        return self._data


class NodeIntInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type is str and data == 'bang':
            self.received_bang = True
            data = self._data
        int_data = any_to_int(data, validate=True)
        super().receive_data(int_data, self.received_type)


class NodeFloatInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type is str and data == 'bang':
            self.received_bang = True
            data = self._data
        else:
            self.received_bang = False
        float_data = any_to_float(data, validate=True)
        super().receive_data(float_data, self.received_type)


class NodeBoolInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type is str and data == 'bang':
            self.received_bang = True
            data = self._data
        else:
            self.received_bang = False
        bool_data = any_to_bool(data)
        super().receive_data(bool_data, self.received_type)


class NodeStringInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.strip_returns = True

    def set_strip_returns(self, value: bool) -> None:
        self.strip_returns = value

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type == str and data == 'bang':
            self.received_bang = True
            if self.bang_repeats_previous:
                if self.widget:
                    data = self.get_widget_value()
                else:
                    data = self._data
        else:
            self.received_bang = False
        string_data = any_to_string(data, strip_returns=self.strip_returns)
        super().receive_data(string_data, self.received_type)


class NodeListInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type is str and data == 'bang':
            self.received_bang = True
            data = self._data
        else:
            self.received_bang = False
        list_data = any_to_list(data)
        super().receive_data(list_data, self.received_type)


class NodeArrayInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if self.received_type is str and data == 'bang':
            self.received_bang = True
            data = self._data
        else:
            self.received_bang = False
        array_data = any_to_array(data, validate=True)
        super().receive_data(array_data, self.received_type)


class NodeTensorInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        self.received_type = type(data)
        if torch_available:
            if self.received_type is str and data == 'bang':
                data = self._data
                self.received_bang = True
            else:
                self.received_bang = False
            tensor_data = any_to_tensor(data, validate=True)
            super().receive_data(tensor_data, self.received_type)


class NodeNumericalInput(NodeInput):
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        super().__init__(label, uuid, node, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.numerical_data = None
        if default_value is not None:
            self.to_numerical(default_value)

    def to_numerical(self, data: Any) -> None:
        t = type(data)
        if t == str:
            self.numerical_data = string_to_float_or_int(data)
        elif t == list:
            numerical_data = list_to_array(data)
            if numerical_data is not None:
                self.numerical_data = numerical_data
        elif t in [bool, np.bool_]:
            if data:
                self.numerical_data = 1
            else:
                self.numerical_data = 0
        else:
            self.numerical_data = data

    def receive_data(self, data: Any, orig_type: Optional[Type] = None) -> None:
        super().receive_data(data, orig_type)
        self.to_numerical(data)

    def __call__(self) -> Any:
        self.fresh_input = False
        if self.widget:
            return self.get_widget_value()
        return self.numerical_data


class Conduit:
    def __init__(self, label: str) -> None:
        self.label = label
        self.clients = []

    def transmit(self, data: Any, from_client: Optional[Any] = None) -> None:
        for client in self.clients:
            if client != from_client:
                client.receive(self.label, data)

    def attach_client(self, client: Any) -> None:
        if client not in self.clients:
            self.clients.append(client)

    def detach_client(self, client: Any) -> None:
        if client in self.clients:
            self.clients.remove(client)


class Variable:
    def __init__(self, label: str, default_value=0.0, setter=None, getter=None) -> None:
        self.label = label
        self.property = None
        self.clients = []
        self.value = default_value
        self.set_callback = setter
        self.get_callback = getter

    def notify_clients_of_value_change(self, from_client: Optional[Any] = None) -> None:
        for client in self.clients:
            if client != from_client:
                client.variable_update()

    def set(self, data: Any, from_client: Optional[Any] = None) -> None:
        if self.property and from_client != self.property.node:
            self.property.set(data, propagate=False)
        self.set_value(data)
        self.notify_clients_of_value_change(from_client)

    def __call__(self) -> Any:
        if self.property:
            self.value = self.property()
        else:
            self.value = self.get_value()
        return self.value

    def get(self) -> Any:
        if self.property:
            self.value = self.property()
        else:
            self.value = self.get_value()
        return self.value

    def attach_client(self, client: Any) -> None:
        if client not in self.clients:
            self.clients.append(client)

    def detach_client(self, client: Any) -> None:
        if client in self.clients:
            if self.property is not None and self.property.node == client:
                self.property = None
            self.clients.remove(client)

    def set_value(self, data: Any) -> None:  # does not notify_clients_of_value_change
        self.value = data
        if self.set_callback:
            self.set_callback(data)

    def get_value(self) -> Any:
        if self.get_callback:
            return self.get_callback()
        return self.value


# Action allows you to set a function in your python code that will be called
# if bound button is clicked or bound menu option is selected
class Action:
    def __init__(self, label: str, action_function):
        self.label = label
        self.property = None
        self.clients = []
        self.action_function = action_function

    def perform(self) -> None:
        if self.action_function is not None:
            self.action_function()

class NodeGroup:
    """A wrapper for a Dear PyGui group widget within a Node."""

    def __init__(self, tag: Union[str, int], parent_node: 'Node', **kwargs):
        self.tag = tag
        self.parent_node = parent_node
        self.kwargs = kwargs  # Store extra dpg options like 'horizontal'

class Node:
    app = None
    inactive_theme = None
    active_theme = None
    active_theme_red = None
    active_theme_green = None
    active_theme_yellow = None
    active_theme_blue = None
    active_theme_pink = None
    active_theme_base = None

    def __init__(self, label: str, data: Any, args: Optional[List[str]] = None) -> None:
        self.label = label
        self.data = data
        self.label = label
        self.uuid = dpg.generate_uuid()
        self.static_uuid = dpg.generate_uuid()
        self.inputs = []
        self.outputs = []
        self.options = []
        self.properties = []
        self.displays = []
        self.ordered_elements = []
        self.message_handlers = {}
        self.message_handlers['set_preset'] = self.set_preset_state
        self.message_handlers['get_preset'] = self.get_preset_state
        self.message_handled = False

        self.property_registery = {}

        self._data = data
        self.unparsed_args = args
        self.ordered_args = []
        self.horizontal = False
        self.loaded_uuid = -1
        self.options_visible = False
        self.has_frame_task = False
        self.created = False
        self.parse_args()
        self.my_editor = self.app.get_current_editor()
        self.draggable = True
        self.visibility = 'show_all'
        self.presentation_state = 'show_all'
        self.do_not_delete = False
        self.active_input = None
        self.in_loading_process = False
        self.show_options_check = None
        self.help_file_name = None
        if Node.active_theme is None:
            self.create_button_themes()

    def custom_cleanup(self) -> None:
        pass

    def get_patcher_path(self):
        path = ''
        if self.my_editor is not None:
            patcher = self.my_editor
            path = '/' + patcher.patch_name
            while patcher.parent_patcher is not None:
                parent_patcher = patcher.parent_patcher
                path = '/' + parent_patcher.patch_name + path
                patcher = patcher.parent_patcher
        return path

    def get_patcher_name(self):
        if self.my_editor is not None:
            return self.my_editor.patch_name

    def cleanup(self) -> None:
        self.remove_frame_tasks()
        self.custom_cleanup()
        for input_ in self.inputs:
            input_.delete_parents()
            if input_.widget is not None:
                dpg.delete_item(input_.widget.uuid)
            dpg.delete_item(input_.uuid)
        for output_ in self.outputs:
            output_.remove_links()
            dpg.delete_item(output_.uuid)
        for property_ in self.properties:
            if property_.widget:
                dpg.delete_item(property_.widget.uuid)
            dpg.delete_item(property_.uuid)

    def set_draggable(self, can_drag: bool) -> None:
        self.draggable = can_drag
        dpg.configure_item(self.uuid, draggable=self.draggable)

    def set_visibility(self, visibility_state: str = 'show_all') -> None:
        self.visibility = visibility_state
        if visibility_state == 'show_all':
            if self.do_not_delete:
                dpg.bind_item_theme(self.uuid, theme=self.app.do_not_delete_theme)
            elif not self.draggable:
                dpg.bind_item_theme(self.uuid, theme=self.app.locked_position_theme)
            else:
                dpg.bind_item_theme(self.uuid, theme=self.app.global_theme)
            dpg.configure_item(self.uuid, draggable=self.draggable)
            dpg.configure_item(self.uuid, label=self.label)
        elif visibility_state == 'widgets_only':
            dpg.bind_item_theme(self.uuid, theme=self.app.widget_only_theme)
            dpg.configure_item(self.uuid, draggable=self.draggable)
            dpg.configure_item(self.uuid, label='')
        else:
            dpg.bind_item_theme(self.uuid, theme=self.app.invisible_theme)
            dpg.configure_item(self.uuid, draggable=False)

        for in_ in self.inputs:
            in_.set_visibility(visibility_state)
        for out_ in self.outputs:
            out_.set_visibility(visibility_state)
        for property_ in self.properties:
            property_.set_visibility(visibility_state)
        for display_ in self.displays:
            display_.set_visibility(visibility_state)

        self.set_custom_visibility()

    def create_button_themes(self):
        with dpg.theme() as Node.active_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.inactive_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_green:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 208, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_blue:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (29, 151, 236, 103), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (29, 151, 236, 180), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_red:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (192, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_pink:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (192, 0, 128), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 0, 192), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_yellow:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 128), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as Node.active_theme_base:
            with dpg.theme_component(dpg.mvAll):
                # dpg.add_theme_color(dpg.mvThemeCol_Button, (32, 64, 64), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (128, 128, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)

    # Add this method inside your base Node class
    def _install_group(self, group: 'NodeGroup'):
        """Creates the dpg.group widget and parents it to this node."""
        # self.uuid should be the tag of the main dpg.node item
        dpg.add_group(tag=group.tag, parent=self.uuid, **group.kwargs)

    def set_custom_visibility(self) -> None:
        pass

    def set_font(self, font: Any) -> None:
        dpg.bind_item_font(self.uuid, font)

    def set_title(self, title: str) -> None:
        dpg.configure_item(self.uuid, label=title)

    def post_load_callback(self) -> None:
        pass

    def send_all(self) -> None:
        for i in range(len(self.outputs)):
            j = len(self.outputs) - i - 1
            output = self.outputs[j]
            output.send_internal()  # should not always trigger!!! make flag to indicate trigger always or trigger on change...

    def add_label(self, label: str = "") -> None:
        new_property = NodeProperty(label, widget_type='label')
        # self.properties.append(new_property)
        self.ordered_elements.append(new_property)
        return new_property

    def add_spacer(self) -> None:
        new_property = NodeProperty('', widget_type='spacer')
        # self.properties.append(new_property)
        self.ordered_elements.append(new_property)
        return new_property

    def add_property(self, label: str = "", uuid: Optional[int] = None,
                    widget_type: Optional[str] = None, width: int = 80,
                    triggers_execution: bool = False, trigger_button: bool = False,
                    default_value: Any = None, min: Optional[float] = None,
                    max: Optional[float] = None, callback: Optional[Callable] = None) -> 'NodeProperty':
        new_property = NodeProperty(label, uuid, self, widget_type, width, triggers_execution, trigger_button, default_value, min, max)
        self.properties.append(new_property)
        self.ordered_elements.append(new_property)
        if callback is not None:
            new_property.add_callback(callback=callback, user_data=self)
        self.property_registery[label] = new_property
        self.message_handlers[label] = self.property_message
        return new_property

    def add_option(self, label: str = "", uuid: Optional[int] = None,
                    widget_type: Optional[str] = None, width: int = 80,
                    triggers_execution: bool = False, trigger_button: bool = False,
                    default_value: Any = None, min: Optional[float] = None,
                    max: Optional[float] = None, callback: Optional[Callable] = None) -> 'NodeProperty':
        if self.show_options_check is None and self.app.easy_mode:
            self.show_options_check = self.add_property('show options', widget_type='checkbox', default_value=False, callback=self.show_hide_options)

        new_option = NodeProperty(label, uuid, self, widget_type, width, triggers_execution, trigger_button, default_value, min, max)
        self.options.append(new_option)
        self.ordered_elements.append(new_option)
        if callback is not None:
            new_option.add_callback(callback=callback, user_data=self)
        self.property_registery[label] = new_option
        self.message_handlers[label] = self.property_message
        return new_option

    def get_preset_state(self):
        preset = {}
        return preset

    def set_preset_state(self, preset):
        pass

    def get_help(self):
        if os.path.exists('dpg_system/help'):
            if self.help_file_name is not None:
                temp_path = 'dpg_system/help/' + self.help_file_name + '.json'
            else:
                temp_path = 'dpg_system/help/' + self.label + '_help.json'
            if os.path.exists(temp_path):
                # if patcher is already open?
                tabs = Node.app.tabs
                for tab in tabs:
                    config = dpg.get_item_configuration(tab)
                    if 'label' in config:
                        title = config['label']
                        if title == self.label + '_help':
                            Node.app.select_tab(tab)
                            return
                Node.app.load_from_file(temp_path)

    def add_display(self, label: str = "", uuid=None, width=80, callback=None):
        new_display = NodeDisplay(label, uuid, self, width)
        self.displays.append(new_display)
        self.ordered_elements.append(new_display)
        # new_display.node = self
        if callback is not None:
            new_display.add_callback(callback=callback, user_data=self)
        return new_display

    def add_group(self, tag: Optional[Union[str, int]] = None,
                  horizontal: bool = False,
                  width: int = -1) -> 'NodeGroup':
        """
        Adds a container group to the node.

        Args:
            tag (Optional[Union[str, int]]): The unique identifier for this group.
                If None, DPG will generate one.
            horizontal (bool): If True, widgets added to this group will be
                arranged horizontally. Defaults to False.
            width (int): The width of the group. -1 means auto-fit.

        Returns:
            NodeGroup: A wrapper object representing the created group.
                Use its .tag attribute to parent other widgets to it.
        """
        # If no tag is provided, generate a unique one using DPG's internal system
        group_tag = dpg.generate_uuid() if tag is None else tag

        # Create the wrapper object, storing the dpg options in kwargs
        new_group = NodeGroup(group_tag, self, horizontal=horizontal, width=width)

        # Use the helper to create the actual DPG widget
        self._install_group(new_group)
        return new_group

    def add_input(self, label: str = "", uuid: Optional[int] = None,
                 widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                 widget_width: int = 80, triggers_execution: bool = False,
                 trigger_button: bool = False, default_value: Any = None,
                 min: Optional[float] = None, max: Optional[float] = None,
                 callback: Optional[Callable] = None,
                 trigger_callback: Optional[Callable] = None,
                 **kwargs) -> 'NodeInput':
        # print('add_input()', kwargs)
        new_input = NodeInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max, **kwargs)
        self.install_input(new_input, callback=callback, trigger_callback=trigger_callback)
        return new_input

    def install_input(self, new_input, callback, trigger_callback=None):
        self.inputs.append(new_input)
        new_input.input_index = len(self.inputs) - 1

        self.ordered_elements.append(new_input)
        if callback is not None:
            new_input.add_callback(callback=callback, user_data=self)
        if trigger_callback is not None:
            new_input.add_trigger_callback(callback=trigger_callback, user_data=self)
        if new_input.widget is not None:
            self.property_registery[new_input.get_label()] = new_input
            self.message_handlers[new_input.get_label()] = self.property_message

    def add_int_input(self, label: str = "", uuid: Optional[int] = None,
                     widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                     widget_width: int = 80, triggers_execution: bool = False,
                     trigger_button: bool = False, default_value: Optional[int] = None,
                     min: Optional[int] = None, max: Optional[int] = None,
                     callback: Optional[Callable] = None) -> 'NodeIntInput':
        new_input = NodeIntInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_float_input(self, label: str = "", uuid: Optional[int] = None,
                       widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                       widget_width: int = 80, triggers_execution: bool = False,
                       trigger_button: bool = False, default_value: Optional[float] = None,
                       min: Optional[float] = None, max: Optional[float] = None,
                       callback: Optional[Callable] = None) -> 'NodeFloatInput':
        new_input = NodeFloatInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_bool_input(self, label: str = "", uuid: Optional[int] = None,
                       widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                       widget_width: int = 80, triggers_execution: bool = False,
                       trigger_button: bool = False, default_value: Any = None,
                       min: Optional[float] = None, max: Optional[float] = None,
                       callback: Optional[Callable] = None) -> 'NodeBoolInput':
        new_input = NodeBoolInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_string_input(self, label: str = "", uuid: Optional[int] = None,
                         widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                         widget_width: int = 80, triggers_execution: bool = False,
                         trigger_button: bool = False, default_value: Any = None,
                         min: Optional[float] = None, max: Optional[float] = None,
                         callback: Optional[Callable] = None) -> 'NodeStringInput':
        new_input = NodeStringInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_list_input(self, label: str = "", uuid: Optional[int] = None,
                       widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                       widget_width: int = 80, triggers_execution: bool = False,
                       trigger_button: bool = False, default_value: Any = None,
                       min: Optional[float] = None, max: Optional[float] = None,
                       callback: Optional[Callable] = None) -> 'NodeListInput':
        new_input = NodeListInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_array_input(self, label: str = "", uuid: Optional[int] = None,
                        widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                        widget_width: int = 80, triggers_execution: bool = False,
                        trigger_button: bool = False, default_value: Any = None,
                        min: Optional[float] = None, max: Optional[float] = None,
                        callback: Optional[Callable] = None) -> 'NodeArrayInput':
        new_input = NodeArrayInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_tensor_input(self, label: str = "", uuid: Optional[int] = None,
                         widget_type: Optional[str] = None, widget_uuid: Optional[int] = None,
                         widget_width: int = 80, triggers_execution: bool = False,
                         trigger_button: bool = False, default_value: Any = None,
                         min: Optional[float] = None, max: Optional[float] = None,
                         callback: Optional[Callable] = None) -> 'NodeTensorInput':
        new_input = NodeTensorInput(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.install_input(new_input, callback=callback)
        return new_input

    def add_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeOutput':
        new_output = NodeOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_int_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeIntOutput':
        new_output = NodeIntOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_float_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeFloatOutput':
        new_output = NodeFloatOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_bool_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeBoolOutput':
        new_output = NodeBoolOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_string_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeStringOutput':
        new_output = NodeStringOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_list_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeListOutput':
        new_output = NodeListOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_array_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeArrayOutput':
        new_output = NodeArrayOutput(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_tensor_output(self, label: str = "output", pos: Optional[List[float]] = None) -> 'NodeTensorOutput':
        if torch_available:
            new_output = NodeTensorOutput(label, self, pos)
            self.outputs.append(new_output)
            self.ordered_elements.append(new_output)
            return new_output
        return None

    def add_handler_to_widgets(self):
        for input_ in self.inputs:
            if input_.widget and input_.widget.widget not in['checkbox', 'button', 'combo', 'knob_float', 'label', 'table']:
                for uuid in input_.widget.uuids:
                    dpg.bind_item_handler_registry(uuid, "widget handler")

        for property_ in self.properties:
            if property_.widget.widget not in ['checkbox', 'button', 'spacer', 'label', 'table']:
                for uuid in property_.widget.uuids:
                    dpg.bind_item_handler_registry(uuid, "widget handler")
                # dpg.bind_item_handler_registry(property_.widget.uuid, "widget handler")

        for option in self.options:
            if option.widget.widget not in ['checkbox', 'button', 'spacer', 'label', 'table']:
                for uuid in option.widget.uuids:
                    dpg.bind_item_handler_registry(uuid, "widget handler")
                # dpg.bind_item_handler_registry(option.widget.uuid, "widget handler")


    def value_changed(self, widget_uuid, force=False):
        pass

    def variable_update(self) -> None:
        self.execute()

    def execute(self) -> None:
        for attribute in self.outputs:
            attribute.set_value(self._data)
        self.send_all()

    def custom_create(self, from_file):
        pass

    def trigger(self) -> None:
        pass

    def increment_widget(self, widget):
        widget.increment()

    def decrement_widget(self, widget):
        widget.decrement()

    def add_frame_task(self) -> None:
        Node.app.add_frame_task(self)
        self.has_frame_task = True

    def remove_frame_tasks(self) -> None:
        if self.has_frame_task:
            Node.app.remove_frame_task(self)
            self.has_frame_task = False

    def create(self, parent: int, pos: List[float], from_file: bool = False) -> None:
        with dpg.node(parent=parent, label=self.label, tag=self.uuid, pos=pos):
            dpg.set_item_pos(self.uuid, pos)
            self.handle_parsed_args()

            if len(self.ordered_elements) > 0:
                for attribute in self.ordered_elements:
                    attribute.create(self.uuid)
                self.custom_create(from_file)
                self.update_parsed_args()
            else:
                self.custom_create(from_file)
        dpg.set_item_user_data(self.uuid, self)
        self.add_handler_to_widgets()
        for option_att in self.options:
            dpg.hide_item(option_att.uuid)
            dpg.hide_item(option_att.widget.uuid)
        self.created = True

    def show_hide_options(self) -> None:
        if self.show_options_check():
            if not self.options_visible:
                self.toggle_show_hide_options()
        else:
            if self.options_visible:
                self.toggle_show_hide_options()

    def args_as_list(self, supplied_args: Optional[List[str]] = None) -> List[str]:
        if supplied_args is not None:
            return supplied_args
        return self.ordered_args

    def arg_as_number(self, default_value: Union[int, float] = 0, index: int = 0) -> Union[int, float]:
        value = default_value
        if len(self.ordered_args) > index:
            val, t = decode_arg(self.ordered_args, index)
            if t == float:
                value = val
            else:
                value = any_to_int(val)
        return value

    def arg_as_int(self, default_value: int = 0, index: int = 0) -> int:
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_int(self.ordered_args[index])
        return value

    def arg_as_float(self, default_value: float = 0.0, index: int = 0) -> float:
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_float(self.ordered_args[index])
        return value

    def arg_as_string(self, default_value: str = '', index: int = 0) -> str:
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_string(self.ordered_args[index])
        return value

    def arg_as_bool(self, default_value: bool = False, index: int = 0) -> bool:
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_bool(self.ordered_args[index])
        return value

    def toggle_show_hide_options(self) -> None:
        if len(self.options) > 0:
            self.options_visible = not self.options_visible
            if self.options_visible:
                for option_att in self.options:
                    dpg.show_item(option_att.uuid)
                    dpg.show_item(option_att.widget.uuid)
            else:
                for option_att in self.options:
                    dpg.hide_item(option_att.uuid)
                    dpg.hide_item(option_att.widget.uuid)

    def check_for_messages(self, in_data: Union[str, List[Any]]) -> bool:
        self.message_handled = False
        if len(self.message_handlers) > 0:
            message = ''
            message_data = []
            t = type(in_data)
            if t == str:
                message_list = in_data.split(' ')
                message = message_list[0]
                message_data = message_list[1:]
            elif t == list:
                list_len = len(in_data)
                if list_len > 0:
                    if type(in_data[0]) == str:
                        message = in_data[0]
                        message_data = in_data[1:]
            if message != '':
                if message in self.message_handlers:
                    self.message_handlers[message](message, message_data)
                    self.message_handled = True
                # else:  # maybe two words in message header
                #     message = ' '.join(message.split('_'))
                #     if message in self.message_handlers:
                #         self.message_handlers[message](message, message_data)
                #         handled = True
        return self.message_handled

    def save_custom(self, container: Dict[str, Any]):
        pass

    def load_custom(self, container: Dict[str, Any]):
        pass

    def on_edit(self, widget):
        pass

    def on_deactivate(self, widget):
        pass

    def property_message(self, message: str = '', args: List[Any] = []) -> None:
        property = None
        if message in self.property_registery:
            property = self.property_registery[message]
        if len(args) == 1:
            property.set(args[0])
        else:
            property.set(args)
        if property.widget is not None:
            if property.widget.callback is not None:
                property.widget.callback()

    def copy_to_clipboard(self) -> Dict[str, Any]:
        node_container = {}
        self.save(node_container, 0)
        nodes_container = {self.label: node_container}
        clipboard_container = {'nodes': nodes_container}
        return clipboard_container

    def save(self, node_container: Dict[str, Any], index: int) -> None:
        if node_container is not None:
            if self.unparsed_args and len(self.unparsed_args) > 0:
                args_container = {}
                args_string = self.label
                for index, arg in enumerate(self.unparsed_args):
                    args_container[index] = arg
                    args_string = args_string + ' ' + arg
                node_container['init'] = args_string
            node_container['name'] = self.label
            node_container['id'] = self.uuid
            pos = dpg.get_item_pos(self.uuid)
            node_container['position_x'] = pos[0]
            node_container['position_y'] = pos[1]
            size = dpg.get_item_rect_size(self.uuid)
            node_container['width'] = size[0]
            node_container['height'] = size[1]
            node_container['visibility'] = self.visibility
            node_container['draggable'] = self.draggable
            if self.do_not_delete:
                node_container['protected'] = True
            node_container['presentation_state'] = self.presentation_state

            self.store_properties(node_container)

    def post_creation_callback(self) -> None:
        pass

    def load(self, node_container: Optional[Dict[str, Any]], offset: Optional[List[float]] = None) -> None:
        self.in_loading_process = True
        if offset is None:
            offset = [0, 0]
        if node_container is not None:
            if 'init' in node_container:
                arg_container = node_container['init']
                self.unparsed_args = arg_container.split(' ')[1:]
            if 'name' in node_container:
                self.label = node_container['name']
            if 'id' in node_container:
                self.loaded_uuid = node_container['id']
            if 'position_x' in node_container and 'position_y' in node_container:
                pos = [0, 0]
                pos[0] = node_container['position_x'] + offset[0]
                pos[1] = node_container['position_y'] + offset[1]
                dpg.set_item_pos(self.uuid, pos)
            if 'protected' in node_container:
                self.do_not_delete = True
            if 'visibility' in node_container:
                self.set_visibility(node_container['visibility'])
            if 'draggable' in node_container:
                self.set_draggable(node_container['draggable'])
            if 'presentation_state' in node_container:
                self.presentation_state = node_container['presentation_state']
            self.restore_properties(node_container)
        self.in_loading_process = False

    def store_properties(self, node_container: Dict[str, Any]) -> None:
        properties_container = {}
        property_number = 0
        for index, _input in enumerate(self.inputs):
            input_container = {}
            if _input.save(input_container):
                properties_container[property_number] = input_container
                property_number += 1
        for index, _property in enumerate(self.properties):
            property_container = {}
            _property.save(property_container)
            properties_container[property_number] = property_container
            property_number += 1
        for index, _option in enumerate(self.options):
            option_container = {}
            _option.save(option_container)
            properties_container[property_number] = option_container
            property_number += 1
        if property_number > 0:
            node_container['properties'] = properties_container
        self.save_custom(node_container)

    def restore_properties(self, node_container: Dict[str, Any]) -> None:
        if 'properties' in node_container:
            properties_container = node_container['properties']
            for index, property_index in enumerate(properties_container):
                property_container = properties_container[property_index]
                if 'name' in property_container:
                    property_label = property_container['name'].strip('#')
                    org_label = property_container['name']
                    found = False

                    for input in self.inputs:
                        if input.widget is not None:
                            a_label = dpg.get_item_label(input.widget.uuid)
                            a_label = a_label.strip('#')
                            if a_label == property_label or a_label == org_label:
                                if 'value' in property_container:
                                    value = property_container['value']
                                    if input.widget.widget != 'button':
                                        input.widget.set(value)
                                        self.active_input = input
                                        input.widget.value_changed(force=True)
                                found = True
                                break
                    if not found:
                        for property in self.properties:
                            if property.widget is not None:
                                a_label = dpg.get_item_label(property.widget.uuid)
                                if a_label == property_label:
                                    if 'value' in property_container:
                                        value = property_container['value']
                                        if property.widget.widget != 'button':
                                            property.widget.set(value)
                                            property.widget.value_changed(force=True)
                                    found = True
                                    break
                    if not found:
                        for option in self.options:
                            if option.widget:
                                a_label = dpg.get_item_label(option.widget.uuid)
                                if a_label == property_label:
                                    if 'value' in property_container:
                                        value = property_container['value']
                                        if option.widget.widget != 'button':
                                            option.widget.set(value)
                                            option.widget.value_changed(force=True)
                                    found = True
                                    break
                    if not found:
                        if property_label == '':
                            if len(self.inputs) > 0:
                                input = self.inputs[0]
                                if input.widget is not None:
                                    if 'value' in property_container:
                                        value = property_container['value']
                                        if input.widget.widget != 'button':
                                            input.widget.set(value)
                                            self.active_input = input
                                            input.widget.value_changed(force=True)


        self.load_custom(node_container)
        self.update_parameters_from_widgets()

    def update_parameters_from_widgets(self) -> None:
        pass

    # def set_value(self, uuid: int, type: Any, input: Any) -> None:
    #     if type in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
    #         dpg.set_value(uuid, any_to_float(input))
    #     elif type in ['drag_int', 'input_int', 'slider_int']:
    #         dpg.set_value(uuid, any_to_int(input))
    #     elif type in ['input_text']:
    #         dpg.set_value(uuid, any_to_string(input))
    #     elif type in ['toggle']:
    #         dpg.set_value(uuid, any_to_bool(input))

    def parse_args(self) -> None:
        self.ordered_args = []
        self.parsed_args = {}
        if self.unparsed_args is not None:
            for arg in self.unparsed_args:
                handled = False
                if '=' in arg and arg != '=':
                    arg_parts = arg.split('=')
                    if len(arg_parts) == 2:
                        if arg_parts[0][-1] == ' ':
                            arg_parts[0] = arg_parts[0][:-1]
                        if arg_parts[1][0] == ' ':
                            arg_parts[1] = arg_parts[1][1:]
                        self.parsed_args[arg_parts[0]] = arg_parts[1]
                        handled = True
                if not handled:
                    self.ordered_args.append(arg)

    def handle_parsed_args(self) -> None:
        for arg_name in self.parsed_args:
            if arg_name in self.property_registery:
                property = self.property_registery[arg_name]
                property.set_default_value(self.parsed_args[arg_name])

    def update_parsed_args(self) -> None:
        for arg_name in self.parsed_args:
            if arg_name in self.property_registery:
                property = self.property_registery[arg_name]
                if property.widget is not None:
                    if property.widget.callback is not None:
                        property.widget.callback()


def register_base_nodes():
    Node.app.register_node('patcher', PatcherNode.factory)
    Node.app.register_node('p', PatcherNode.factory)
    Node.app.register_node('in', PatcherInputNode.factory)
    Node.app.register_node('out', PatcherOutputNode.factory)


class PatcherInputNode(Node):
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'PatcherInputNode':
        node = PatcherInputNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_name = ''
        if len(args) > 0:
            s, t = decode_arg(args, 0)
            if t == str:
                self.input_name = s

        text_width = dpg.get_text_size('source')
        if text_width is None:
            text_width = [80, 14]
        self.go_property = self.add_property('source', widget_type='button', width=text_width[0] + 8, callback=self.jump_to_patcher)
        self.input_out = self.add_output(self.input_name)
        self.patcher_node = self.app.get_current_editor().patcher_node
        self.name_option = self.add_option('input name', widget_type='text_input', default_value=self.input_name, callback=self.name_changed)

    def name_changed(self):
        self.input_name = self.name_option()
        old_name = self.input_out.get_label()
        self.input_out.set_label(self.input_name)
        if self.patcher_node:
            self.patcher_node.change_input_name(old_name, self.input_name)

    def jump_to_patcher(self):
        parent_patcher = self.app.get_current_editor().parent_patcher
        if parent_patcher is not None:
            for i, editor in enumerate(self.app.node_editors):
                if editor == parent_patcher:
                    tab = self.app.tabs[i]
                    self.app.current_node_editor = i
                    dpg.set_value(self.app.tab_bar, tab)
                    break

    def connect_to_parent(self, patcher_node: 'PatcherNode') -> None:
        # print('in connect to parent', self.input_name)
        self.patcher_node = patcher_node
        self.custom_create(False)  # ??

    def custom_create(self, from_file):
        if self.patcher_node is not None:
            remote_input = self.patcher_node.add_patcher_input(self.input_name, self.input_out)
            if self.input_name == '':
                self.input_name = remote_input.get_label()
                self.input_out.set_label(self.input_name)
                self.name_option.set(self.input_name)

    def custom_cleanup(self):
        if self.patcher_node is not None:
            self.patcher_node.remove_patcher_input(self.input_name, self.input_out)


class PatcherOutputNode(Node):
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'PatcherOutputNode':
        node = PatcherOutputNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # print('create out')
        self.output_name = ''
        if len(args) > 0:
            s, t = decode_arg(args, 0)
            if t == str:
                self.output_name = s
        text_width = dpg.get_text_size('dest')
        if text_width is None:
            text_width = [80, 14]
        self.go_property = self.add_property('dest', widget_type='button', width=text_width[0] + 8, callback=self.jump_to_patcher)
        self.output_in = self.add_input(self.output_name, callback=self.send_to_patcher)
        self.patcher_node = self.app.get_current_editor().patcher_node
        self.target_output = None
        self.name_option = self.add_option('output name', widget_type='text_input', default_value=self.output_name, callback=self.name_changed)

    def name_changed(self):
        self.output_name = self.name_option()
        old_name = self.output_in.get_label()
        self.output_in.set_label(self.output_name)
        if self.patcher_node:
            self.patcher_node.change_output_name(old_name, self.output_name)

    def send_to_patcher(self):
        if self.target_output is not None:
            if self.output_in.fresh_input:
                self.output_in.fresh_input = False
                data = self.output_in()
                if data is not None:
                    self.target_output.send(data)

    def jump_to_patcher(self):
        parent_patcher = self.app.get_current_editor().parent_patcher
        if parent_patcher is not None:
            for i, editor in enumerate(self.app.node_editors):
                if editor == parent_patcher:
                    tab = self.app.tabs[i]
                    self.app.current_node_editor = i
                    dpg.set_value(self.app.tab_bar, tab)
                    break

    def connect_to_parent(self, patcher_node: 'PatcherNode') -> None:
        # print('out connect to parent', self.output_name)
        self.patcher_node = patcher_node
        self.custom_create(False)

    def custom_create(self, from_file):
        if self.patcher_node is not None:
            self.target_output = self.patcher_node.add_patcher_output(self.output_name, self.output_in)
            if self.output_name == '':
                self.output_name = self.target_output.get_label()
                self.output_in.set_label(self.output_name)
                self.name_option.set(self.output_name)

    def custom_cleanup(self):
        if self.patcher_node is not None:
            self.patcher_node.remove_patcher_output(self.output_name, self.output_in)


class PatcherNode(Node):
    '''
    PatcherNode instantiates a new Node Editor
    Inside this editor window, create InputNodes and OutputNodes
    These open up (make visible) previously hidden inputs and outputs on the PatcherNode
    Node Editor must have a PatcherNode member that can be set when created this way
    This provides the communication conduit
    Things received in inputs of the PatcherNode are passed to the inputs in the attached Editor
    Things sent to an output of an attached Editor, are passed to the outputs of the PatcherNode
    - How are these saved (as groups) - patcher node must save attached editor nodes

    '''
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'PatcherNode':
        node = PatcherNode('patcher', data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.max_input_count = 20
        self.max_output_count = 20
        self.subpatcher_loaded_uuid = -1

        self.patcher_name = 'untitled'
        if len(args) > 0:
            s, t = decode_arg(args, 0)
            if t == str:
                self.patcher_name = s
        self.home_editor = self.app.get_current_editor()
        self.home_editor_index = self.app.current_node_editor
        self.patch_editor = None
        text_size = dpg.get_text_size(text=self.patcher_name)
        if text_size is None:
            text_size = [80, 14]
        self.button = self.add_property(self.patcher_name, widget_type='button', width=text_size[0] + 8, callback=self.open_patcher)
        self.show_input = [False] * self.max_input_count
        self.show_output = [False] * self.max_output_count
        self.patcher_inputs = [None] * self.max_input_count
        self.patcher_outputs = [None] * self.max_output_count
        self.input_outs = [None] * self.max_input_count
        self.output_ins = [None] * self.max_output_count
        for i in range(self.max_input_count):
            in_temp = self.add_input('in ' + str(i), callback=self.receive_)
            self.patcher_inputs[i] = in_temp
        for i in range(self.max_output_count):
            out_temp = self.add_output('out ' + str(i))
            self.patcher_outputs[i] = out_temp
        self.name_option = self.add_option('patcher name', widget_type='text_input', default_value=self.patcher_name, callback=self.name_changed)

    def set_name(self, name: str) -> None:
        self.name_option.set(name)
        self.name_changed()

    def name_changed(self) -> None:
        new_name = self.name_option()
        self.unparsed_args = [new_name]
        self.button.set_label(new_name)
        size = dpg.get_text_size(new_name, font=self.app.font_24)
        if size is not None:
            dpg.set_item_width(self.button.widget.uuid, int(size[0] * self.app.font_scale_variable() + 12))

        if self.patch_editor:
            self.patch_editor.set_name(new_name)

    def change_input_name(self, old_name: str, new_name: str) -> None:
        for in_ in self.inputs:
            if in_.get_label() is old_name:
                in_.set_label(new_name)

    def change_output_name(self, old_name: str, new_name: str) -> None:
        for out_ in self.outputs:
            if out_.get_label() == old_name:
                out_.set_label(new_name)

    def receive_(self, input: Optional['NodeInput'] = None) -> None:
        if self.active_input is not None:
            index = self.active_input.input_index
            if self.input_outs[index] is not None:
                data = self.patcher_inputs[index]()
                self.input_outs[index].send(data)

    def open_patcher(self) -> None:
        for i, editor in enumerate(self.app.node_editors):
            if editor == self.patch_editor:
                tab = self.app.tabs[i]
                self.app.current_node_editor = i
                dpg.set_value(self.app.tab_bar, tab)
                break

    def add_patcher_output(self, output_name: str, output: 'NodeOutput') -> Optional['NodeOutput']:
        for i in range(self.max_output_count):
            if self.output_ins[i] is None:
                self.output_ins[i] = output
                self.show_output[i] = True
                if output_name != '':
                    self.patcher_outputs[i].set_label(output_name)
                self.update_outputs()
                return self.patcher_outputs[i]
        return None

    def remove_patcher_output(self, output_name: str, output: 'NodeOutput') -> None:
        for i in range(self.max_output_count):
            if self.output_ins[i] == output:
                self.output_ins[i] = None
                self.show_output[i] = False
                self.patcher_outputs[i].set_label('out ' + str(i))
                self.patcher_outputs[i].remove_links()
            # must remove connections!
        self.update_outputs()

    def update_outputs(self) -> None:
        for i in range(self.max_output_count):
            if self.show_output[i]:
                dpg.show_item(self.patcher_outputs[i].uuid)
            else:
                dpg.hide_item(self.patcher_outputs[i].uuid)

    def add_patcher_input(self, input_name: str, output: 'NodeOutput') -> Optional['NodeInput']:
        for i in range(self.max_input_count):
            if self.input_outs[i] is None:
                self.input_outs[i] = output
                self.show_input[i] = True
                if input_name != '':
                    self.patcher_inputs[i].set_label(input_name)
                self.update_inputs()
                return self.patcher_inputs[i]

    def remove_patcher_input(self, input_name: str, input: 'NodeInput') -> None:
        for i in range(self.max_input_count):
            if self.input_outs[i] == input:
                self.input_outs[i] = None
                self.show_input[i] = False
                dpg.set_value(self.patcher_inputs[i].label_uuid, 'in ' + str(i))
                self.patcher_inputs[i].delete_parents()
        self.update_inputs()

    def update_inputs(self) -> None:
        for i in range(self.max_input_count):
            if self.show_input[i]:
                dpg.show_item(self.patcher_inputs[i].uuid)
            else:
                dpg.hide_item(self.patcher_inputs[i].uuid)

    def custom_create(self, from_file: bool) -> None:
        if not from_file:
            self.patch_editor = self.app.add_node_editor()
            self.patch_editor.set_name(self.patcher_name)
            self.app.set_tab_title(len(self.app.node_editors) - 1, self.patcher_name)

            self.connect()

        # note that this happens before custom load setup... so 'self.subpatcher_loaded_uuid' is not yet valid
        # self.patch_editor = self.app.find_orphaned_subpatch(self.patcher_name, self.subpatcher_loaded_uuid)
        # # note that patch_editor may not have been opened yet if opening from file
        # reattached = False
        # if self.patch_editor is None and not self.app.loading:
        #     self.patch_editor = self.app.add_node_editor(self.patcher_name)
        #     self.app.set_tab_title(len(self.app.node_editors) - 1, self.patcher_name)
        # elif self.patch_editor is not None:
        #     reattached = True
        #
        # if self.patch_editor is not None:
        #     self.reconnect(self.patch_editor, attaching=reattached)
        # else:
        #     print('patch editor not loaded yet')
        #     self.app.loaded_patcher_nodes.append(self)
        #     # patch not yet loaded so will be attached when it loads

    def connect(self, patch_editor: Optional['NodeEditor'] = None) -> None:
        if patch_editor is not None:
            self.patch_editor = patch_editor
        if self.patch_editor is not None:
            self.home_editor.add_subpatch(self.patch_editor)
            self.patch_editor.patcher_node = self
            self.patch_editor.parent_patcher = self.home_editor
            self.update_inputs()
            self.update_outputs()
            # self.loaded_uuid = -1
            self.patch_editor.reconnect_to_parent(self)

    def custom_cleanup(self) -> None:  # should delete associated patcher
       self.app.remove_node_editor(self.patch_editor)

    def save_custom(self, container: Dict[str, Any]) -> None:
        container['show inputs'] = self.show_input
        container['show outputs'] = self.show_output
        container['patcher id'] = self.patch_editor.uuid

    def load_custom(self, container: Dict[str, Any]) -> None:  # called after custom setup...
        # print('load_custom', self.patcher_name)
        if 'show inputs' in container:
            self.show_input = container['show inputs']
        if 'show outputs' in container:
            self.show_output = container['show outputs']
        if 'patcher id' in container:
            self.subpatcher_loaded_uuid = container['patcher id']

        self.patch_editor = self.app.find_orphaned_subpatch(self.patcher_name, self.subpatcher_loaded_uuid)
        # note that patch_editor may not have been opened yet if opening from file
        if self.patch_editor is None:
            # print('added node editor for', self.patcher_name)
            self.patch_editor = self.app.add_node_editor()
            self.app.set_tab_title(len(self.app.node_editors) - 1, self.patcher_name)
        elif self.patch_editor is not None:
            # print('found existing node editor for', self.patcher_name)
            self.update_inputs()
            self.update_outputs()

        if self.patch_editor is not None:
            # print('connected patcher')
            self.connect()
        else:
            # print('patch editor not loaded yet')
            self.app.loaded_patcher_nodes.append(self)
            # patch not yet loaded so will be attached when it loads

        self.app.current_node_editor = self.home_editor_index



class OriginNode(Node):
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'OriginNode':
        node = OriginNode('', data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.ref_property = self.add_property('', widget_type='button', width=1)

    def custom_create(self, from_file: bool) -> None:
        with dpg.theme() as item_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodesCol_MiniMapNodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodesCol_MiniMapBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodesCol_MiniMapNodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodesCol_MiniMapNodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)

                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 0, 0, 0), category=dpg.mvThemeCat_Nodes)

        dpg.bind_item_theme(self.ref_property.uuid, self.app.invisible_theme)
        dpg.bind_item_theme(self.uuid, self.app.invisible_theme)


class PlaceholderNameNode(Node):
    node_list: List[str] = []

    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'PlaceholderNameNode':
        node = PlaceholderNameNode('New Node', data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.filtered_list: List[str] = []
        self.name_property = self.add_property(label='##node_name', widget_type='text_input', width=180)
        if len(self.node_list) == 0:
            self.node_list = self.app.node_factory_container.get_node_list()
        self.variable_list = self.app.get_variable_list()
        self.patcher_list = self.app.patchers
        self.action_list = list(self.app.actions.keys())
        self.node_list_box = self.add_property('###options', widget_type='list_box', width=180)
        self.list_box_arrowed: bool = False
        self.current_name: str = ''
        self.enter_args = False

    def custom_create(self, from_file: bool) -> None:
        dpg.configure_item(self.node_list_box.widget.uuid, show=False)

    def calc_fuzz(self, test: str, node_name: str) -> float:
        ratio = fuzz.partial_ratio(node_name.lower(), test.lower())
        full_ratio = fuzz.ratio(node_name.lower(), test.lower())

        if ratio == 100:
            test_len = len(test)
            node_len = len(node_name)
            if test_len > node_len:
                if node_name[:node_len] != test[:node_len]:
                    ratio = (full_ratio * 2 + ratio) / 3
            else:
                if node_name[:test_len] != test[:test_len]:
                    ratio = (full_ratio * 2 + ratio) / 3
        len_ratio = len(test) / len(node_name)

        if len_ratio < 1:
            len_ratio = pow(len_ratio, 4)
        len_ratio = len_ratio * .5 + 0.5
        final_ratio = (ratio * (1 - len_ratio) + full_ratio * len_ratio)
        return final_ratio

    def fuzzy_score(self, test: str) -> None:
        scores: Dict[str, float] = {}
        for index, node_name in enumerate(self.node_list):
            final_ratio = self.calc_fuzz(test, node_name)
            scores[node_name] = final_ratio

        for index, variable_name in enumerate(self.variable_list):
            final_ratio = self.calc_fuzz(test, variable_name)
            scores[variable_name] = final_ratio

        for index, patcher_name in enumerate(self.patcher_list):
            final_ratio = self.calc_fuzz(test, patcher_name)
            scores[patcher_name] = final_ratio

        for index, action_name in enumerate(self.action_list):
            final_ratio = self.calc_fuzz(test, action_name)
            scores[action_name] = final_ratio

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.filtered_list = []
        for index, item in enumerate(sorted_scores):
            if item[1] == 100:
                self.filtered_list.append(item[0])
            elif item[1] > 20 and len(self.filtered_list) < 10:
                self.filtered_list.append(item[0])

    def increment_widget(self, widget: PropertyWidget) -> None:
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        if filter_name in self.filtered_list:
            index = self.filtered_list.index(filter_name)
            index -= 1
            if index >= 0:
                self.list_box_arrowed = True
                filter_name = self.filtered_list[index]
                self.node_list_box.set(filter_name)

    def decrement_widget(self, widget: PropertyWidget) -> None:
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        if filter_name in self.filtered_list:
            index = self.filtered_list.index(filter_name)
            index += 1
            if index < len(self.filtered_list):
                self.list_box_arrowed = True
                filter_name = self.filtered_list[index]
                self.node_list_box.set(filter_name)

    def on_edit(self, widget: PropertyWidget) -> None:
        if widget == self.name_property.widget and len(self.node_list) > 0:
            self.list_box_arrowed = False
            self.filtered_list = []
            filter_name = dpg.get_value(self.name_property.widget.uuid)
            self.current_name = filter_name

            if len(filter_name) > 0:
                dpg.configure_item(self.node_list_box.widget.uuid, show=True)
            if len(filter_name) > 0 and filter_name[-1] == ' ':
                selection = dpg.get_value(self.node_list_box.widget.uuid)
                self.current_name = selection
                self.enter_args = True
                self.execute()
            else:
                f = filter_name.lower()
                self.fuzzy_score(f)
                dpg.configure_item(self.node_list_box.widget.uuid, items=self.filtered_list)
                if len(self.filtered_list) > 0:
                    dpg.set_value(self.node_list_box.widget.uuid, self.filtered_list[0])
                    self.current_name = self.filtered_list[0]

        elif widget == self.node_list_box.widget:
            selection = dpg.get_value(self.node_list_box.widget.uuid)
            self.current_name = selection
            self.execute()

    def on_deactivate(self, widget: PropertyWidget) -> None:
        if widget == self.name_property.widget:
            if dpg.is_item_hovered(self.node_list_box.widget.uuid) or dpg.is_item_clicked(self.node_list_box.widget.uuid):
                pass
            else:
                self.execute()
        elif widget == self.node_list_box.widget:
            self.execute()

    def execute(self) -> None:
        if self.enter_args:
            self.current_name = dpg.get_value(self.node_list_box.widget.uuid)
            from dpg_system.node_editor import NodeFactory
            node_model = NodeFactory('new_node', PlaceholderArgsNode.factory, data=None)
            pos = dpg.get_item_pos(self.uuid)
            if node_model:
                new_node = Node.app.create_node_from_model(node_model, pos, args=[self.current_name])
            # else:
            #     new_node = Node.app.create_var_node_for_variable('placeholder_2', pos)
            editor = self.my_editor
            if editor is not None:
                editor.remove_node(self)
        else:
            self.current_name = dpg.get_value(self.node_list_box.widget.uuid)
            found = False
            v = None
            action = False
            if self.current_name in self.node_list:
                found = True
                node_model = Node.app.node_factory_container.locate_node_by_name(self.current_name)
                pos = dpg.get_item_pos(self.uuid)

                if node_model:
                    new_node = Node.app.create_node_from_model(node_model, pos, args=[])
                else:
                    new_node = Node.app.create_var_node_for_variable(self.current_name, pos)
                editor = self.my_editor
                if editor is not None:
                    editor.remove_node(self)

            elif self.current_name in self.variable_list:
                v = self.app.find_variable(self.current_name)
                if v is not None:
                    found = True
            elif self.current_name in self.action_list:
                v = self.app.find_action(self.current_name)
                if v is not None:
                    found = True
                    action = True
            new_node_args = None
            if found and v is not None:
                found = False
                new_node_args = []
                if not action:
                    t = type(v.value)
                    if t == int:
                        new_node_args = ['int', self.current_name]
                        found = True
                    elif t == float:
                        new_node_args = ['float', self.current_name]
                        found = True
                    elif t == str:
                        new_node_args = ['string', self.current_name]
                        found = True
                    elif t == bool:
                        new_node_args = ['toggle', self.current_name]
                        found = True
                    elif t == list:
                        new_node_args = ['message', self.current_name]
                        found = True
                else:
                    new_node_args = ['button', self.current_name]
                    found = True
            if found and new_node_args is not None:
                Node.app.create_node_by_name(new_node_args[0], self, args = new_node_args[1:])
            else:
                if self.current_name in self.patcher_list:
                    hold_node_editor_index = Node.app.current_node_editor
                    patcher_node = Node.app.create_node_by_name('patcher', self, )
                    Node.app.current_node_editor = len(Node.app.node_editors) - 1
                    Node.app.load_from_file('dpg_system/patch_library/' + self.current_name + '.json')
                    if patcher_node is not None:
                        patcher_node.set_name(self.current_name)
                    Node.app.current_node_editor = hold_node_editor_index
                    tab = self.app.tabs[Node.app.current_node_editor]
                    dpg.set_value(self.app.tab_bar, tab)
                    # Node.app.select_tab(Node.app.tabs[Node.app.current_node_editor])
            editor = self.my_editor
            if editor is not None:
                editor.remove_node(self)


class PlaceholderArgsNode(Node):
    node_list: List[str] = []

    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'PlaceholderArgsNode':
        node = PlaceholderArgsNode('New Node', data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.name = ''

        if len(args) > 0:
            self.name = any_to_string(args[0])
        self.static_name = self.add_property(label='##static_name', widget_type='label', width=180)
        self.args_property = self.add_property(label='args', widget_type='text_input', width=180)
        self.node_list = self.app.node_factory_container.get_node_list()
        self.variable_list = self.app.get_variable_list()
        self.patcher_list = self.app.patchers
        self.action_list = list(self.app.actions.keys())

    def prompt_for_args(self) -> None:
        pass

    def on_edit(self, widget: PropertyWidget) -> None:
        pass

    def custom_create(self, from_file):
        self.static_name.set(self.name)
        Node.app.set_widget_focus(self.args_property.widget.uuid)

    def on_deactivate(self, widget: PropertyWidget) -> None:
        self.execute()

    def execute(self) -> None:
        arg_string = dpg.get_value(self.args_property.widget.uuid)
        new_node_args: List[str] = []
        selection_name = self.name
        if len(arg_string) > 0:
            args = arg_string.split(' ')
            new_node_args = [selection_name] + args
        else:
            new_node_args = [selection_name]

        node_model = None
        found = False
        v = None
        action = False
        if new_node_args[0] in self.node_list:
            found = True

        if found:
            if len(new_node_args) > 1:
                # print(new_node_args)
                Node.app.create_node_by_name(new_node_args[0], self, new_node_args[1:])
            else:
                Node.app.create_node_by_name(new_node_args[0], self, )
            return
        elif new_node_args[0] in self.variable_list:
            v = self.app.find_variable(new_node_args[0])
            if v is not None:
                found = True
        elif selection_name in self.variable_list:
            new_node_args[0] = selection_name
            v = self.app.find_variable(new_node_args[0])
            if v is not None:
                found = True
        elif selection_name in self.action_list:
            new_node_args[0] = selection_name
            v = self.app.find_action(new_node_args[0])
            if v is not None:
                found = True
                action = True
        if found:
            additional = []
            if len(new_node_args) > 1:
                additional = new_node_args[1:]
            found = False

            if not action:
                t = type(v.value)
                if t == int:
                    new_node_args = ['int', new_node_args[0]]
                    found = True
                elif t == float:
                    new_node_args = ['float', new_node_args[0]]
                    found = True
                elif t == str:
                    new_node_args = ['string', new_node_args[0]]
                    found = True
                elif t == bool:
                    new_node_args = ['toggle', new_node_args[0]]
                    found = True
                elif t == list:
                    new_node_args = ['message', new_node_args[0]]
                    found = True
            else:
                new_node_args = ['button', new_node_args[0]]
                found = True
            if found:
                if len(additional) > 0:
                    new_node_args += additional
                if len(new_node_args) > 1:
                    Node.app.create_node_by_name(new_node_args[0], self, new_node_args[1:])
                else:
                    Node.app.create_node_by_name(new_node_args[0], self, )


def dialog_cancel_callback(sender, app_data):
    try:
        if sender is not None:
            dpg.delete_item(sender)
    except Exception as e:
        print(e)
    Node.app.active_widget = -1

class LoadDialog:
    def __init__(self, parent, callback, extensions, default_path=''):
        Node.app.active_widget = 1
        self.callback = callback
        self.parent = parent
        self.load_take_task = None
        with dpg.file_dialog(modal=True, default_path=default_path, directory_selector=False, show=True, height=400, width=800,
                             callback=self.load_callback, cancel_callback=dialog_cancel_callback,
                             tag='load_dialog') as self.save_take_task:
            for extension in extensions:
                dpg.add_file_extension(extension)

    def load_callback(self, sender, app_data):
        try:
            if app_data is not None and 'file_path_name' in app_data:
                load_path = app_data['file_path_name']
                self.callback(load_path)
            else:
                print('no file chosen')
            if sender is not None:
                dpg.delete_item(sender)
        except Exception as e:
            print(e)
        Node.app.active_widget = -1


class SaveDialog:
    def __init__(self, parent, callback, extensions, default_path=''):
        Node.app.active_widget = 1
        self.callback = callback
        self.parent = parent
        self.save_take_task = None
        with dpg.file_dialog(modal=True, default_path=default_path, directory_selector=False, show=True, height=400, width=800,
                             callback=self.save_callback, cancel_callback=dialog_cancel_callback,
                             tag='save_dialog') as self.save_take_task:
            for extension in extensions:
                dpg.add_file_extension(extension)

    def save_callback(self, sender, app_data):
        try:
            if app_data is not None and 'file_path_name' in app_data:
                save_path = app_data['file_path_name']
                self.callback(save_path)
            else:
                print('no file chosen')
            if sender is not None:
                dpg.delete_item(sender)
        except Exception as e:
            print(e)
        Node.app.active_widget = -1

