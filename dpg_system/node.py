import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.conversion_utils import *
import json
from typing import List, Any, Callable, Union, Tuple
from fuzzywuzzy import fuzz


class OutputNodeAttribute:
    _pin_active_theme = None
    _pin_active_and_connected_theme = None
    _pin_theme_created = False

    def __init__(self, label: str = "output", node=None, pos=None):
        if not self._pin_theme_created:
            self.create_pin_themes()
        self.uuid = -1
        self._label = label
        self._children = []  # output attributes
        self.links = []
        self.pos = pos
        self.node = node
        self.output_always = True
        self.new_output = False
        self.loaded_uuid = -1
        self.loaded_children = []

    def create_pin_themes(self):
        with dpg.theme() as self._pin_active_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (51, 170, 255), category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self._pin_active_and_connected_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (153, 212, 255), category=dpg.mvThemeCat_Nodes)

    def add_child(self, child, parent):
        link_uuid = dpg.add_node_link(self.uuid, child.uuid, parent=parent)
        # print('added link', link_uuid, self.uuid, child.uuid, self.node.label)
        child.set_parent(self)
        self._children.append(child)
        self.links.append(link_uuid)
        dpg.set_item_user_data(link_uuid, [self, child])

    def remove_link(self, link, child):
        # print('remove_link')
        if link in self.links:
            self.links.remove(link)
        if child in self._children:
            self._children.remove(child)
        child.remove_parent(self)
        dpg.delete_item(link)

    def remove_child(self, child):
        for kid in self._children:
            if kid == child:
                # print('outlet remove child', kid.uuid, kid._label)
                self._children.remove(kid)
                break

    def set_value(self, data):
        self.new_output = True
        for child in self._children:
            child.receive_data(data)

    def send(self, data=None):  # called every time
        if data is not None:
            self.set_value(data)
        if self.output_always or self.new_output:
            if len(self._children) > 0:
                dpg.bind_item_theme(self.uuid, self._pin_active_and_connected_theme)
            else:
                dpg.bind_item_theme(self.uuid, self._pin_active_theme)
            Node.app.node_editors[Node.app.current_node_editor].add_active_pin(self.uuid)
            for child in self._children:
                child.trigger()
        self.new_output = False

    def submit(self, parent):
        if self.pos is not None:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self, pos=self.pos) as self.uuid:
                dpg.add_text(self._label)
        else:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self) as self.uuid:
                dpg.add_text(self._label)

    def get_children(self):
        return self._children

    def save(self, output_container):
        output_container['name'] = self._label
        output_container['id'] = self.uuid
        children = {}
        for index, child in enumerate(self._children):
            children[index] = child.uuid
        output_container['children'] = children

    def load(self, output_container):
        self.loaded_uuid = output_container['id']
        self.loaded_children = []
        kids = output_container['children']
        for i in kids:
            self.loaded_children.append(kids[i])


def value_trigger_callback(s, a, u):
    if u is not None:
        u.execute()


class DisplayNodeAttribute:
    def __init__(self, label: str = "", uuid=None, node=None, width=80):
        self.uuid = dpg.generate_uuid()
        self.callback = None
        self.user_data = None
        self.node = node
        self.submit_callback = None

    def submit(self, parent):
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Static, user_data=self, id=self.uuid)
        with self.node_attribute:
            if self.submit_callback is not None:
                self.submit_callback()
            if self.callback is not None:
                dpg.set_item_callback(self.widget.uuid, self.callback)
                dpg.set_item_user_data(self.widget.uuid, self.user_data)

    def load(self, property_container):
        pass

    def save(self, property_container):
        pass

    def add_callback(self, callback, user_data=None):
        if user_data is None:
            self.user_data = self
        else:
            self.user_data = user_data
        self.callback = callback

    def get_widget_value(self):
        return 0

    def set(self, data):
        pass


class PropertyNodeAttribute:
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        self.label = label
        if uuid == None:
            self.uuid = dpg.generate_uuid()
        self.widget = PropertyWidget(label, uuid, node=node, widget_type=widget_type, width=width, triggers_execution=triggers_execution, trigger_button=trigger_button, default_value=default_value, min=min, max=max)
        self.callback = None
        self.user_data = None
        self.node_attribute = None
        self.variable = None
        self.node = node

    def submit(self, parent):
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Static, user_data=self, id=self.uuid)
        with self.node_attribute:
            if self.callback is not None:
                self.widget.callback = self.callback
                self.widget.user_data = self.user_data
            self.widget.submit()

    # def set_node(self, node):
    #     self.node = node
    #     if self.widget:
    #         self.widget.node = node
    def set_default_value(self, data):
        self.widget.set_default_value(data)

    def attach_to_variable(self, variable):
        self.variable = variable
        self.variable.property = self
        self.widget.attach_to_variable(variable)

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

    def get_widget_value(self):
        return self.widget.value

    def set(self, data, propagate=True):
        if type(data) == list:
            if len(data) == 1:
                data = data[0]
            else:
                if self.widget.widget in ['text_input', 'combo']:
                    data = any_to_string(data)
                else:
                    data = data[0]
        self.widget.set(data, propagate)
        if self.variable and propagate:
            self.variable.set_value(data)  # will propagate to all instances


class PropertyWidget:
    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        self._label = label
        if uuid is None:
            self.uuid = dpg.generate_uuid()
        else:
            self.uuid = uuid
        self.widget_has_trigger = trigger_button
        self.widget = widget_type
        self.widget_width = width
        self.triggers_execution = triggers_execution
        self.default_value = default_value
        self.value = None
        self.min = min
        self.max = max
        self.combo_items = []
        self.callback = None
        self.user_data = self
        self.tag = self.uuid
        if widget_type == 'drag_float':
            self.speed = 0.01
        else:
            self.speed = 1
        if widget_type == 'input_float':
            self.step = .1
        else:
            self.step = 1
        self.variable = None
        self.node = node

    def submit(self):
        if self.widget in ['drag_float', 'slider_float', 'input_float', 'knob_float']:
            if self.default_value is None:
                self.default_value = 0.0
            if type(self.default_value) is not float:
                if type(self.default_value) is not int:
                    self.default_value = 0.0
            self.value = self.default_value
        elif self.widget in ['drag_int', 'slider_int', 'input_int', 'knob_int']:
            if self.default_value is None:
                self.default_value = 0
            if type(self.default_value) is not int:
                if type(self.default_value) is not float:
                    self.default_value = 0
            self.value = self.default_value
        elif self.widget in ['text_input', 'combo']:
            if self.default_value is None:
                self.default_value = ''
            self.value = self.default_value
        elif self.widget == 'checkbox':
            if self.default_value is None:
                self.default_value = False
            if type(self.default_value) is not bool:
                    self.default_value = False
            self.value = self.default_value

        with dpg.group(horizontal=self.widget_has_trigger):
            if self.widget == 'drag_float':
                if self.min is not None:
                    min = self.min
                else:
                    min = -math.inf
                if self.max is not None:
                    max = self.max
                else:
                    max = math.inf
                dpg.add_drag_float(width=self.widget_width, clamped=True, label=self._label, tag=self.uuid, max_value=max, min_value=min, user_data=self.node, default_value=self.default_value, speed=self.speed)
            elif self.widget == 'drag_int':
                if self.min is not None:
                    min = self.min
                else:
                    min = -math.inf
                if self.max is not None:
                    max = self.max
                else:
                    max = math.inf
                dpg.add_drag_int(label=self._label, width=self.widget_width, tag=self.uuid, max_value=max, min_value=min, user_data=self.node, default_value=self.default_value)
            elif self.widget == 'slider_float':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_slider_float(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, min_value=min, max_value=max)
            elif self.widget == 'slider_int':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_slider_int(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, min_value=min, max_value=max)
            elif self.widget == 'knob_float':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_knob_float(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, min_value=min, max_value=max)
            elif self.widget == 'knob_int':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_knob_int(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, min_value=min, max_value=max)
            elif self.widget == 'input_float':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_input_float(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, step=self.step, min_value=min, max_value=max)
            elif self.widget == 'input_int':
                if self.min is not None:
                    min = self.min
                else:
                    min = 0
                if self.max is not None:
                    max = self.max
                else:
                    max = 100
                dpg.add_input_int(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, step=self.step, min_value=min, max_value=max)
            elif self.widget == 'checkbox':
                dpg.add_checkbox(label=self._label, tag=self.uuid, default_value=self.default_value, user_data=self)
                dpg.set_item_user_data(self.uuid, user_data=self)
                dpg.set_item_callback(self.uuid, callback=lambda: self.clickable_changed())
            elif self.widget == 'button':
                button = dpg.add_button(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node)
                with dpg.theme() as item_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
                dpg.bind_item_theme(button, item_theme)
                dpg.set_item_callback(self.uuid, callback=lambda: self.clickable_changed())
            elif self.widget == 'text_input':
                dpg.add_input_text(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value, on_enter=True)
            elif self.widget == 'combo':
                dpg.add_combo(self.combo_items, label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, default_value=self.default_value)
                dpg.set_item_callback(self.uuid, callback=lambda: self.clickable_changed())
            elif self.widget == 'color_picker':
                dpg.add_color_picker(label='color', width=self.widget_width, display_type=dpg.mvColorEdit_uint8, tag=self.uuid, picker_mode=dpg.mvColorPicker_wheel, no_side_preview=False, no_alpha=False, alpha_bar=True, alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf, user_data=self.node, no_inputs=True, default_value=self.default_value)
                dpg.set_item_callback(self.uuid, callback=lambda: self.clickable_changed())
            elif self.widget == 'list_box':
                dpg.add_listbox(label=self._label, width=self.widget_width, tag=self.uuid, user_data=self.node, num_items=8)
            if self.widget not in ['checkbox', 'button', 'combo']:
                dpg.set_item_user_data(self.uuid, user_data=self)
                dpg.set_item_callback(self.uuid, callback=lambda s, a, u: self.value_changed(a))
            if self.widget_has_trigger:
                button = dpg.add_button(label='', width=14, callback=value_trigger_callback, user_data=self.node)
                with dpg.theme() as item_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
                dpg.bind_item_theme(button, item_theme)
        return

    def set_default_value(self, data):
        if self.widget in ['drag_float', 'slider_float', 'knob_float', 'input_float']:
            self.default_value = any_to_float(data)
        elif self.widget in ['drag_int', 'slider_int', 'knob_int', 'input_int']:
            self.default_value = any_to_int(data)
        elif self.widget == 'checkbox':
            self.default_value = any_to_bool(data)
        elif self.widget in ['text_input', 'combo', 'list_box']:
            self.default_value = any_to_string(data)
        elif self.widget == 'color_picker':
            self.default_value = tuple(any_to_list(data))

    def set_limits(self, min_, max_):
        self.min = min_
        self.max = max_
        dpg.configure_item(self.uuid, min_value=self.min, max_value=self.max)

    def set_format(self, format):
        dpg.configure_item(self.uuid, format=format)

    def attach_to_variable(self, variable):
        self.variable = variable

    def value_changed(self, uuid=-1, force=False):
        if not dpg.is_mouse_button_down(0) and not force:
            return
        self.value = dpg.get_value(self.uuid)
        if self.variable:
            self.variable.set_value(self.value)
        if self.callback is not None:
            self.callback()
        if self.triggers_execution:
            self.node.execute()

    def clickable_changed(self):
        self.value = dpg.get_value(self.uuid)
        if self.variable:
            self.variable.set_value(self.value)
        if self.callback is not None:
            self.callback()
        if self.triggers_execution:
            self.node.execute()

    def increment(self):
        if self.widget == 'checkbox':
            val = dpg.get_value(self.uuid)
            val = not val
            self.set(val)
        elif self.widget in ['drag_int', 'slider_int', 'input_int', 'knob_int']:
            val = dpg.get_value(self.uuid)
            self.set(val + self.speed)
        elif self.widget in ['drag_float', 'slider_float', 'input_float', 'knob_float']:
            val = dpg.get_value(self.uuid)
            self.set(val + self.speed)
        elif self.widget == 'combo':
            val = dpg.get_value(self.uuid)
            index = self.combo_items.index(val)
            index += 1
            if index < len(self.combo_items):
                val = self.combo_items[index]
                self.set(val)
        else:
            self.node.increment_widget(self.widget)
        if self.callback is not None:
            self.callback()

    def decrement(self):
        if self.widget == 'checkbox':
            val = dpg.get_value(self.uuid)
            val = not val
            self.set(val)
        elif self.widget in ['drag_int', 'slider_int', 'input_int', 'knob_int']:
            val = dpg.get_value(self.uuid)
            self.set(val - self.speed)
        elif self.widget in ['drag_float', 'slider_float', 'input_float', 'knob_float']:
            val = dpg.get_value(self.uuid)
            self.set(val - self.speed)
        elif self.widget == 'combo':
            val = dpg.get_value(self.uuid)
            index = self.combo_items.index(val)
            index -= 1
            if index >= 0:
                val = self.combo_items[index]
                self.set(val)
        else:
            self.node.decrement_widget(self.widget)
        if self.callback is not None:
            self.callback()

    def set(self, data, propagate=True):
        if self.widget == 'checkbox':
            val = any_to_bool(data)
            dpg.set_value(self.uuid, val)
            self.value = val
        elif self.widget == 'combo':
            val = any_to_string(data)
            dpg.set_value(self.uuid, val)
            self.value = val
        elif self.widget == 'list_box':
            val = any_to_string(data)
            dpg.set_value(self.uuid, val)
            self.value = val
        elif self.widget == 'text_input':
            val = any_to_string(data)
            dpg.set_value(self.uuid, val)
            self.value = val
            size = dpg.get_text_size(self.value, font=dpg.get_item_font(self.uuid))
            width = size[0]
            if width > 1024:
                width = 1024
            if width > dpg.get_item_width(self.uuid):
                dpg.set_item_width(self.uuid, width)
        elif self.widget in ['drag_int', 'input_int', 'slider_int', 'knob_int']:
            val = any_to_int(data)
            if val:
                if self.min != self.max:
                    if self.max and val > self.max:
                        val = self.max
                    if self.min and val < self.min:
                        val = self.min
            dpg.set_value(self.uuid, val)
            self.value = val
        elif self.widget in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
            val = any_to_float(data)
            if val:
                if self.min != self.max:
                    if self.max and val > self.max:
                        val = self.max
                    if self.min and val < self.min:
                        val = self.min
            dpg.set_value(self.uuid, val)
            self.value = val
        elif self.widget == 'color_picker':
            if type(data) != tuple:
                val = tuple(any_to_list(data))
            else:
                val = data
            dpg.set_value(self.uuid, val)
            self.value = val
        if self.variable and propagate:
            self.variable.set_value(self.value)

    def load(self, widget_container):
        if 'value' in widget_container:
            val = widget_container['value']
            self.set(val)

    def save(self, widget_container):
        widget_container['name'] = self._label
        value = dpg.get_value(self.uuid)
        widget_container['value'] = value
        value_type = type(value).__name__
        if value_type == 'str':
            widget_container['value_type'] = 'string'
        else:
            widget_container['value_type'] = value_type

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


class InputNodeAttribute:
    _pin_active_theme = None
    _pin_theme_created = False

    def __init__(self, label: str = "", uuid=None, node=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None):
        if not self._pin_theme_created:
            self.create_pin_themes()
        self._label = label
        if uuid is None:
            self.uuid = dpg.generate_uuid()
        else:
            self.uuid = uuid
        self._parents = []  # input attribute
        self._data = 0
        self.executor = False
        self.triggers_execution = triggers_execution
        self.node = node
        self.fresh_input = False
        self.node_attribute = None
        self.bang_repeats_previous = True
        self.widget = None
        self.widget_has_trigger = trigger_button
        if widget_type:
            self.widget = PropertyWidget(label, uuid=widget_uuid, node=node, widget_type=widget_type, width=widget_width, triggers_execution=triggers_execution, trigger_button=trigger_button, default_value=default_value, min=min, max=max)
        self.callback = None
        self.user_data = None
        self.variable = None

    def create_pin_themes(self):
        self._pin_inactive_theme = dpg.theme()
        with dpg.theme() as self._pin_active_theme:
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvNodeCol_Pin, (153, 212, 255), category=dpg.mvThemeCat_Nodes)

    def submit(self, parent):
        self.node_attribute = dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Input, user_data=self, id=self.uuid)
        with self.node_attribute:
            if self.widget is None:
                if self._label != "":
                    dpg.add_text(self._label)
            else:
                if self.callback:
                    self.widget.callback = self.callback
                    self.widget.user_data = self.user_data
                self.widget.submit()

    def set_default_value(self, data):
        if self.widget is not None:
            self.widget.set_default_value(data)

    def add_callback(self, callback, user_data=None):
        self.callback = callback
        self.user_data = user_data

    def get_received_data(self):
        self.fresh_input = False
        return self._data

    def get_data(self):
        if self.widget:
            return self.get_widget_value()
        return self._data

    def attach_to_variable(self, variable):
        self.variable = variable
        self.variable.property = self
        self.widget.attach_to_variable(variable)

    def delete_parents(self):
        # print(self._parents)
        for p in self._parents:  # output linking to this
            p.remove_child(self)
            for l in p.links:
                if dpg.does_item_exist(l):
                    d = dpg.get_item_user_data(l)
                    input_ = d[1]
                    if input_.uuid == self.uuid:
                        # print('deleting link', l)
                        dpg.delete_item(l)

            # find link in p.links whose user_data links it to me
            # print(self._parents)
        self._parents = []

    def remove_parent(self, parent):
        self._parents.remove(parent)

    def receive_data(self, data):
        if not self.node.check_for_messages(data):
            if type(data) == str and data == 'bang':
                if self.widget:
                    if self.widget.widget != 'text_input':
                        data = self.get_widget_value()
                elif self.bang_repeats_previous:
                    data = self._data
            self._data = data
            self.fresh_input = True
            dpg.bind_item_theme(self.uuid, self._pin_active_theme)
            Node.app.node_editors[Node.app.current_node_editor].add_active_pin(self.uuid)
            if self.widget:
                self.widget.set(data)
                if self.callback:
                    self.callback()

    def trigger(self):
        if self.triggers_execution:
            self.node.execute()

    def set_parent(self, parent: OutputNodeAttribute):
        if parent not in self._parents:
            self._parents.append(parent)

    def save(self, input_container):
        if self.widget:
            self.widget.save(input_container)
            return True
        return False

    def load(self, input_container):
        if self.widget:
            self.widget.load(input_container)

    def get_widget_value(self):
        if self.widget:
            return self.widget.value

    def set(self, data, propagate=True):
        if type(data) == list:
            if len(data) == 1:
                data = data[0]
            else:
                if self.widget.widget in ['text_input', 'combo']:
                    data = any_to_string(data)
                else:
                    data = data[0]
        if self.widget:
            self.widget.set(data, False)
        if self.variable and propagate:
            self.variable.set_value(data)

    @property
    def data(self):
        return self._data


class Variable:
    def __init__(self, label: str, default_value=0.0, setter=None, getter=None):
        self.label = label
        self.property = None
        self.clients = []
        self.value = default_value
        self.set_callback = setter
        self.get_callback = getter

    def notify_clients_of_value_change(self, from_client=None):
        for client in self.clients:
            if client != from_client:
                client.variable_update()

    def set(self, data, from_client=None):
        if self.property and from_client != self.property.node:
            self.property.set(data, propagate=False)
        self.set_value(data)
        self.notify_clients_of_value_change(from_client)

    def get(self):
        if self.property:
            self.value = self.property.get_widget_value()
        else:
            self.value = self.get_value()
        return self.value

    def attach_client(self, client):
        if client not in self.clients:
            self.clients.append(client)

    def detach_client(self, client):
        if client in self.clients:
            if self.property is not None and self.property.node == client:
                self.property = None
            self.clients.remove(client)

    def set_value(self, data):  # does not notify_clients_of_value_change
        if self.set_callback:
            self.set_callback(data)
        else:
            self.value = data

    def get_value(self):
        if self.get_callback:
            return self.get_callback()
        return self.value


class Node:
    app = None

    def __init__(self, label: str, data, args=None):
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

    def custom_cleanup(self):
        pass

    def cleanup(self):
        self.remove_frame_tasks()
        self.custom_cleanup()
        for input_ in self.inputs:
            input_.delete_parents()

    def send_all(self):
        for output in self.outputs:
            output.send()  # should not always trigger!!! make flag to indicate trigger always or trigger on change...

    def add_property(self, label: str = "", uuid=None, widget_type=None, width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None, callback=None):
        new_property = PropertyNodeAttribute(label, uuid, self, widget_type, width, triggers_execution, trigger_button, default_value, min, max)
        self.properties.append(new_property)
        self.ordered_elements.append(new_property)
        if callback is not None:
            new_property.add_callback(callback=callback, user_data=self)
        self.property_registery[label] = new_property
        self.message_handlers[label] = self.property_message
        return new_property

    def add_option(self, label: str = "", uuid=None, widget_type=None, width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None, callback=None):
        new_option = PropertyNodeAttribute(label, uuid, self, widget_type, width, triggers_execution, trigger_button, default_value, min, max)
        self.options.append(new_option)
        self.ordered_elements.append(new_option)
        if callback is not None:
            new_option.add_callback(callback=callback, user_data=self)
        self.property_registery[label] = new_option
        self.message_handlers[label] = self.property_message
        return new_option

    def add_display(self, label: str = "", uuid=None, width=80, callback=None):
        new_display = DisplayNodeAttribute(label, uuid, self, width)
        self.displays.append(new_display)
        self.ordered_elements.append(new_display)
        # new_display.node = self
        if callback is not None:
            new_display.add_callback(callback=callback, user_data=self)
        return new_display

    def add_input(self, label: str = "", uuid=None, widget_type=None, widget_uuid=None, widget_width=80, triggers_execution=False, trigger_button=False, default_value=None, min=None, max=None, callback=None):
        new_input = InputNodeAttribute(label, uuid, self, widget_type, widget_uuid, widget_width, triggers_execution, trigger_button, default_value, min, max)
        self.inputs.append(new_input)
        self.ordered_elements.append(new_input)
        if callback is not None:
            new_input.add_callback(callback=callback, user_data=self)
        if widget_type is not None:
            self.property_registery[label] = new_input
            self.message_handlers[label] = self.property_message
        return new_input

    def add_output(self, label: str = "output", pos=None):
        new_output = OutputNodeAttribute(label, self, pos)
        self.outputs.append(new_output)
        self.ordered_elements.append(new_output)
        return new_output

    def add_handler_to_widgets(self):
        for input_ in self.inputs:
            if input_.widget and input_.widget.widget not in['button', 'combo', 'knob_float', 'knob_int']:
                dpg.bind_item_handler_registry(input_.widget.uuid, "widget handler")
        for property_ in self.properties:
            if property_.widget.widget != 'button':
                dpg.bind_item_handler_registry(property_.widget.uuid, "widget handler")
        for option in self.options:
            dpg.bind_item_handler_registry(option.widget.uuid, "widget handler")

    def value_changed(self, widget_uuid, force=False):
        pass

    def variable_update(self):
        self.execute()

    def execute(self):
        for attribute in self.outputs:
            attribute.set_value(self._data)
        self.send_all()

    def custom_setup(self):
        pass

    def trigger(self):
        pass

    def increment_widget(self, widget):
        widget.increment()

    def decrement_widget(self, widget):
        widget.decrement()

    def add_frame_task(self):
        Node.app.add_frame_task(self)
        self.has_frame_task = True

    def remove_frame_tasks(self):
        if self.has_frame_task:
            Node.app.remove_frame_task(self)
            self.has_frame_task = False

    def submit(self, parent, pos):
        with dpg.node(parent=parent, label=self.label, tag=self.uuid, pos=pos):
            dpg.set_item_pos(self.uuid, pos)
            self.handle_parsed_args()
            if len(self.ordered_elements) > 0:
                for attribute in self.ordered_elements:
                    attribute.submit(self.uuid)
                self.custom_setup()
                self.update_parsed_args()

        dpg.set_item_user_data(self.uuid, self)
        self.add_handler_to_widgets()
        for option_att in self.options:
            dpg.hide_item(option_att.uuid)
            dpg.hide_item(option_att.widget.uuid)
        self.created = True

    def args_as_list(self):
        return self.ordered_args

    def arg_as_number(self, default_value=0, index=0):
        value = default_value
        if len(self.ordered_args) > index:
            val, t = decode_arg(self.ordered_args, index)
            if t == float:
                value = val
            else:
                value = any_to_int(val)
        return value

    def arg_as_int(self, default_value=0.0, index=0):
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_int(self.ordered_args[index])
        return value

    def arg_as_float(self, default_value=0.0, index=0):
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_float(self.ordered_args[index])
        return value

    def arg_as_string(self, default_value='', index=0):
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_string(self.ordered_args[index])
        return value

    def arg_as_bool(self, default_value=False, index=0):
        value = default_value
        if len(self.ordered_args) > index:
            value = any_to_bool(self.ordered_args[index])
        return value

    def toggle_show_hide_options(self):
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

    def check_for_messages(self, in_data):
        handled = False
        if len(self.message_handlers) > 0:
            message = ''
            message_data = []
            t = type(in_data)
            if t == str:
                message = in_data
            elif t == list:
                list_len = len(in_data)
                if list_len > 0:
                    if type(in_data[0]) == str:
                        message = in_data[0]
                        message_data = in_data[1:]
            if message != '':
                if message in self.message_handlers:
                    self.message_handlers[message](message, message_data)
                    handled = True
                else:  # maybe two words in message header
                    message = ' '.join(message.split('_'))
                    if message in self.message_handlers:
                        self.message_handlers[message](message, message_data)
                        handled = True
        return handled

    def save_custom_setup(self, container):
        pass

    def load_custom_setup(self, container):
        pass

    def on_edit(self, widget):
        pass

    def on_deactivate(self, widget):
        pass

    def property_message(self, message='', args=[]):
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

    def save(self, node_container, index):
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
            self.save_custom_setup(node_container)

    def load(self, node_container, offset=None):
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
            # if 'width' in node_container and 'height' in node_container:
            #     size = [0, 0]
            #     size[0] = node_container['width']
            #     size[1] = node_container['height']
            #     dpg.set_item_width(self.uuid, size[0])
            #     dpg.set_item_height(self.uuid, size[1])
            if 'properties' in node_container:
                properties_container = node_container['properties']
                for index, property_index in enumerate(properties_container):
                    property_container = properties_container[property_index]
                    if 'name' in property_container:
                        property_label = property_container['name']
                        found = False
                        for input in self.inputs:
                            if input.widget:
                                a_label = dpg.get_item_label(input.widget.uuid)
                                if a_label == property_label:
                                    if 'value' in property_container:
                                        value = property_container['value']
                                        input.widget.set(value)
                                        input.widget.value_changed(force=True)
                                    found = True
                                    break
                        if not found:
                            for property in self.properties:
                                if property.widget:
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
                                            option.widget.set(value)
                                            option.widget.value_changed(force=True)
                                        found = True
                                        break
            self.load_custom_setup(node_container)
        self.update_parameters_from_widgets()

    def update_parameters_from_widgets(self):
        pass

    def set_value(self, uuid, type, input):
        if type in ['drag_float', 'input_float', 'slider_float', 'knob_float']:
            dpg.set_value(uuid, any_to_float(input))
        elif type in ['drag_int', 'input_int', 'slider_int', 'knob_int']:
            dpg.set_value(uuid, any_to_int(input))
        elif type in ['input_text']:
            dpg.set_value(uuid, any_to_string(input))
        elif type in ['toggle']:
            dpg.set_value(uuid, any_to_bool(input))

    def parse_args(self):
        self.ordered_args = []
        self.parsed_args = {}
        if self.unparsed_args is not None:
            for arg in self.unparsed_args:
                handled = False
                if '=' in arg:
                    print('found', arg)
                    arg_parts = arg.split('=')
                    if len(arg_parts) == 2:
                        print(arg_parts)
                        if arg_parts[0][-1] == ' ':
                            arg_parts[0] = arg_parts[0][:-1]
                        if arg_parts[1][0] == ' ':
                            arg_parts[1] = arg_parts[1][1:]
                        self.parsed_args[arg_parts[0]] = arg_parts[1]
                        handled = True
                if not handled:
                    self.ordered_args.append(arg)
        print(self.parsed_args, self.ordered_args)

    def handle_parsed_args(self):
        for arg_name in self.parsed_args:
            if arg_name in self.property_registery:
                property = self.property_registery[arg_name]
                property.set_default_value(self.parsed_args[arg_name])

    def update_parsed_args(self):
        for arg_name in self.parsed_args:
            if arg_name in self.property_registery:
                property = self.property_registery[arg_name]
                if property.widget is not None:
                    if property.widget.callback is not None:
                        property.widget.callback()


class PlaceholderNode(Node):
    node_list = []

    @staticmethod
    def factory(name, data, args=None):
        node = PlaceholderNode('New Node', data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.filtered_list = []
        self.name_property = self.add_property(label='##node_name', widget_type='text_input', width=180)
        self.static_name = self.add_property(label='##static_name', widget_type='text_input', width=180)
        self.args_property = self.add_property(label='args', widget_type='text_input', width=180, callback=self.execute)
        if len(self.node_list) == 0:
            self.node_list = self.app.node_factory_container.get_node_list()
        self.variable_list = self.app.get_variable_list()

        self.node_list_box = self.add_property('###options', widget_type='list_box', width=180)

    def custom_setup(self):
        dpg.configure_item(self.args_property.widget.uuid, show=False, on_enter=True)
        dpg.configure_item(self.static_name.widget.uuid, show=False)
        dpg.configure_item(self.node_list_box.widget.uuid, show=False)

    def fuzzy_score(self, test):
        scores = {}
        for index, node_name in enumerate(self.node_list):
            ratio = fuzz.partial_ratio(node_name.lower(), test.lower())
            if ratio == 100:
                len_diff = abs(len(test.lower()) - len(node_name.lower()))
                full_ratio = fuzz.ratio(node_name.lower(), test.lower())
                ratio = (ratio * len_diff + full_ratio) / (1 + len_diff)
                if test.lower() == node_name.lower()[:len(test)]:
                    ratio += 10
            scores[node_name] = ratio
        for index, variable_name in enumerate(self.variable_list):
            ratio = fuzz.partial_ratio(variable_name.lower(), test.lower())
            if ratio == 100:
                len_diff = abs(len(test.lower()) - len(variable_name.lower()))
                full_ratio = fuzz.ratio(variable_name.lower(), test.lower())
                ratio = (ratio * len_diff + full_ratio) / (1 + len_diff)
                if test.lower() == variable_name.lower()[:len(test)]:
                    ratio += 10
            scores[variable_name] = ratio
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.filtered_list = []
        for index, item in enumerate(sorted_scores):
            if item[1] == 100:
                self.filtered_list.append(item[0])
            elif item[1] > 20 and len(self.filtered_list) < 10:
                self.filtered_list.append(item[0])

    def increment_widget(self, widget):
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        # print(filter_name)
        if filter_name in self.filtered_list:
            # print('name in list')
            index = self.filtered_list.index(filter_name)
            # print(index)
            index -= 1
            # print(index)
            if index >= 0:
                # print('ok index')
                filter_name = self.filtered_list[index]
                # print(filter_name)
                self.node_list_box.set(filter_name)

    def decrement_widget(self, widget):
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        # print(filter_name)
        if filter_name in self.filtered_list:
            # print('name in list')
            index = self.filtered_list.index(filter_name)
            # print(index)
            index += 1
            # print(index)
            if index < len(self.filtered_list):
                # print('ok index')
                filter_name = self.filtered_list[index]
                # print(filter_name)
                self.node_list_box.set(filter_name)

    def on_edit(self, widget):
        if widget == self.static_name:
            return
        if widget == self.name_property.widget and len(self.node_list) > 0:
            self.filtered_list = []
            filter_name = dpg.get_value(self.name_property.widget.uuid)
            if len(filter_name) > 0:
                dpg.configure_item(self.node_list_box.widget.uuid, show=True)
            if len(filter_name) > 0 and filter_name[-1] == ' ':
                selection = dpg.get_value(self.node_list_box.widget.uuid)
                dpg.focus_item(self.node_list_box.widget.uuid)
                dpg.configure_item(self.name_property.widget.uuid, enabled=False)
                dpg.configure_item(self.node_list_box.widget.uuid, items=[], show=False)
                dpg.configure_item(self.name_property.widget.uuid, show=False)
                dpg.configure_item(self.static_name.widget.uuid, show=True)
                dpg.configure_item(self.args_property.widget.uuid, show=True, on_enter=True)
                self.static_name.set(selection)
                dpg.focus_item(self.args_property.widget.uuid)
            else:
                f = filter_name.lower()
                self.fuzzy_score(f)
                dpg.configure_item(self.node_list_box.widget.uuid, items=self.filtered_list)
                if len(self.filtered_list) > 0:
                    dpg.set_value(self.node_list_box.widget.uuid, self.filtered_list[0])

        elif widget == self.node_list_box.widget:
            selection = dpg.get_value(self.node_list_box.widget.uuid)
            dpg.focus_item(self.node_list_box.widget.uuid)
            dpg.configure_item(self.name_property.widget.uuid, enabled=False)
            dpg.configure_item(self.node_list_box.widget.uuid, items=[], show=False)
            dpg.configure_item(self.name_property.widget.uuid, show=False)
            dpg.configure_item(self.static_name.widget.uuid, show=True)
            dpg.configure_item(self.args_property.widget.uuid, show=True, on_enter=True)
            self.static_name.set(selection)
            dpg.focus_item(self.args_property.widget.uuid)

    def on_deactivate(self, widget):
        if widget == self.args_property.widget:
            self.execute()

    def execute(self):
        if dpg.is_item_active(self.name_property.widget.uuid):
            print(self.name_property.get_widget_value())
        else:
            selection_name = dpg.get_value(self.node_list_box.widget.uuid)
            new_node_name = dpg.get_value(self.name_property.widget.uuid)
            arg_string = dpg.get_value(self.args_property.widget.uuid)
            new_node_args = []
            if len(arg_string) > 0:
                args = arg_string.split(' ')
                new_node_args = [new_node_name] + args
            else:
                new_node_args = [new_node_name]
            node_model = None
            found = False
            if new_node_args[0] in self.node_list:
                found = True
            elif selection_name in self.node_list:
                new_node_args[0] = selection_name
                found = True
            if found:
                if len(new_node_args) > 1:
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
            if found:
                additional = []
                if len(new_node_args) > 1:
                    additional = new_node_args[1:]
                t = type(v.value)
                found = False
                if t == int:
                    new_node_args = ['int', new_node_args[0]]
                    found = True
                elif t == float:
                    new_node_args = ['float', new_node_args[0]]
                    found = True
                elif t == str:
                    new_node_args = ['message', new_node_args[0]]
                    found = True
                elif t == bool:
                    new_node_args = ['toggle', new_node_args[0]]
                    found = True
                if found:
                    if len(additional) > 0:
                        new_node_args += additional
                    if len(new_node_args) > 1:
                        Node.app.create_node_by_name(new_node_args[0], self, new_node_args[1:])
                    else:
                        Node.app.create_node_by_name(new_node_args[0], self, )




