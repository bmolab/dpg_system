import dearpygui.dearpygui as dpg
import math
import time
import numpy as np
import random

from dpg_system.node import Node
import json


class NodeEditor:
    @staticmethod
    def _link_callback(sender, app_data, user_data):
        # print('link callback')
        output_attr_uuid, input_attr_uuid = app_data
        input_attr = dpg.get_item_user_data(input_attr_uuid)
        output_attr = dpg.get_item_user_data(output_attr_uuid)
        output_attr.add_child(input_attr, sender)

    @staticmethod
    def _unlink_callback(sender, app_data, user_data):
        # print('unlink')
        dat = dpg.get_item_user_data(app_data)
        out = dat[0]
        child = dat[1]
        out.remove_link(app_data, child)

    def __init__(self, height=0, width=0):
        self._nodes = []
        self._links = []
        self.height = height
        self.width = width
        self.uuid = dpg.generate_uuid()
        self.active_pins = []
        self.num_nodes = 0
        self.node_theme = None
        self.setup_theme()
        self.patch_name = ''
        self.file_path = ''
        self.mini_map = False

    def add_node(self, node: Node):
        self._nodes.append(node)
        self.num_nodes = len(self._nodes)

    def node_cleanup(self, node_uuid):
        # print('deleting', node_uuid)
        for node in self._nodes:
            if node.uuid == node_uuid:
                for element in node.ordered_elements:
                    print(element.uuid)
                # for property in node._property_attributes:
                #     print(property.label, property.uuid)
                # for option in node._option_attributes:
                #     print(option.label, option.uuid)
                # for output in node._output_attributes:
                #     print(output.label, output.uuid)
                self.remove_node(node)
                break
        # we need to have a registry to node_uuid to Node

    def remove_node(self, node):
        for n in self._nodes:
            if n == node:
                # need to remove links as well
                node.cleanup()
                self._nodes.remove(node)
                dpg.delete_item(node.uuid)
                self.num_nodes = len(self._nodes)
                break

    def submit(self, parent):
        with dpg.child_window(width=0, parent=parent, user_data=self):
            with dpg.node_editor(tag=self.uuid, callback=NodeEditor._link_callback, height=self.height, width=self.width, delink_callback=NodeEditor._unlink_callback):
                for node in self._nodes:
                    node.submit(self.uuid)
        dpg.bind_theme(self.node_theme)

    def add_active_pin(self, uuid):
        self.active_pins.append(uuid)

    def reset_pins(self):
        for uuid in self.active_pins:
            if dpg.does_item_exist(uuid):
                dpg.bind_item_theme(uuid, 0)
        self.active_pins = []

    def show_minimap(self, show):
        if show:
            dpg.configure_item(self.uuid, minimap=True)
            self.mini_map = True
        else:
            dpg.configure_item(self.uuid, minimap=False)
            self.mini_map = False

    def duplicate_selection(self):
        file_container = {}
        nodes_container = {}
        selected_node_objects = []
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        for index, node in enumerate(self._nodes):
            if node.uuid in selected_nodes:
                selected_node_objects.append(node)
                node_container = {}
                node.save(node_container, index)
                nodes_container[index] = node_container
        file_container['nodes'] = nodes_container
        offset = [50, 50]
        links_container = {}
        link_index = 0
        for node_index, node in enumerate(selected_node_objects):
            # save links
            for out_index, output in enumerate(node.outputs):
                if len(output._children) > 0:
                    for in_index, input in enumerate(output._children):
                        link_container = {}
                        link_container['source_node'] = node.uuid
                        link_container['source_output_index'] = out_index
                        dest_node = input.node
                        link_container['dest_node'] = dest_node.uuid
                        for node_in_index, test_input in enumerate(dest_node.inputs):
                            if test_input.uuid == input.uuid:
                                link_container['dest_input_index'] = node_in_index
                                links_container[link_index] = link_container
                                link_index += 1
                                break
        file_container['links'] = links_container

        dpg.clear_selected_nodes(self.uuid)
        self.uncontainerize(file_container, offset=offset)

    def containerize(self, patch_container=None):
        if patch_container is None:
            patch_container = {}
        nodes_container = {}
        patch_container['height'] = dpg.get_viewport_height()
        patch_container['width'] = dpg.get_viewport_width()
        patch_container['position'] = dpg.get_viewport_pos()
        if self.patch_name != ' ':
            patch_container['name'] = self.patch_name
        if self.file_path != '':
            patch_container['path'] = self.file_path

        for index, node in enumerate(self._nodes):
            node_container = {}
            node.save(node_container, index)
            nodes_container[index] = node_container
        patch_container['nodes'] = nodes_container

        links_container = {}
        link_index = 0
        for node_index, node in enumerate(self._nodes):
            # save links
            for out_index, output in enumerate(node.outputs):
                if len(output._children) > 0:
                    for in_index, input in enumerate(output._children):
                        link_container = {}
                        link_container['source_node'] = node.uuid
                        link_container['source_output_index'] = out_index
                        dest_node = input.node
                        link_container['dest_node'] = dest_node.uuid
                        for node_in_index, test_input in enumerate(dest_node.inputs):
                            if test_input.uuid == input.uuid:
                                link_container['dest_input_index'] = node_in_index
                                links_container[link_index] = link_container
                                link_index += 1
                                break
        patch_container['links'] = links_container
        return patch_container

    def save_into(self, patch_container):
        patch_container = self.containerize(patch_container)

    def save(self, path=None):
        if path is None:
            return
        self.patch_name = path.split('/')[-1]
        print(path, self.patch_name)
        if '.' in self.patch_name:
            parts = self.patch_name.split('.')
            if len(parts) == 2:
                if parts[1] == 'json':
                    self.patch_name = parts[0]
        self.file_path = path
        with open(path, 'w') as f:
            file_container = self.containerize()
            json.dump(file_container, f, indent=4)
        Node.app.set_current_tab_title(self.patch_name)

    def uncontainerize(self, file_container, offset=None):
        created_nodes = {}
        if offset is None:
            offset = [0, 0]

        if 'name' in file_container:
            self.patch_name = file_container['name']
        if 'path' in file_container:
            self.file_path = file_container['path']
        height = dpg.get_viewport_height()
        if 'height' in file_container:
            height = file_container['height']
        width = dpg.get_viewport_width()
        if 'width' in file_container:
            width = file_container['width']
        position = dpg.get_viewport_pos()
        if 'position' in file_container:
            position = file_container['position']
        # print(height, width, position)

        if 'nodes' in file_container:
            nodes_container = file_container['nodes']
            for index, node_index in enumerate(nodes_container):
                node_container = nodes_container[node_index]
                pos = [0, 0]
                if 'position_x' in node_container:
                    pos[0] = node_container['position_x'] + offset[0]
                if 'position_y' in node_container:
                    pos[1] = node_container['position_y'] + offset[1]
                args = []
                if 'init' in node_container:
                    args_container = node_container['init']
                    args = args_container.split(' ')
                if len(args) > 1:
                    new_node = Node.app.create_node_by_name_from_file(args[0], pos, args[1:])
                elif len(args) > 0:
                    new_node = Node.app.create_node_by_name_from_file(args[0], pos, )
                else:
                    l = node_container['name']
                    new_node = Node.app.create_node_by_name_from_file(l, pos, )
                if new_node != None:
                    new_node.load(node_container, offset=offset)
                    created_nodes[new_node.loaded_uuid] = new_node
                    dpg.focus_item(new_node.uuid)

        if 'links' in file_container:
            links_container = file_container['links']
            # print(links_container)
            for index, link_index in enumerate(links_container):
                source_node = None
                dest_node = None
                link_container = links_container[link_index]
                source_node_loaded_uuid = link_container['source_node']
                if source_node_loaded_uuid in created_nodes:
                    source_node = created_nodes[source_node_loaded_uuid]
                dest_node_loaded_uuid = link_container['dest_node']
                if dest_node_loaded_uuid in created_nodes:
                    dest_node = created_nodes[dest_node_loaded_uuid]
                if source_node is not None and dest_node is not None:
                    source_output_index = link_container['source_output_index']
                    dest_input_index = link_container['dest_input_index']
                    if source_output_index < len(source_node.outputs):
                        source_output = source_node.outputs[source_output_index]
                        if dest_input_index < len(dest_node.inputs):
                            dest_input = dest_node.inputs[dest_input_index]
                            source_output.add_child(dest_input, self.uuid)
        dpg.configure_viewport(0, height=height, width=width, x_pos=int(position[0]), y_pos=int(position[1]))
        dpg.set_viewport_pos(position)
        # conf = dpg.get_viewport_configuration(0)
        # print(conf)

    def load_(self, patch_container, path='', name=''):
        print(path, name)
        self.file_path = path
        self.patch_name = name
        self.uncontainerize(patch_container)
        if self.patch_name == '':
            self.patch_name = 'node patch'
        Node.app.set_current_tab_title(self.patch_name)

    def load(self, path=''):
        try:
            if len(path) > 0:
                with open(path, 'r') as f:
                    file_container = json.load(f)
                    self.file_path = path
                    self.patch_name = self.file_path.split('/')[-1]
                    if '.' in self.patch_name:
                        parts = self.patch_name.split('.')
                        if len(parts) == 2:
                            if parts[1] == 'json':
                                self.patch_name = parts[0]
                    self.uncontainerize(file_container)
                    self.file_path = path
                    self.patch_name = self.file_path.split('/')[-1]
                    if '.' in self.patch_name:
                        parts = self.patch_name.split('.')
                        if len(parts) == 2:
                            if parts[1] == 'json':
                                self.patch_name = parts[0]
                    Node.app.set_current_tab_title(self.patch_name)
 #                   dpg.set_viewport_title(self.patch_name)
        except:
            print('exception occurred during load')
            pass

    def setup_theme(self):
        self.node_scalers = {}
        with dpg.theme() as self.node_theme:
            with dpg.theme_component(dpg.mvAll):
                self.node_scalers[dpg.mvNodeStyleVar_GridSpacing] = 16
                dpg.add_theme_style(dpg.mvNodeStyleVar_GridSpacing, self.node_scalers[dpg.mvNodeStyleVar_GridSpacing], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeCol_GridLine] = [60, 60, 60]
                dpg.add_theme_color(dpg.mvNodeCol_GridLine, self.node_scalers[dpg.mvNodeCol_GridLine], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeStyleVar_NodePadding] = [4, 1]
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding,  self.node_scalers[dpg.mvNodeStyleVar_NodePadding][0],  self.node_scalers[dpg.mvNodeStyleVar_NodePadding][1], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeStyleVar_PinOffset] = 2
                dpg.add_theme_style(dpg.mvNodeStyleVar_PinOffset, self.node_scalers[dpg.mvNodeStyleVar_PinOffset], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeStyleVar_LinkThickness] = 1
                dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, self.node_scalers[dpg.mvNodeStyleVar_LinkThickness], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeCol_Pin] = [30, 100, 150]
                dpg.add_theme_color(dpg.mvNodeCol_Pin, self.node_scalers[dpg.mvNodeCol_Pin], category=dpg.mvThemeCat_Nodes)

    # def scale_nodes(self, scale):
    #     with dpg.theme() as self.node_theme:
    #         with dpg.theme_component(dpg.mvAll):
    #             dpg.add_theme_style(dpg.mvNodeStyleVar_GridSpacing, round(self.node_scalers[dpg.mvNodeStyleVar_GridSpacing] * scale), category=dpg.mvThemeCat_Nodes)
    #             new_grid = self.node_scalers[dpg.mvNodeCol_GridLine]
    #             new_grid[0] = round(new_grid[0] * scale)
    #             new_grid[1] = round(new_grid[1] * scale)
    #             new_grid[2] = round(new_grid[2] * scale)
    #             dpg.add_theme_color(dpg.mvNodeCol_GridLine, new_grid, category=dpg.mvThemeCat_Nodes)
    #             node_padding = self.node_scalers[dpg.mvNodeStyleVar_NodePadding]
    #             node_padding[0] *= scale
    #             node_padding[1] *= scale
    #             dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, node_padding[0], node_padding[1], category=dpg.mvThemeCat_Nodes)
    #             dpg.add_theme_style(dpg.mvNodeStyleVar_PinOffset, round(self.node_scalers[dpg.mvNodeStyleVar_PinOffset] * scale), category=dpg.mvThemeCat_Nodes)
    #             dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, round(self.node_scalers[dpg.mvNodeStyleVar_LinkThickness] * scale), category=dpg.mvThemeCat_Nodes)
    #     dpg.bind_theme(self.node_theme)
#     def scale_theme(self):
#         dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding)
#         dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding)
#         dpg.add_theme_style(dpg.mvNodeStyleVar_GridSpacing)
#         dpg.add_theme_style(dpg.mvNodeStyleVar_PinCircleRadius)
#         dpg.add_theme_style(dpg.mvNodeStyleVar_PinHoverRadius)
#         dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness)
#
#         dpg.add_theme_style(dpg.mvStyleVar_FrameRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_WindowPadding)
#         dpg.add_theme_style(dpg.mvStyleVar_CellPadding)
#         dpg.add_theme_style(dpg.mvStyleVar_WindowPadding)
#         dpg.add_theme_style(dpg.mvStyleVar_WindowRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_WindowMinSize)
#         dpg.add_theme_style(dpg.mvStyleVar_ChildRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_PopupRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_FramePadding)
#         dpg.add_theme_style(dpg.mvStyleVar_FrameRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing)
#         dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing)
#         dpg.add_theme_style(dpg.mvStyleVar_CellPadding)
# #        dpg.add_theme_style(dpg.mvStyleVar_TouchExtraPadding)
#         dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing)
# #        dpg.add_theme_style(dpg.mvStyleVar_ColumnsMinSpacing)
#         dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize)
#         dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding)
#         dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize)
#         dpg.add_theme_style(dpg.mvStyleVar_GrabRounding)
# #        dpg.add_theme_style(dpg.mvStyleVar_LogSliderDeadzone)
#         dpg.add_theme_style(dpg.mvStyleVar_TabRounding)
# #        dpg.add_theme_style(dpg.mvStyleVar_TabMinWidthForCloseButton)
# #        dpg.add_theme_style(dpg.mvStyleVar_DisplayWindowPadding)
# #        dpg.add_theme_style(dpg.mvStyleVar_DisplaySafeAreaPadding)
# #        dpg.add_theme_style(dpg.mvStyleVar_MouseCursorScale)


########################################################################################################################
# Drag & Drop
########################################################################################################################
class NodeFactory:

    def __init__(self, label: str, node_generator, data):
        self.label = label
        self._generator = node_generator
        self._data = data

    def submit(self, parent):
        dpg.add_button(label=self.label, parent=parent, width=-1)
#        dpg.bind_item_theme(dpg.last_item(), global_theme)

        with dpg.drag_payload(parent=dpg.last_item(), drag_data=(self, self._generator, self._data)):
            dpg.add_text(f"Name: {self.label}")

    def create(self, name=None, args=None):
        if name is not None:
            return self._generator(name, self._data, args)
        else:
            return self._generator(self.label, self._data, args)


class NodeFactoryContainer:
    def __init__(self, label: str, width: int = 150, height: int = -1):
        self._label = label
        self._width = width
        self._height = height
        self._uuid = dpg.generate_uuid()
        self._children = []  # drag sources

    def add_node_factory(self, source: NodeFactory):
       self._children.append(source)

    def locate_node_by_name(self, name):
        for child in self._children:
            if child.label == name:
                return child
        return None

    def get_node_list(self):
        list = []
        for child in self._children:
            list.append(child.label)
        return list

    def submit(self, parent):
        with dpg.child_window(parent=parent, width=self._width, height=self._height, tag=self._uuid,
                              menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label)

            for child in self._children:
                child.submit(child_parent)
