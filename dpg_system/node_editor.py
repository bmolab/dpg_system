import dearpygui.dearpygui as dpg
import math
import time
import numpy as np
import random
from dpg_system.node import Node, OriginNode, PatcherNode
import json


class NodeEditor:
    app = None
    @staticmethod
    def _link_callback(sender, app_data, user_data):
        output_attr_uuid, input_attr_uuid = app_data
        input_attr = dpg.get_item_user_data(input_attr_uuid)
        output_attr = dpg.get_item_user_data(output_attr_uuid)
        output_attr.add_child(input_attr, sender)
        editor = NodeEditor.app.get_current_editor()
        if editor is not None:
            editor.modified = True

    @staticmethod
    def _unlink_callback(sender, app_data, user_data):
        # print('unlink')
        dat = dpg.get_item_user_data(app_data)
        out = dat[0]
        child = dat[1]
        out.remove_link(app_data, child)
        NodeEditor.app.get_current_editor().modified = True

    def __init__(self, height=0, width=0):
        self._nodes = []
        self._links = []
        self.subpatches = []
        self.height = height
        self.width = width
        self.uuid = dpg.generate_uuid()
        self.loaded_uuid = -1
        self.loaded_parent_node_uuid = -1
        self.active_pins = []
        self.app = NodeEditor.app
        self.num_nodes = 0
        self.node_theme = None
        self.comment_theme = None
        self.setup_theme()
        self.patch_name = ''
        self.file_path = ''
        self.mini_map = False
        self.origin = None
        self.patcher_node = None
        self.parent_patcher = None
        self.modified = False
        self.duplicated_subpatch_nodes = {}
        self.saving_preset = False
        self.presenting = False

    def set_name(self, name):
        self.patch_name = name
        self.app.set_editor_tab_title(self, name)

    def add_subpatch(self, subpatch_editor):
        self.subpatches.append(subpatch_editor)

    def align_selected(self):
        # find dominant axis
        # align to that axis
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        x_min = 100000
        x_max = -100000
        y_min = 100000
        y_max = -100000

        for node_uuid in selected_nodes:
            pos = dpg.get_item_pos(node_uuid)
            if pos[0] < x_min:
                x_min = pos[0]
            if pos[0] > x_max:
                x_max = pos[0]
            if pos[1] < y_min:
                y_min = pos[1]
            if pos[1] > y_max:
                y_max = pos[1]

        if (x_max - x_min) > (y_max - y_min):
            y_mean = (y_max + y_min) / 2
            for node_uuid in selected_nodes:
                pos = dpg.get_item_pos(node_uuid)
                pos[1] = y_mean
                dpg.set_item_pos(node_uuid, pos)
        else:
            x_mean = (x_max + x_min) / 2
            for node_uuid in selected_nodes:
                pos = dpg.get_item_pos(node_uuid)
                pos[0] = x_mean
                dpg.set_item_pos(node_uuid, pos)


    def align_and_distribute_selected(self, align, spread_factor=1.0):
        # find dominant axis
        # align and distribute to that axis
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        if len(selected_nodes) <= 1:
            return

        x_min = 100000
        x_max = -100000
        y_min = 100000
        y_max = -100000

        for node_uuid in selected_nodes:
            pos = dpg.get_item_pos(node_uuid)
            if pos[0] < x_min:
                x_min = pos[0]
            if pos[0] > x_max:
                x_max = pos[0]
            if pos[1] < y_min:
                y_min = pos[1]
            if pos[1] > y_max:
                y_max = pos[1]

        if (x_max - x_min) > (y_max - y_min):
            y_mean = (y_max + y_min) / 2

            dest_dict = {}
            total_node_width = 0
            real_x_max = -10000
            center_acc = 0
            bottom_acc = 0
            for index, node_uuid in enumerate(selected_nodes):
                s = dpg.get_item_rect_size(node_uuid)
                pos = dpg.get_item_pos(node_uuid)
                dest_dict[index] = [pos, node_uuid, s]
                total_node_width += s[0]
                if pos[0] + s[0] > real_x_max:
                    real_x_max = pos[0] + s[0]
                center_acc += (s[1] / 2 + pos[1])
                bottom_acc += (s[1] + pos[1])

            total_width = (real_x_max - x_min) * spread_factor
            total_gap = total_width - total_node_width
            gap = total_gap / (len(selected_nodes) - 1)
            center = center_acc / len(selected_nodes)
            sorted_dest = sorted(dest_dict.items(), key=lambda item: item[1][0][0])
            x_acc = x_min

            for index, dest_data in enumerate(sorted_dest):
                uuid = dest_data[1][1]
                if align == -1:
                    pos = [x_acc, y_mean]
                elif align == 0:
                    pos = [x_acc, center - dest_data[1][2][1] / 2]
                elif align == 1:
                    pos = [x_acc, bottom_acc - dest_data[1][2][1]]
                else:
                    pos = [x_acc, dest_data[1][0][1]]

                x_acc += (dest_data[1][2][0] + gap)
                dpg.set_item_pos(uuid, pos)
        else:
            x_mean = (x_max + x_min) / 2

            dest_dict = {}
            total_node_height = 0
            real_y_max = -10000
            center_acc = 0

            for index, node_uuid in enumerate(selected_nodes):
                s = dpg.get_item_rect_size(node_uuid)
                pos = dpg.get_item_pos(node_uuid)
                dest_dict[index] = [pos, node_uuid, s]
                total_node_height += s[1]
                if pos[1] + s[1] > real_y_max:
                    real_y_max = pos[1] + s[1]
                center_acc += (s[0] / 2 + pos[0])

            total_height = (real_y_max - y_min) * spread_factor
            total_gap = total_height - total_node_height
            gap = total_gap / (len(selected_nodes) - 1)
            center = center_acc / len(selected_nodes)
            sorted_dest = sorted(dest_dict.items(), key=lambda item: item[1][0][1])
            y_acc = y_min

            for index, dest_data in enumerate(sorted_dest):
                uuid = dest_data[1][1]
                if align == -1:
                    pos = [x_mean, y_acc]
                elif align == 0:
                    pos = [center - dest_data[1][2][0] / 2, y_acc]
                elif align == -2:
                    pos = [dest_data[1][0][0], y_acc]
                y_acc += (dest_data[1][2][1] + gap)
                dpg.set_item_pos(uuid, pos)

    def connect_nodes_to_nodes(self, source_nodes, dest_nodes):
        print('connect_nodes_to_nodes')
        connect_count = len(dest_nodes)
        if len(source_nodes) < connect_count:
            connect_count = len(source_nodes)
        source_dict = {}
        dest_dict = {}
        for i in range(connect_count):
            pos = dpg.get_item_pos(source_nodes[i].uuid)
            source_dict[pos[1]] = source_nodes[i]
            pos = dpg.get_item_pos(dest_nodes[i].uuid)
            dest_dict[pos[1]] = dest_nodes[i]
        sorted_source = dict(sorted(source_dict.items()))
        sorted_dest = dict(sorted(dest_dict.items()))
        source_keys = list(sorted_source.keys())
        dest_keys = list(sorted_dest.keys())
        for index, source in enumerate(source_keys):
            source_output = sorted_source[source].outputs[0]
            dest_input = sorted_dest[dest_keys[index]].inputs[0]
            source_output.add_child(dest_input, self.uuid)

    def connect_single_node_output_to_nodes(self, source_nodes, dest_nodes):
        print('connect_single_node_output_to_nodes')
        out_count = len(source_nodes[0].outputs)
        dest_count = len(dest_nodes)
        connect_count = dest_count
        dest_dict = {}
        for i in range(connect_count):
            pos = dpg.get_item_pos(dest_nodes[i].uuid)
            dest_dict[pos[1]] = dest_nodes[i]
        sorted_dest = dict(sorted(dest_dict.items()))
        out_ = source_nodes[0].outputs[0]
        for dest_key in sorted_dest:
            in_ = sorted_dest[dest_key].inputs[0]
            out_.add_child(in_, self.uuid)

    def connect_single_node_multi_outputs_to_single_node_multi_inputs(self, source_nodes, dest_nodes):
        print('connect_single_node_multi_outputs_to_single_node_multi_inputs')
        input_count = len(dest_nodes[0].inputs)
        output_count = len(source_nodes[0].outputs)
        connect_count = input_count
        if output_count < connect_count:
            connect_count = output_count
        for i in range(connect_count):
            source_output = source_nodes[0].outputs[i]
            dest_input = dest_nodes[0].inputs[i]
            source_output.add_child(dest_input, self.uuid)

    def connect_single_node_multi_outputs_to_nodes(self, source_nodes, dest_nodes):
        print('connect_single_node_multi_outputs_to_nodes')
        out_count = len(source_nodes[0].outputs)
        dest_count = len(dest_nodes)
        connect_count = out_count
        if dest_count < connect_count:
            connect_count = dest_count
        dest_dict = {}
        for i in range(connect_count):
            pos = dpg.get_item_pos(dest_nodes[i].uuid)
            dest_dict[pos[1]] = dest_nodes[i]
        sorted_dest = dict(sorted(dest_dict.items()))
        for index, dest_key in enumerate(sorted_dest):
            out_ = source_nodes[0].outputs[index]
            in_ = sorted_dest[dest_key].inputs[0]
            out_.add_child(in_, self.uuid)

    def connect_nodes_to_single_node_multi_output(self, source_nodes, dest_nodes):
        print('connect_nodes_to_single_node_multi_output')
        in_count = len(dest_nodes[0].inputs)
        source_count = len(source_nodes)
        connect_count = in_count
        if source_count < connect_count:
            connect_count = source_count
        source_dict = {}
        for i in range(connect_count):
            pos = dpg.get_item_pos(source_nodes[i].uuid)
            source_dict[pos[1]] = source_nodes[i]
        sorted_source = dict(sorted(source_dict.items()))
        for index, source_key in enumerate(sorted_source):
            in_ = dest_nodes[0].inputs[index]
            out_ = sorted_source[source_key].outputs[0]
            out_.add_child(in_, self.uuid)

    def connect_nodes_to_single_node_input(self, source_nodes, dest_nodes):
        print('connect_nodes_to_single_node_input')
        source_count = len(source_nodes)
        connect_count = source_count
        source_dict = {}
        for i in range(connect_count):
            pos = dpg.get_item_pos(source_nodes[i].uuid)
            source_dict[pos[1]] = source_nodes[i]
        sorted_source = dict(sorted(source_dict.items()))
        in_ = dest_nodes[0].inputs[0]
        for source_key in sorted_source:
            out_ = sorted_source[source_key].outputs[0]
            out_.add_child(in_, self.uuid)

    def connect_selected(self):
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        if len(selected_nodes) <= 1:
            return
        x_min = 100000
        x_max = -100000
        for node_uuid in selected_nodes:
            pos = dpg.get_item_pos(node_uuid)
            if pos[0] < x_min:
                x_min = pos[0]
            if pos[0] > x_max:
                x_max = pos[0]
        x_mean = (x_max + x_min) / 2

        source_nodes = []
        dest_nodes = []

        for node_uuid in selected_nodes:
            pos = dpg.get_item_pos(node_uuid)
            if pos[0] <= x_mean:
                source_nodes.append(dpg.get_item_user_data(node_uuid))
            else:
                dest_nodes.append(dpg.get_item_user_data(node_uuid))
#       if dest_nodes count > 0 and source_nodes count > 0
#           connect first out of each source to first out of each dest
        if len(source_nodes) > 1 and len(dest_nodes) > 1:
            self.connect_nodes_to_nodes(source_nodes, dest_nodes)
        else:
            # if source_nodes count == 1
            if len(source_nodes) == 1:
                source_output_count = len(source_nodes[0].outputs)
                dest_count = len(dest_nodes)
                # if source_nodes output count == 1
                if source_output_count == 1:
                    # connect first out of source to first out of each dest
                    self.connect_single_node_output_to_nodes(source_nodes, dest_nodes)
                # else
                else:
                    # if dest_nodes count == 1
                    if dest_count == 1:
                        # if dest_nodes input count > 1
                        if len(dest_nodes[0].inputs) > 1:
                            # connect source outs to corresponding dest ins
                            self.connect_single_node_multi_outputs_to_single_node_multi_inputs(source_nodes, dest_nodes)
                        # else
                        else:
                            # connect first out of source to first in of dest
                            self.connect_nodes_to_nodes(source_nodes, dest_nodes)
                    # else
                    else:
                        # connect each out of source to first out of different dest
                        self.connect_single_node_multi_outputs_to_nodes(source_nodes, dest_nodes)
            # else (len(dest_nodes) == 1, len(source_nodes) > 1)
            else:
                # dest_nodes input count == 1
                if len(dest_nodes[0].inputs) == 1:
                    # connect first out of each source to first in of dest
                    self.connect_nodes_to_single_node_input(source_nodes, dest_nodes)
                # else (dest_nodes input count > 1)
                else:
                    # connect first out of each source to inputs of dest in order
                    self.connect_nodes_to_single_node_multi_output(source_nodes, dest_nodes)


        # if len(source_nodes) == 1:
        #     out_count = len(source_nodes[0].outputs)
        #     dest_count = len(dest_nodes)
        #     if out_count > 1:
        #         connect_count = out_count
        #     else:
        #         connect_count = dest_count
        #     if dest_count < out_count:
        #         connect_count = dest_count
        #     dest_dict = {}
        #     for i in range(connect_count):
        #         pos = dpg.get_item_pos(dest_nodes[i].uuid)
        #         dest_dict[pos[1]] = dest_nodes[i]
        #     sorted_dest = dict(sorted(dest_dict.items()))
        #     if out_count == 1:
        #         for index, dest_key in enumerate(sorted_dest):
        #             out_ = source_nodes[0].outputs[0]
        #             in_ = sorted_dest[dest_key].inputs[0]
        #             out_.add_child(in_, self.uuid)
        #     else:
        #         for index, dest_key in enumerate(sorted_dest):
        #             out_ = source_nodes[0].outputs[index]
        #             in_ = sorted_dest[dest_key].inputs[0]
        #             out_.add_child(in_, self.uuid)
        # elif len(dest_nodes) == 1:
        #     in_count = len(dest_nodes[0].inputs)
        #     source_count = len(source_nodes)
        #     if in_count > 1:
        #         connect_count = in_count
        #     else:
        #         connect_count = source_count
        #     if source_count < in_count:
        #         connect_count = source_count
        #     if in_count == 1:
        #         for i in range(connect_count):
        #             out_ = source_nodes[i].outputs[0]
        #             in_ = dest_nodes[0].inputs[0]
        #             out_.add_child(in_, self.uuid)
        #     else:
        #         for i in range(connect_count):
        #             out_ = source_nodes[i].outputs[0]
        #             in_ = dest_nodes[0].inputs[i]
        #             out_.add_child(in_, self.uuid)
        # else:
        #     connect_count = len(dest_nodes)
        #     if len(source_nodes) < connect_count:
        #         connect_count = len(source_nodes)
        #     source_dict = {}
        #     dest_dict = {}
        #     for i in range(connect_count):
        #         pos = dpg.get_item_pos(source_nodes[i].uuid)
        #         source_dict[pos[1]] = source_nodes[i]
        #         pos = dpg.get_item_pos(dest_nodes[i].uuid)
        #         dest_dict[pos[1]] = dest_nodes[i]
        #     sorted_source = dict(sorted(source_dict.items()))
        #     sorted_dest = dict(sorted(dest_dict.items()))
        #     source_keys = list(sorted_source.keys())
        #     dest_keys = list(sorted_dest.keys())
        #     for index, source in enumerate(source_keys):
        #         source_output = sorted_source[source].outputs[0]
        #         dest_input = sorted_dest[dest_keys[index]].inputs[0]
        #         source_output.add_child(dest_input, self.uuid)

    def find_patcher_node(self, patcher_name):
        # print('find_patcher_node in ', self.patch_name, self._nodes)
        for node in self._nodes:
            if node.label == 'patcher':
                # print('found a patcher', node.patcher_name, patcher_name)
                # might have to have load_uuids in case multiple of same name in patcher
                if node.patcher_name == patcher_name:
                    return node
        return None

    def reconnect_to_parent(self, parent_patcher_node=None):
        if self.parent_patcher is not None:
            # print('reconnect_to_parent', self.patch_name)
            if parent_patcher_node is None:
                parent_patcher_node = self.parent_patcher.find_patcher_node(self.patch_name)
            if parent_patcher_node is not None:
                # print('found parent')
                for node in self._nodes:
                    if node.label == 'in':
                        node.connect_to_parent(parent_patcher_node)
                    elif node.label == 'out':
                        node.connect_to_parent(parent_patcher_node)

    def add_node(self, node: Node):
        self._nodes.append(node)
        self.num_nodes = len(self._nodes)
        self.modified = True

    def remove_all_nodes(self):
        self.app.clear_remembered_ids()
        for node in self._nodes:
            node.cleanup()
            dpg.delete_item(node.uuid)
        self._nodes = []
        self.num_nodes = len(self._nodes)
        self.modified = True
        for link_uuid in self._links:
            dat = dpg.get_item_user_data(link_uuid)
            out = dat[0]
            child = dat[1]
            out.remove_link(link_uuid, child)
        self.active_pins = []

    def node_cleanup(self, node_uuid):
        # print('deleting', node_uuid)
        for node in self._nodes:
            if node.uuid == node_uuid:
                # for element in node.ordered_elements:
                #     print(element.uuid)
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
                self.remove_active_pins(node)
                node.cleanup()
                self._nodes.remove(node)
                dpg.delete_item(node.uuid)
                self.num_nodes = len(self._nodes)
                self.modified = True
                break

    def remove_active_pins(self, node):
        for input in node.inputs:
            if input.uuid in self.active_pins:
                self.active_pins.remove(input.uuid)
        for output in node.outputs:
            if output.uuid in self.active_pins:
                self.active_pins.remove(output.uuid)

    def create(self, parent):
        with dpg.child_window(width=0, parent=parent, user_data=self):
            with dpg.node_editor(tag=self.uuid, callback=NodeEditor._link_callback, height=self.height, width=self.width, delink_callback=NodeEditor._unlink_callback):
                for node in self._nodes:
                    node.create(self.uuid)

        dpg.bind_theme(self.node_theme)
        # add invisible node that is reference for panning

        self.origin = OriginNode.factory('origin', None)
        self.origin.create(self.uuid, pos=[0, 0])
        self.add_node(self.origin)

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

    def cut_selection(self):
        clip = self.copy_selection()
        node_uuids = dpg.get_selected_nodes(self.uuid)
        for node_uuid in node_uuids:
            # somehow we have to connect to the actual Node object
            self.node_cleanup(node_uuid)
        link_uuids = dpg.get_selected_links(self.uuid)
        for link_uuid in link_uuids:
            dat = dpg.get_item_user_data(link_uuid)
            out = dat[0]
            child = dat[1]
            out.remove_link(link_uuid, child)
        return clip

    def reset_origin(self):
        area = self.calc_node_area()
        offset_x = - (area[0] - 16)
        offset_y = - (area[1] - 16)
        for index, node in enumerate(self._nodes):
            if node.label != '':
                pos = dpg.get_item_pos(node.uuid)
                pos[0] = pos[0] + offset_x
                pos[1] = pos[1] + offset_y
                dpg.set_item_pos(node.uuid, pos)
            else:
                dpg.set_item_pos(node.uuid, [0.0, 0.0])

    def calc_node_area(self):
        top = None
        right = None
        left = None
        bottom = None

        for index, node in enumerate(self._nodes):
            if node.label != '':
                pos = dpg.get_item_pos(node.uuid)
                width = dpg.get_item_width(node.uuid)
                height = dpg.get_item_height(node.uuid)
                if top is None:
                    top = pos[1]
                else:
                    if pos[1] < top:
                        top = pos[1]

                if left is None:
                    left = pos[0]
                else:
                    if pos[0] < left:
                        left = pos[0]

                b = pos[1] + height
                if bottom is None:
                    bottom = b
                else:
                    if b > bottom:
                        bottom = b

                r = pos[0] + width
                if right is None:
                    right = r
                else:
                    if r > right:
                        right = r

        return [left, top, right, bottom]

    def copy_selection(self):
        self.app.loading = True
        file_container = {}
        nodes_container = {}
        selected_node_objects = []
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        subpatch_container = {}

        # NOTE - sub-patcher objects should not move with mouse
        for index, node in enumerate(self._nodes):
            if node.uuid in selected_nodes:
                if node.label == 'patcher':
                    if node.patch_editor is not None:
                        patch_container = {}
                        node.patch_editor.containerize(patch_container)
                        subpatch_container[index] = patch_container
                        # we need to remove all subpatch nodes from drag list...

        if len(subpatch_container) > 0:
            file_container['patches'] = subpatch_container
        for index, node in enumerate(self._nodes):
            if node.uuid in selected_nodes:
                selected_node_objects.append(node)
                node_container = {}
                node.save(node_container, index)
                nodes_container[index] = node_container
        file_container['nodes'] = nodes_container

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
        # dpg.clear_selected_nodes(self.uuid)
        self.app.loading = False
        return file_container

    def clear_loaded_uuids(self):
        # for uuid in self.app.created_nodes:
        #     node = self.app.created_nodes[uuid]
        #     if node is not None:
        #         node.post_load_callback()
        for uuid in self.app.created_nodes:
            node = self.app.created_nodes[uuid]
            node.loaded_uuid = -1

        for editor in self.app.node_editors:
            editor.loaded_uuid = -1
            editor.loaded_parent_node_uuid = -1

    def paste(self, file_container, drag=True, origin=False, clear_loaded_uuids=True, previously_created_nodes=None):
        if previously_created_nodes is None:
            self.app.created_nodes = {}
        else:
            self.app.created_nodes = previously_created_nodes
        self.app.loading = True
        if len(file_container) == 0:
            return

        # print('about to uncontainerize in paste')
        self.uncontainerize(file_container, create_origin=origin)
        # print('done uncontainerize in paste')
        # print('establishing links')
        for node_editor_uuid in self.app.links_containers:
            new_links = {}
            links_container = self.app.links_containers[node_editor_uuid]
            # print(links_container)
            # print(self.app.created_nodes)
            for index, link_index in enumerate(links_container):
                source_node = None
                dest_node = None
                link_container = links_container[link_index]
                new_link = link_container

                source_node_loaded_uuid = link_container['source_node']
                if source_node_loaded_uuid in self.app.created_nodes:
                    source_node = self.app.created_nodes[source_node_loaded_uuid]
                    new_link['source_node'] = source_node.uuid
                dest_node_loaded_uuid = link_container['dest_node']
                if dest_node_loaded_uuid in self.app.created_nodes:
                    dest_node = self.app.created_nodes[dest_node_loaded_uuid]
                    new_link['dest_node'] = dest_node.uuid
                if source_node is not None and dest_node is not None:
                    source_output_index = link_container['source_output_index']
                    dest_input_index = link_container['dest_input_index']
                    if source_output_index < len(source_node.outputs):
                        source_output = source_node.outputs[source_output_index]
                        if dest_input_index < len(dest_node.inputs):
                            dest_input = dest_node.inputs[dest_input_index]
                            source_output.add_child(dest_input, node_editor_uuid)
                new_links[link_index] = new_link
                self.app.links_containers[node_editor_uuid] = new_links.copy()
            # print('adjusted links', new_links)

        # print('links established')
        if clear_loaded_uuids:
            for uuid in self.app.created_nodes:
                node = self.app.created_nodes[uuid]
                if node is not None:
                    node.post_load_callback()

            self.clear_loaded_uuids()
            # print('uuids cleared')

        # now self.app.created_nodes dict has all created nodes
        if drag:
            self.app.dragging_created_nodes = True
            self.app.drag_starts = {}

            # we do not drag subpatch nodes
            for node_uuid in self.duplicated_subpatch_nodes:
                if node_uuid in self.app.created_nodes:
                    del self.app.created_nodes[node_uuid]

            for created_node in self.app.created_nodes:
                node = self.app.created_nodes[created_node]
                self.app.drag_starts[created_node] = dpg.get_item_pos(node.uuid)
                # but top left most node should be at mouse...
            if len(self.app.drag_starts) > 0:
                sort = sorted(self.app.drag_starts.items(), key=lambda item: item[1][0])
                left_most = sort[0][1][0]
                sort = sorted(self.app.drag_starts.items(), key=lambda item: item[1][1])
                top_most = sort[0][1][1]
                left_top = [left_most, top_most]
                self.app.dragging_ref = self.editor_pos_to_global_pos(left_top)
                self.modified = True
                self.app.loading = False
                self.app.drag_create_nodes()
        # else:
        #     print('no drag starts')

    def global_pos_to_editor_pos(self, pos):
        panel_pos = dpg.get_item_pos(self.app.center_panel)
        origin_pos = dpg.get_item_pos(self.origin.ref_property.widget.uuid)
        origin_node_pos = dpg.get_item_pos(self.origin.uuid)
        editor_mouse_pos = pos
        editor_mouse_pos[0] -= (panel_pos[0] + 8 + (origin_pos[0] - origin_node_pos[0]) - 4)
        editor_mouse_pos[1] -= (panel_pos[1] + 8 + (origin_pos[1] - origin_node_pos[1]) - 15)
        return editor_mouse_pos

    def editor_pos_to_global_pos(self, pos):
        panel_pos = dpg.get_item_pos(self.app.center_panel)
        origin_pos = dpg.get_item_pos(self.origin.ref_property.widget.uuid)
        origin_node_pos = dpg.get_item_pos(self.origin.uuid)
        global_pos = pos
        global_pos[0] += (panel_pos[0] + 8 + (origin_pos[0] - origin_node_pos[0]) - 4)
        global_pos[1] += (panel_pos[1] + 8 + (origin_pos[1] - origin_node_pos[1]) - 30)
        return global_pos

    def reveal_hidden(self):
        for node in self._nodes:
            node.set_visibility('show_all')

    def remember_presentation(self):
        for node in self._nodes:
            node.presentation_state = node.visibility

    def enter_presentation_state(self):
        self.presenting = True
        for node in self._nodes:
            node.set_visibility(node.presentation_state)
            node.set_draggable(False)
        dpg.bind_theme(self.node_presentation_theme)

    def enter_edit_state(self):
        self.presenting = False
        for node in self._nodes:
            node.set_visibility('show_all')
            node.set_draggable(True)
        dpg.bind_theme(self.node_theme)


    def patchify_selection(self):
        #  find centre of patch
        #  create patcher at centre
        #  find outside links
        self.app.created_nodes = {}
        centre = self.app.centre_of_selection()
        external_sources, external_targets = self.app.get_links_into_selection()
        clipboard = self.cut_selection()
        source_patch_tab = self.app.current_node_editor
        patcher_node = self.app.create_node_by_name('patcher', placeholder=None, args=['embedded_' + str(self.app.new_patcher_index)], pos=centre)
        # patcher_node.uuid
        self.app.current_node_editor = len(self.app.node_editors) - 1
        sub_patch_tab = self.app.current_node_editor
        sub_patch_editor = self.app.get_current_editor()
        sub_patch_editor.uncontainerize(clipboard)
        if 'links' in clipboard:
            links_container = clipboard['links']
            for index, link_index in enumerate(links_container):
                source_node = None
                dest_node = None
                link_container = links_container[link_index]
                source_node_loaded_uuid = link_container['source_node']
                if source_node_loaded_uuid in self.app.created_nodes:
                    source_node = self.app.created_nodes[source_node_loaded_uuid]
                dest_node_loaded_uuid = link_container['dest_node']
                if dest_node_loaded_uuid in self.app.created_nodes:
                    dest_node = self.app.created_nodes[dest_node_loaded_uuid]
                if source_node is not None and dest_node is not None:
                    source_output_index = link_container['source_output_index']
                    dest_input_index = link_container['dest_input_index']
                    if source_output_index < len(source_node.outputs):
                        source_output = source_node.outputs[source_output_index]
                        if dest_input_index < len(dest_node.inputs):
                            dest_input = dest_node.inputs[dest_input_index]
                            source_output.add_child(dest_input, sub_patch_editor.uuid)
        dpg.set_value(self.app.tab_bar, sub_patch_tab)

        input_node_index = 0
        for source in external_sources:
            source_output = source[0]
            dest_input = source[1]
            dest_node_uuid = source[2]
            dest_input_index_in_node = source[3]

            source_pos = dpg.get_item_pos(source_output.node.uuid)
            input_node = self.app.create_node_by_name('in', placeholder=None, args=[], pos=source_pos)
            for create_node_uuid in self.app.created_nodes:
                node = self.app.created_nodes[create_node_uuid]

                if node.loaded_uuid == dest_node_uuid:
                    if dest_input_index_in_node < len(node.inputs):
                        new_dest_input = node.inputs[dest_input_index_in_node]
                        input_node.input_out.add_child(new_dest_input, sub_patch_editor.uuid)
            self.app.current_node_editor = source_patch_tab
            source_output.add_child(patcher_node.inputs[input_node_index], self.app.get_current_editor().uuid)
            input_node_index += 1
            self.app.current_node_editor = sub_patch_tab


        output_node_index = 0
        for target in external_targets:
            target_input = target[0]
            source_output = target[1]
            source_node_uuid = target[2]
            source_output_index_in_node = target[3]

            target_pos = dpg.get_item_pos(target_input.node.uuid)
            output_node = self.app.create_node_by_name('out', placeholder=None, args=[], pos=target_pos)
            for create_node_uuid in self.app.created_nodes:
                node = self.app.created_nodes[create_node_uuid]

                if node.loaded_uuid == source_node_uuid:
                    if source_output_index_in_node < len(node.outputs):
                        new_source_output = node.outputs[source_output_index_in_node]
                        new_source_output.add_child(output_node.output_in, sub_patch_editor.uuid)
            self.app.current_node_editor = source_patch_tab
            patcher_node.outputs[output_node_index].add_child(target_input, self.app.get_current_editor().uuid)
            output_node_index += 1
            self.app.current_node_editor = sub_patch_tab

        self.app.current_node_editor = source_patch_tab

    def duplicate_selection(self):
        clipboard = self.copy_selection()
        self.paste(clipboard)

    def containerize(self, patch_container=None, exclude_list=None):
        if exclude_list is None:
            exclude_list = []
        if patch_container is None:
            patch_container = {}
        nodes_container = {}
        patch_container['height'] = dpg.get_viewport_height()
        patch_container['width'] = dpg.get_viewport_width()
        patch_container['position'] = dpg.get_viewport_pos()
        patch_container['id'] = self.uuid
        if self.patcher_node is not None:
            patch_container['parent_node_uuid'] = self.patcher_node.uuid
        if self.patch_name != '':
            patch_container['name'] = self.patch_name
        if self.file_path != '':
            patch_container['path'] = self.file_path

        for index, node in enumerate(self._nodes):
            if node in exclude_list:
                continue
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

    def save_into(self, patch_container=None):
        if patch_container is None:
            patch_container = {}
        patch_container = self.containerize(patch_container)
        self.modified = False
        return patch_container

    def save(self, path=None):
        if path is None:
            return
        self.patch_name = path.split('/')[-1]
        if '.' in self.patch_name:
            parts = self.patch_name.split('.')
            if len(parts) == 2:
                if parts[1] == 'json':
                    self.patch_name = parts[0]
        self.file_path = path
        with open(path, 'w') as f:
            file_container = self.containerize()
            # print(file_container)
            json.dump(file_container, f, indent=4)
        self.app.set_current_tab_title(self.patch_name)
        self.modified = False

    def uncontainerize(self, file_container, offset=None, create_origin=False):
        if offset is None:
            offset = [0, 0]
        hold_editor = self.app.current_node_editor

        if 'patches' in file_container:
            patch_container = file_container['patches']
            for patch_key in patch_container:
                patch = patch_container[patch_key]
                sub_patch_editor = self.app.add_node_editor()
                self.app.current_node_editor = len(self.app.node_editors) - 1
                sub_patch_editor.uncontainerize(patch)
                self.app.set_current_tab_title(sub_patch_editor.patch_name)
        self.duplicated_subpatch_nodes = self.app.created_nodes.copy()

        self.app.current_node_editor = hold_editor
        if 'name' in file_container:
            self.patch_name = file_container['name']
        if 'path' in file_container:
            self.file_path = file_container['path']
        if 'id' in file_container:
            self.loaded_uuid = file_container['id']
        if 'parent_node_uuid' in file_container:
            self.loaded_parent_node_uuid = file_container['parent_node_uuid']

        height = dpg.get_viewport_height()
        if 'height' in file_container:
            height = file_container['height']
        width = dpg.get_viewport_width()
        if 'width' in file_container:
            width = file_container['width']
        position = dpg.get_viewport_pos()
        if 'position' in file_container:
            position = file_container['position']

        node_name = ''
        if 'nodes' in file_container:
            nodes_container = file_container['nodes']
            for index, node_index in enumerate(nodes_container):
                node_container = nodes_container[node_index]
                if not create_origin and 'name' in node_container:
                    if node_container['name'] == '':
                        # origin
                        continue
                    else:
                        node_name = node_container['name']
                elif create_origin:
                    if 'name' in node_container:
                        if node_container['name'] == '':
                            node_name = 'origin'
                        else:
                            node_name = node_container['name']
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
                    new_node = self.app.create_node_by_name_from_file(args[0], pos, args[1:])
                elif len(args) > 0:
                    new_node = self.app.create_node_by_name_from_file(args[0], pos, )
                else:
                    if node_name == 'origin':
                        self.origin = OriginNode.factory('origin', None)
                        self.origin.create(self.uuid, pos=pos)
                        new_node = self.add_node(self.origin)
                    else:
                        new_node = self.app.create_node_by_name_from_file(node_name, pos, )
                if new_node != None:
                    new_node.load(node_container, offset=offset)
                    self.app.created_nodes[new_node.loaded_uuid] = new_node
                    dpg.focus_item(new_node.uuid)

        if self.loaded_parent_node_uuid != -1:
            parent_node = self.app.find_loaded_parent(self.loaded_parent_node_uuid)
            if parent_node is not None:
                parent_node.connect(self)

        if 'links' in file_container:
            self.app.links_containers[self.uuid] = file_container['links']

        dpg.configure_viewport(0, height=height, width=width)
        # dpg.configure_viewport(0, height=height, width=width, x_pos=int(position[0]), y_pos=int(position[1]))
        # dpg.set_viewport_pos(position)
        self.modified = False

    def load_(self, patch_container, path='', name=''):
        self.file_path = path
        self.patch_name = name
        self.uncontainerize(patch_container)
        if self.patch_name == '':
            self.patch_name = 'node patch'
        self.app.set_current_tab_title(self.patch_name)
        self.modified = False

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
                    self.app.add_to_recent(self.patch_name, path)
                    self.uncontainerize(file_container)
                    self.file_path = path
                    self.patch_name = self.file_path.split('/')[-1]
                    if '.' in self.patch_name:
                        parts = self.patch_name.split('.')
                        if len(parts) == 2:
                            if parts[1] == 'json':
                                self.patch_name = parts[0]
                    self.app.set_current_tab_title(self.patch_name)
                    self.modified = False
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
                self.node_scalers[dpg.mvNodeStyleVar_LinkThickness] = 2
                dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness, self.node_scalers[dpg.mvNodeStyleVar_LinkThickness], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeCol_Pin] = [30, 100, 150]
                dpg.add_theme_color(dpg.mvNodeCol_Pin, self.node_scalers[dpg.mvNodeCol_Pin], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, [255, 255, 0, 255], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvNodeCol_Link, [66, 150, 250, 100], category=dpg.mvThemeCat_Nodes)
        with dpg.theme() as self.node_presentation_theme:
            with dpg.theme_component(dpg.mvAll):
                self.node_scalers[dpg.mvNodeStyleVar_GridSpacing] = 16
                dpg.add_theme_style(dpg.mvNodeStyleVar_GridSpacing, self.node_scalers[dpg.mvNodeStyleVar_GridSpacing], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeCol_GridLine] = [60, 60, 60]
                dpg.add_theme_color(dpg.mvNodeCol_GridLine, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeStyleVar_NodePadding] = [4, 1]
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding,  self.node_scalers[dpg.mvNodeStyleVar_NodePadding][0],  self.node_scalers[dpg.mvNodeStyleVar_NodePadding][1], category=dpg.mvThemeCat_Nodes)
                self.node_scalers[dpg.mvNodeStyleVar_PinOffset] = 2
                dpg.add_theme_style(dpg.mvNodeStyleVar_PinOffset, self.node_scalers[dpg.mvNodeStyleVar_PinOffset], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_Link, [0.0, 0.0, 0.0, 0.0], category=dpg.mvThemeCat_Nodes)
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
        # dpg.bind_item_theme(dpg.last_item(), global_theme)

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

    def create(self, parent):
        with dpg.child_window(parent=parent, width=self._width, height=self._height, tag=self._uuid,
                              menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label)

            for child in self._children:
                child.create(child_parent)
