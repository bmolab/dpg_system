import dearpygui.dearpygui as dpg
import math
import time
import numpy as np
import random
import traceback
from dpg_system.node import Node, OriginNode, PatcherNode
import json


class NodeEditor:
    app = None
    @staticmethod
    def _link_callback(sender, app_data, user_data):
        # Mouse-driven link create. (Programmatic add_child during load/paste does NOT
        # come through here — it goes via connect_link, which calls add_child directly.)
        NodeEditor.app.snapshot_for_undo()
        output_attr_uuid, input_attr_uuid = app_data
        input_attr = dpg.get_item_user_data(input_attr_uuid)
        output_attr = dpg.get_item_user_data(output_attr_uuid)
        output_attr.add_child(input_attr, sender)
        editor = NodeEditor.app.get_current_editor()
        if editor is not None:
            editor.modified = True

    @staticmethod
    def _unlink_callback(sender, app_data, user_data):
        NodeEditor.app.snapshot_for_undo()
        dat = dpg.get_item_user_data(app_data)
        out = dat[0]
        child = dat[1]
        out.remove_link(app_data, child)
        NodeEditor.app.get_current_editor().modified = True

    def __init__(self, height=0, width=0, top_level_patcher=True):
        self._nodes = []
        self.subpatches = []
        self.parent_patcher = None
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
        self.patch_name = 'patch ' + str(Node.app.new_patcher_index)
        self.file_path = ''
        self.mini_map = False
        self.origin = None
        self.patcher_node = None
        self.modified = False
        self.duplicated_subpatch_nodes = {}
        self.saving_preset = False
        self.presenting = False
        self.is_first_frame = True
        self._editor_padding = [0, 0]
        self._next_stable_id = 1

    def set_name(self, name):
        old_name = getattr(self, 'patch_name', None)
        self.patch_name = name
        self.app.set_editor_tab_title(self, name)
        
        # Notify nodes of the rename so they can update namespaces or proxy targets
        if old_name is not None and old_name != name:
            for node in self._nodes:
                if hasattr(node, 'patcher_name_changed'):
                    node.patcher_name_changed(old_name, name)

    def set_path(self, path):
        self.file_path = path

    def add_subpatch(self, subpatch_editor):
        self.subpatches.append(subpatch_editor)

    def align_selected(self):
        # find dominant axis
        # align to that axis
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        if len(selected_nodes) <= 1:
            return
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

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

        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

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
            real_x_max = float('-inf')
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
            bottom = bottom_acc / len(selected_nodes)
            sorted_dest = sorted(dest_dict.items(), key=lambda item: item[1][0][0])
            x_acc = x_min

            for index, dest_data in enumerate(sorted_dest):
                uuid = dest_data[1][1]
                if align == -1:
                    pos = [x_acc, y_mean]
                elif align == 0:
                    pos = [x_acc, center - dest_data[1][2][1] / 2]
                elif align == 1:
                    pos = [x_acc, bottom - dest_data[1][2][1]]
                else:
                    pos = [x_acc, dest_data[1][0][1]]

                x_acc += (dest_data[1][2][0] + gap)
                dpg.set_item_pos(uuid, pos)
        else:
            x_mean = (x_max + x_min) / 2

            dest_dict = {}
            total_node_height = 0
            real_y_max = float('-inf')
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
        x_min = float('inf')
        x_max = float('-inf')
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

        # Filter so downstream connect_* methods can safely index outputs[0]/inputs[0].
        # Sources need at least one output; dests need at least one input.
        source_nodes = [n for n in source_nodes if n is not None and len(n.outputs) > 0]
        dest_nodes = [n for n in dest_nodes if n is not None and len(n.inputs) > 0]
        if len(source_nodes) == 0 or len(dest_nodes) == 0:
            return
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
        if node.stable_id is None:
            node.stable_id = self._next_stable_id
            self._next_stable_id += 1
        else:
            # Externally set (file load, snapshot recreate). Keep our counter ahead.
            if node.stable_id >= self._next_stable_id:
                self._next_stable_id = node.stable_id + 1
        self.modified = True

    def first_frame(self):
        for node in self._nodes:
            node.first_frame()
        # Capture the internal padding between editor widget and content area.
        # On the first frame, panning is [0, 0] and origin is at [0, 0],
        # so the difference in screen coords IS the padding.
        if self.origin is not None:
            try:
                editor_screen = dpg.get_item_rect_min(self.uuid)
                origin_screen = dpg.get_item_rect_min(self.origin.uuid)
                self._editor_padding = [
                    origin_screen[0] - editor_screen[0],
                    origin_screen[1] - editor_screen[1]
                ]
            except Exception as e:
                print(f'first_frame: editor padding probe failed: {type(e).__name__}: {e}')
                self._editor_padding = [0, 0]
        self.is_first_frame = False

    def remove_all_nodes(self):
        self.app.clear_remembered_ids()
        for node in self._nodes:
            node.cleanup()
            dpg.delete_item(node.uuid)
        self._nodes = []
        self.num_nodes = len(self._nodes)
        self.modified = True
        self.active_pins = []

    def node_cleanup(self, node_uuid):
        node = dpg.get_item_user_data(node_uuid)
        if node is not None:
            self.remove_node(node)
        # for node in self._nodes:
        #     if node.uuid == node_uuid:
        #         if node != self.origin:
        #             self.remove_node(node)
        #         break

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
                # i don;t think there are ever any nodes to create here?
                for node in self._nodes:
                    node.create(self.uuid)

        dpg.bind_theme(self.node_theme)
        # add invisible node that is reference for panning

        self.origin = OriginNode.factory('origin', None)
        self.origin.do_not_delete = True
        self.origin.presentation_state = 'hidden'
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

    def apply_snapshot(self, snap):
        """Reconcile this editor against a previously captured snapshot.
        Nodes that exist in both: update in place via node.load() — no teardown,
        runtime state preserved.
        Nodes only in snapshot: recreated.
        Nodes only live: deleted (origin is never deleted).
        Links: set-diffed by (src_id, src_out_idx, dst_id, dst_in_idx).
        """
        if snap is None or 'nodes' not in snap:
            return

        live_by_sid = {n.stable_id: n for n in self._nodes if n.stable_id is not None}
        origin_sid = self.origin.stable_id if self.origin is not None else None

        snap_nodes = snap['nodes']
        snap_sids = set()
        # snap_sid -> live_uuid (after reconcile). Used by the link pass to translate
        # snapshot-side node references to current DPG uuids.
        sid_to_uuid = {}

        # Pass 1: matched + snap-only
        for nc in snap_nodes.values():
            snap_sid = nc.get('sid')
            if snap_sid is None:
                continue
            snap_sids.add(snap_sid)
            live = live_by_sid.get(snap_sid)
            if live is not None:
                # Update in place. node.load() handles position, visibility, properties.
                try:
                    live.load(nc)
                except Exception as e:
                    print(f'apply_snapshot: load failed for {live.label} (sid={snap_sid}): '
                          f'{type(e).__name__}: {e}')
                    traceback.print_exc()
                sid_to_uuid[snap_sid] = live.uuid
            else:
                # Recreate. node.load() will pick up sid from nc and re-establish stable_id.
                if nc.get('name') == '':
                    continue  # origin node sentinel; never recreate
                pos = [nc.get('position_x', 0), nc.get('position_y', 0)]
                args = []
                if 'init' in nc:
                    args = nc['init'].split(' ')
                node_name = args[0] if args else nc.get('name', '')
                node_args = args[1:] if len(args) > 1 else []
                try:
                    new_node = self.app.create_node_by_name_from_file(node_name, pos, node_args)
                    if new_node is not None:
                        new_node.load(nc)
                        new_node.post_creation_callback()
                        sid_to_uuid[snap_sid] = new_node.uuid
                except Exception as e:
                    print(f'apply_snapshot: create failed for {node_name} (sid={snap_sid}): '
                          f'{type(e).__name__}: {e}')
                    traceback.print_exc()

        # Pass 2: live-only -> delete (skip origin and protected)
        for live_sid, live in list(live_by_sid.items()):
            if live_sid in snap_sids:
                continue
            if live_sid == origin_sid or live.do_not_delete:
                continue
            try:
                self.remove_node(live)
            except Exception as e:
                print(f'apply_snapshot: remove failed for {live.label} (sid={live_sid}): '
                      f'{type(e).__name__}: {e}')
                traceback.print_exc()

        # Pass 3: reconcile links
        # Build live tuples: (src_uuid, src_out_idx, dst_uuid, dst_in_idx) -> (output, link_uuid, dest_input)
        live_links = {}
        for n in self._nodes:
            for out_idx, output in enumerate(n.outputs):
                for child_idx, child in enumerate(output._children):
                    dest_node = child.node
                    dest_in_idx = None
                    for i, inp in enumerate(dest_node.inputs):
                        if inp.uuid == child.uuid:
                            dest_in_idx = i
                            break
                    if dest_in_idx is None:
                        continue
                    link_uuid = output.links[child_idx] if child_idx < len(output.links) else None
                    key = (n.uuid, out_idx, dest_node.uuid, dest_in_idx)
                    live_links[key] = (output, link_uuid, child)

        # snap link entries reference DPG uuids from save time; translate uuid -> sid -> current live uuid.
        snap_uuid_to_sid = {nc.get('id'): nc.get('sid')
                            for nc in snap_nodes.values()
                            if nc.get('id') is not None and nc.get('sid') is not None}
        snap_link_keys = set()
        for lc in snap.get('links', {}).values():
            src_sid = snap_uuid_to_sid.get(lc.get('source_node'))
            dst_sid = snap_uuid_to_sid.get(lc.get('dest_node'))
            src_uuid = sid_to_uuid.get(src_sid)
            dst_uuid = sid_to_uuid.get(dst_sid)
            if src_uuid is None or dst_uuid is None:
                continue
            key = (src_uuid, lc.get('source_output_index'), dst_uuid, lc.get('dest_input_index'))
            snap_link_keys.add(key)

        # Remove live links not in snap
        for key, (output, link_uuid, dest_input) in live_links.items():
            if key not in snap_link_keys and link_uuid is not None:
                try:
                    output.remove_link(link_uuid, dest_input)
                except Exception as e:
                    print(f'apply_snapshot: remove_link failed for key={key}: '
                          f'{type(e).__name__}: {e}')
                    traceback.print_exc()

        # Add snap links not in live
        live_by_uuid = {n.uuid: n for n in self._nodes}  # rebuild after deletes/creates
        for key in snap_link_keys:
            if key in live_links:
                continue
            src_uuid, src_out_idx, dst_uuid, dst_in_idx = key
            src = live_by_uuid.get(src_uuid)
            dst = live_by_uuid.get(dst_uuid)
            if src is None or dst is None:
                continue
            if src_out_idx >= len(src.outputs) or dst_in_idx >= len(dst.inputs):
                continue
            try:
                src.outputs[src_out_idx].add_child(dst.inputs[dst_in_idx], self.uuid)
            except Exception as e:
                print(f'apply_snapshot: add_link failed for key={key}: '
                      f'{type(e).__name__}: {e}')
                traceback.print_exc()

        self.modified = True

    def capture_snapshot(self):
        """Capture a snapshot dict suitable for apply_snapshot().
        Positions are stored in absolute (un-origin-adjusted) form so undo doesn't
        fight with intervening pans."""
        snap = self.containerize({})
        if self.origin is not None and 'nodes' in snap:
            try:
                origin_offset = dpg.get_item_pos(self.origin.uuid)
                if abs(origin_offset[0]) > 0.5 or abs(origin_offset[1]) > 0.5:
                    for nc in snap['nodes'].values():
                        if 'position_x' in nc:
                            nc['position_x'] += origin_offset[0]
                            nc['position_y'] += origin_offset[1]
            except Exception:
                pass
        return snap

    def pan_nodes(self, dx, dy):
        """Shift every node by (dx, dy). Positive dx moves nodes right, positive dy moves nodes down."""
        if len(self._nodes) == 0 or (dx == 0 and dy == 0):
            return
        for node in self._nodes:
            try:
                pos = dpg.get_item_pos(node.uuid)
                dpg.set_item_pos(node.uuid, [pos[0] + dx, pos[1] + dy])
            except Exception as e:
                print(f'pan_nodes: failed to move node {getattr(node, "label", "?")} '
                      f'(uuid={node.uuid}): {type(e).__name__}: {e}')

    def home_nodes(self):
        """Shift all nodes so the origin node appears at the top-left of the editor.
        All nodes keep their relative positions to each other."""
        if self.origin is None or len(self._nodes) == 0:
            return

        try:
            # Both are viewport screen coordinates — directly comparable
            editor_screen = dpg.get_item_rect_min(self.uuid)
            origin_screen = dpg.get_item_rect_min(self.origin.uuid)

            # How far the origin is from the editor content area
            # _editor_padding accounts for internal padding between editor rect and content
            shift_x = origin_screen[0] - editor_screen[0] - self._editor_padding[0]
            shift_y = origin_screen[1] - editor_screen[1] - self._editor_padding[1]

            if abs(shift_x) < 1 and abs(shift_y) < 1:
                return  # Already home

            # Move all nodes so origin lands at editor top-left
            for node in self._nodes:
                try:
                    pos = dpg.get_item_pos(node.uuid)
                    dpg.set_item_pos(node.uuid, [pos[0] - shift_x, pos[1] - shift_y])
                except Exception as e:
                    print(f'home_nodes: failed to move node {getattr(node, "label", "?")} '
                          f'(uuid={node.uuid}): {type(e).__name__}: {e}')
        except Exception as e:
            print(f'home_nodes error: {type(e).__name__}: {e}')

    def cut_selection(self):
        clip = self.copy_selection()
        node_uuids = dpg.get_selected_nodes(self.uuid)

        self.delete_selected_items()
        # for node_uuid in node_uuids:
        #     self.node_cleanup(node_uuid)
        #
        # link_uuids = dpg.get_selected_links(self.uuid)
        # for link_uuid in link_uuids:
        #     if dpg.does_item_exist(link_uuid):
        #         dat = dpg.get_item_user_data(link_uuid)
        #         out = dat[0]
        #         child = dat[1]
        #         if len(node_uuids) == 0 or out.node.uuid in node_uuids or child.node.uuid in node_uuids:
        #         #  remove only if link connects to selected node or no selected node
        #             out.remove_link(link_uuid, child)
        return clip

    def delete_selected_items(self):
        node_uuids = dpg.get_selected_nodes(self.uuid)
        for node_uuid in node_uuids:
            node = dpg.get_item_user_data(node_uuid)
            if node is not None and node.visibility == 'show_all' and not node.do_not_delete: # do not delete invisible nodes
                self.node_cleanup(node_uuid)

        link_uuids = dpg.get_selected_links(self.uuid)
        for link_uuid in link_uuids:
            if dpg.does_item_exist(link_uuid):
                dat = dpg.get_item_user_data(link_uuid)
                out = dat[0]
                child = dat[1]
                if len(node_uuids) == 0 or out.node.uuid in node_uuids or child.node.uuid in node_uuids:
                #  remove only if link connects to selected node or no selected node
                    out.remove_link(link_uuid, child)


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
        try:
            file_container = {}
            nodes_container = {}
            selected_node_objects = []
            selected_nodes = dpg.get_selected_nodes(self.uuid)
            subpatch_container = {}

            def _is_copyable(node):
                # Origin (label == '') and other protected singletons must not be copied.
                # uncontainerize() also skips empty-name nodes on the load side, but
                # excluding them here keeps them out of the link-enumeration pass below
                # and avoids serializing useless container entries.
                if node.label == '' or getattr(node, 'do_not_delete', False):
                    return False
                return True

            # NOTE - sub-patcher objects should not move with mouse
            for index, node in enumerate(self._nodes):
                if node.uuid in selected_nodes and _is_copyable(node):
                    if node.label == 'patcher':
                        if node.patch_editor is not None:
                            patch_container = {}
                            node.patch_editor.containerize(patch_container)
                            subpatch_container[index] = patch_container
                            # we need to remove all subpatch nodes from drag list...

            if len(subpatch_container) > 0:
                file_container['patches'] = subpatch_container
            for index, node in enumerate(self._nodes):
                if node.uuid in selected_nodes and _is_copyable(node):
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
                            link_container['source_node_name'] = node.label
                            link_container['source_output_index'] = out_index
                            link_container['source_output_name'] = output.get_label()
                            dest_node = input.node
                            link_container['dest_node'] = dest_node.uuid
                            link_container['dest_node_name'] = dest_node.label
                            for node_in_index, test_input in enumerate(dest_node.inputs):
                                if test_input.uuid == input.uuid:
                                    link_container['dest_input_index'] = node_in_index
                                    link_container['dest_input_name'] = test_input.get_label()
                                    links_container[link_index] = link_container
                                    link_index += 1
                                    break
            file_container['links'] = links_container
            # dpg.clear_selected_nodes(self.uuid)
            return file_container
        finally:
            self.app.loading = False

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
        try:
            if len(file_container) == 0:
                return

            # Pasted/duplicated nodes are NEW nodes — must not inherit stable_ids from the source.
            if 'nodes' in file_container:
                for nc in file_container['nodes'].values():
                    if isinstance(nc, dict):
                        nc.pop('sid', None)

            self.uncontainerize(file_container, create_origin=origin)
            for node_editor_uuid in self.app.links_containers:
                new_links = {}
                links_container = self.app.links_containers[node_editor_uuid]
                for index, link_index in enumerate(links_container):
                    new_link = self.app.connect_link(links_container, index, link_index, node_editor_uuid)
                    new_links[link_index] = new_link
                    self.app.links_containers[node_editor_uuid] = new_links.copy()

            if clear_loaded_uuids:
                for uuid in self.app.created_nodes:
                    node = self.app.created_nodes[uuid]
                    if node is not None:
                        node.post_load_callback()

                self.clear_loaded_uuids()

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
                    left_top = [left_most + 30, top_most + 10]
                    self.app.dragging_ref = self.editor_pos_to_global_pos(left_top)
                    self.modified = True
                    # Clear loading before drag_create_nodes — callee depends on the flag.
                    self.app.loading = False
                    self.app.drag_create_nodes()
            # else:
            #     print('no drag starts')
        finally:
            self.app.loading = False

    def _origin_widget_uuid(self):
        # Resolve self.origin.ref_property.widget.uuid safely; returns None if any
        # link in the chain is missing (e.g. before create() finishes or during teardown).
        if self.origin is None:
            return None
        ref_property = getattr(self.origin, 'ref_property', None)
        if ref_property is None:
            return None
        widget = getattr(ref_property, 'widget', None)
        if widget is None:
            return None
        return widget.uuid

    def _safe_get_item_pos(self, uuid, label):
        # Returns [x, y] or None if the item is gone / DPG state is unqueryable.
        # SystemError covers the "<built-in function get_item_state> returned a result
        # with an exception set" case where DPG's C side is in a bad spot.
        if uuid is None or not dpg.does_item_exist(uuid):
            return None
        try:
            return dpg.get_item_pos(uuid)
        except (SystemError, Exception) as e:
            print(f'_safe_get_item_pos({label}, uuid={uuid}) failed: {type(e).__name__}: {e}')
            return None

    def _validate_origin(self):
        # If self.origin points at a node whose DPG item is gone, clear the reference
        # so subsequent geometry queries don't repeatedly hit the same bad UUID.
        if self.origin is None:
            return
        origin_uuid = getattr(self.origin, 'uuid', None)
        if origin_uuid is None or not dpg.does_item_exist(origin_uuid):
            print(f'_validate_origin: dropping stale origin (uuid={origin_uuid})')
            self.origin = None

    def global_pos_to_editor_pos(self, pos):
        self._validate_origin()
        panel_pos = self._safe_get_item_pos(self.app.center_panel, 'center_panel') or [0, 0]
        origin_widget_uuid = self._origin_widget_uuid()
        if origin_widget_uuid is None or self.origin is None:
            editor_mouse_pos = pos
            editor_mouse_pos[0] -= (panel_pos[0] + 8 - 4)
            editor_mouse_pos[1] -= (panel_pos[1] + 8 - 15)
            return editor_mouse_pos
        origin_pos = self._safe_get_item_pos(origin_widget_uuid, 'origin_widget')
        origin_node_pos = self._safe_get_item_pos(self.origin.uuid, 'origin_node')
        if origin_pos is None or origin_node_pos is None:
            # Fall back to the no-origin formula rather than propagating bad state.
            editor_mouse_pos = pos
            editor_mouse_pos[0] -= (panel_pos[0] + 8 - 4)
            editor_mouse_pos[1] -= (panel_pos[1] + 8 - 15)
            return editor_mouse_pos
        editor_mouse_pos = pos
        editor_mouse_pos[0] -= (panel_pos[0] + 8 + (origin_pos[0] - origin_node_pos[0]) - 4)
        editor_mouse_pos[1] -= (panel_pos[1] + 8 + (origin_pos[1] - origin_node_pos[1]) - 15)
        return editor_mouse_pos

    def editor_pos_to_global_pos(self, pos):
        self._validate_origin()
        panel_pos = self._safe_get_item_pos(self.app.center_panel, 'center_panel') or [0, 0]
        origin_widget_uuid = self._origin_widget_uuid()
        if origin_widget_uuid is None or self.origin is None:
            global_pos = pos
            global_pos[0] += (panel_pos[0] + 8 - 4)
            global_pos[1] += (panel_pos[1] + 8 - 30)
            return global_pos
        origin_pos = self._safe_get_item_pos(origin_widget_uuid, 'origin_widget')
        origin_node_pos = self._safe_get_item_pos(self.origin.uuid, 'origin_node')
        if origin_pos is None or origin_node_pos is None:
            global_pos = pos
            global_pos[0] += (panel_pos[0] + 8 - 4)
            global_pos[1] += (panel_pos[1] + 8 - 30)
            return global_pos
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
            if node.label != '':
                node.set_visibility(node.presentation_state)
                node.set_draggable(False)
        dpg.bind_theme(self.node_presentation_theme)

    def enter_edit_state(self):
        self.presenting = False
        for node in self._nodes:
            if node.label != '':
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
                self.app.connect_link(links_container, index, link_index, sub_patch_editor.uuid)
                # source_node = None
                # dest_node = None
                # link_container = links_container[link_index]
                # source_node_loaded_uuid = link_container['source_node']
                # if source_node_loaded_uuid in self.app.created_nodes:
                #     source_node = self.app.created_nodes[source_node_loaded_uuid]
                # dest_node_loaded_uuid = link_container['dest_node']
                # if dest_node_loaded_uuid in self.app.created_nodes:
                #     dest_node = self.app.created_nodes[dest_node_loaded_uuid]
                # if source_node is not None and dest_node is not None:
                #     source_output_index = link_container['source_output_index']
                #     dest_input_index = link_container['dest_input_index']
                #     if source_output_index < len(source_node.outputs):
                #         source_output = source_node.outputs[source_output_index]
                #         if dest_input_index < len(dest_node.inputs):
                #             dest_input = dest_node.inputs[dest_input_index]
                #             source_output.add_child(dest_input, sub_patch_editor.uuid)
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

        # Correct saved positions to be relative to origin at [0, 0].
        # The origin node may have accumulated an offset from home_nodes().
        if self.origin is not None:
            origin_offset = dpg.get_item_pos(self.origin.uuid)
            if abs(origin_offset[0]) > 0.5 or abs(origin_offset[1]) > 0.5:
                for node_key in nodes_container:
                    nc = nodes_container[node_key]
                    if 'position_x' in nc:
                        nc['position_x'] -= origin_offset[0]
                        nc['position_y'] -= origin_offset[1]

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
                        link_container['source_node_name'] = node.label
                        link_container['source_output_index'] = out_index
                        link_container['source_output_name'] = output.get_label()
                        dest_node = input.node
                        link_container['dest_node'] = dest_node.uuid
                        link_container['dest_node_name'] = dest_node.label
                        for node_in_index, test_input in enumerate(dest_node.inputs):
                            if test_input.uuid == input.uuid:
                                link_container['dest_input_index'] = node_in_index
                                link_container['dest_input_name'] = test_input.get_label()
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

        old_patch_name = getattr(self, 'patch_name', None)

        self.patch_name = self._patch_name_from_path(path)
        self.file_path = path

        # Broadcast the new name to nodes BEFORE we serialize their JSON state.
        # This allows proxy widgets targeting the old patcher alias to organically rewrite their
        # internal `target_name_property`, meaning they save the *new* name to disk.
        if old_patch_name is not None and old_patch_name != self.patch_name:
            for node in self._nodes:
                if hasattr(node, 'patcher_name_changed'):
                    node.patcher_name_changed(old_patch_name, self.patch_name)

        try:
            file_container = self.containerize()
            with open(path, 'w') as f:
                json.dump(file_container, f, indent=4)
        except Exception as e:
            # Keep self.modified = True so the user knows the patch is unsaved.
            print(f'save: failed to write {path}: {type(e).__name__}: {e}')
            traceback.print_exc()
            return
        self.app.set_current_tab_title(self.patch_name)
        self.modified = False

    def is_root_patcher(self):
        return self.parent_patcher is not None

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
        self.app.currently_loading_patch_name = self.patch_name

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
                            self.app.currently_loading_node_name = node_name
                pos = [0, 0]
                if 'position_x' in node_container:
                    pos[0] = node_container['position_x'] + offset[0]
                if 'position_y' in node_container:
                    pos[1] = node_container['position_y'] + offset[1]
                args = []
                if 'init' in node_container:
                    args_container = node_container['init']
                    # Skip the split when init is empty — ''.split(' ') == [''] which would
                    # otherwise route into the len==1 branch and call create with name=''.
                    if args_container != '':
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
                    try:
                        new_node.load(node_container, offset=offset)
                        self.app.created_nodes[new_node.loaded_uuid] = new_node
                        dpg.focus_item(new_node.uuid)
                        new_node.post_creation_callback()
                    except Exception as e:
                        print('error loading node', new_node.label, e)

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
        self.is_first_frame = True

    def load_(self, patch_container, path='', name=''):
        self.file_path = path
        self.patch_name = name
        self.uncontainerize(patch_container)
        if self.patch_name == '':
            self.patch_name = 'patch ' + str(Node.app.new_patcher_index)
            Node.app.new_patcher_index += 1
        self.app.set_current_tab_title(self.patch_name)
        self.modified = False
        self.is_first_frame = True

    def _patch_name_from_path(self, path):
        # Strip directory and a trailing .json extension to produce the display name.
        name = path.split('/')[-1]
        if '.' in name:
            parts = name.split('.')
            if len(parts) == 2 and parts[1] == 'json':
                name = parts[0]
        return name

    def load(self, path=''):
        try:
            if len(path) > 0:
                with open(path, 'r') as f:
                    file_container = json.load(f)
                    self.file_path = path
                    self.patch_name = self._patch_name_from_path(path)
                    self.app.add_to_recent(self.patch_name, path)
                    self.uncontainerize(file_container)
                    # uncontainerize() resets file_path / patch_name from the file
                    # container — restore the on-disk path/name so the editor reflects
                    # where it was actually loaded from.
                    self.file_path = path
                    self.patch_name = self._patch_name_from_path(path)
                    self.app.set_current_tab_title(self.patch_name)
                    self.modified = False
                    self.is_first_frame = True
        except Exception as e:
            print(f'exception occurred during load of {path}: {type(e).__name__}: {e}')
            traceback.print_exc()

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
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 255, 0, 128), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 4, category=dpg.mvThemeCat_Core)
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
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, [255, 255, 0, 255], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 255, 0, 128), category=dpg.mvThemeCat_Core)
                self.node_scalers[dpg.mvNodeCol_Pin] = [30, 100, 150]
                dpg.add_theme_color(dpg.mvNodeCol_Pin, self.node_scalers[dpg.mvNodeCol_Pin], category=dpg.mvThemeCat_Nodes)


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
