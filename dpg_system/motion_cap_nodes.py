import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import numpy as np
from dpg_system.body_base import *
#import torch

def register_mocap_nodes():
    Node.app.register_node('gl_body', MoCapGLBody.factory)
    Node.app.register_node('take', MoCapTakeNode.factory)
    Node.app.register_node('body_to_joints', MoCapBody.factory)


class MoCapNode(Node):
    joint_map = {
        'base_of_skull': 2,
        'upper_vertebrae': 17,
        'mid_vertebrae': 1,
        'lower_vertebrae': 32,
        'spine_pelvis': 31,
        'pelvis_anchor': 4,
        'left_hip': 14,
        'left_knee': 12,
        'left_ankle': 8,
        'right_hip': 28,
        'right_knee': 26,
        'right_ankle': 22,
        'left_shoulder_blade': 13,
        'left_shoulder': 5,
        'left_elbow': 9,
        'left_wrist': 10,
        'right_shoulder_blade': 27,
        'right_shoulder': 19,
        'right_elbow': 23,
        'right_wrist': 24
    }

    @staticmethod
    def factory(name, data, args=None):
        node = MoCapNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)


class MoCapTakeNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoCapTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.speed = 1
        self.buffer = None
        self.frames = 0
        self.current_frame = 0

        self.quat_buffer = None
        self.position_buffer = None
        self.label_buffer = None
        self.file_name = ''
        self.streaming = False

        self.on_off_property = self.add_property('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed_property = self.add_input('speed', widget_type='drag_float', default_value=self.speed, callback=self.speed_changed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.load_button = self.add_property('load', widget_type='button', callback=self.load_take)
        self.file_name_property = self.add_property('', widget_type='text_input')
        self.quaternions_out = self.add_output('quaternions')
        self.positions_out = self.add_output('positions')
        self.labels_out = self.add_output('labels')
        self.load_path = ''
        self.load_path_option = self.add_option('path', widget_type='text_input', default_value=self.load_path, callback=self.load_from_load_path)
        self.message_handlers['load'] = self.load_take_message

    def speed_changed(self):
        self.speed = self.speed_property.get_widget_value()

    def start_stop_streaming(self):
        if self.on_off_property.get_widget_value():
            if not self.streaming and self.load_path != '':
                self.add_frame_task()
                self.streaming = True
        else:
            if self.streaming:
                self.remove_frame_tasks()
                self.streaming = False

    def frame_task(self):
        self.current_frame += self.speed
        if self.current_frame > self.frames:
            self.current_frame = 0
        self.input.set(self.current_frame)
        frame = int(self.current_frame)
        self.quaternions_out.set_value(self.quat_buffer[frame])
        self.positions_out.set_value(self.position_buffer[frame])
        self.labels_out.set_value(self.label_buffer[frame])
        self.send_all()

    def load_from_load_path(self):
        path = self.load_path_option.get_widget_value()
        if path != '':
            self.load_take_from_npz(path)

    def load_take_from_npz(self, path):
        take_file = np.load(path)
        self.file_name = path.split('/')[-1]
        self.file_name_property.set(self.file_name)
        self.load_path = path
        self.load_path_option.set(self.load_path)
        self.quat_buffer = take_file['quats']
        for idx, quat in enumerate(self.quat_buffer):
            if quat[10, 0] < 0:
                self.quat_buffer[idx, 10] *= -1
        self.frames = self.quat_buffer.shape[0]
        self.position_buffer = take_file['positions']
        self.label_buffer = take_file['labels']
        self.current_frame = 0
        self.start_stop_streaming()

    def frame_widget_changed(self):
        data = self.input.get_widget_value()
        if data < self.frames:
            self.current_frame = data
            self.quaternions_out.set_value(self.quat_buffer[self.current_frame])
            self.positions_out.set_value(self.position_buffer[self.current_frame])
            self.labels_out.set_value(self.label_buffer[self.current_frame])
            self.send_all()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            if t == int:
                if data < self.frames:
                    self.current_frame = int(data)
                    self.quaternions_out.set_value(self.quat_buffer[self.current_frame])
                    self.positions_out.set_value(self.position_buffer[self.current_frame])
                    self.labels_out.set_value(self.label_buffer[self.current_frame])
                    self.send_all()

    def load_take_message(self, message='', args=[]):
        if len(args) > 0:
            path = any_to_string(args[0])
            self.load_take_from_npz(path)
        else:
            self.load_take(args)

    def load_take(self, args=None):
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400,
                             user_data=self, callback=self.load_npz_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".npz")

    def load_npz_callback(self, sender, app_data):
        # print('self=', self, 'sender=', sender, 'app_data=', app_data)
        if 'file_path_name' in app_data:
            self.load_path = app_data['file_path_name']
            if self.load_path != '':
                self.load_take_from_npz(self.load_path)
        else:
            print('no file chosen')
        dpg.delete_item(sender)

    # def load_take_from_pt_file(self, path):
    #     take_container = torch.jit.load(path)
    #     self.take_quats = take_container.quaternions.numpy()
    #     self.take_positions = take_container.positions.numpy()
    #     self.take_labels = take_container.labels.numpy()
    #
    # def save_take_as_pt_file(self, path):
    #     quat_tensor = torch.from_numpy(self.take_quats)
    #     position_tensor = torch.from_numpy(self.take_positions)
    #     label_tensor = torch.from_numpy(self.take_labels)
    #     d = {'quaternions': quat_tensor, 'positions': position_tensor, 'labels': label_tensor}
    #     container = torch.jit.script(Container(d))
    #     container.save(path + '_container.pt')
    #
    #     torch.save(d, path + '.pt')


class MoCapBody(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoCapBody(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.joint_offsets = []
        for key in self.joint_map:
            index = self.joint_map[key]
            self.joint_offsets.append(index)

        self.input = self.add_input('pose in', triggers_execution=True)
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.joint_outputs = []

        for key in self.joint_map:
            stripped_key = key.replace('_', ' ')
            output = self.add_output(stripped_key)
            self.joint_outputs.append(output)

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input.get_received_data()
            t = type(incoming)
            if t == np.ndarray:
                for i, index in enumerate(self.joint_offsets):
                    if index < incoming.shape[0]:
                        joint_value = incoming[index]
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()


class MoCapGLBody(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoCapGLBody(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.show_joint_activity = False
        self.input = self.add_input('pose in', triggers_execution=True)
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.gl_chain_output = self.add_output('gl_chain')
        self.show_joint_spheres_option = self.add_option('show joint motion', widget_type='checkbox', default_value=self.show_joint_activity)
        self.joint_motion_scale_option = self.add_option('joint motion scale', widget_type='drag_float', default_value=5)
        self.diff_quat_smoothing_option = self.add_option('joint motion smoothing', widget_type='drag_float', default_value=0.8, max=1.0, min=0.0)
        self.joint_disk_alpha_option = self.add_option('joint motion alpha', widget_type='drag_float', default_value=0.5, max=1.0, min=0.0)
        self.body = BodyData()
        self.body.node = self

    def joint_callback(self):
        self.gl_chain_output.send('draw')

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input.get_received_data()
            t = type(incoming)
            if t == np.ndarray:
                # work on this!!!
                if incoming.shape[0] == 37:
                    for joint_name in self.joint_map:
                        joint_id = self.joint_map[joint_name]
                        self.body.update(joint_index=joint_id, quat=incoming[joint_id])
        elif self.gl_chain_input.fresh_input:
            incoming = self.gl_chain_input.get_received_data()
            t = type(incoming)
            if t == str and incoming == 'draw':
                scale = self.joint_motion_scale_option.get_widget_value()
                smoothing = self.diff_quat_smoothing_option.get_widget_value()
                self.body.joint_motion_scale = scale
                self.body.diffQuatSmoothingA = smoothing
                self.body.joint_disk_alpha = self.joint_disk_alpha_option.get_widget_value()
                self.body.draw(self.show_joint_spheres_option.get_widget_value())
