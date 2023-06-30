from dpg_system.body_base import *
import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
import os
import scipy
from scipy import signal
from dpg_system.node import Node
from dpg_system.conversion_utils import *

def register_smpl_nodes():
    Node.app.register_node("smpl_take", SMPLTakeNode.factory)
    Node.app.register_node("smpl_pose_to_joints", SMPLPoseToJointsNode.factory)

class SMPLNode(Node):
    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2',
        'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1',
        'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3',
        'right_index1', 'right_index2', 'right_index3', 'right_middle1',
        'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2',
        'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
        'right_thumb1', 'right_thumb2', 'right_thumb3'
    ]

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def load_smpl_file(self, in_path):
        try:
            data = np.load(in_path)
            return data['poses']
        except Exception as e:
            return None

    def extract_joint_data(self, pose_data):
        # joint_data_size = len(self.joint_names) * 3
        joint_data = pose_data[:, :22 * 3]
        return joint_data


class SMPLTakeNode(SMPLNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        speed = 1
        self.smpl_data = None
        self.joint_data = None
        self.frames = 0
        self.streaming = False
        self.current_frame = 0
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.load_button = self.add_property('load', widget_type='button', callback=self.load_take)
        self.file_name = self.add_label('')
        self.joint_data_out = self.add_output('joint_data')
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path, callback=self.load_smpl_callback)
        # self.message_handlers['load'] = self.load_take_message

    def start_stop_streaming(self):
        if self.on_off():
            if not self.streaming and self.load_path() != '':
                self.add_frame_task()
                self.streaming = True
        else:
            if self.streaming:
                self.remove_frame_tasks()
                self.streaming = False

    def frame_task(self):
        self.current_frame += self.speed()
        if self.current_frame >= self.frames:
            self.current_frame = 0
        self.input.set(self.current_frame)
        frame = int(self.current_frame)
        if self.joint_data is not None and frame < self.frames:
            self.joint_data_out.send(self.joint_data[frame])

    def load_smpl_callback(self):
        in_path = self.load_path()
        self.load_smpl(in_path)

    def load_smpl(self, in_path):
        if os.path.isfile(in_path):
            self.smpl_data = self.load_smpl_file(in_path)
            if self.smpl_data is not None:
                self.file_name.set(in_path.split('/')[-1])
                self.load_path.set(in_path)
                self.joint_data = self.extract_joint_data(self.smpl_data)
                self.joint_data = self.joint_data.reshape((self.joint_data.shape[0], self.joint_data.shape[1] // 3, 3))
                self.frames = self.joint_data.shape[0]
                self.current_frame = 0
                self.start_stop_streaming()

    def frame_widget_changed(self):
        data = self.input()
        if self.joint_data is not None and int(data) < self.frames:
            self.current_frame = int(data)
            self.joint_data_out.send(self.joint_data[self.current_frame])

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            if t == int:
                if self.joint_data is not None and int(data) < self.frames:
                    self.current_frame = int(data)
                    self.joint_data_out.send(self.joint_data[self.current_frame])

    def load_take(self, args=None):
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=640,
                             user_data=self, callback=self.load_npz_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".npz")

    def load_npz_callback(self, sender, app_data):
        if 'file_path_name' in app_data:
            load_path = app_data['file_path_name']
            if load_path != '':
                self.load_smpl(load_path)
        else:
            print('no file chosen')
        dpg.delete_item(sender)


class SMPLPoseToJointsNode(SMPLNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLPoseToJointsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.joint_offsets = []
        for index, key in enumerate(self.joint_names):
            if index < 22:
                self.joint_offsets.append(index)

        self.input = self.add_input('pose in', triggers_execution=True)
        self.output_as = self.add_property('output_as', widget_type='combo', default_value='quaternions')
        self.output_as.widget.combo_items = ['quaternions', 'euler angles']
        # self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.joint_outputs = []

        for index, key in enumerate(self.joint_names):
            if index < 22:
                stripped_key = key.replace('_', ' ')
                output = self.add_output(stripped_key)
                self.joint_outputs.append(output)

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input()
            output_quaternions = (self.output_as() == 'quaternions')

            t = type(incoming)
            if t == np.ndarray:
                for i, index in enumerate(self.joint_offsets):
                    if index < incoming.shape[0]:
                        joint_value = incoming[index]
                        if output_quaternions:
                            rot = scipy.spatial.transform.Rotation.from_euler('xyz', any_to_list(joint_value), degrees=False)
                            q = rot.as_quat()
                            joint_value = np.array([q[3], q[0], q[1], q[2]])
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()

