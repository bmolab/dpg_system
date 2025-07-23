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
import pickle
from dpg_system.conversion_utils import *
import pickle
import chumpy as ch
from chumpy.ch import MatVecMult
import cv2
import scipy.sparse as sp
import torch.utils.data as tudata
from pathlib import Path
import hashlib
from _hashlib import HASH as Hash
import json
import argparse
import functools
import os
from shutil import unpack_archive
import joblib
from tqdm.auto import tqdm
import gzip
import scipy



def register_smpl_nodes():
    Node.app.register_node("smpl_take", SMPLTakeNode.factory)
    Node.app.register_node("smpl_pose_to_joints", SMPLPoseToJointsNode.factory)
    Node.app.register_node("smpl_body", SMPLBodyNode.factory)

    Node.app.register_node("smpl_quats_to_joints", SMPLPoseQuatsToJointsNode.factory)

    Node.app.register_node("smpl_to_active", SMPLToActivePoseNode.factory)
    Node.app.register_node("active_to_smpl", ActiveToSMPLPoseNode.factory)



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

    def load_smpl_take_file(self, in_path):
        try:
            data = np.load(in_path)
            print(list(data.keys()))
            print(data['trans'].shape)
            return data['poses'], data['trans']
        except Exception as e:
            return None, None

    def extract_joint_data(self, pose_data):
        # joint_data_size = len(self.joint_names) * 3
        joint_data = pose_data[:, :22 * 3]
        return joint_data

    def load_SMPL_R_model_file(self, in_path):
        suffix = in_path.split('.')[-1]
        try:
            if suffix == 'npz':
                data = np.load(in_path)
                return data, 'npz'
            elif suffix == 'pkl':
                data = load_model(in_path)
                # data = pickle.load(open(in_path, 'rb'), encoding='latin1')
                return data, 'pkl'
        except Exception as e:
            print('load_SMPL_R_model_file failed', e)
            return None
        return None


class SMPLShadowTranslator():
    smpl_joints = {
        'pelvis': 0,
        'left_hip': 1,
        'right_hip': 2,
        'spine1': 3,
        'left_knee': 4,
        'right_knee': 5,
        'spine2': 6,
        'left_ankle': 7,
        'right_ankle': 8,
        'spine3': 9,
        'left_foot': 10,
        'right_foot': 11,
        'neck': 12,
        'left_collar': 13,
        'right_collar': 14,
        'head': 15,
        'left_shoulder': 16,
        'right_shoulder': 17,
        'left_elbow': 18,
        'right_elbow': 19,
        'left_wrist': 20,
        'right_wrist': 21
    }

    active_joints = {
        'base_of_skull': 0,
        'upper_vertebrae': 1,
        'mid_vertebrae': 2,
        'lower_vertebrae': 3,
        'spine_pelvis': 4,
        'pelvis_anchor': 5,
        'left_hip': 6,
        'left_knee': 7,
        'left_ankle': 8,
        'right_hip': 9,
        'right_knee': 10,
        'right_ankle': 11,
        'left_shoulder_blade': 12,
        'left_shoulder': 13,
        'left_elbow': 14,
        'left_wrist': 15,
        'right_shoulder_blade': 16,
        'right_shoulder': 17,
        'right_elbow': 18,
        'right_wrist': 19
    }

    smpl_to_active_joint_map = {
        'head': 'base_of_skull',
        'neck': 'upper_vertebrae',
        'spine3': 'mid_vertebrae',
        'spine2': 'lower_vertebrae',
        'spine1': 'spine_pelvis',
        'pelvis': 'pelvis_anchor',
        'left_hip': 'left_hip',
        'left_knee': 'left_knee',
        'left_ankle': 'left_ankle',
        'right_hip': 'right_hip',
        'right_knee': 'right_knee',
        'right_ankle': 'right_ankle',
        'left_collar': 'left_shoulder_blade',
        'left_shoulder': 'left_shoulder',
        'left_elbow': 'left_elbow',
        'left_wrist': 'left_wrist',
        'right_collar': 'right_shoulder_blade',
        'right_shoulder': 'right_shoulder',
        'right_elbow': 'right_elbow',
        'right_wrist': 'right_wrist'
    }

    smpl_from_active_joint_map = {
        'head': 'base_of_skull',
        'neck': 'upper_vertebrae',
        'spine3': 'mid_vertebrae',
        'spine2': 'lower_vertebrae',
        'spine1': 'spine_pelvis',
        'pelvis': 'pelvis_anchor',
        'left_hip': 'left_hip',
        'left_knee': 'left_knee',
        'left_ankle': 'left_ankle',
        'left_foot': 'empty',
        'right_hip': 'right_hip',
        'right_knee': 'right_knee',
        'right_ankle': 'right_ankle',
        'right_foot': 'empty',
        'left_collar': 'left_shoulder_blade',
        'left_shoulder': 'left_shoulder',
        'left_elbow': 'left_elbow',
        'left_wrist': 'left_wrist',
        'right_collar': 'right_shoulder_blade',
        'right_shoulder': 'right_shoulder',
        'right_elbow': 'right_elbow',
        'right_wrist': 'right_wrist'
    }

    @staticmethod
    def translate_from_smpl_to_active(smpl_pose): #  expects n x 3 in, outputs 20 x 3
        output_size = len(SMPLShadowTranslator.smpl_to_active_joint_map)
        active_pose = np.zeros((output_size, smpl_pose.shape[-1]), dtype=np.float32)

        for smpl_joint in SMPLShadowTranslator.smpl_to_active_joint_map:
            smpl_index = SMPLShadowTranslator.smpl_joints[smpl_joint]
            active_joint = SMPLShadowTranslator.smpl_to_active_joint_map[smpl_joint]
            active_index = SMPLShadowTranslator.active_joints[active_joint]
            active_pose[active_index] = smpl_pose[smpl_index]
        return active_pose

    @staticmethod
    def translate_from_active_to_smpl(active_pose): #  expects 20 x 3 in, outputs 20 x 3
        output_size = len(SMPLShadowTranslator.smpl_from_active_joint_map)
        smpl_pose = np.zeros((output_size, active_pose.shape[-1]), dtype=np.float32)

        if active_pose.shape[1] == 3:
            empty = [0.0, 0.0, 0.0]
        elif active_pose.shape[1] == 4:
            empty = [1.0, 0.0, 0.0, 0.0]
        elif active_pose.shape[1] == 2:
            empty = [0.0, 0.0]
        elif active_pose.shape[1] == 2:
            empty = [0.0]

        for smpl_joint in SMPLShadowTranslator.smpl_from_active_joint_map:
            smpl_index = SMPLShadowTranslator.smpl_joints[smpl_joint]
            active_joint = SMPLShadowTranslator.smpl_from_active_joint_map[smpl_joint]
            if active_joint in SMPLShadowTranslator.active_joints:
                active_index = SMPLShadowTranslator.active_joints[active_joint]
                smpl_pose[smpl_index] = active_pose[active_index]
            else:
                smpl_pose[smpl_index] = empty
        return smpl_pose

    def __init__(self, label, data, args):
        pass

    
class SMPLToActivePoseNode(SMPLShadowTranslator, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLToActivePoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        SMPLShadowTranslator.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.input = self.add_input('smpl pose', triggers_execution=True)
        self.output_format_in = self.add_input('output_format', widget_type='combo', default_value='quaternions')
        self.output_format_in.widget.combo_items = ['quaternions', 'rotation_vectors', 'generic']
        self.y_is_up = self.add_property('y is up', widget_type='checkbox')
        self.output = self.add_output('active pose')
        self.y_up = np.array([0.7071067811865475, 0.0, -0.7071067811865475, 0.0])

    def execute(self):
        smpl_pose = self.input()
        smpl_pose = any_to_array(smpl_pose)
        if len(smpl_pose.shape) == 1:
            smpl_pose = np.reshape(smpl_pose, (-1, 3))
        if self.output_format_in() == 'quaternions':
            active_pose = np.zeros((22, 4))
            active_pose[:, 0] = 1.0
        elif self.output_format_in() == 'rotation_vectors':
            active_pose = np.zeros((22, 3))
        else:
            if len(smpl_pose.shape) == 1:
                smpl_pose = np.expand_dims(smpl_pose, axis=1)
            active_pose = np.zeros((22, smpl_pose.shape[-1]))

        # NOTE: smpl seems to assume z is up vector which messes up axes of root rotation
        # we should force root rotation to rotate -90 around x axis

        if len(smpl_pose.shape) > 1:
            if smpl_pose.shape[1] == 3:
                # if self.y_is_up():
                #     smpl_pose[0] = rotate_vector_rodrigues(smpl_pose[0], np.array([1.0, 0.0, 0.0]), -90)
                active_pose = SMPLShadowTranslator.translate_from_smpl_to_active(smpl_pose)
                if self.output_format_in() == 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_rotvec(active_pose)
                    active_pose = rot.as_quat(scalar_first=True)
                    if self.y_is_up():
                        active_pose[5] = active_pose[5] * self.y_up
            elif smpl_pose.shape[1] == 4:
                active_pose = SMPLShadowTranslator.translate_from_smpl_to_active(smpl_pose)
                if self.output_format_in() == 'rotation_vectors':
                    rot = scipy.spatial.transform.Rotation.from_quat(active_pose, scalar_first=True)
                    active_pose = rot.as_rotvec()
                elif self.y_is_up():
                    active_pose[5] = active_pose[5] * self.y_up
            else:
                active_pose = SMPLShadowTranslator.translate_from_smpl_to_active(smpl_pose)

        self.output.send(active_pose)

def rotate_vector_rodrigues(v, k, theta):
    """
    Rotates a 3D vector 'v' around an axis 'k' by an angle 'theta' using Rodrigues' rotation formula.

    Args:
        v (np.array): The 3D vector to be rotated.
        k (np.array): The 3D axis of rotation (will be normalized).
        theta (float): The angle of rotation in radians.

    Returns:
        np.array: The rotated 3D vector.
    """
    v = np.asarray(v)
    k = np.asarray(k) / np.linalg.norm(k)  # Normalize the axis vector to a unit vector

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rodrigues' formula components
    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1 - cos_theta)

    return term1 + term2 + term3

class ActiveToSMPLPoseNode(SMPLShadowTranslator, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ActiveToSMPLPoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        SMPLShadowTranslator.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.input = self.add_input('smpl pose', triggers_execution=True)
        self.output_format_in = self.add_input('output_format', widget_type='combo', default_value='rotation_vectors')
        self.output_format_in.widget.combo_items = ['quaternions', 'rotation_vectors', 'generic']
        self.output = self.add_output('active pose')

    def execute(self):
        active_pose = self.input()
        active_pose = any_to_array(active_pose)
        if self.output_format_in() == 'quaternions':
            smpl_pose = np.zeros((20, 4))
            smpl_pose[:, 0] = 1.0
        elif self.output_format_in() == 'rotation_vectors':
            smpl_pose = np.zeros((20, 3))
        else:
            if len(active_pose.shape) == 1:
                active_pose = np.expand_dims(active_pose, axis=1)
            smpl_pose = np.zeros((20, active_pose.shape[-1]))

        if len(active_pose.shape) > 1:
            if active_pose.shape[1] == 3:
                smpl_pose = SMPLShadowTranslator.translate_from_active_to_smpl(active_pose)
                if self.output_format_in() == 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_rotvec(smpl_pose)
                    smpl_pose = rot.as_quat(scalar_first=True)
            elif active_pose.shape[1] == 4:
                smpl_pose = SMPLShadowTranslator.translate_from_active_to_smpl(active_pose)
                if self.output_format_in() != 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_quat(smpl_pose, scalar_first=True)
                    smpl_pose = rot.as_rotvec()
            else:
                smpl_pose = SMPLShadowTranslator.translate_from_active_to_smpl(active_pose)
        self.output.send(smpl_pose)


class SMPLBodyNode(SMPLNode):
    limb_dict = {
        'pelvis_left_hip': 'left_hip',
        'pelvis_right_hip': 'right_hip',
        'pelvis_spine1': 'lower_back',
        'left_hip_left_knee': 'left_thigh',
        'right_hip_right_knee': 'right_thigh',
        'spine1_spine2': 'mid_back',
        'left_knee_left_ankle': 'left_lower_leg',
        'right_knee_right_ankle': 'right_lower_leg',
        'spine2_spine3': 'upper_back',
        'left_ankle_left_foot': 'left_foot',
        'right_ankle_right_foot': 'right_foot',
        'spine3_neck': 'lower_neck',
        'spine3_left_collar': 'left_shoulder_blade',
        'spine3_right_collar': 'right_shoulder_blade',
        'neck_head': 'upper_neck',
        'left_collar_left_shoulder': 'left_shoulder',
        'right_collar_right_shoulder': 'right_shoulder',
        'left_shoulder_left_elbow': 'left_upper_arm',
        'right_shoulder_right_elbow': 'right_upper_arm',
        'left_elbow_left_wrist': 'left_forearm',
        'right_elbow_right_wrist': 'right_forearm',
        'left_wrist_left_index1': 'left_hand',
        'right_wrist_left_index2': 'right_hand'
    }
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLBodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.smpl_model = None
        self.file_name = ''
        self.model_type = 'pkl'
        self.limbs = {}
        self.joint_positions = None
        self.betas = self.add_input('betas', callback=self.receive_betas)
        self.betas.set([0.0] * 16)
        self.kinematic_tree = None
        self.load_button = self.add_property('load', widget_type='button', callback=self.load_body)
        self.file_name = self.add_label('')
        self.skeleton_data_out = self.add_output('skeleton_data')
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path, callback=self.smpl_path_changed)

    def load_smpl_model(self, in_path):
        if os.path.isfile(in_path):
            self.smpl_model, self.model_type = self.load_SMPL_R_model_file(in_path)
            if self.smpl_model is not None:
                self.file_name.set(in_path.split('/')[-1])
                self.load_path.set(in_path)
                self.process_smpl_model()
                self.execute()

    def process_smpl_model(self):
        if self.model_type == 'npz':
            self.betas.set(self.smpl_model['betas'])
            self.kinematic_tree = self.smpl_model['kintree_table']
            self.joint_positions = self.smpl_model['J']
        else:
            # print(list(self.smpl_model.keys()))
            self.betas.set(self.smpl_model.betas)
            self.kinematic_tree = self.smpl_model.kintree_table
            self.joint_positions = self.smpl_model.J
        self.update_betas()

    def update_betas(self):
        for index, start in enumerate(self.kinematic_tree[0]):
            if 0 <= start <= 23:
                end = self.kinematic_tree[1, index]
                start_joint = self.joint_names[start]
                end_joint = self.joint_names[end]
                limb_name = self.limb_dict[start_joint + '_' + end_joint]
                limb_diff = self.chumpy_to_numpy(self.joint_positions[end] - self.joint_positions[start])
                self.limbs[limb_name] = limb_diff

    def chumpy_to_numpy(self, chumpy_array):
        numpy_array = np.array(chumpy_array.ravel()).reshape(chumpy_array.shape)
        return numpy_array

    def receive_betas(self):
        if self.model_type == 'npz':
            self.smpl_model['betas'] = self.betas()
        else:
            self.smpl_model.betas = self.betas()
        self.update_betas()
        self.execute()

    def smpl_path_changed(self):
        in_path = self.load_path()
        self.load_smpl_model(in_path)

    def execute(self):
        limb_keys = list(self.limbs.keys())
        for limb in limb_keys:
            limb_data = list(self.limbs[limb])
            limb_data = [limb] + limb_data
            self.skeleton_data_out.send(limb_data)

    def load_body(self, args=None):
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
                             user_data=self, callback=self.load_smpl_callback, tag="file_dialog_id"):
            # dpg.add_file_extension(".npz")
            dpg.add_file_extension(".pkl")

    def load_smpl_callback(self, sender, app_data):
        if 'file_path_name' in app_data:
            load_path = app_data['file_path_name']
            if load_path != '':
                self.load_smpl_model(load_path)
        else:
            print('no file chosen')
        dpg.delete_item(sender)


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
        self.root_positions = None
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.load_button = self.add_property('load', widget_type='button', callback=self.load_take)
        self.file_name = self.add_label('')
        self.joint_data_out = self.add_output('joint_data')
        self.root_position_out = self.add_output('root_position')
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
            self.root_position_out.send(self.root_positions[frame])

    def load_smpl_callback(self):
        in_path = self.load_path()
        self.load_smpl(in_path)

    def load_smpl(self, in_path):
        if os.path.isfile(in_path):
            self.smpl_data, self.root_positions = self.load_smpl_take_file(in_path)
            print(self.root_positions[0])
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
            frame = int(self.current_frame)
            self.joint_data_out.send(self.joint_data[frame])
            self.root_position_out.send(self.root_positions[frame])

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            if t == int:
                if self.joint_data is not None and int(data) < self.frames:
                    self.current_frame = int(data)
                    frame = int(self.current_frame)
                    self.joint_data_out.send(self.joint_data[frame])
                    self.root_position_out.send(self.root_positions[frame])

    def load_take(self, args=None):
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
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
        self.output_as.widget.combo_items = ['quaternions', 'euler angles', 'roll_pitch_yaw']
        self.use_degrees = self.add_property('degrees', widget_type='checkbox', default_value=False)
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
            output_rpy = (self.output_as() == 'roll_pitch_yaw')

            t = type(incoming)
            if t == np.ndarray:
                for i, index in enumerate(self.joint_offsets):
                    if index < incoming.shape[0]:
                        joint_value = incoming[index]
                        if output_quaternions:
                            rot = scipy.spatial.transform.Rotation.from_rotvec(any_to_list(joint_value), degrees=self.use_degrees())
                            q = rot.as_quat()
                            joint_value = np.array([q[3], q[0], q[1], q[2]])
                        elif output_rpy:
                            rot = scipy.spatial.transform.Rotation.from_rotvec(any_to_list(joint_value), degrees=self.use_degrees())
                            q = rot.as_euler('XYZ', degrees=self.use_degrees())
                            joint_value = np.array([q[3], q[0], q[1], q[2]])
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()

class SMPLPoseQuatsToJointsNode(SMPLNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLPoseQuatsToJointsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.joint_offsets = []
        for index, key in enumerate(self.joint_names):
            if index < 22:
                self.joint_offsets.append(index)

        self.input = self.add_input('pose in', triggers_execution=True)
        # self.output_as = self.add_property('output_as', widget_type='combo', default_value='quaternions')
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

            t = type(incoming)
            if t == torch.Tensor:
                incoming = any_to_array(incoming)
                t = type(incoming)
            if t == np.ndarray:
                for i, index in enumerate(self.joint_offsets):
                    if index < incoming.shape[0]:
                        joint_value = incoming[index]
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()

# from smpl

def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'


def ready_arguments(fname_or_dict):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        dd = fname_or_dict

    backwards_compatibility_replacements(dd)

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))

    return dd


def load_model(fname_or_dict):
    dd = ready_arguments(fname_or_dict)

    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True
    }

    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1, 3))
    result.J_transformed = Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result

def ischumpy(x): return hasattr(x, 'dterms')

def verts_decorated(trans, pose,
                    v_template, J, weights, kintree_table, bs_style, f,
                    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):
    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = ch.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v

    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
    else:
        v_posed = v_shaped

    v = v_posed

    if sp.issparse(J):
        regressor = J
        J_tmpx = MatVecMult(regressor, v_shaped[:, 0])
        J_tmpy = MatVecMult(regressor, v_shaped[:, 1])
        J_tmpz = MatVecMult(regressor, v_shaped[:, 2])
        J = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
    else:
        assert (ischumpy(J))

    assert (bs_style == 'lbs')
    result, Jtr = verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, xp=ch)

    tr = trans.reshape((1, 3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type = bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result


# def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=ch):
#     if xp == ch:
#         assert (hasattr(pose, 'dterms'))
#         assert (hasattr(v, 'dterms'))
#         assert (hasattr(J, 'dterms'))
#         assert (hasattr(weights, 'dterms'))
#
#     assert (bs_style == 'lbs')
#     result = verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)
#
#     return result
#

class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1,3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))


def global_rigid_transformation(pose, J, kintree_table, xp):
    results = {}
    pose = pose.reshape((-1, 3))
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    if xp == ch:
        rodrigues = lambda x: Rodrigues(x)
    else:
        import cv2
        rodrigues = lambda x: cv2.Rodrigues(x)[0]

    with_zeros = lambda x: xp.vstack((x, xp.array([[0.0, 0.0, 0.0, 1.0]])))
    results[0] = with_zeros(xp.hstack((rodrigues(pose[0, :]), J[0, :].reshape((3, 1)))))

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(xp.hstack((
            rodrigues(pose[i, :]),
            ((J[i, :] - J[parent[i], :]).reshape((3, 1)))
        ))))

    pack = lambda x: xp.hstack([np.zeros((4, 3)), x.reshape((4, 1))])

    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [results[i] - (pack(
            results[i].dot(xp.concatenate(((J[i, :]), 0))))
        ) for i in range(len(results))]
        results = results2
    result = xp.dstack(results)
    return result, results_global


def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False, xp=ch):
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    T = A.dot(weights.T)

    rest_shape_h = xp.vstack((v.T, np.ones((1, v.shape[0]))))

    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:, :3]

    if not want_Jtr:
        return v
    Jtr = xp.vstack([g[:3, 3] for g in A_global])
    return (v, Jtr)


#######

hashes = {
    "ACCAD.tar.bz2": {
        "unpacks_to": "ACCAD",
        "hash": "193442a2ab66cb116932b8bce08ecb89",
    },
    "BMLhandball.tar.bz2": {
        "unpacks_to": "BMLhandball",
        "hash": "8947df17dd59d052ae618daf24ccace3",
    },
    "BMLmovi.tar.bz2": {
        "unpacks_to": "BMLmovi",
        "hash": "6dfb134273f284152aa2d0838d7529d5",
    },
    "CMU.tar.bz2": {"unpacks_to": "CMU", "hash": "f04bc3f37f3eafebfb12ba0cf706ca72"},
    "DFaust67.tar.bz2": {
        "unpacks_to": "DFaust_67",
        "hash": "7e5f11ed897da72c5159ef3c747383b8",
    },
    "EKUT.tar.bz2": {"unpacks_to": "EKUT", "hash": "221ee4a27a03afd1808cbb11af067879"},
    "HumanEva.tar.bz2": {
        "unpacks_to": "HumanEva",
        "hash": "ca781438b08caafd8a42b91cce905a03",
    },
    "KIT.tar.bz2": {"unpacks_to": "KIT", "hash": "3813500a3909f6ded1a1fffbd27ff35a"},
    "MPIHDM05.tar.bz2": {
        "unpacks_to": "MPI_HDM05",
        "hash": "f76da8deb9e583c65c618d57fbad1be4",
    },
    "MPILimits.tar.bz2": {
        "unpacks_to": "MPI_Limits",
        "hash": "72398ec89ff8ac8550813686cdb07b00",
    },
    "MPImosh.tar.bz2": {
        "unpacks_to": "MPI_mosh",
        "hash": "a00019cac611816b7ac5b7e2035f3a8a",
    },
    "SFU.tar.bz2": {"unpacks_to": "SFU", "hash": "cb10b931509566c0a49d72456e0909e2"},
    "SSMsynced.tar.bz2": {
        "unpacks_to": "SSM_synced",
        "hash": "7cc15af6bf95c34e481d58ed04587b58",
    },
    "TCDhandMocap.tar.bz2": {
        "unpacks_to": "TCD_handMocap",
        "hash": "c500aa07973bf33ac1587a521b7d66d3",
    },
    "TotalCapture.tar.bz2": {
        "unpacks_to": "TotalCapture",
        "hash": "b2c6833d3341816f4550799b460a1b27",
    },
    "Transitionsmocap.tar.bz2": {
        "unpacks_to": "Transitions_mocap",
        "hash": "705e8020405357d9d65d17580a6e9b39",
    },
    "EyesJapanDataset.tar.bz2": {
        "unpacks_to": "Eyes_Japan_Dataset",
        "hash": "d19fc19771cfdbe8efe2422719e5f3f1",
    },
    "BMLrub.tar.bz2": {
        "unpacks_to": "BioMotionLab_NTroje",
        "hash": "8b82ffa6c79d42a920f5dde1dcd087c3",
    },
    "DanceDB.tar.bz2": {
        "unpacks_to": "DanceDB",
        "hash": "9ce35953c4234489036ecb1c26ae38bc",
    },
}

class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm(total=kwargs["total"]) as self._pbar:
            del kwargs["total"]
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def md5_update_from_file(filename: Union[str, Path], hash: Hash) -> Hash:
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash


def md5_file(filename: Union[str, Path]) -> str:
    return str(md5_update_from_file(filename, hashlib.md5()).hexdigest())


def md5_update_from_dir(directory: Union[str, Path], hash: Hash) -> Hash:
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            hash = md5_update_from_file(path, hash)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash


def md5_dir(directory: Union[str, Path]) -> str:
    return str(md5_update_from_dir(directory, hashlib.md5()).hexdigest())


def npz_paths(npz_directory):
    npz_directory = Path(npz_directory).resolve()
    npz_paths = []
    for r, d, f in os.walk(npz_directory, followlinks=True):
        for fname in f:
            if "npz" == fname.split(".")[-1] and fname != "shape.npz":
                yield os.path.join(npz_directory, r, fname)


def npz_len(npz_path, strict=True):
    cdata = np.load(npz_path)
    h = md5_file(npz_path)
    dirs = [hashes[h]['unpacks_to'] for h in hashes]
    if strict:
        m = []
        for p in Path(npz_path).parents:
            m += [d for d in dirs if p.name == d]
        assert len(m) == 1, f"Subdir of {npz_path} contains {len(m)} of {dirs}"
        subdir = m[0]
    else:
        subdir = Path(npz_path).parts[-2]
    return subdir, h, cdata["poses"].shape[0]


def npz_lens(unpacked_directory, n_jobs, strict=True):
    paths = [p for p in npz_paths(unpacked_directory)]
    return ProgressParallel(n_jobs=n_jobs)(
        [joblib.delayed(npz_len)(npz_path, strict=strict) for npz_path in paths], total=len(paths)
    )


def save_lens(save_path, npz_file_lens):
    with gzip.open(save_path, "wt") as f:
        f.write(json.dumps(npz_file_lens))

def keep_slice(n, keep):
    drop = (1.0 - keep) / 2.0
    return slice(int(n * drop), int(n * keep + n * drop))


def viable_slice(cdata, keep):
    """
    Inspects a dictionary loaded from `.npz` numpy dumps
    and creates a slice of the viable indexes.
    args:

        - `cdata`: dictionary containing keys:
            ['poses', 'gender', 'mocap_framerate', 'betas',
             'marker_data', 'dmpls', 'marker_labels', 'trans']
        - `keep`: ratio of the file to keep, between zero and 1.,
            drops leading and trailing ends of the arrays

    returns:

        - viable: slice that can access frames in the arrays:
            cdata['poses'], cdata['marker_data'], cdata['dmpls'], cdata['trans']
    """
    assert (
        keep > 0.0 and keep <= 1.0
    ), "Proportion of array to keep must be between zero and one"
    n = cdata["poses"].shape[0]
    return keep_slice(n, keep)

# Cell
def npz_contents(
    npz_path,
    clip_length,
    overlapping,
    keep=0.8,
    keys=("poses", "dmpls", "trans", "betas", "gender"),
    shuffle=False,
    seed=None,
):
    # cache this because we will often be accessing the same file multiple times
    cdata = np.load(npz_path)

    # slice of viable indices
    viable = viable_slice(cdata, keep)

    # slice iterator
    # every time the file is opened the non-overlapping slices will be the same
    # this may not be preferred, but loading overlapping means a lot of repetitive data
    def clip_slices(viable, clip_length, overlapping):
        i = 0
        step = 1 if overlapping else clip_length
        for i in range(viable.start, viable.stop, step):
            if i + clip_length < viable.stop:
                yield slice(i, i + clip_length)

    # buffer the iterator and shuffle here, when implementing that
    buf_clip_slices = [s for s in clip_slices(viable, clip_length, overlapping)]
    if shuffle:
        # this will be correlated over workers
        # seed should be passed drawn from torch Generator
        seed = seed if seed else random.randint(1e6)
        random.Random(seed).shuffle(buf_clip_slices)

    # iterate over slices
    for s in buf_clip_slices:
        data = {}
        # unpack and enforce data type
        to_load = [k for k in ("poses", "dmpls", "trans") if k in keys]
        for k in to_load:
            data[k] = cdata[k][s].astype(np.float32)
        if "betas" in keys:
            r = s.stop - s.start
            data["betas"] = np.repeat(
                cdata["betas"][np.newaxis].astype(np.float32), repeats=r, axis=0
            )
        if "gender" in keys:

            def gender_to_int(g):
                # casting gender to integer will raise a warning in future
                g = str(g.astype(str))
                return {"male": -1, "neutral": 0, "female": 1}[g]

            data["gender"] = np.array(
                [gender_to_int(cdata["gender"]) for _ in range(s.start, s.stop)]
            )
        yield data


class AMASS(tudata.IterableDataset):
    def __init__(
        self,
        amass_location,
        clip_length,
        overlapping,
        keep=0.8,
        transform=None,
        data_keys=("poses", "trans"),
        file_list_seed=0,
        shuffle=False,
        seed=None,
        strict=True
    ):
        assert clip_length > 0 and type(clip_length) is int
        self.transform = transform
        self.data_keys = data_keys
        self.amass_location = amass_location
        # these should be shuffled but pull shuffle argument out of dataloader worker arguments
        self._npz_paths = [npz_path for npz_path in npz_paths(amass_location)]
        random.Random(file_list_seed).shuffle(self._npz_paths)
        self._npz_paths = tuple(self._npz_paths)
        self.npz_paths = self._npz_paths
        self.clip_length = clip_length
        self.overlapping = overlapping
        self.keep = keep
        self.shuffle = shuffle
        self.seed = seed if seed else random.randint(0, 1e6)
        self.strict = strict

    def infer_len(self, n_jobs=4):
        # uses known dimensions of the npz files in the AMASS dataset to infer the length
        # with clip_length and overlapping settings stored
        lenfile = Path(self.amass_location) / Path("npz_file_lens.json.gz")
        # try to load file
        if lenfile.exists():
            with gzip.open(lenfile, "rt") as f:
                self.npz_lens = json.load(f)
                def filter_lens(npz_lens):
                    # filter out file length information to only existing dirs
                    datasets = [p.name for p in Path(self.amass_location).glob('*') if p.is_dir()]
                    return [(p, h, l) for p, h, l in npz_lens
                            if p in datasets]
                self.npz_lens = filter_lens(self.npz_lens)
        else:  # if it's not there, recompute it and create the file
            print(f'Inspecting {len(self.npz_paths)} files to determine dataset length'
                  f', saving the result to {lenfile}')
            self.npz_lens = npz_lens(self.amass_location, n_jobs, strict=self.strict)
            save_lens(lenfile, self.npz_lens)

        # using stored lengths to infer the total dataset length
        def lenslice(s):
            if self.overlapping:
                return (s.stop - s.start) - (self.clip_length - 1)
            else:
                return math.floor((s.stop - s.start) / self.clip_length)

        N = 0
        for p, h, l in self.npz_lens:
            s = keep_slice(l, keep=self.keep)
            N += lenslice(s)

        return N

    def __len__(self):
        if hasattr(self, "N"):
            return self.N
        else:
            self.N = self.infer_len()
            return self.N

    def __iter__(self):
        if self.shuffle:
            self.npz_paths = list(self.npz_paths)
            random.Random(self.seed).shuffle(self.npz_paths)
        for npz_path in self.npz_paths:
            for data in npz_contents(
                npz_path,
                self.clip_length,
                self.overlapping,
                keys=self.data_keys,
                keep=self.keep,
                shuffle=self.shuffle,
                seed=self.seed,
            ):
                self.seed += 1  # increment to vary shuffle over files
                yield {k: self.transform(data[k]) for k in data}

