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
from dpg_system.interface_nodes import Vector2DNode
import pickle
from dpg_system.conversion_utils import *
from dpg_system.smpl_dynamics import SMPLDynamicsModel
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
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
from scipy.spatial.transform import Rotation
# from body_defs import JointTranslator

def register_smpl_nodes():
    Node.app.register_node("smpl_take", SMPLTakeNode.factory)
    Node.app.register_node("smpl_pose_to_joints", SMPLPoseToJointsNode.factory)
    Node.app.register_node("smpl_body", SMPLBodyNode.factory)
    Node.app.register_node("smpl_pose", SMPLPoseNode.factory)

    Node.app.register_node("smpl_quats_to_joints", SMPLPoseQuatsToJointsNode.factory)

    Node.app.register_node("smpl_to_active", SMPLToActivePoseNode.factory)
    Node.app.register_node("smpl_to_quats", SMPLToQuaternionsNode.factory)
    Node.app.register_node("active_to_smpl", ActiveToSMPLPoseNode.factory)
    Node.app.register_node("quats_flip_y_z", QuatFlipYZAxesNode.factory)
    Node.app.register_node("smpl_dynamics", SMPLDynamicsNode.factory)
    Node.app.register_node("smpl_torque", SMPLTorqueNode.factory)
    Node.app.register_node("smpl_beta_editor", SMPLBetaEditorNode.factory)
    Node.app.register_node("shadow_to_smpl", ShadowToSMPLNode.factory)


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


class SMPLPoseNode(Vector2DNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLPoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Vector2DNode.__init__(self, label, data, ['22', '4'])

    def custom_create(self, from_file):
        Vector2DNode.custom_create(self, from_file)
        self.zero_input.set_label('reset')
        for i in range(1, 23):
            input = self.inputs[i]

            dpg.configure_item(input.widget.uuids[3], label=SMPLNode.joint_names[i - 1])
            for id in input.widget.uuids:
                dpg.configure_item(id, width=45)
            input.set([1.0, 0.0, 0.0, 0.0])
            self.output_vector[i - 1, 0] = 1.0

    def zero(self):
        not_zeroed = True
        if self.vector_format_input() == 'numpy':
            if self.current_dims[0] == self.output_vector.shape[0]:
                if self.current_dims[1] == 1 and len(self.output_vector.shape) == 1:
                    self.output_vector = np.zeros(self.current_dims[0])
                    self.output_vector[:, 0] = 1.0
                    not_zeroed = False
            if not_zeroed:
                self.output_vector = np.zeros(self.current_dims)
                self.output_vector[:, 0] = 1.0

        elif self.vector_format_input() == 'torch':
            if self.current_dims[0] == self.output_vector.shape[0]:
                if self.current_dims[1] == 1 and len(self.output_vector.shape) == 1:
                    self.output_vector = torch.zeros(self.current_dims[0])
                    self.output_vector[:, 0] = 1.0
                    not_zeroed = False
            if not_zeroed:
                self.output_vector = torch.zeros(self.current_dims)
                self.output_vector[:, 0] = 1.0
        else:
            self.output_vector = [[1.0, 0.0, 0.0, 0.0] * self.current_dims[0]]
        self.execute()



# class JointTranslator():
#     smpl_joints = {
#         'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3,
#         'left_knee': 4, 'right_knee': 5, 'spine2': 6, 'left_ankle': 7,
#         'right_ankle': 8, 'spine3': 9, 'left_foot': 10, 'right_foot': 11,
#         'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15,
#         'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
#         'left_wrist': 20, 'right_wrist': 21
#     }
#
#     active_joints = {
#         'base_of_skull': 0, 'upper_vertebrae': 1, 'mid_vertebrae': 2, 'lower_vertebrae': 3,
#         'spine_pelvis': 4, 'pelvis_anchor': 5, 'left_hip': 6, 'left_knee': 7,
#         'left_ankle': 8, 'right_hip': 9, 'right_knee': 10, 'right_ankle': 11,
#         'left_shoulder_blade': 12, 'left_shoulder': 13, 'left_elbow': 14, 'left_wrist': 15,
#         'right_shoulder_blade': 16, 'right_shoulder': 17, 'right_elbow': 18, 'right_wrist': 19
#     }
#
#     smpl_to_active_joint_map = {
#         'head': 'base_of_skull',
#         'neck': 'upper_vertebrae',
#         'spine3': 'mid_vertebrae',
#         'spine2': 'lower_vertebrae',
#         'spine1': 'spine_pelvis',
#         'pelvis': 'pelvis_anchor',
#         'left_hip': 'left_hip',
#         'left_knee': 'left_knee',
#         'left_ankle': 'left_ankle',
#         'right_hip': 'right_hip',
#         'right_knee': 'right_knee',
#         'right_ankle': 'right_ankle',
#         'left_collar': 'left_shoulder_blade',
#         'left_shoulder': 'left_shoulder',
#         'left_elbow': 'left_elbow',
#         'left_wrist': 'left_wrist',
#         'right_collar': 'right_shoulder_blade',
#         'right_shoulder': 'right_shoulder',
#         'right_elbow': 'right_elbow',
#         'right_wrist': 'right_wrist'
#     }
#
#     smpl_from_active_joint_map = {
#         'head': 'base_of_skull',
#         'neck': 'upper_vertebrae',
#         'spine3': 'mid_vertebrae',
#         'spine2': 'lower_vertebrae',
#         'spine1': 'spine_pelvis',
#         'pelvis': 'pelvis_anchor',
#         'left_hip': 'left_hip',
#         'left_knee': 'left_knee',
#         'left_ankle': 'left_ankle',
#         'left_foot': 'empty',
#         'right_hip': 'right_hip',
#         'right_knee': 'right_knee',
#         'right_ankle': 'right_ankle',
#         'right_foot': 'empty',
#         'left_collar': 'left_shoulder_blade',
#         'left_shoulder': 'left_shoulder',
#         'left_elbow': 'left_elbow',
#         'left_wrist': 'left_wrist',
#         'right_collar': 'right_shoulder_blade',
#         'right_shoulder': 'right_shoulder',
#         'right_elbow': 'right_elbow',
#         'right_wrist': 'right_wrist'
#     }
#
#
#
#
#
#     @staticmethod
#     def translate_from_smpl_to_active(smpl_pose): #  expects n x 3 in, outputs 20 x 3
#         output_size = len(JointTranslator.smpl_to_active_joint_map)
#         active_pose = np.zeros((output_size, smpl_pose.shape[-1]), dtype=np.float32)
#
#         for smpl_joint in JointTranslator.smpl_to_active_joint_map:
#             smpl_index = JointTranslator.smpl_joints[smpl_joint]
#             active_joint = JointTranslator.smpl_to_active_joint_map[smpl_joint]
#             active_index = JointTranslator.active_joints[active_joint]
#             active_pose[active_index] = smpl_pose[smpl_index]
#         return active_pose
#
#     @staticmethod
#     def translate_from_active_to_smpl(active_pose): #  expects 20 x 3 in, outputs 20 x 3
#         output_size = len(JointTranslator.smpl_from_active_joint_map)
#         smpl_pose = np.zeros((output_size, active_pose.shape[-1]), dtype=np.float32)
#
#         if active_pose.shape[1] == 3:
#             empty = [0.0, 0.0, 0.0]
#         elif active_pose.shape[1] == 4:
#             empty = [1.0, 0.0, 0.0, 0.0]
#         elif active_pose.shape[1] == 2:
#             empty = [0.0, 0.0]
#         elif active_pose.shape[1] == 2:
#             empty = [0.0]
#
#         for smpl_joint in JointTranslator.smpl_from_active_joint_map:
#             smpl_index = JointTranslator.smpl_joints[smpl_joint]
#             active_joint = JointTranslator.smpl_from_active_joint_map[smpl_joint]
#             if active_joint in JointTranslator.active_joints:
#                 active_index = JointTranslator.active_joints[active_joint]
#                 smpl_pose[smpl_index] = active_pose[active_index]
#             else:
#                 smpl_pose[smpl_index] = empty
#         return smpl_pose
#
#     def __init__(self, label, data, args):
#         pass

def quaternion_multiply_scalar_first(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions in [w, x, y, z] format.
    This function is vectorized and can handle arrays of quaternions.

    Args:
        q1 (np.ndarray): The first quaternion or batch of quaternions.
                         Shape can be (4,) for a single quaternion or
                         (N, 4) for a batch of N quaternions.
        q2 (np.ndarray): The second quaternion or batch of quaternions.
                         Must be compatible for broadcasting with q1.

    Returns:
        np.ndarray: The resulting quaternion product, with the same shape as the inputs.
    """
    # Extract components using vectorized slicing for efficiency
    # The ellipsis (...) allows this to work for both single (4,) and batch (N, 4) arrays
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Apply the standard quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack the results back into a new array along the last axis
    return np.stack((w, x, y, z), axis=-1)


class SMPLToActivePoseNode(JointTranslator, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLToActivePoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        JointTranslator.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.input = self.add_input('smpl pose', triggers_execution=True)
        self.output_format_in = self.add_input('output_format', widget_type='combo', default_value='quaternions')
        self.output_format_in.widget.combo_items = ['quaternions', 'rotation_vectors', 'generic']
        self.y_is_up = self.add_property('y is up', widget_type='checkbox')
        self.output = self.add_output('active pose')
        self.y_up = np.array([0.7071067811865475, -0.7071067811865475, 0.0, 0.0])

    # def quaternion_multiply_scalar_first(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    #     """
    #     Multiplies two quaternions in [w, x, y, z] format.
    #     This function is vectorized and can handle arrays of quaternions.
    #
    #     Args:
    #         q1 (np.ndarray): The first quaternion or batch of quaternions.
    #                          Shape can be (4,) for a single quaternion or
    #                          (N, 4) for a batch of N quaternions.
    #         q2 (np.ndarray): The second quaternion or batch of quaternions.
    #                          Must be compatible for broadcasting with q1.
    #
    #     Returns:
    #         np.ndarray: The resulting quaternion product, with the same shape as the inputs.
    #     """
    #     # Extract components using vectorized slicing for efficiency
    #     # The ellipsis (...) allows this to work for both single (4,) and batch (N, 4) arrays
    #     w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    #     w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    #
    #     # Apply the standard quaternion multiplication formula
    #     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    #     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    #     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    #     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    #
    #     # Stack the results back into a new array along the last axis
    #     return np.stack((w, x, y, z), axis=-1)

    def swap_yz_axis_angle(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        Swaps the Y and Z axes of a rotation defined in axis-angle format.

        The axis-angle vector itself is transformed as a vector in the original
        coordinate space.

        Args:
            axis_angle (np.ndarray): A 3-element NumPy array representing the
                                     rotation (axis * angle).

        Returns:
            np.ndarray: The transformed 3-element axis-angle vector.
        """
        if not isinstance(axis_angle, np.ndarray):
            axis_angle = np.array(axis_angle)

        if axis_angle.shape != (3,):
            raise ValueError("Input must be a 3-element axis-angle vector.")

        x, y, z = axis_angle[0], axis_angle[1], axis_angle[2]

        return np.array([x, z, -y])

    def execute(self):
        smpl_pose = self.input()
        smpl_pose = any_to_array(smpl_pose).copy()
        if len(smpl_pose.shape) == 1:
            if smpl_pose.shape[0] <= 30:
                smpl_pose = np.expand_dims(smpl_pose, axis=1)
            else:
                smpl_pose = np.reshape(smpl_pose, (-1, 3))
        joint_count = smpl_pose.shape[0]
        if self.output_format_in() == 'quaternions':
            active_pose = np.zeros((joint_count, 4))
            active_pose[:, 0] = 1.0
        elif self.output_format_in() == 'rotation_vectors':
            active_pose = np.zeros((joint_count, 3))
        else:
            if len(smpl_pose.shape) == 1:
                smpl_pose = np.expand_dims(smpl_pose, axis=1)
            active_pose = np.zeros((joint_count, smpl_pose.shape[-1]))

        # NOTE: smpl seems to assume z is up vector which messes up axes of root rotation
        # we should force root rotation to rotate -90 around x axis

        if len(smpl_pose.shape) > 1:
            if smpl_pose.shape[1] == 3:
                # Map to body t_ indices (31 entries) with proper padding
                active_pose = JointTranslator.translate_from_smpl_to_body_joints(smpl_pose)
                if self.output_format_in() == 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_rotvec(active_pose)
                    active_pose = rot.as_quat(scalar_first=True)
                    if self.y_is_up():
                        active_pose[5] = quaternion_multiply_scalar_first(self.y_up, active_pose[5])
                else:
                    if self.y_is_up():
                        rot = scipy.spatial.transform.Rotation.from_rotvec(active_pose[5])
                        root_rot = rot.as_quat(scalar_first=True)
                        root_rot = quaternion_multiply_scalar_first(self.y_up, root_rot)
                        rot = scipy.spatial.transform.Rotation.from_quat(root_rot, scalar_first=True)
                        active_pose[5] = rot.as_rotvec()

            elif smpl_pose.shape[1] == 4:
                active_pose = JointTranslator.translate_from_smpl_to_body_joints(smpl_pose)
                if self.output_format_in() == 'rotation_vectors':
                    rot = scipy.spatial.transform.Rotation.from_quat(active_pose, scalar_first=True)
                    active_pose = rot.as_rotvec()
                elif self.y_is_up():
                    active_pose[5] = quaternion_multiply_scalar_first(self.y_up, active_pose[5])
            else:
                active_pose = JointTranslator.translate_from_smpl_to_body_joints(smpl_pose)

        self.output.send(active_pose)


class SMPLToQuaternionsNode(JointTranslator, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLToQuaternionsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        JointTranslator.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.input = self.add_input('smpl pose', triggers_execution=True)
        self.y_is_up = self.add_property('y is up', widget_type='checkbox')
        self.y_up = np.array([0.7071067811865475, -0.7071067811865475, 0.0, 0.0])
        self.output = self.add_output('smpl pose as quaternions')

    def execute(self):
        smpl_pose = self.input()
        smpl_pose = any_to_array(smpl_pose)
        if len(smpl_pose.shape) == 1:
            smpl_pose = np.reshape(smpl_pose, (-1, 3))

        if len(smpl_pose.shape) > 1:
            if smpl_pose.shape[1] == 3:
                active_smpl_pose = smpl_pose[:22]
                # active_pose = JointTranslator.translate_from_smpl_to_bmolab_active(smpl_pose)
                rot = scipy.spatial.transform.Rotation.from_rotvec(active_smpl_pose)
                active_smpl_pose = rot.as_quat(scalar_first=True)
                if self.y_is_up():
                    active_smpl_pose[0] = quaternion_multiply_scalar_first(self.y_up, active_smpl_pose[0])

                self.output.send(active_smpl_pose)


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


# NOTE: Orientation of SMPL and Shadow are different
# trans also needs to reflect this different orientation

class ActiveToSMPLPoseNode(JointTranslator, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ActiveToSMPLPoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        JointTranslator.__init__(self, label, data, args)
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
                smpl_pose = JointTranslator.translate_from_bmolab_active_to_smpl(active_pose)
                if self.output_format_in() == 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_rotvec(smpl_pose)
                    smpl_pose = rot.as_quat(scalar_first=True)
            elif active_pose.shape[1] == 4:
                smpl_pose = JointTranslator.translate_from_bmolab_active_to_smpl(active_pose)
                if self.output_format_in() != 'quaternions':
                    rot = scipy.spatial.transform.Rotation.from_quat(smpl_pose, scalar_first=True)
                    smpl_pose = rot.as_rotvec()
            else:
                smpl_pose = JointTranslator.translate_from_bmolab_active_to_smpl(active_pose)
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
        LoadDialog(self, callback=self.load_smpl_callback, extensions=['.pkl'])
        # with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
        #                      user_data=self, callback=self.load_smpl_callback, tag="file_dialog_id"):
        #     # dpg.add_file_extension(".npz")
        #     dpg.add_file_extension(".pkl")

    def load_smpl_callback(self, load_path):
        if load_path != '':
            self.load_smpl_model(load_path)
        else:
            print('no file chosen')


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
        LoadDialog(self, callback=self.load_npz_callback, extensions=['.npz'])
        # with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
        #                      user_data=self, callback=self.load_npz_callback, tag="file_dialog_id"):
        #     dpg.add_file_extension(".npz")

    def load_npz_callback(self, load_path):
        if load_path != '':
            self.load_smpl(load_path)
        else:
            print('no file chosen')


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


class QuatFlipYZAxesNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuatFlipYZAxesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ROT_X_NEG_90 = Rotation.from_euler('x', -90, degrees=True)
        self.quats_input = self.add_input('pose in', triggers_execution=True)
        self.quats_output = self.add_output('flipped pose out')
        self.rotate_joint = JointTranslator.bmolab_active_joints['pelvis_anchor']

    def execute(self):
        original_quats = self.quats_input().copy()
        original_quat = original_quats[self.rotate_joint]
        original_rot = Rotation.from_quat(original_quat, scalar_first=True)

        # Compose the rotations: New Rotation = (-90 deg X-Rot) * (Original Rot)
        new_rot = self.ROT_X_NEG_90 * original_rot

        # Convert the resulting Rotation object back to a quaternion numpy array
        original_quats[self.rotate_joint] = new_rot.as_quat(scalar_first=True)
        self.quats_output.send(original_quats)

class SMPLDynamicsNode(SMPLNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLDynamicsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)


        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.trans_input = self.add_input('trans in')
        self.metadata_input = self.add_input('metadata')
        
        self.torques_output = self.add_output('torques out')
        self.metrics_output = self.add_output('metrics out')
        self.contact_probs_output = self.add_output('contact probs out')
        
        self.gender = self.add_property('gender', widget_type='combo', default_value='neutral', callback=self.params_changed)
        self.gender.widget.combo_items = ['male', 'female', 'neutral']
        
        self.total_mass = self.add_property('total_mass', widget_type='drag_float', default_value=75.0, callback=self.params_changed)
        self.model_path = self.add_property('model_path', widget_type='text_input', default_value='', callback=self.params_changed)
        
        self.input_smoothing = self.add_property('input_smoothing', widget_type='drag_float', default_value=0.0)
        self.calibrate_pose = self.add_property('calibrate_pose', widget_type='button', callback=self.calibrate_callback)
        self.up_axis = self.add_property('input_up_axis', widget_type='combo', default_value='Y-up', callback=self.params_changed)
        self.up_axis.widget.combo_items = ['Y-up', 'Z-up']

        self.calibrate_trigger = self.add_input('calibrate', triggers_execution=True)
        self.smoothing_input = self.add_input('smoothing')
        self.mass_input = self.add_input('mass')
        
        self.current_betas = None
        self.current_dt = 1.0/60.0
        self.model = None


    def custom_create(self, from_file):
        self._initialize_model()

    def calibrate_callback(self):
        if self.model:
            self.model.calibrate_balance()

    def params_changed(self):
        self._initialize_model()

    def _initialize_model(self):
        gender = self.gender()
        mass = self.total_mass()
        path = self.model_path()
        # Pass betas if we have them
        self.model = SMPLDynamicsModel(total_mass=mass, gender=gender, betas=self.current_betas, model_path=path)

    def execute(self):
        # Handle Metadata Input
        if self.metadata_input.fresh_input:
            meta = self.metadata_input()
            if isinstance(meta, dict):
                needs_reinit = False
                
                # Framerate -> DT
                if 'mocap_framerate' in meta:
                    try:
                        fps = float(meta['mocap_framerate'])
                        if fps > 0:
                            self.current_dt = 1.0 / fps
                    except:
                        pass
                        
                # Gender
                if 'gender' in meta:
                    new_gender = str(meta['gender'])
                    if new_gender != self.gender():
                        self.gender.set(new_gender)
                        needs_reinit = True # Mass ratios change
                        
                # Total Mass
                if 'total_mass' in meta:
                    try:
                        new_mass = float(meta['total_mass'])
                        if abs(new_mass - self.total_mass()) > 0.001:
                            self.total_mass.set(new_mass)
                            needs_reinit = True
                    except:
                        pass
                        
                # Betas (Shape)
                if 'betas' in meta:
                    new_betas = meta['betas']
                    # Check if changed (using numpy array comparison if needed, or just assuming diff)
                    # Simple equality check might fail for arrays
                    if self.current_betas is None or not np.array_equal(self.current_betas, new_betas):
                         self.current_betas = new_betas
                         needs_reinit = True
                         
                if needs_reinit:
                    self._initialize_model()

        if self.pose_input.fresh_input:
            pose = self.pose_input()
            trans = self.trans_input()
            
            # Handle formats
            # Pose: Expect (22, 3) or (66,)
            pose = any_to_array(pose).copy()
            if trans is None:
                trans = np.zeros(3)
            else:
                trans = any_to_array(trans).copy()
                
            if pose is not None and self.model is not None:
                dt = self.current_dt
                
                # Coordinate System Conversion & Format Handling
                # Physics Model is Z-up. 
                # Inputs can be (22, 3) Axis-Angle OR (22, 4) Quaternions.
                
                up_axis_mode = self.up_axis()
                from scipy.spatial.transform import Rotation
                
                # Check for Quaternions (Last dim 4)
                is_quaternion = (pose.shape[-1] == 4)
                
                # 1. Apply Y-up -> Z-up Correction (if needed)
                if up_axis_mode == 'Y-up':
                     # Rotation Matrix for -90 deg X
                     R_fix = Rotation.from_euler('x', -90, degrees=True).as_matrix()
                     
                     # 1a. Fix Translation
                     # v_new = R @ v_old
                     trans = R_fix @ trans
                     
                     # 1b. Fix Root Orientation
                     if is_quaternion:
                         # Root is pose[0] (4,)
                         root_quat = pose[0] # (x, y, z, w) usually from scipy
                         root_r = Rotation.from_quat(root_quat)
                         
                         # R_new = R_fix @ R_root
                         new_root_r = Rotation.from_matrix(R_fix @ root_r.as_matrix())
                         pose[0] = new_root_r.as_quat()
                     else:
                         # Axis-Angle
                         root_rotvec = pose[0]
                         root_mat = Rotation.from_rotvec(root_rotvec).as_matrix()
                         new_root_mat = R_fix @ root_mat
                         pose[0] = Rotation.from_matrix(new_root_mat).as_rotvec()
                
                # 2. Convert to Axis-Angle (if Quaternion)
                # SMPLDynamicsModel expects (22, 3) Axis-Angle
                if is_quaternion:
                    # Convert (22, 4) -> (22, 3)
                    # We can batch convert if shape is (22, 4)
                    if pose.ndim == 2:
                        # scipy Rotation can take (N, 4)
                        # Note: scipy expects (x, y, z, w). Ensure upstream provides this.
                        # Assuming DPG standard is consistent with scipy? Usually yes.
                        r_objs = Rotation.from_quat(pose)
                        pose = r_objs.as_rotvec() # (22, 3)
                    else:
                        # Maybe flattened (88,)? Handle if needed, but any_to_array usually preserves shape?
                        # If simple flat array, reshape first?
                        # For now assume (22, 4)
                        pass

                # Calibration Input

                # Calibration Input
                if self.calibrate_trigger.fresh_input:
                    if self.model:
                        self.model.calibrate_balance()

                # Calibration Button (Handled by callback, but check purely for legacy/safety if needed? No, callback is better)
                if self.calibrate_pose():
                     # This might return True if clicked? dpg_system button properties return True on click frame.
                     # But we added a callback, so we might double-trigger if we check here too.
                     # Remove this check if callback handles it.
                     pass 
                
                smoothing = self.input_smoothing()
                if self.smoothing_input.fresh_input:
                    s_val = self.smoothing_input()
                    if s_val is not None:
                        smoothing = float(s_val)
                        self.input_smoothing.set(smoothing)
                
                if smoothing < 0: smoothing = 0.0

                # Mass Input processing (should be rare)
                if self.mass_input.fresh_input:
                    m_val = self.mass_input()
                    if m_val is not None:
                         m_float = float(m_val)
                         if abs(m_float - self.total_mass()) > 0.001:
                             self.total_mass.set(m_float)
                             self._initialize_model()
                
                try:
                    torques, metrics = self.model.update_frame(trans, pose, dt, smoothing_sigma=smoothing)
                    
                    # Convert torques dict to (22, 3) numpy array
                    torque_array = np.zeros((22, 3))
                    from dpg_system.smpl_dynamics import SMPL_JOINT_NAMES
                    
                    for i, name in enumerate(SMPL_JOINT_NAMES):
                        if name in torques:
                            torque_array[i] = torques[name]
                    
                    self.torques_output.send(torque_array)
                    self.metrics_output.send(metrics)
                    
                except Exception as e:
                    # On first frames history might be insufficient? 
                    # update_frame handles it gracefully (returns 0s)
                    pass


class SMPLTorqueNode(SMPLNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SMPLTorqueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.config_input = self.add_input('config') # gender, betas, framerate
        
        # self.torques_output removed

        self.effort_output = self.add_output('effort')
        self.gravity_effort_output = self.add_output('gravity_effort')
        self.combined_effort_output = self.add_output('combined_effort')

        self.output_positions = self.add_output('joint_positions')
        self.output_root = self.add_output('root_torque')
        
        self.message_handlers['set_max_torque'] = self.set_max_torque_handler
        self.message_handlers['set_max_torque'] = self.set_max_torque_handler
        self.message_handlers['print_max_torque'] = self.print_max_torque
        
        self.torque_ratios = {} # Format: {generic_joint_key: ratio_vector}
        
        self.output_torque_vectors = self.add_output("torque_vectors")
        self.inertia_output = self.add_output("inertias")
        self.gravity_torque_vec_output = self.add_output('gravity_torque_vectors')
        self.dynamic_torque_vec_output = self.add_output('dynamic_torque_vectors')
        self.passive_torque_vec_output = self.add_output('passive_torque_vectors')
        self.contact_probs_output = self.add_output('contact_probs')
        self.contact_probs_fusion_output = self.add_output('contact_probs_fusion')
        self.contact_pressure_output = self.add_output('contact_pressure')
        self.output_com = self.add_output('com_pos')
        self.output_zmp = self.add_output('zmp_pos')
        self.output_limb_lengths = self.add_output('limb_lengths')
        self.noise_score_output = self.add_output('noise_score')
        self.noise_report_output = self.add_output('noise_report')
        self.floor_level_output = self.add_output('floor_level')
        self.balance_score_output = self.add_output('balance_score')
        self.imbalance_vector_output = self.add_output('imbalance_vector')
        self.support_polygon_output = self.add_output('support_polygon')
        
        # Noise stats controls
        self.reset_noise_stats_input = self.add_input('reset_noise_stats', widget_type='button', callback=self._reset_noise_stats)
        

        self.zero_root_torque = self.add_option('zero_root_torque', widget_type='checkbox', default_value=True)
        self.add_gravity_prop = self.add_option('add_gravity', widget_type='checkbox', default_value=True)
        self.enable_app_gravity_prop = self.add_option('enable_apparent_gravity', widget_type='checkbox', default_value=True)
        self.up_axis_prop = self.add_property('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']
        self.axis_perm_prop = self.add_property('axis_permutation', widget_type='text_input', default_value='x, z, -y', callback=self._on_axis_perm_changed)
        self.quat_format_prop = self.add_property('quat_format', widget_type='combo', default_value='wxyz')
        self.quat_format_prop.widget.combo_items = ['xyzw', 'wxyz']
        
        self.enable_passive_limits = self.add_option('enable_passive_limits', widget_type='checkbox', default_value=True)
        
        self.enable_one_euro_prop = self.add_option('enable_one_euro_filter', widget_type='checkbox', default_value=True)
        self.min_cutoff_prop = self.add_option('one_euro_min_cutoff', widget_type='drag_float', default_value=1.0)
        self.beta_prop = self.add_option('one_euro_beta', widget_type='drag_float', default_value=0.05)
        

        self.floor_enable_prop = self.add_option('floor_contact_enable', widget_type='checkbox', default_value=True)
        self.floor_height_prop = self.add_option('floor_height', widget_type='drag_float', default_value=0.0)
        self.floor_tol_prop = self.add_option('floor_tolerance', widget_type='drag_float', default_value=0.15)
        self.reset_floor_input = self.add_input('reset_floor', widget_type='button', callback=self._reset_floor)
        
        # Bias: Negative = Toe Preference, Positive = Heel Preference
        self.heel_toe_bias_prop = self.add_option('heel_toe_bias', widget_type='drag_float', default_value=0.02)
        
        # Contact Method Selection
        self.contact_method_prop = self.add_option('contact_method', widget_type='combo', default_value='fusion')
        self.contact_method_prop.widget.combo_items = ['fusion', 'stability', 'com_driven', 'consensus']
        
        # --- Rate Limiting ---
        self.enable_rate_limiting_prop = self.add_option('enable_rate_limiting', widget_type='checkbox', default_value=True)
        self.rate_limit_strength_prop = self.add_option('rate_limit_strength', widget_type='drag_float', default_value=1.0)
        self.enable_jitter_damping_prop = self.add_option('enable_jitter_damping', widget_type='checkbox', default_value=True)
        self.enable_velocity_gate_prop = self.add_option('enable_velocity_gate', widget_type='checkbox', default_value=True)
        self.enable_kf_smoothing_prop = self.add_option('enable_kf_smoothing', widget_type='checkbox', default_value=True)
        self.kf_responsiveness_prop = self.add_option('kf_responsiveness', widget_type='drag_float', default_value=10.0)
        self.kf_smoothness_prop = self.add_option('kf_smoothness', widget_type='drag_float', default_value=1.0)
        self.kf_clamp_radius_prop = self.add_option('kf_clamp_radius', widget_type='drag_float', default_value=15.0)
        
        # --- World-Frame Dynamics ---
        self.world_frame_dynamics_prop = self.add_option('world_frame_dynamics', widget_type='checkbox', default_value=False)
        self.com_pos_mc_prop = self.add_option('com_pos_min_cutoff', widget_type='drag_float', default_value=8.0)
        self.com_pos_beta_prop = self.add_option('com_pos_beta', widget_type='drag_float', default_value=0.05)
        self.com_vel_mc_prop = self.add_option('com_vel_min_cutoff', widget_type='drag_float', default_value=3.0)
        self.com_vel_beta_prop = self.add_option('com_vel_beta', widget_type='drag_float', default_value=0.05)
        self.com_acc_mc_prop = self.add_option('com_acc_min_cutoff', widget_type='drag_float', default_value=2.0)
        self.com_acc_beta_prop = self.add_option('com_acc_beta', widget_type='drag_float', default_value=0.8)
        self.smooth_input_window_prop = self.add_property('smooth_input_window', widget_type='drag_int', default_value=0)
        self.magnetometer_cadence_prop = self.add_property('magnetometer_cadence', widget_type='drag_int', default_value=0)
        
        # --- Spine Geometry ---
        self.use_s_curve_spine_prop = self.add_option('use_s_curve_spine', widget_type='checkbox', default_value=True)
        
        # Calibrated default for Subject_81: [0.0, -0.14, 0.05]


        self.processor = None
        # Default config
        self.framerate = 60.0
        self.gender = 'neutral'
        self.betas = None
        self.total_mass = 75.0
        
    def _to_array(self, d):
        d = any_to_array(d)
        return d

    def _on_axis_perm_changed(self):
        if self.processor:
            self.processor.set_axis_permutation(self.axis_perm_prop())

    def _reset_noise_stats(self):
        """Reset noise statistics for a new file evaluation."""
        if self.processor:
            self.processor.reset_noise_stats()

    def _reset_floor(self):
        """Reset the adaptive floor height estimate back to the configured floor_height."""
        if self.processor:
            base_height = self.floor_height_prop() if hasattr(self, 'floor_height_prop') else 0.0
            self.processor._inferred_floor_height = base_height
            print(f"SMPLTorqueNode: Adaptive floor reset to {base_height:.3f}")

    def execute(self):
        # 1. Handle Config

        if self.config_input.fresh_input:
            cfg = self.config_input()
            if isinstance(cfg, dict):
                changed = False
                
                if 'motioncapture_framerate' in cfg:
                    fr = float(cfg['motioncapture_framerate'])
                    if fr != self.framerate:
                        self.framerate = fr
                        changed = True
                        
                if 'gender' in cfg:
                    g = str(cfg['gender'])
                    if g != self.gender:
                        self.gender = g
                        changed = True
                        
                if 'betas' in cfg:
                    b = self._to_array(cfg['betas'])
                    # Simple check
                    if self.betas is None or not np.array_equal(self.betas, b):
                        self.betas = b
                        changed = True
                
                # Re-init processor if needed
                if changed or self.processor is None:
                    self.processor = SMPLProcessor(
                        framerate=self.framerate,
                        betas=self.betas,
                        gender=self.gender,
                        total_mass_kg=self.total_mass,
                        model_path=os.path.dirname(os.path.abspath(__file__)) # Use absolute path to dpg_system
                    )
                    self._on_axis_perm_changed()
                    self._restore_max_torque_overrides()
                    
                    # Send updated limb lengths immediately upon config change
                    if hasattr(self.processor, 'skeleton_offsets'):
                         offsets = self.processor.skeleton_offsets # (24, 3)
                         lengths = np.linalg.norm(offsets, axis=-1) # (24,)
                         self.output_limb_lengths.send(lengths)
        
        # Ensure processor exists
        if self.processor is None:
             self.processor = SMPLProcessor(
                framerate=self.framerate,
                betas=self.betas,
                gender=self.gender,
                total_mass_kg=self.total_mass,
                model_path=os.path.dirname(os.path.abspath(__file__))
            )
             self._on_axis_perm_changed()
             self._restore_max_torque_overrides()
             
             # Send initial limb lengths
             if hasattr(self.processor, 'skeleton_offsets'):
                 offsets = self.processor.skeleton_offsets
                 lengths = np.linalg.norm(offsets, axis=-1)
                 self.output_limb_lengths.send(lengths)

        # 2. Process Data
        if self.pose_input.fresh_input:
            pose = self._to_array(self.pose_input())
            trans = self.trans_input()
            if trans is None:
                trans = np.zeros(3)
            else:
                trans = self._to_array(trans)
                
            # Input format handling
            # SMPLProcessor expects (F, 24, 3) or (F, 24, 4) or flattened
            # DPG pose might be (22, 3) or (24, 3) or flattened
            
            # Helper to reshape if flattened [22*3] or [24*3]
            def reshape_pose_input(p_in):
                if p_in.ndim == 1:
                    if p_in.size == 72: # 24*3
                        return p_in.reshape(1, 24, 3)
                    elif p_in.size == 66: # 22*3 (missing hands?)
                        # Pad with identity/zeros.
                        p24 = np.zeros((1, 24, 3))
                        p24[0, :22, :] = p_in.reshape(22, 3)
                        return p24
                    elif p_in.size == 88: # 22*4 quats
                        # Pad to 24*4
                        p24 = np.zeros((1, 24, 4))
                        p24[:, :, 0] = 1.0 # Identity w=1
                        p24[0, :22, :] = p_in.reshape(22, 4)
                        return p24
                    elif p_in.size == 96: # 24*4
                        return p_in.reshape(1, 24, 4)
                elif p_in.ndim == 2:
                    # (22, 3), (24, 3), (22, 4), (24, 4)
                    if p_in.shape[0] == 22:
                        # Pad to 24
                        if p_in.shape[1] == 3:
                            p24 = np.zeros((1, 24, 3))
                            p24[0, :22, :] = p_in
                            return p24
                        elif p_in.shape[1] == 4:
                            p24 = np.zeros((1, 24, 4))
                            p24[:, :, 0] = 1.0
                            p24[0, :22, :] = p_in
                            return p24
                    elif p_in.shape[0] == 24:
                        return p_in[np.newaxis, ...] # Add F dim
                return p_in

            pose = reshape_pose_input(pose)

            # Determine input type
            # Shape is now (F, 24, C)
            input_type = 'axis_angle'
            if pose.shape[-1] == 4:
                input_type = 'quat'
            
            # Process
            # Prepare Config
            options = SMPLProcessingOptions(
                input_type=input_type,
                return_quats=True,
                
                # Coordinates
                input_up_axis=self.up_axis_prop(),
                axis_permutation=self.processor.axis_permutation if hasattr(self.processor, 'axis_permutation') else None,
                quat_format=self.quat_format_prop(),
                
                # Physics
                dt=1.0/max(self.framerate, 1.0),
                add_gravity=self.add_gravity_prop(),
                enable_passive_limits=self.enable_passive_limits(),
                enable_apparent_gravity=self.enable_app_gravity_prop(),
                
                # Filtering
                enable_one_euro_filter=self.enable_one_euro_prop(),
                filter_min_cutoff=self.min_cutoff_prop() if hasattr(self, 'min_cutoff_prop') else 1.0,
                filter_beta=self.beta_prop() if hasattr(self, 'beta_prop') else 0.05,
                
                
                # Floor
                floor_enable=self.floor_enable_prop(),
                floor_height=self.floor_height_prop() if hasattr(self, 'floor_height_prop') else 0.0,
                floor_tolerance=self.floor_tol_prop() if hasattr(self, 'floor_tol_prop') else 0.15,
                heel_toe_bias=self.heel_toe_bias_prop() if hasattr(self, 'heel_toe_bias_prop') else 0.0,
                contact_method=self.contact_method_prop() if hasattr(self, 'contact_method_prop') else 'fusion',
                
                # Rate Limiting
                enable_rate_limiting=self.enable_rate_limiting_prop() if hasattr(self, 'enable_rate_limiting_prop') else True,
                rate_limit_strength=self.rate_limit_strength_prop() if hasattr(self, 'rate_limit_strength_prop') else 1.0,
                enable_jitter_damping=self.enable_jitter_damping_prop() if hasattr(self, 'enable_jitter_damping_prop') else True,
                enable_velocity_gate=self.enable_velocity_gate_prop() if hasattr(self, 'enable_velocity_gate_prop') else True,
                enable_kf_smoothing=self.enable_kf_smoothing_prop() if hasattr(self, 'enable_kf_smoothing_prop') else True,
                kf_responsiveness=self.kf_responsiveness_prop() if hasattr(self, 'kf_responsiveness_prop') else 10.0,
                kf_smoothness=self.kf_smoothness_prop() if hasattr(self, 'kf_smoothness_prop') else 1.0,
                kf_clamp_radius=self.kf_clamp_radius_prop() if hasattr(self, 'kf_clamp_radius_prop') else 15.0,
                world_frame_dynamics=self.world_frame_dynamics_prop() if hasattr(self, 'world_frame_dynamics_prop') else False,
                com_pos_min_cutoff=self.com_pos_mc_prop() if hasattr(self, 'com_pos_mc_prop') else 8.0,
                com_pos_beta=self.com_pos_beta_prop() if hasattr(self, 'com_pos_beta_prop') else 0.05,
                com_vel_min_cutoff=self.com_vel_mc_prop() if hasattr(self, 'com_vel_mc_prop') else 3.0,
                com_vel_beta=self.com_vel_beta_prop() if hasattr(self, 'com_vel_beta_prop') else 0.05,
                com_acc_min_cutoff=self.com_acc_mc_prop() if hasattr(self, 'com_acc_mc_prop') else 2.0,
                com_acc_beta=self.com_acc_beta_prop() if hasattr(self, 'com_acc_beta_prop') else 0.8,
                smooth_input_window=self.smooth_input_window_prop() if hasattr(self, 'smooth_input_window_prop') else 0,
                magnetometer_cadence=self.magnetometer_cadence_prop() if hasattr(self, 'magnetometer_cadence_prop') else 0,
                use_s_curve_spine=self.use_s_curve_spine_prop() if hasattr(self, 'use_s_curve_spine_prop') else True,
            )
            
            # Process
            try:
                res = self.processor.process_frame(pose, trans, options)
                # Output Torques: (F, 22) -> (22) if F=1
                # Output Inertias: (22,)
                
                # torques = res['torques'] # REMOVED
                inertias = res['inertias']
                efforts_dyn = res.get('efforts_dyn', np.zeros_like(inertias)) 
                efforts_grav = res.get('efforts_grav', np.zeros_like(inertias))
                efforts_net = res.get('efforts_net', np.zeros_like(inertias))
                
                # Vectors
                # If key missing in older processor versions (shouldn't happen), fallback to zeros
                torques_vec = res.get('torques_vec', np.zeros((inertias.shape[0], 24, 3)))
                torques_grav_vec = res.get('torques_grav_vec', np.zeros((inertias.shape[0], 24, 3)))
                torques_passive_vec = res.get('torques_passive_vec', np.zeros((inertias.shape[0], 24, 3)))
                
                # CoM / ZMP
                com_out = getattr(self.processor, 'current_com', np.zeros((inertias.shape[0], 3)))
                zmp_out = getattr(self.processor, 'current_zmp', np.zeros((inertias.shape[0], 3)))
                
                if inertias.shape[0] == 1:
                    # torques = torques[0] # REMOVED
                    efforts_dyn = efforts_dyn[0]
                    efforts_grav = efforts_grav[0]
                    efforts_net = efforts_net[0]
                    com_out = com_out[0] if com_out.ndim > 1 else com_out
                    zmp_out = zmp_out[0] if zmp_out.ndim > 1 else zmp_out
                    inertias = inertias[0]
                    torques_vec = torques_vec[0] # (24, 3)
                    torques_grav_vec = torques_grav_vec[0]
                    torques_passive_vec = torques_passive_vec[0]
                    
                # Zero Root Torque if requested
                if self.zero_root_torque():
                     # Send original root torque (scalar) before zeroing vectors
                     root_val = np.linalg.norm(torques_vec[0])
                     self.output_root.send(root_val)
                    
                     # Zero out index 0
                     inertias = inertias.copy()
                     inertias[0] = 0.0
                     
                     efforts_dyn = efforts_dyn.copy()
                     efforts_dyn[0] = 0.0
                     
                     efforts_grav = efforts_grav.copy()
                     efforts_grav[0] = 0.0
                     
                     efforts_net = efforts_net.copy()
                     efforts_net[0] = 0.0
                     
                     torques_vec = torques_vec.copy()
                     torques_vec[0] = np.zeros(3)
                     
                     torques_grav_vec = torques_grav_vec.copy()
                     torques_grav_vec[0] = np.zeros(3)

                # self.torques_output.send(torques) - Removed

                self.effort_output.send(efforts_dyn)
                self.gravity_effort_output.send(efforts_grav)
                self.combined_effort_output.send(efforts_net)
                
                if 'torques_dyn_vec' in res:
                    self.output_torque_vectors.send(torques_vec)

                if 'inertias' in res:
                    self.inertia_output.send(inertias)

                if 'torques_dyn_vec' in res:
                    t_dyn = res['torques_dyn_vec']
                    if t_dyn.shape[0] == 1:
                        # Unpack if batch=1 (J,3)
                        self.dynamic_torque_vec_output.send(t_dyn[0])
                    else:
                        self.dynamic_torque_vec_output.send(t_dyn)

                self.gravity_torque_vec_output.send(torques_grav_vec)
                self.passive_torque_vec_output.send(torques_passive_vec)


                if 'positions' in res:
                     pos = res['positions']
                     if pos.shape[0] == 1:
                         pos = pos[0] # (24, 3)
                     self.output_positions.send(pos)
                
                if 'contact_probs' in res:
                     probs = res['contact_probs']
                     if probs.shape[0] == 1:
                         probs = probs[0]
                     self.contact_probs_output.send(probs)

                if 'contact_probs_fusion' in res:
                     probs_fusion = res['contact_probs_fusion']
                     if probs_fusion.shape[0] == 1:
                         probs_fusion = probs_fusion[0]
                     self.contact_probs_fusion_output.send(probs_fusion)
                     
                press_out = getattr(self.processor, 'contact_pressure', None)
                if press_out is not None:
                     if press_out.ndim > 1:
                         press_out = press_out.flatten()
                     self.contact_pressure_output.send(press_out)
                     
                self.output_com.send(com_out)
                self.output_zmp.send(zmp_out)
                
                # Send inferred floor level
                floor_level = getattr(self.processor, '_inferred_floor_height', None)
                if floor_level is not None:
                    self.floor_level_output.send(float(floor_level))
                
                # Send noise statistics
                noise_score = self.processor.get_noise_score()
                noise_report = self.processor.get_noise_report()
                self.noise_score_output.send(noise_score)
                self.noise_report_output.send(noise_report)
                
                # Send balance stability
                if 'balance' in res:
                    bal = res['balance']
                    self.balance_score_output.send(bal['stability_score'])
                    self.imbalance_vector_output.send(bal['imbalance_vector'])
                    self.support_polygon_output.send(bal['support_polygon'])
            
            except Exception as e:
                # Catch processing errors (e.g. shape mismatch on first frame)
                print(f"SMPLTorqueNode Error: {e}")
                import traceback
                traceback.print_exc()
                pass

    def print_max_torque(self, message='', args=[]):
        max = self.processor.max_torque_array
        print(max)

    def set_max_torque_handler(self, message='', args=[]):
        """
        Handler for 'set_max_torque' message.
        Expected args: 
        1. [joint_name_filter, value] where value is float or list of 3 floats.
        2. [(24,3) numpy array] -> Sets full profile.
        3. [(24,) numpy array] -> Sets full profile (isotropic).
        """
        if self.processor is None:
            return
            
        # Case 1: Direct Array Payload (Single Argument)
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            arr = args[0]
            if arr.shape == (24, 3) or arr.shape == (24, 1) or arr.shape == (24,):
                # --- Persistence Logic for Arrays (Biometric Back-Projection + Heuristic) ---
                # Strategy: 
                # 1. Analyze Input attributes against Defaults.
                # 2. If Input matches Default BUT we have a saved override, prefer the Override (Heuristic).
                # 3. Construct "Effective Array" that merges Input and Overrides.
                # 4. Apply Effective Array to Processor.
                
                defaults = self.processor._compute_max_torque_profile()
                effective_arr = arr.copy() # Start with input
                
                # Function to ensure we have a vector for math
                def to_vec(v):
                    v = np.array(v, dtype=float)
                    if v.ndim == 0: return np.full(3, v)
                    if v.shape == (1,): return np.full(3, v[0])
                    return v

                # Capture valid joint indices for mapping
                # We need to apply logic per Biometric Group
                
                # Copy old ratios to reference during decision making
                old_ratios = getattr(self, 'torque_ratios', {}).copy()
                new_ratios = {}
                
                overrides_applied = 0
                
                for k in defaults:
                    default_vec = to_vec(defaults[k])
                    
                    # Find ALL joint indices belonging to this group
                    # e.g., 'knee' -> left_knee_idx, right_knee_idx
                    indices = []
                    for i, name in enumerate(self.processor.joint_names):
                        if k in name:
                            indices.append(i)
                            
                    if not indices: continue
                    
                    # Check the First representative for logic (assuming isotropic input for the group if from biometric source)
                    # If input array varies with group (e.g. left knee != right knee), this logic is imperfect but sufficient for biometric groups.
                    rep_idx = indices[0]
                    val_vec = arr[rep_idx]
                    
                    # --- Heuristic Check ---
                    is_default = np.allclose(val_vec, default_vec, rtol=1e-3, atol=1e-3)
                    
                    has_override = False
                    start_ratio = 1.0
                    if k in old_ratios:
                        start_ratio = old_ratios[k]
                        if not np.allclose(start_ratio, 1.0, rtol=1e-3, atol=1e-3):
                            has_override = True
                            
                    final_ratio = 1.0
                    
                    if is_default and has_override:
                        # IGNORE Reset. Restore Override.
                        # Calculate restored value based on CURRENT default (which is what we are comparing against)
                        restored_val = default_vec * start_ratio
                        
                        # Apply to Effective Array (All indices in group)
                        for idx in indices:
                            effective_arr[idx] = restored_val
                            
                        # Keep Old Ratio
                        final_ratio = start_ratio
                        overrides_applied += 1
                        
                    else:
                        # Accept Input (New Custom or Confirmed Default)
                        final_ratio = val_vec / (default_vec + 1e-6)
                        # No change to effective_arr needed (it has input value)
                        
                    # Store
                    new_ratios[k] = final_ratio
                    
                # 1. Apply Effective Array
                self.processor.set_full_max_torque_profile(effective_arr)
                
                # 2. Persist Ratios
                self.torque_ratios = new_ratios
                
                if overrides_applied > 0:
                    print(f"SMPLTorqueNode: Array Input processed. Preserved {overrides_applied} custom overrides against default reset.")
                # else:
                #     print(f"SMPLTorqueNode: Array Input processed. No overrides active or Input was non-default.")
                
                return
                        
                print(f"SMPLTorqueNode: Persisted array settings via {len(self.torque_ratios)} biometric groups.")
                return
            else:
                 print(f"SMPLTorqueNode: Invalid max torque array shape {arr.shape}. Expected (24, 3) or (24,).")
                 return

        # Case 2: Standard [Filter, Value]
        if len(args) >= 2:
            joint_filter = str(args[0])
            try:
                # Value might be float, list, or numpy array
                val = args[1]
                # count = self.processor.set_max_torque(joint_filter, val) # MOVED BELOW CHECK
                
                # --- Persistence Logic (Scaling) ---
                # Calculate ratio relative to current biometric default
                defaults = self.processor._compute_max_torque_profile()
                
                # Function to ensure we have a vector for math
                def to_vec(v):
                    v = np.array(v, dtype=float)
                    if v.ndim == 0: return np.full(3, v)
                    if v.shape == (1,): return np.full(3, v[0])
                    return v

                val_vec = to_vec(val)
                
                # Check which keys were updated
                # Logic mirrors SMPLProcessor.set_max_torque: `if joint_filter in k:`
                for k in defaults:
                    if joint_filter in k:
                        # Calculate Ratio
                        default_vec = to_vec(defaults[k])
                        
                        # --- Heuristic: Default Rejection ---
                        # If the incoming value 'val_vec' matches 'default_vec', AND we already have a custom override,
                        # we assume this is a 'System Reset' (e.g. from file loader) and ignore it to preserve user intent.
                        
                        is_default = np.allclose(val_vec, default_vec, rtol=1e-3, atol=1e-3)
                        
                        existing_ratio = 1.0
                        if hasattr(self, 'torque_ratios') and k in self.torque_ratios:
                             # Check if existing ratio is non-trivial from 1.0 vector
                             r = self.torque_ratios[k] # vector
                             if not np.allclose(r, 1.0, rtol=1e-3, atol=1e-3):
                                 existing_ratio = 2.0 # Just a flag that it's "Custom"
                        
                        if is_default and existing_ratio != 1.0:
                             print(f"SMPLTorqueNode: Ignoring reset to default for '{k}' because custom override is active.")
                             # Skip update!
                             # But wait, set_max_torque was ALREADY called above!
                             # We need to perform this check BEFORE calling set_max_torque.
                             continue
                        
                        # Calculate Ratio
                        ratio = val_vec / (default_vec + 1e-6)
                        
                        # Store intent
                        if not hasattr(self, 'torque_ratios'): self.torque_ratios = {}
                        self.torque_ratios[k] = ratio
                        # print(f"Stored Max Torque Ratio for '{k}': {ratio}")
                
                # --- APPLY Update (Moved below check) ---
                # We can't use the simple 'count = processor.set_max_torque' because we might skip some keys.
                # However, set_max_torque applies to ALL matching keys.
                # If we want to selectively skip, we might differ from processor logic.
                # But typically 'joint_filter' maps to specific joints.
                # If the heuristic triggers, it usually triggers for ALL joints in the filter (if filter is broad and defaults match).
                
                # Let's Refactor: Call set_max_torque ONLY if we didn't skip everything?
                # Or better: Iterate keys and call set_max_torque per key if valid.
                
                count = 0
                for k in defaults:
                    if joint_filter in k:
                         default_vec = to_vec(defaults[k])
                         is_default = np.allclose(val_vec, default_vec, rtol=1e-3, atol=1e-3)
                         
                         has_override = False
                         if hasattr(self, 'torque_ratios') and k in self.torque_ratios:
                             r = self.torque_ratios[k]
                             if not np.allclose(r, 1.0, rtol=1e-3, atol=1e-3):
                                 has_override = True
                                 
                         if is_default and has_override:
                             print(f"SMPLTorqueNode: Ignoring reset to default for '{k}' (Override Active).")
                             continue
                         
                         # Apply Update
                         self.processor.set_max_torque(k, val)
                         count += 1
                         
                         # Update Storage
                         ratio = val_vec / (default_vec + 1e-6)
                         if not hasattr(self, 'torque_ratios'): self.torque_ratios = {}
                         self.torque_ratios[k] = ratio

                if count > 0:
                    print(f"SMPLTorqueNode: Updated max torque for {count} joints matching '{joint_filter}' to {val}")
                else:
                    if count == 0:
                         print(f"SMPLTorqueNode: No update performed (filtered or rejected).")
            except ValueError:
                print(f"SMPLTorqueNode: Invalid torque value '{args[1]}'")
                
            except Exception as e:
                print(f"SMPLTorqueNode: Error updating torque: {e}")
                import traceback
                traceback.print_exc()
                pass

    def _restore_max_torque_overrides(self):
        """
        Re-applies user Max Torque settings by scaling current defaults by stored Ratios.
        """
        if hasattr(self, 'torque_ratios') and self.torque_ratios and self.processor:
            defaults = self.processor._compute_max_torque_profile()
            count = 0
            for k, ratio in self.torque_ratios.items():
                if k in defaults:
                    default_val = defaults[k]
                    # Ensure Vector for safe math
                    v_def = np.array(default_val, dtype=float)
                    if v_def.ndim == 0: v_def = np.full(3, v_def)
                    elif v_def.size == 1: v_def = np.full(3, v_def.item())
                    
                    new_val = v_def * ratio
                    self.processor.set_max_torque(k, new_val)
                    count += 1
            if count > 0:
                 print(f"SMPLTorqueNode: Restored max torque for {count} joint groups based on biometric scaling.")


class SMPLBetaEditorNode(Node):
    """Node for manually tuning SMPL betas and outputting a config dict for smpl_torque."""

    # Semantic labels for beta components (from SMPL-H PCA analysis)
    BETA_LABELS = [
        'beta_0 size',       # Overall body scale (dominant)
        'beta_1 weight',     # Girth / BMI (legs > arms)
        'beta_2 arm_ratio',  # Arm length relative to body
        'beta_3 leg_ratio',  # Leg length relative to body
        'beta_4',
        'beta_5',
        'beta_6',
        'beta_7',
        'beta_8',
        'beta_9',
    ]

    @staticmethod
    def factory(name, data, args=None):
        node = SMPLBetaEditorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Beta properties (all 10)
        self.beta_props = []
        for i in range(10):
            prop = self.add_property(self.BETA_LABELS[i], widget_type='slider_float', default_value=0.0, min=-2.5, max=2.5, callback=self._on_param_changed)
            self.beta_props.append(prop)

        # Body parameters
        self.gender_prop = self.add_property('gender', widget_type='combo', default_value='neutral', callback=self._on_param_changed)
        self.gender_prop.widget.combo_items = ['male', 'female', 'neutral']
        self.total_mass_prop = self.add_property('total_mass', widget_type='drag_float', default_value=75.0, callback=self._on_param_changed)

        # Optional input: full 10-element beta array overrides properties
        self.betas_input = self.add_input('betas_in', callback=self._on_betas_received)

        # Reset button
        self.reset_input = self.add_input('reset', widget_type='button', callback=self._reset_betas)

        # Outputs
        self.config_output = self.add_output('config')
        self.limb_lengths_output = self.add_output('limb_lengths')

    def _reset_betas(self):
        for prop in self.beta_props:
            prop.widget.set(0.0)
        self._recompute_and_send()

    def custom_create(self, from_file):
        self._recompute_and_send()

    def _get_betas_array(self):
        """Assemble a 10-element betas array from the property widgets."""
        betas = np.zeros(10)
        for i, prop in enumerate(self.beta_props):
            betas[i] = prop()
        return betas

    def _on_betas_received(self):
        """When a full betas array or config dict arrives via input, update the property widgets."""
        raw = self.betas_input()
        if raw is None:
            return
        if isinstance(raw, dict):
            if 'betas' in raw:
                b = any_to_array(raw['betas']).flatten()
                for i in range(min(len(b), 10)):
                    self.beta_props[i].widget.set(float(b[i]))
            if 'gender' in raw:
                self.gender_prop.widget.set(raw['gender'])
        else:
            b = any_to_array(raw).flatten()
            for i in range(min(len(b), 10)):
                self.beta_props[i].widget.set(float(b[i]))
        self._recompute_and_send()

    def _on_param_changed(self):
        """Any property changed — recompute and send."""
        self._recompute_and_send()

    def _recompute_and_send(self):
        """Recompute limb properties and send config + limb lengths."""
        betas = self._get_betas_array()
        gender = self.gender_prop()
        total_mass = self.total_mass_prop()

        # Build config dict (matches smpl_torque config input format)
        config = {
            'betas': betas,
            'gender': gender,
        }
        self.config_output.send(config)

        # Compute limb lengths using SMPLProcessor
        try:
            processor = SMPLProcessor(
                framerate=60.0,
                betas=betas,
                gender=gender,
                total_mass_kg=total_mass,
                model_path=os.path.dirname(os.path.abspath(__file__))
            )
            lengths = processor.limb_data.get('lengths', {})
            masses = processor.limb_data.get('masses', {})

            # Also compute total scaled mass for display
            scale_mass = 1.0
            if len(betas) > 1:
                scale_mass += betas[0] * 0.08
                scale_mass += betas[1] * 0.20
            scaled_total = total_mass * max(0.4, scale_mass)

            result = {
                'lengths': lengths,
                'masses': masses,
                'total_mass_scaled': scaled_total,
            }
            offsets = processor.limb_data.get('offsets')
            if offsets is not None:
                result['offsets'] = offsets
            self.limb_lengths_output.send(result)
        except Exception as e:
            print(f"SMPLBetaEditorNode: Error computing limb properties: {e}")

    def execute(self):
        """Triggered by betas_in input if it has triggers_execution."""
        if self.betas_input.fresh_input:
            self._on_betas_received()


class ShadowToSMPLNode(Node):
    """
    Converts Shadow mocap rotations to SMPL format so that the
    SMPL skeleton reproduces the same physical world-space pose.

    Uses per-joint correction: q_smpl_i = C_parent(i) * q_shadow_i * C_i^-1
    where C_i maps joint i's children's Shadow offset directions to SMPL.

    Accepts quaternion (20x4, wxyz) or axis-angle (20x3) input.
    Outputs quaternion (24x4, wxyz) or axis-angle (22x3) in SMPL joint order.
    """

    # SMPL parent indices
    SMPL_PARENTS = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
        18, 19, 20, 21,
    ]

    # SMPL children (built from parents)
    SMPL_CHILDREN = None  # {parent_idx: [child_idx, ...]}

    @classmethod
    def _build_children_map(cls):
        children = {i: [] for i in range(24)}
        for i in range(1, 24):
            p = cls.SMPL_PARENTS[i]
            children[p].append(i)
        cls.SMPL_CHILDREN = children

    @staticmethod
    def factory(name, data, args=None):
        return ShadowToSMPLNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.config_input = self.add_input('config')

        self.output_format_prop = self.add_property('output_format', widget_type='combo', default_value='quaternions')
        self.output_format_prop.widget.combo_items = ['quaternions', 'axis_angle']

        self.pose_output = self.add_output('pose')
        self.trans_output = self.add_output('trans')

        if ShadowToSMPLNode.SMPL_CHILDREN is None:
            ShadowToSMPLNode._build_children_map()

        self._betas = None
        self._gender = 'neutral'

        # C_i corrections (24 Rotation objects)
        self._C = [Rotation.identity()] * 24
        self._C_inv = [Rotation.identity()] * 24

        # Load both skeletons and compute corrections
        self._shadow_offsets = self._load_shadow_offsets()  # {bmolab_idx: offset_vec}
        self._recompute_corrections()

    def _load_shadow_offsets(self):
        """Load Shadow bone offset vectors, keyed by bmolab_active index."""
        def_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'definition.xml'
        if not def_path.exists():
            def_path = Path('dpg_system') / 'definition.xml'
        if not def_path.exists():
            print("ShadowToSMPLNode: definition.xml not found")
            return {}

        # Step 1: Load all offsets keyed by body joint index (joint_name_to_bmolab_index)
        tree = ET.parse(str(def_path.resolve()))
        root = tree.getroot()
        body_idx_offsets = {}
        for node in root.iter('node'):
            if 'translate' in node.attrib:
                limb_name = node.attrib['id']
                ji = JointTranslator.shadow_limb_name_to_bmolab_index(limb_name)
                if ji != -1:
                    vals = list(map(float, node.attrib['translate'].split(' ')))
                    body_idx_offsets[ji] = np.array(vals) / 100.0

        # Step 2: Build body_joint_index → bmolab_active_index mapping
        # For 0-19: identical. For >= 20: diverge because body has TopOfHead at 20
        # and different ordering. This explicit map handles the ≥20 cases.
        body_to_active = {i: i for i in range(20)}  # 0-19 are identical
        # body joint name (t_* index) → bmolab_active name (index)
        body_to_active[t_TopOfHead] = -1            # 20 → no bmolab_active equivalent
        body_to_active[t_LeftBallOfFoot] = 20        # 21 → left_foot (20)
        body_to_active[t_LeftToeTip] = 24            # 22 → left_toe_tip (24)
        body_to_active[t_RightBallOfFoot] = 21       # 23 → right_foot (21)
        body_to_active[t_RightToeTip] = 25           # 24 → right_toe_tip (25)
        body_to_active[t_LeftKnuckle] = 22           # 25 → left_hand (22)
        body_to_active[t_LeftFingerTip] = 26         # 26 → left_finger_tip (26)
        body_to_active[t_RightKnuckle] = 23          # 27 → right_hand (23)
        body_to_active[t_RightFingerTip] = 27        # 28 → right_finger_tip (27)
        body_to_active[t_LeftHeel] = 28              # 29 → left_heel (28)
        body_to_active[t_RightHeel] = 29             # 30 → right_heel (29)

        # Step 3: Re-key offsets to bmolab_active indices
        offsets = {}
        for body_idx, vec in body_idx_offsets.items():
            active_idx = body_to_active.get(body_idx, body_idx)
            offsets[active_idx] = vec

        print(f"ShadowToSMPLNode: loaded {len(offsets)} shadow offsets")
        return offsets

    def _load_smpl_offsets(self):
        """Load SMPL offset vectors from smplx model. Returns (24, 3)."""
        try:
            import torch
            import smplx
        except ImportError:
            print("ShadowToSMPLNode: smplx/torch unavailable")
            return None

        model_path = os.path.dirname(os.path.abspath(__file__))
        gender_map = {'male': 'MALE', 'female': 'FEMALE', 'neutral': 'MALE'}
        g_tag = gender_map.get(self._gender, 'MALE')

        try:
            model = smplx.create(model_path=model_path, model_type='smplh',
                                 gender=g_tag, num_betas=10, ext='pkl')
            betas_tensor = torch.zeros(1, 10)
            if self._betas is not None:
                b = torch.tensor(self._betas, dtype=torch.float32).flatten()
                n = min(len(b), 10)
                betas_tensor[0, :n] = b[:n]

            output = model(betas=betas_tensor)
            joints = output.joints[0].detach().cpu().numpy()
            parents = self.SMPL_PARENTS
            offsets = np.zeros((24, 3))
            for i in range(1, 24):
                child_idx = i
                if joints.shape[0] > 24 and i == 23:
                    child_idx = 37
                offsets[i] = joints[child_idx] - joints[parents[i]]
            return offsets
        except Exception as e:
            print(f"ShadowToSMPLNode: Error loading SMPL model: {e}")
            return None

    def _bmolab_to_smpl_index(self):
        """Build bmolab active index → SMPL index map."""
        m = {}
        for smpl_name, bmolab_name in JointTranslator.smpl_to_bmolab_active_joint_map.items():
            if smpl_name in JointTranslator.smpl_joints and bmolab_name in JointTranslator.bmolab_active_joints:
                si = JointTranslator.smpl_joints[smpl_name]
                bi = JointTranslator.bmolab_active_joints[bmolab_name]
                m[bi] = si
        return m

    def _smpl_to_bmolab_index(self):
        """Build SMPL index → bmolab active index map."""
        m = {}
        for smpl_name, bmolab_name in JointTranslator.smpl_to_bmolab_active_joint_map.items():
            if smpl_name in JointTranslator.smpl_joints and bmolab_name in JointTranslator.bmolab_active_joints:
                si = JointTranslator.smpl_joints[smpl_name]
                bi = JointTranslator.bmolab_active_joints[bmolab_name]
                m[si] = bi
        return m

    def _recompute_corrections(self):
        """Compute per-joint C_i corrections based on children's offset directions."""
        smpl_offsets = self._load_smpl_offsets()
        if smpl_offsets is None or not self._shadow_offsets:
            print("ShadowToSMPLNode: Cannot compute corrections (missing skeleton data)")
            return

        s2b = self._smpl_to_bmolab_index()  # SMPL idx → bmolab idx
        children = self.SMPL_CHILDREN
        C = [Rotation.identity()] * 24

        for i in range(24):
            child_indices = children[i]
            if not child_indices:
                continue  # leaf: C = identity

            # Gather child offset direction pairs
            shadow_dirs = []
            smpl_dirs = []
            for ci in child_indices:
                bi = s2b.get(ci, -1)
                if bi < 0 or bi not in self._shadow_offsets:
                    continue
                sv = self._shadow_offsets[bi]
                mv = smpl_offsets[ci]
                sn = np.linalg.norm(sv)
                mn = np.linalg.norm(mv)
                if sn < 1e-6 or mn < 1e-6:
                    continue
                shadow_dirs.append(sv / sn)
                smpl_dirs.append(mv / mn)

            if not shadow_dirs:
                continue

            if len(shadow_dirs) == 1:
                # Single child: exact rotation between directions
                C[i] = self._rotation_between(shadow_dirs[0], smpl_dirs[0])
            else:
                # Multi-child: SVD best-fit rotation (Wahba's problem)
                try:
                    r, _ = Rotation.align_vectors(
                        np.array(smpl_dirs),
                        np.array(shadow_dirs)
                    )
                    C[i] = r
                except Exception:
                    C[i] = self._rotation_between(shadow_dirs[0], smpl_dirs[0])

        self._C = C
        self._C_inv = [c.inv() for c in C]

        # Debug: print correction angles
        smpl_names = list(JointTranslator.smpl_joints.keys())
        for i in range(24):
            angle = C[i].magnitude() * 180.0 / np.pi
            name = smpl_names[i] if i < len(smpl_names) else f'j{i}'
            if angle > 0.1:
                print(f"  ShadowToSMPL C[{name}]: {angle:.1f}°")

    @staticmethod
    def _rotation_between(v_from, v_to):
        """Rotation taking unit vector v_from to v_to."""
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        if dot > 0.9999:
            return Rotation.identity()
        if dot < -0.9999:
            perp = np.array([1., 0., 0.])
            if abs(np.dot(v_from, perp)) > 0.9:
                perp = np.array([0., 1., 0.])
            axis = np.cross(v_from, perp)
            axis /= np.linalg.norm(axis)
            return Rotation.from_rotvec(axis * np.pi)
        sn = np.linalg.norm(cross)
        axis = cross / sn
        angle = np.arctan2(sn, dot)
        return Rotation.from_rotvec(axis * angle)

    def execute(self):
        # Handle config changes (betas/gender affect SMPL offsets)
        if self.config_input.fresh_input:
            cfg = self.config_input()
            if isinstance(cfg, dict):
                changed = False
                if 'betas' in cfg:
                    b = any_to_array(cfg['betas']).flatten()
                    if self._betas is None or not np.array_equal(self._betas, b):
                        self._betas = b
                        changed = True
                if 'gender' in cfg:
                    g = str(cfg['gender'])
                    if g != self._gender:
                        self._gender = g
                        changed = True
                if changed:
                    self._recompute_corrections()

        if not self.pose_input.fresh_input:
            return

        raw_pose = self.pose_input()
        if raw_pose is None:
            return

        bmolab_data = any_to_array(raw_pose)

        # Reshape flattened input
        if bmolab_data.ndim == 1:
            if bmolab_data.size == 80:
                bmolab_data = bmolab_data.reshape(20, 4)
            elif bmolab_data.size == 60:
                bmolab_data = bmolab_data.reshape(20, 3)
            else:
                return

        is_quat = bmolab_data.shape[-1] == 4

        # Build SMPL index → bmolab index map
        s2b = self._smpl_to_bmolab_index()
        parents = self.SMPL_PARENTS
        C = self._C
        C_inv = self._C_inv

        # Output: 24 joints × 4 (quaternion wxyz)
        smpl_quats = np.zeros((24, 4))
        smpl_quats[:, 0] = 1.0  # identity default

        for smpl_i in range(24):
            bmolab_i = s2b.get(smpl_i, -1)
            if bmolab_i < 0 or bmolab_i >= bmolab_data.shape[0]:
                # No Shadow data for this joint — apply T-pose correction only
                parent_i = parents[smpl_i]
                if parent_i >= 0:
                    r = C[parent_i] * C_inv[smpl_i]
                else:
                    r = C_inv[smpl_i]
                smpl_quats[smpl_i] = r.as_quat(scalar_first=True)
                continue

            # Get Shadow local rotation
            if is_quat:
                q_shadow = Rotation.from_quat(bmolab_data[bmolab_i], scalar_first=True)
            else:
                q_shadow = Rotation.from_rotvec(bmolab_data[bmolab_i])

            # Apply correction: q_smpl = C_parent * q_shadow * C_self^-1
            parent_i = parents[smpl_i]
            if parent_i >= 0:
                q_smpl = C[parent_i] * q_shadow * C_inv[smpl_i]
            else:
                # Root: no parent correction
                q_smpl = q_shadow * C_inv[smpl_i]

            smpl_quats[smpl_i] = q_smpl.as_quat(scalar_first=True)

        # Output format
        if self.output_format_prop() == 'axis_angle':
            rots = Rotation.from_quat(smpl_quats, scalar_first=True)
            self.pose_output.send(rots.as_rotvec().astype(np.float32))
        else:
            self.pose_output.send(smpl_quats)

        # Pass through translation
        trans = self.trans_input()
        if trans is not None:
            self.trans_output.send(any_to_array(trans))



