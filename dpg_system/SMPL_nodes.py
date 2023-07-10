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

def register_smpl_nodes():
    Node.app.register_node("smpl_take", SMPLTakeNode.factory)
    Node.app.register_node("smpl_pose_to_joints", SMPLPoseToJointsNode.factory)
    Node.app.register_node("smpl_body", SMPLBodyNode.factory)

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

