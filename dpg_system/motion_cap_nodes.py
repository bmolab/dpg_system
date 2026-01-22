import platform as _platform
import subprocess
import os

import numpy as np
import quaternion
import torch
from dpg_system.body_base import *
from pyquaternion import Quaternion
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML
import scipy
import copy
import datetime
import json

#
# print('before body_defs')
# import body_defs
# # from body_defs import JointTranslator
# print('after body_defs')
import dpg_system.MotionSDK as shadow
import random
from dpg_system.smpl_nodes import SMPLNode

def register_motion_cap_nodes():
    Node.app.register_node('gl_body', MoCapGLBody.factory)
    Node.app.register_node('gl_simple_body', SimpleMoCapGLBody.factory)
    Node.app.register_node('gl_alt_body', AlternateMoCapGLBody.factory)
    Node.app.register_node('take', MoCapTakeNode.factory)
    Node.app.register_node('take_dict', OpenTakeNode.factory)
    Node.app.register_node('body_to_joints', MoCapBody.factory)
    Node.app.register_node('shadow_body_to_joints', MoCapBody.factory)
    Node.app.register_node('pose', PoseNode.factory)
    Node.app.register_node('shadow_pose', PoseNode.factory)
    Node.app.register_node('active_joints', ActiveJointsNode.factory)
    Node.app.register_node('shadow', MotionShadowNode.factory)
    Node.app.register_node('local_to_global_body', LocalToGlobalBodyNode.factory)
    Node.app.register_node('global_to_local_body', GlobalToLocalBodyNode.factory)
    Node.app.register_node('target_pose', TargetPoseNode.factory)
    Node.app.register_node('calibrate_pose', PoseCalibrateNode.factory)
    Node.app.register_node('limb_size', LimbSizingNode.factory)
    Node.app.register_node('quaternion_diff_and_axis', DiffQuaternionsNode.factory)
    Node.app.register_node('check_burst', CheckBurstNode.factory)
    Node.app.register_node('json_npz_frame_picker', JsonRandomEventWindowNode.factory)

def find_process_id(process_name):
    try:
        result = subprocess.run(['pgrep', process_name], capture_output=True, text=True, check=True)
        pid = result.stdout.split('\n')
        return pid
    except subprocess.CalledProcessError:
        return None
    except ValueError:
         return None

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

    active_joint_map = {
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

    active_to_shadow_map = [2, 17, 1, 32, 31, 4, 14, 12, 8, 28, 26, 22, 13, 5, 9, 10, 27, 19, 23, 24]
    shadow_to_active_map = JointTranslator.joints_to_input_vector

    @staticmethod
    def factory(name, data, args=None):
        node = MoCapNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.joint_offsets = []
        for key in self.joint_map:
            index = self.joint_map[key]
            self.joint_offsets.append(index)


class MoCapTakeNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoCapTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        speed = 1
        self.buffer = None
        self.frames = 0
        self.current_frame = 0

        self.quat_buffer = None
        self.position_buffer = None
        self.label_buffer = None
        self.streaming = False
        self.record_quat_sequence = []
        self.record_position_sequence = []

        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.load_button = self.add_input('load', widget_type='button', callback=self.load_take)
        self.file_name = self.add_label('')
        self.frame_input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.quaternions_out = self.add_output('quaternions')
        self.positions_out = self.add_output('positions')
        self.labels_out = self.add_output('labels')
        self.dump_button = self.add_input('dump', widget_type='button', callback=self.dump_take)
        self.dump_out = self.add_output('dump')
        self.add_spacer()

        self.record_button = self.add_input('record', widget_type='button', callback=self.start_recording)
        self.recording = False
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop_recording)
        self.quaternions_input = self.add_input('quaternions in', callback=self.quaternions_received)
        self.record_positions_input = self.add_input('record positions', widget_type='checkbox', default_value=False)
        self.positions_input = self.add_input('positions in', callback=self.positions_received)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_sequence)
        self.temp_save_name = ''

        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path, callback=self.load_from_load_path)
        self.message_handlers['load'] = self.load_take_message
        self.new_positions = False
        self.load_take_task = -1

    def start_recording(self):
        if self.streaming:
            self.remove_frame_tasks()
            self.streaming = False
            self.on_off.set(False)
        self.record_quat_sequence = []
        self.record_position_sequence = []
        self.frame_count = 0
        self.recording = True

    def stop_recording(self):
        if self.recording:
            if len(self.record_quat_sequence) > 0:
                self.quat_buffer = np.array(self.record_quat_sequence)
                if self.record_positions_input():
                    self.position_buffer = np.array(self.record_position_sequence)
                else:
                    self.position_buffer = None
                self.label_buffer = None
                self.frames = self.quat_buffer.shape[0]
                self.current_frame = 0
                self.frame_input.set(self.current_frame)
                starttime = datetime.datetime.now()
                self.temp_save_name = datetime.datetime.strftime(starttime, 'temp_mocap_take_%Y%m%d_%H%M%S.npz')
                self.save_take(self.temp_save_name)
            self.recording = False

    def save_sequence(self):
        arg = self.save_button()
        if type(arg) == str:
            save_path = arg
            if self.save_take(save_path):
                return
        SaveDialog(self, self.save_file_callback, extensions=['.npz'])

    def save_file_callback(self, save_path):
        if not self.save_take(save_path):
            print('failed to save')

    def save_take(self, save_path):
        if save_path != '':
            if self.quat_buffer is not None:
                if self.position_buffer is not None:
                    np.savez(save_path, quats=self.quat_buffer, positions=self.position_buffer)
                else:
                    np.savez(save_path, quaternions=self.quat_buffer)
                if self.load_path() == self.temp_save_name and self.temp_save_name[:15] == 'temp_mocap_take':
                    os.remove(self.load_path())
                self.load_path.set(save_path)
                self.file_name.set(save_path)
                return True
        return False

    def positions_received(self):
        self.new_positions = True

    def quaternions_received(self):
        if self.recording:
            if self.record_positions_input():
                if self.new_positions:
                    self.record_position_sequence.append(any_to_array(self.positions_input()).copy())
                    self.new_positions = False
                else:
                    print('take: positions expected but not received')
                    return
            self.record_quat_sequence.append(any_to_array(self.quaternions_input()).copy())
            self.frame_count = len(self.record_quat_sequence)
            self.frame_input.set(self.frame_count, propagate=False)

    def dump_take(self):
        self.dump_out.send(self.position_buffer)

    def start_stop_streaming(self):
        if self.on_off():
            if not self.streaming and self.frames > 0:
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
        self.frame_input.set(self.current_frame)
        frame = int(self.current_frame)
        if self.label_buffer is not None:
            self.labels_out.send(self.label_buffer[frame])
        if self.position_buffer is not None:
            self.positions_out.send(self.position_buffer[frame])
        if self.quat_buffer is not None:
            self.quaternions_out.send(self.quat_buffer[frame])

    def frame_widget_changed(self):
        data = self.frame_input()
        if data < self.frames:
            self.current_frame = data
            if self.label_buffer is not None:
                self.labels_out.send(self.label_buffer[self.current_frame])
            if self.position_buffer is not None:
                self.positions_out.send(self.position_buffer[self.current_frame])
            self.quaternions_out.send(self.quat_buffer[self.current_frame])

    def execute(self):
        if self.frame_input.fresh_input:
            data = self.frame_input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            if t == int:
                if data < self.frames:
                    self.current_frame = int(data)
                    self.labels_out.send(self.label_buffer[self.current_frame])
                    self.positions_out.send(self.position_buffer[self.current_frame])
                    self.quaternions_out.send(self.quat_buffer[self.current_frame])

    def load_take_message(self, message='', args=None):
        if args is not None:
            if len(args) > 0:
                path = any_to_string(args[0])
                self.load_take_from_npz(path)
            else:
                self.load_take(args)

    def load_take(self, args=None):
        arg = self.load_button()
        if type(arg) == str:
            if os.path.exists(arg):
                try:
                    self.load_path.set(arg)
                    self.load_take_from_npz(arg)
                except Exception as e:
                    print('load_npz_callback: error loading take file:', e, arg)
                return

        LoadDialog(self, self.load_npz_callback, extensions=['.npz'])

    def load_npz_callback(self, load_path):
        if load_path != '':
            try:
                self.load_path.set(load_path)
                self.load_take_from_npz(load_path)
            except Exception as e:
                print('load_npz_callback: error loading take file:', e, load_path)

    def load_from_load_path(self):
        path = self.load_path()
        if path != '':
            try:
                self.load_take_from_npz(path)
            except Exception as e:
                print('no take file found:', path)

    def load_take_from_npz(self, path):
        take_file = np.load(path)
        file_name = path.split('/')[-1]
        self.file_name.set(file_name)
        self.load_path.set(path)
        if 'quats' in take_file:
            self.quat_buffer = take_file['quats']
            for idx, quat in enumerate(self.quat_buffer):
                if quat[10, 0] < 0:
                    self.quat_buffer[idx, 10] *= -1
        else:
            self.quat_buffer = None

        self.frames = self.quat_buffer.shape[0]
        if 'positions' in take_file:
            self.position_buffer = take_file['positions']
        else:
            self.position_buffer = None
        if 'labels' in take_file:
            self.label_buffer = take_file['labels']
        else:
            self.label_buffer = None
        self.current_frame = 0
        self.start_stop_streaming()


def take_cancel_callback(sender, app_data):
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1


class OpenTakeNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = OpenTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        speed = 1
        self.buffer = None
        self.frame_count = 0
        self.current_frame = 0
        self.clip_start = 0
        self.clip_end = -1

        self.streaming = False
        self.take_dict = {}
        self.global_dict = {}
        self.sequence_keys = []
        self.global_keys = []

        self.take_data_in = self.add_input('take data in', callback=self.take_data_received)
        self.global_data_in = self.add_input('global data in', callback=self.receive_globals)

        self.load_button = self.add_input('load', widget_type='button', callback=self.load_take)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_sequence)

        self.file_name = self.add_label('')
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path,
                                         callback=self.load_from_load_path)
        self.record_button = self.add_input('record', widget_type='button', callback=self.record_button_clicked)
        self.record_button.name_archive.append('stop record')
        self.play_pause_button = self.add_input('play', widget_type='button', callback=self.play_button_clicked)
        self.play_pause_button.name_archive.append('pause')
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop_button_clicked)
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=True)
        self.external_clock_enable_input = self.add_input('enable external clock', widget_type='checkbox', default_value=False)
        self.external_clock_input = self.add_input('external clock', callback=self.external_play)
        self.add_spacer()
        self.frame_input = self.add_input('frame', widget_type='drag_int', callback=self.frame_widget_changed)
        self.length_property = self.add_input('length: 0', widget_type='label')
        self.speed = self.add_input('play speed', widget_type='drag_float', default_value=speed)
        self.add_spacer()

        self.clip_start_input = self.add_input('clip start frame', widget_type='drag_int', callback=self.clip_changed, trigger_button=True, trigger_callback=self.clip_start_set)
        self.clip_end_input = self.add_input('clip end frame', widget_type='drag_int', callback=self.clip_changed, trigger_button=True, trigger_callback=self.clip_end_set)
        self.save_clip_button = self.add_input('save clip', widget_type='button', callback=self.save_clip)

        self.reset_clip_button = self.add_input('reset clip', widget_type='button', callback=self.reset_clip)

        self.add_spacer()
        self.dump_button = self.add_input('dump', widget_type='button', callback=self.dump_take)
        self.dump_out = self.add_output('dump')
        self.global_params_out = self.add_output('globals')
        self.take_data_out = self.add_output('take data out')
        self.frame_out = self.add_output('frame')
        self.done_out = self.add_output('done')
        self.temp_save_name = ''
        self.last_frame_out = -1
        self.recording = False

        self.message_handlers['load'] = self.load_take_message
        self.new_positions = False
        self.load_take_task = -1

    def custom_create(self, from_file):
        self.load_button.widget.set_active_theme(Node.active_theme_blue)
        self.load_button.widget.set_height(24)
        self.save_button.widget.set_active_theme(Node.active_theme_blue)
        self.save_button.widget.set_height(24)
        self.record_button.widget.set_active_theme(Node.active_theme_pink)
        self.record_button.widget.set_height(24)
        self.stop_button.widget.set_active_theme(Node.active_theme_red)
        self.stop_button.widget.set_height(24)
        self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
        self.play_pause_button.widget.set_height(24)

    def receive_globals(self):
        incoming = self.global_data_in()
        t = type(incoming)
        if t is dict:
            self.global_dict = self.global_data_in()
        elif t is list:
            key = incoming[0]
            data = incoming[1:]
            self.global_dict[key] = data

    def reset_clip(self):
        self.clip_start = 0
        self.clip_end = self.frame_count - 1
        self.clip_start_input.set(self.clip_start)
        self.clip_end_input.set(self.clip_end)

    def clip_changed(self):
        self.clip_start = self.clip_start_input()
        self.clip_end = self.clip_end_input()
        if self.clip_start < 0:
            self.clip_start = 0
        elif self.clip_start >= self.frame_count:
            self.clip_start = self.frame_count - 1

        if self.clip_end < self.clip_start:
            self.clip_end = self.clip_start
        elif self.clip_end >= self.frame_count:
            self.clip_end = self.frame_count - 1

        self.clip_start_input.set(self.clip_start)
        self.clip_end_input.set(self.clip_end)

    def clip_start_set(self):
        self.clip_start = self.current_frame
        self.clip_start_input.set(self.clip_start)

    def clip_end_set(self):
        self.clip_end = self.current_frame
        self.clip_end_input.set(self.clip_end)

    def stop_button_clicked(self):
        if self.streaming:
            self.play_button_clicked()
            self.current_frame = 0
        elif self.recording:
            self.record_button_clicked()

    def play_button_clicked(self):
        if not self.streaming and self.frame_count > 0:
            self.last_frame_out = -1
            self.current_frame = self.speed() * -1.0
#            print('play_button_clicked', self.current_frame)
            self.start_playing()
            self.streaming = True
            self.play_pause_button.set_label('pause')
            self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)
        else:
            if self.streaming:
                self.streaming = False
                self.stop_playing()
                self.play_pause_button.set_label('play')
                self.play_pause_button.widget.set_active_theme(Node.active_theme_green)

    def record_button_clicked(self):
        if self.streaming:
            self.streaming = False
            self.stop_playing()
            self.play_pause_button.set_label('play')

        if not self.recording:
            self.take_dict = {}
            self.frame_count = 0
            self.recording = True
            self.record_button.set_label('stop record')
            self.reset_clip()
        else:

            self.record_button.set_label('record')
            self.recording = False

            if len(self.take_dict) > 0:
                for key, value in self.take_dict.items():

                    value = np.array(value)
                    self.take_dict[key] = value
                for key, value in self.global_dict.items():
                    self.take_dict[key] = value
                key = list(self.take_dict.keys())[0]
                self.frame_count = self.take_dict[key].shape[0]
                self.current_frame = 0
                self.frame_input.set(self.current_frame)
                self.clip_start = 0
                self.clip_end = self.frame_count - 1
                self.frame_input.widget.max = self.clip_end
                self.frame_input.widget.min = self.clip_start
                self.clip_start_input.set(self.clip_start)
                self.clip_end_input.set(self.clip_end)
                self.length_property.set('length: ' + str(self.frame_count))
                self.frame_input.widget.max = self.frame_count - 1
                self.frame_input.widget.min = 0
                starttime = datetime.datetime.now()
                path_start = os.getcwd()
                self.temp_save_name = datetime.datetime.strftime(starttime, 'temp_take_%Y%m%d_%H%M%S.npz')
                self.save_take(path_start + '/' + self.temp_save_name)
                self.reset_clip()

    def save_sequence(self):
        arg = self.save_button()
        if type(arg) is list:
            arg = ' '.join(arg)
        if type(arg) == str:
            save_path = arg
            if self.save_take(save_path):
                return
        SaveDialog(self, self.save_file_callback, extensions=['.npz'])

    def save_file_callback(self, save_path):
        if not self.save_take(save_path):
            print('failed to save')

    def save_take(self, save_path):
        if save_path != '':
            try:
                for key, value in self.global_dict.items():
                    self.take_dict[key] = value
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, **self.take_dict)
                if self.load_path() == self.temp_save_name and self.temp_save_name[:9] == 'temp_take':
                    os.remove(self.load_path())
                self.load_path.set(save_path)
                self.file_name.set(save_path)
            except Exception as e:
                print(e)
            return True
        return False

    def save_clip(self):
        arg = self.save_button()
        if type(arg) == str:
            save_path = arg
            if self.save_clip_only(save_path):
                return
        SaveDialog(self, self.save_clip_callback, extensions=['.npz'])

    def save_clip_callback(self, save_path):
        if not self.save_clip_only(save_path):
            print('failed to save clip')

    def save_clip_only(self, save_path):
        if save_path != '':
            clip_dict = {}
            for key, value in self.take_dict.items():
                value = np.array(value)
                clip_value = value[self.clip_start:self.clip_end + 1]
                clip_dict[key] = clip_value
            for key, value in self.global_dict.items():
                clip_dict[key] = value
            np.savez(save_path, **clip_dict)
            return True
        return False

    def take_data_received(self):
        if self.recording:
            incoming_dict = self.take_data_in()
            max_len = 0
            if type(incoming_dict) is dict:
                keys_list = list(incoming_dict.keys())
                self.sequence_keys = keys_list.copy()
                for key in keys_list:
                    if key in self.take_dict:
                        self.take_dict[key].append(incoming_dict[key])
                    else:
                        self.take_dict[key] = [incoming_dict[key]]
                    if len(self.take_dict[key]) > max_len:
                        max_len = len(self.take_dict[key])
                self.frame_count = max_len
            else:
                self.frame_count += 1
            self.frame_input.set(self.frame_count, propagate=False)

    def dump_take(self):
        self.dump_out.send(self.take_dict)

    def start_playing(self):
        if not self.external_clock_enable_input():
            self.add_frame_task()

    def stop_playing(self):
        if not self.external_clock_enable_input():
            self.remove_frame_tasks()

    def frame_task(self):
        self.step()

    def external_play(self):
        if self.external_clock_enable_input():
            if self.streaming:
                self.step()

    def step(self):
        self.current_frame += self.speed()
        if int(self.current_frame) > self.last_frame_out:
            if self.current_frame > self.clip_end:
                if self.loop_input():
                    self.current_frame = self.clip_start
                else:
                    self.stop_button_clicked()
                self.done_out.send('done')
                if not self.loop_input():
                    return

            self.frame_input.set(self.current_frame)
            frame = int(self.current_frame)

            self.last_frame_out = frame
            frame_dict = {}
            for key in self.sequence_keys:
                frame_dict[key] = self.take_dict[key][frame]
            self.frame_out.send(self.last_frame_out)
            self.take_data_out.send(frame_dict)

    def load_from_load_path(self):
        path = self.load_path()
        if path != '':
            try:
                self.load_take_from_npz(path)
            except Exception as e:
                print('no take file found:', path)

    def load_take_from_npz(self, path):
        if self.streaming:
            self.stop_button_clicked()
        take_file = np.load(path, allow_pickle=True)
        file_name = path.split('/')[-1]
        self.file_name.set(file_name)
        self.load_path.set(path)
        self.take_dict = dict(take_file)
        keys_list = list(self.take_dict.keys())
        # print('keys_list', keys_list)
        sequence_length = 0
        if len(keys_list) > 0:
            for key in keys_list:
                data = self.take_dict[key]
                t = type(data)
                if t is np.ndarray:
                    if len(data.shape) > 0:
                        if data.shape[0] > sequence_length:
                            sequence_length = data.shape[0]
            self.sequence_keys = []
            self.global_keys = []
            for key in keys_list:
                data = self.take_dict[key]
                t = type(data)
                if t is np.ndarray:
                    if len(data.shape) > 0 and data.shape[0] == sequence_length:
                        self.sequence_keys.append(key)
                    else:
                        self.global_keys.append(key)
                else:
                    self.global_keys.append(key)

            self.global_dict = {}
            for key in self.global_keys:
                data = self.take_dict[key]
                self.global_dict[key] = data
            self.global_dict['length'] = sequence_length
            if len(self.global_dict) > 0:
                self.global_params_out.send(self.global_dict)

            self.frame_count = sequence_length
            self.length_property.set('length: ' + str(self.frame_count))
            self.clip_start = 0
            self.clip_end = self.frame_count - 1
            self.frame_input.widget.max = self.clip_end
            self.frame_input.widget.min = self.clip_start
            self.clip_start_input.set(self.clip_start)
            self.clip_end_input.set(self.clip_end)
            self.current_frame = 0
            self.frame_input.set(self.current_frame)
        self.current_frame = 0

    def frame_widget_changed(self):
        data = self.frame_input()
        if data < 0:
            data = 0
            self.current_frame = data
            self.frame_input.set(self.current_frame)
        if data < self.frame_count:
            self.current_frame = data
            frame_dict = {}
            for key in self.sequence_keys:
                frame_dict[key] = self.take_dict[key][self.current_frame]
            self.frame_out.send(self.current_frame)
            self.take_data_out.send(frame_dict)
        else:
            data = self.frame_count - 1
            self.current_frame = data
            self.frame_input.set(self.current_frame)

    def execute(self):
        if self.frame_input.fresh_input:
            data = self.frame_input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            if t == int:
                if data < self.frame_count:
                    frame_dict = {}
                    for key in self.sequence_keys:
                        frame_dict[key] = frame_dict[key][self.current_frame]
                    self.frame_out.send(self.current_frame)
                    self.take_data_out.send(frame_dict)

    def load_take_message(self, message='', args=None):
        if args is not None:
            if len(args) > 0:
                path = any_to_string(args[0])
                self.load_take_from_npz(path)
            else:
                self.load_take(args)

    def load_take(self, args=None):
        arg = self.load_button()
        if type(arg) == str:
            if os.path.exists(arg):
                try:
                    self.load_path.set(arg)
                    self.load_take_from_npz(arg)
                except Exception as e:
                    print('load_npz_callback: error loading take file:', e, arg)
                return
        LoadDialog(self, self.load_npz_callback, extensions=['.npz'])

    def load_npz_callback(self, load_path):
        if load_path != '':
            try:
                self.load_path.set(load_path)
                self.load_take_from_npz(load_path)
            except Exception as e:
                print('load_npz_callback: error loading take file:', e, load_path)


class PoseNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = PoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shadow = False
        self.joint_inputs = []
        if label == 'shadow_pose':
            self.shadow = True
            for key in self.joint_map:
                stripped_key = key.replace('_', ' ')
                input_ = self.add_input(stripped_key, triggers_execution=True, default_value=[1.0, 0.0, 0.0, 0.0])
                self.joint_inputs.append(input_)
        else:
            for key in self.active_joint_map:
                stripped_key = key.replace('_', ' ')
                input_ = self.add_input(stripped_key, triggers_execution=True, default_value=[1.0, 0.0, 0.0, 0.0])
                self.joint_inputs.append(input_)

        self.output = self.add_output('pose out')

    def execute(self):
        # we need to know the size of the pose vector...
        if self.shadow:
            pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 37)
            for index, joint_input in enumerate(self.joint_inputs):
                incoming = joint_input()
                t = type(incoming)
                if t == torch.Tensor:
                    incoming = tensor_to_array(incoming)
                    t = np.ndarray
                if t == np.ndarray:
                    offset = self.joint_offsets[index]
                    pose[offset] = incoming
        else:
            pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 20)
            for index, joint_input in enumerate(self.joint_inputs):
                incoming = joint_input()
                t = type(incoming)
                if t == torch.Tensor:
                    incoming = tensor_to_array(incoming)
                    t = np.ndarray
                if t == np.ndarray:
                    pose[index] = incoming
        self.output.send(pose)


class PoseCalibrateNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = PoseCalibrateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_correction = None
        self.input = self.add_input('pose input', triggers_execution=True)
        self.calibration_input = self.add_input('calibration input', callback=self.set_calibration)
        self.calibrate_input = self.add_input('calibrate input', widget_type='button', callback=self.calibrate)

        self.output = self.add_output('pose out')
        self.calibration_output = self.add_output('calibration output')

    def set_calibration(self):
        self.pose_correction = any_to_array(self.calibration_input())

    def calibrate(self):
        # calculate inverse of input pose
        in_pose = self.input()
        if in_pose is not None and type(in_pose) is np.ndarray:
            self.pose_correction = np.zeros_like(in_pose)
            for i in range(in_pose.shape[0]):
                q = Quaternion(in_pose[i])
                qq = q.inverse
                self.pose_correction[i] = np.array(qq.q)
        self.calibration_output.send(self.pose_correction)

    def execute(self):
        in_pose = self.input()
        if self.pose_correction is not None:
            out_pose = np.zeros_like(in_pose)
            for i in range(in_pose.shape[0]):
                q = Quaternion(in_pose[i])
                pc = Quaternion(self.pose_correction[i])
                qq = q * pc
                out_pose[i] = [qq.w, qq.x, qq.y, qq.z]
            self.output.send(out_pose)


class MoCapBody(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoCapBody(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('pose in', triggers_execution=True)
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.joint_outputs = []
        self.shadow = False
        if label == 'shadow_body_to_joints':
            self.shadow = True

        if self.shadow:
            for key in self.joint_map:
                stripped_key = key.replace('_', ' ')
                output = self.add_output(stripped_key)
                self.joint_outputs.append(output)
        else:
            for key in self.active_joint_map:
                stripped_key = key.replace('_', ' ')
                output = self.add_output(stripped_key)
                self.joint_outputs.append(output)

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input()
            t = type(incoming)
            if t == torch.Tensor:
                incoming = tensor_to_array(incoming)
                t = np.ndarray
            if t == np.ndarray:
                if self.shadow:
                    if incoming.shape[0] == 37:
                        for i, index in enumerate(self.joint_offsets):
                            if index < incoming.shape[0]:
                                joint_value = incoming[index]
                                self.joint_outputs[i].set_value(joint_value)
                    elif incoming.shape[0] == 20:
                        for i, active_index in enumerate(self.active_to_shadow_map):
                            if active_index < incoming.shape[0]:
                                joint_value = incoming[active_index]
                                self.joint_outputs[i].set_value(joint_value)
                else:
                    if incoming.shape[0] == 37:
                        for i, index in enumerate(self.shadow_to_active_map):
                            if index != -1 and index < len(self.joint_outputs):
                                joint_value = incoming[i]
                                self.joint_outputs[index].set_value(joint_value)
                    elif incoming.shape[0] == 20:
                        for i in range(20):
                            self.joint_outputs[i].set_value(incoming[i])
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
        self.gl_chain_input = self.add_input('gl chain', callback=self.draw)
        self.gl_chain_output = self.add_output('gl_chain')
        self.capture_pose_input = self.add_input('capture pose', widget_type='button', callback=self.capture_pose)
        self.joint_data_input = self.add_input('joint data', callback=self.receive_joint_data)
        self.clear_joint_data_input = self.add_input('clear joint data', widget_type='button', callback=self.clear_joint_data)
        # y-up
        # transl
        self.current_joint_output = self.add_output('current_joint_name')
        self.current_joint_data_output = self.add_output('current_joint_data')
        self.current_joint_gl_output = self.add_output('current_joint_gl_chain')

        self.absolute_quats_input = self.add_option('absolute quats', widget_type='checkbox')
        self.calc_diff_quats = self.add_option('calc_diff_quats', widget_type='checkbox', default_value=False, callback=self.set_calc_diff)
        self.skeleton_only = self.add_option('skeleton_only', widget_type='checkbox', default_value=False)
        self.z_up_option = self.add_option('z_up', widget_type='checkbox', default_value=False)
        self.joint_axes = self.add_option('joint_axes', widget_type='checkbox', default_value=False)
        self.show_joint_spheres = self.add_option('show joint motion', widget_type='checkbox', default_value=self.show_joint_activity)
        self.joint_indicator = self.add_option('joint indicator', widget_type='combo', default_value='sphere')
        self.joint_indicator.widget.combo_items = ['sphere', 'disk']
        self.joint_data_selection = self.add_option('joint data type', widget_type='combo', default_value='diff_axis-angle')
        self.joint_data_selection.widget.combo_items = ['diff_quaternion', 'diff_axis-angle']
        self.joint_motion_scale = self.add_option('joint motion scale', widget_type='drag_float', default_value=1)
        self.diff_quat_smoothing_A = self.add_option('joint motion smoothing a', widget_type='drag_float', default_value=0.8, max=1.0, min=0.0)
        self.diff_quat_smoothing_B = self.add_option('joint motion smoothing b', widget_type='drag_float',
                                                   default_value=0.9, max=1.0, min=0.0)

        self.orientation_before = self.add_option('orientation before rotation', widget_type='checkbox', default_value=False)
        self.joint_disk_alpha = self.add_option('joint motion alpha', widget_type='drag_float', default_value=0.5, max=1.0, min=0.0)
        self.body_color_id = self.add_option('colour id', widget_type='input_int', default_value=0)
        self.limb_sizes_out = self.add_output('limb_sizes')
        self.body = BodyData()
        self.body.node = self
        self.external_joint_data = None

    def clear_joint_data(self):
        self.external_joint_data = None

    def set_calc_diff(self):
        self.body.calc_diff = self.calc_diff_quats()

    def capture_pose(self):
        self.body.capture_current_pose()

    def process_commands(self, command):
        if type(command[0]) == str:
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in JointTranslator.joint_name_to_bmolab_index:
                    target_joint_index = JointTranslator.joint_name_to_bmolab_index[target_joint]
                    self.body.joints[target_joint_index].bone_translation = command[1:]
                    self.body.joints[target_joint_index].set_matrix()
                # self.body.joints[target_joint_index].set_mass()
            elif command[0] == 'limb_vertices':
                joint_name = command[1]
                limb_vertices = []
                for i in range(len(command) - 2):
                    spec = any_to_list(command[i + 2])
                    if len(spec) < 3:
                        for i in range(3 - len(spec)):
                            spec.append(0)
                    limb_vertices.append(spec)
                self.set_limb_vertices(joint_name, limb_vertices)
            elif command[0] == 'limb_size':
                if len(command) > 1:
                    joint_name = command[1]
                    if joint_name in JointTranslator.joint_name_to_bmolab_index:
                        joint_index = JointTranslator.joint_name_to_bmolab_index[joint_name]
                        if len(command) > 2:
                            dims = command[2:]
                            for i in range(len(dims)):
                                dims[i] = any_to_float(dims[i])
                            self.body.set_limb_dims(joint_index, dims)
            elif command[0] == 'dump_limb_sizes':
                self.dump_limb_sizes()

    def set_limb_vertices(self, name, vertices):
        self.body.set_limb_vertices(name, vertices)

    def custom_create(self, from_file):
        self.dump_limb_sizes()

    def dump_limb_sizes(self):
        limb_sizes = {}
        for joint in self.body.joints:
            limb_sizes[joint.name] = joint.dims
        self.limb_sizes_out.send(limb_sizes)

    def joint_callback(self, joint_index):
        local_index = joint_index
        if joint_index in [20]:
            return
        if joint_index == 21:
            local_index = 20  # handle left foot
        elif joint_index == 23:
            local_index = 21  # handle right foot
        elif joint_index == 25:
            local_index = 22 # handle left hand
        elif joint_index == 27:
            local_index = 23 # handle right hand
        elif joint_index == 22:
            local_index = 24
        elif joint_index == 24:
            local_index = 25
        elif joint_index == 26:
            local_index = 26
        elif joint_index == 28:
            local_index = 27

        if local_index > 27:
            return

        # if joint_index >= t_ActiveJointCount:
        #     return
        if local_index < 0:
            return

        glPushMatrix()

        mode = self.joint_data_selection()
        # joint_name = joint_index_to_name[joint_index]
        self.current_joint_output.send(local_index)
        if self.external_joint_data is not None:
            # in all cases, what if incoming data.shape[0] is 22
            # then we need to remap the joint data from smpl to active
            if type(self.external_joint_data) is np.ndarray:
                if self.external_joint_data.shape[0] >= 20:
                    if local_index < self.external_joint_data.shape[0]:
                        self.current_joint_data_output.send(self.external_joint_data[local_index])
                elif self.external_joint_data.shape[0] == 1:
                    if self.external_joint_data.shape[1] >= 20:
                        if local_index < self.external_joint_data.shape[0]:
                            self.current_joint_data_output.send(self.external_joint_data[0][local_index])
            elif type(self.external_joint_data) is torch.Tensor:
                if self.external_joint_data.shape[0] >= 20:
                    if local_index < self.external_joint_data.shape[0]:
                        self.current_joint_data_output.send(self.external_joint_data[local_index])
                elif self.external_joint_data.shape[0] == 1:
                    if self.external_joint_data.shape[1] >= 20:
                        if local_index < self.external_joint_data.shape[0]:
                            self.current_joint_data_output.send(self.external_joint_data[0][local_index])
            elif type(self.external_joint_data) is list:
                if len(self.external_joint_data) >= 20:
                    if local_index < self.external_joint_data.shape[0]:
                        self.current_joint_data_output.send(self.external_joint_data[local_index])
                elif len(self.external_joint_data) == 1:
                    if len(self.external_joint_data[0]) >= 20:
                        if local_index < self.external_joint_data.shape[0]:
                            self.current_joint_data_output.send(self.external_joint_data[0][local_index])
        elif mode == 'diff_axis-angle':
            if self.body.normalized_axes is not None:
                current_axis = self.body.normalized_axes[0, joint_index]
                if self.body.magnitudes is not None:
                    current_magnitude = self.body.magnitudes[0, joint_index]
                    output_value = np.ndarray(shape=(4))
                    output_value[:3] = current_axis
                    output_value[3] = current_magnitude
                    self.current_joint_data_output.send(output_value)
        elif mode == 'diff_quaternion':
            if self.body.magnitudes is not None:
                value = self.body.magnitudes[0, joint_index]
                self.current_joint_data_output.send(value)
        self.current_joint_gl_output.send('draw')
        glPopMatrix()

    def receive_joint_data(self):
        data = self.joint_data_input()
        if type(data) is torch.Tensor:
            self.external_joint_data = data.clone()
        elif type(data) is np.ndarray:
            self.external_joint_data = data.copy()
        elif type(data) is list:
            self.external_joint_data = data.copy()
        else:
            self.external_joint_data = None

    def execute(self):
        incoming = self.input()
        t = type(incoming)
        if t == torch.Tensor:
            incoming = tensor_to_array(incoming)
            t = np.ndarray
        if t == np.ndarray:
            if incoming.shape[0] == 80:
                self.body.update_quats(np.reshape(incoming, [20, 4]), self.z_up_option())
            elif incoming.shape[0] == 20:
                if incoming.shape[1] == 4:
                    self.body.update_quats(incoming, self.z_up_option())
            elif incoming.shape[0] == 22:
                if incoming.shape[1] == 4:
                    # smpl joint order
                    converted_joints = JointTranslator.translate_from_smpl_to_bmolab_active(incoming)
                    self.body.update_quats(converted_joints, self.z_up_option())
            elif incoming.shape[0] == 37:
                active_joints = incoming[self.active_to_shadow_map]
                self.body.update_quats(active_joints, self.z_up_option())
            elif incoming.shape[0] == 148:
                incoming = np.reshape(incoming, [37, 4])
                active_joints = incoming[self.active_to_shadow_map]
                self.body.update_quats(active_joints, self.z_up_option())
        elif t in [list, str]:
            if t == str:
                incoming = [incoming]
            self.process_commands(incoming)

    def draw(self):
        incoming = self.gl_chain_input()
        self.body.orientation_before = self.orientation_before()
        t = type(incoming)
        if t == str and incoming == 'draw':
            self.body.joint_display = self.joint_indicator()
            scale = self.joint_motion_scale()
            smoothing_a = self.diff_quat_smoothing_A()
            smoothing_b = self.diff_quat_smoothing_B()

            self.body.joint_motion_scale = scale
            self.body.diffQuatSmoothingA = smoothing_a
            self.body.diffQuatSmoothingB = smoothing_b
            self.body.joint_disk_alpha = self.joint_disk_alpha()
            # if self.absolute_quats_input():
            #     self.body.draw_absolute_quats(self.show_joint_spheres(), self.skeleton_only())
            # else:
            self.body.draw(self.show_joint_spheres(), self.skeleton_only(), self.joint_axes())
            self.gl_chain_output.send('draw')
            # self.external_joint_data = None


class DiffQuaternionsNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DiffQuaternionsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('quaternions in', triggers_execution=True)
        self.smooth_a_input = self.add_input('smoothing A (0-1)', widget_type='drag_float', default_value=0.8, min=0.0, max=1.0)
        self.smooth_b_input = self.add_input('smoothing B (0-1)', widget_type='drag_float', default_value=0.9, min=0.0, max=1.0)
        self.magnitude_out = self.add_output('magnitudes')
        self.normalized_axes_out = self.add_output('axes')
        self.restart_cal_input = self.add_input('restart calculation', widget_type='button', callback=self.restart_cal)
        self.axes = None
        self.normalized_axes = None
        self.magnitudes = None
        self.smoothed_quaternions_a = None
        self.smoothed_quaternions_b = None
        self.diff_quats = None
        self.diff_angles = None

    def execute(self):
        incoming_quats = any_to_array(self.input())
        self.calc_diff_quaternions(incoming_quats)
        self.magnitude_out.send(self.magnitudes)
        self.normalized_axes_out.send(self.normalized_axes)

    def restart_cal(self):
        print('DiffQuaternionsNode armed reset')
        self.smoothed_quaternions_a = None

    def calc_diff_quaternions(self, incoming_quats):
        if self.smoothed_quaternions_a is None:
            print('DiffQuaternionsNode reset')
            self.smoothed_quaternions_a = np.copy(incoming_quats)
            self.smoothed_quaternions_b = np.copy(incoming_quats)
        else:
            self.smoothed_quaternions_a = self.smoothed_quaternions_a * self.smooth_a_input() + incoming_quats * (1.0 - self.smooth_a_input())
            self.smoothed_quaternions_b = self.smoothed_quaternions_b * self.smooth_b_input() + incoming_quats * (1.0 - self.smooth_b_input())
        # we want axis, magnitude from diff_quat

        a = quaternion.as_quat_array(self.smoothed_quaternions_a)
        b = quaternion.as_quat_array(self.smoothed_quaternions_b)

        b_inv = np.conjugate(b) / np.abs(b) ** 2
        diff_quats = a * b_inv

        self.diff_quats = np.expand_dims(diff_quats, axis=0)
        try:
            self.axes = quaternion.as_rotation_vector(self.diff_quats)
        except Exception as a:
            print(e)

        angles = np.linalg.norm(self.axes, axis=2) + 1e-8
        self.magnitudes = self.quaternion_distances(self.smoothed_quaternions_a, self.smoothed_quaternions_b)
        self.normalized_axes = (self.axes / np.expand_dims(angles, axis=-1))
        self.diff_angles = angles

    def quaternion_distances(self, q1, q2):
        q1_ = q1 / np.expand_dims(np.linalg.norm(q1, axis=-1), -1)
        q2_ = q2 / np.expand_dims(np.linalg.norm(q2, axis=-1), -1)
        q1__ = quaternion.as_quat_array(q1_)
        q2__ = quaternion.as_quat_array(q2_)

        distances = quaternion.rotation_intrinsic_distance(q1__, q2__)
        return distances


class CheckBurstNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CheckBurstNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('diff array', triggers_execution=True)
        self.input2 = self.add_input('previous frame array', callback=self.receive_prev)
        self.prev_diff_input = self.add_input('previous diff array', callback=self.receive_prev_diff)
        self.frame_input = self.add_input('frame')
        self.file_input = self.add_input('file')
        self.threshold1_input = self.add_input('threshold 1', widget_type='drag_float', default_value=0.1, min=0.0, max=1.0)
        self.threshold2_input = self.add_input('threshold 2 previous', widget_type='drag_float', default_value=0.01, min=0.0, max=1.0)
        self.threshold_L_input = self.add_input('threshold low', widget_type='drag_float', default_value=0.01, min=0.0, max=1.0)
        self.jerk_threshold = self.add_input('jerk threshold pct', widget_type='drag_float', default_value=0.5, min=0.0, max=5.0)
        self.save_button = self.add_input('save burst files', widget_type='button', callback=self.save_result)
        # add timestamp--node for date and time sting (Datetime node)
        # chack burst date and timestamp begining num (file name, check burst Nov...)
        # take dict: save a temp file--
        # 'date_time'--combine object
        self.save_path_input = self.add_input('save path', widget_type='text_input', default_value='/home/bmolab/Projects/AMASS_2/burst_files.json')
        self.file_dict_out = self.add_output('file_dict')
        self.file_dict = {}

    def receive_prev(self):
        pass

    def receive_prev_diff(self):
        pass

    def execute(self):
        acc = np.asarray(self.input())
        prev_vel = np.asarray(self.input2())
        prev_acc = np.asarray(self.prev_diff_input())

        if self.frame_input() == 0:
            return

        self.detect_sudden_burst(acc, prev_vel, prev_acc)
        self.file_dict_out.send(self.file_dict)

    def detect_sudden_burst(self, acc, prev_vel, prev_acc):

        mag_diff = np.linalg.norm(acc)
        # mag_diff_prev = np.linalg.norm(prev_diff)
        # diff_acceleration = np.abs(mag_diff - mag_diff_prev)
        # avg_mag_diff = (mag_diff + mag_diff_prev) / 2
        # jerk = diff_acceleration > self.jerk_threshold() * avg_mag_diff

        # element-wise
        acc_arr = np.asarray(acc).ravel()
        prev_acc_arr = np.asarray(prev_acc).ravel()
        jerk = np.abs(acc_arr - prev_acc_arr)
        # threshold_arr = np.full(22, 0.2)
        jerk_mask = (jerk > self.threshold_L_input()) & (jerk > self.jerk_threshold() * ((acc_arr + prev_acc_arr) / 2))

        if mag_diff > self.threshold1_input():
            diff_max_i = np.argmax(acc)
            prev_v = prev_vel[diff_max_i]

            if prev_v < self.threshold2_input() or (self.frame_input() > 1 and jerk_mask.any()):
                print(self.file_input())
                print('frame', self.frame_input())
                print('acc', acc)
                print('prev_vel', prev_vel)
                print('prev_v', prev_v)
                print('mag_diff_acc', mag_diff)
                print('prev_acc', prev_acc)
                print('jerk_mask', jerk_mask)

                # jerk = jerk_mask > threshold_arr
                i_jerk = np.where(jerk_mask)[0].tolist()
                values = jerk[i_jerk].tolist()
                print('indices', i_jerk)
                print('values', values)

                # print('mag_diff_prev', mag_diff_prev)
                # print('avg_mag_diff', avg_mag_diff)
                # print('diff_acceleration', diff_acceleration)

                event = {'frame': self.frame_input(),
                         'prev_vel': prev_vel,
                         'prev_v': prev_v,
                         'acc': acc,
                         'prev_acc': prev_acc,
                         'jerk': float(jerk_mask.any()),
                         'mag_diff': mag_diff,
                         'jerk_indices': i_jerk,
                         'jerk_values': values,
                }
                self.file_dict.setdefault(self.file_input(), []).append(event)


    def save_result(self):
        path = self.save_path_input()

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.file_dict, f, indent=2, cls=NumpyEncoder)
                print(f"Saved burst files to {path}")
        except Exception as e:
            print("Error saving burst files:", e)


class JsonRandomEventWindowNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = JsonRandomEventWindowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.events = []
        self.next_event = self.add_input('next', triggers_execution=True, trigger_button=True)
        self.json_path_input = self.add_input('json path', widget_type='text_input',
                                              default_value='/home/bmolab/Projects/AMASS_2/burst_files.json', callback=self.load_json)
        self.joint_input = self.add_input('joint', widget_type='text_input', default_value='')
        self.path_output = self.add_output('npz path')
        self.event_frame_out = self.add_output('event frame')
        self.joints = self.add_output('joints')
        self.jerk_values = self.add_output('jerk values')
        self.jerk_index = self.add_output('jerk index')
        self.prev_acc = self.add_output('prev_acc')
        self.acc = self.add_output('acc')

    def load_json(self):
        json_path = self.json_path_input()
        print("loaded json file", json_path)
        if not json_path or not os.path.exists(json_path):
            self.events = []
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = []

        if isinstance(data, dict):
            for npz_path, event_list in data.items():
                if isinstance(event_list, list):
                    for event in event_list:
                        if isinstance(event, dict) and "frame" in event:
                            joints = []
                            for i in event["jerk_indices"]:
                                joints.append(SMPLNode.joint_names[i])

                            events.append((npz_path, int(event['frame']), joints, event["jerk_indices"], event["jerk_values"],
                                          event['prev_acc'], event['acc']))
        self.events = events

        print(f"loaded {len(self.events)} events")

    def execute(self):
        if not self.events:
            self.load_json()

        joint_str = self.joint_input()
        if joint_str == '':
            npz_path, event_frame, joints, jerk_index , jerk_values, prev_acc, acc = random.choice(self.events)

        else:
            joint_idx = SMPLNode.joint_names.index(joint_str)
            filtered = [(p, fr, joints, jerk_index, jerk_values, acc, prev_acc) for (p, fr, joints, jerk_index, jerk_values, acc, prev_acc) in self.events if joint_idx in jerk_index]
            npz_path, event_frame, joints, jerk_index, jerk_values, acc, prev_acc = random.choice(filtered)

        self.path_output.send(npz_path)
        self.event_frame_out.send(event_frame)
        self.joints.send(joints)
        self.jerk_values.send(jerk_values)
        self.jerk_index.send(jerk_index)
        self.acc.send(acc)
        self.prev_acc.send(prev_acc)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)

class SimpleMoCapGLBody(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SimpleMoCapGLBody(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('pose in', triggers_execution=True)
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.gl_chain_output = self.add_output('gl_chain')

        self.skeleton_only = self.add_option('skeleton_only', widget_type='checkbox', default_value=False)
        self.multi_body_translation_x = self.add_option('multi offset x', widget_type='drag_float', default_value=0.0)
        self.multi_body_translation_y = self.add_option('multi offset y', widget_type='drag_float', default_value=0.0)
        self.multi_body_translation_z = self.add_option('multi offset z', widget_type='drag_float', default_value=0.0)
        self.body = SimpleBodyData()
        self.body.node = self

    def joint_callback(self, index):
        pass
    #
    def process_commands(self, command):
        if type(command[0]) == str:
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in JointTranslator.joint_name_to_bmolab_index:
                    target_joint_index = JointTranslator.joint_name_to_bmolab_index[target_joint]
                    if len(command) >= 2:
                        # self.body.limbs[target_joint_index].dims[2] = any_to_float(command[1])
                        self.body.joints[target_joint_index].set_limb_length(any_to_float(command[3]))
                    if len(command) == 3:
                        self.body.limbs[target_joint_index].dims[1] = any_to_float(command[2])
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[2])
                        #self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    if len(command) == 4:
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[3])
                        # self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    # self.body.joints[target_joint_index].set_matrix()
                    self.body.limbs[target_joint_index] = None
                    # self.body.limbs[target_joint_index].new_shape = True
                # self.body.joints[target_joint_index].set_mass()


    def execute(self):
        if self.input.fresh_input:
            incoming = self.input()
            t = type(incoming)
            if t == torch.Tensor:
                incoming = tensor_to_array(incoming)
                t = np.ndarray
            if t == np.ndarray:
                if len(incoming.shape) == 1:
                    if incoming.shape[0] == 80:
                        self.body.update_quats(np.reshape(incoming, [20, 4]))
                elif len(incoming.shape) == 2:
                    if incoming.shape[0] == 20:
                        if incoming.shape[1] == 4:
                            self.body.update_quats(incoming)
                    elif incoming.shape[0] == 80:
                        count = incoming.shape[1]
                        self.body.update_quats(np.reshape(incoming, [20, 4, count]))
                elif len(incoming.shape) == 3:
                    if incoming.shape[1] == 20:
                        if incoming.shape[2] == 4:
                            self.body.update_quats(incoming)

            elif t == list:
                self.process_commands(incoming)

        elif self.gl_chain_input.fresh_input:
            translation = [self.multi_body_translation_x(), self.multi_body_translation_y(), self.multi_body_translation_z()]
            self.body.multi_body_translation = translation
            incoming = self.gl_chain_input()
            t = type(incoming)
            if t == str and incoming == 'draw':
                self.body.draw(False, self.skeleton_only())


def quaternion_multiply(q1, q2):
    """Multiply quaternions q1*q2"""
    # incoming is x, y, z, w
    index = [3, 0, 1, 2]
    q1 = q1[index]      # w, x, y, z
    q2 = q2[index]      # w, x, y, z
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    # a = q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3] - q1[0]*q2[0]
    # b = q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[0] - q1[0]*q2[3]
    # c = q1[1]*q2[3] - q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2]
    # d = q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2] + q1[0]*q2[1]
    return quaternion_norm(np.array([x, y, z, w]))  # x, y, z, w

def quaternion_multiply_wxyz(q1, q2):
    """Multiply quaternions q1*q2"""
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    # a = q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3] - q1[0]*q2[0]
    # b = q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[0] - q1[0]*q2[3]
    # c = q1[1]*q2[3] - q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2]
    # d = q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2] + q1[0]*q2[1]
    return quaternion_norm(np.array([w, x, y, z]))  # x, y, z, w

def quaternion_divide(q1, q2):
    conj = quaternion_conj(q2)  # x, y, z, w
    div = quaternion_multiply(q1, conj)
    return quaternion_norm(div)

def quaternion_reciprocal_wxyz(q):
    """Return reciprocal (inverse) of quaternion q.inverse"""
    norm = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    return np.array([-q[0] / norm, -q[1] / norm, -q[2] / norm, q[3] / norm])

def quaternion_reciprocal_xyzw(q):
    # incoming is x, y, z, w
    """Return reciprocal (inverse) of quaternion q.inverse"""
    norm = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    return np.array([q[0] / norm, -q[1] / norm, -q[2] / norm, -q[3] / norm])    # x, y, z, w

def quaternion_conj(q):
    index = [3, 0, 1, 2]
    q = q[index]    # w, x, y, z
    """Return quaternion-conjugate of quaternion q̄"""
    return np.array([-q[1], -q[2], -q[3], q[0]])  # x, y, z, w

def quaternion_norm(q):
    return q / np.linalg.norm(q)


# ABSOLUTE TO LOCAL QUATERNIONS!!! and VICE VERSA....
class LocalToGlobalBodyNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = LocalToGlobalBodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_data = np.zeros((20, 4))
        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.absolute_pose_output = self.add_output('absolute pose out')

    def execute(self):
        data = self.pose_input()
        t = type(data)
        if t == torch.Tensor:
            data = tensor_to_array(data)
        self.calc_globals(data)
        self.absolute_pose_output.send(self.pose_data)

    def calc_globals(self, active_joints_data):

        # pelvis
        offset = self.active_joint_map['pelvis_anchor']
        pelvis_anchor_abs_quat = active_joints_data[offset]
        self.pose_data[offset] = pelvis_anchor_abs_quat

        # spine1
        offset = self.active_joint_map['spine_pelvis']
        spine_pelvis_abs_quat = quaternion_multiply(pelvis_anchor_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = spine_pelvis_abs_quat

        # spine2
        offset = self.active_joint_map['lower_vertebrae']
        lower_vertebrae_abs_quat = quaternion_multiply(spine_pelvis_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = lower_vertebrae_abs_quat

        # spine3
        offset = self.active_joint_map['mid_vertebrae']
        mid_vertebrae_abs_quat = quaternion_multiply(lower_vertebrae_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = mid_vertebrae_abs_quat

        # neck
        offset = self.active_joint_map['upper_vertebrae']
        upper_vertebrae_abs_quat = quaternion_multiply(mid_vertebrae_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = upper_vertebrae_abs_quat

        # head
        offset = self.active_joint_map['base_of_skull']
        base_of_skull_abs_quat = quaternion_multiply(upper_vertebrae_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = base_of_skull_abs_quat

        # left_collar
        offset = self.active_joint_map['left_shoulder_blade']
        left_shoulder_blade_abs_quat = quaternion_multiply(mid_vertebrae_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_shoulder_blade_abs_quat

        # left_shoulder
        offset = self.active_joint_map['left_shoulder']
        left_shoulder_abs_quat = quaternion_multiply(left_shoulder_blade_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_shoulder_abs_quat

        # left_elbow
        offset = self.active_joint_map['left_elbow']
        left_elbow_abs_quat = quaternion_multiply(left_shoulder_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_elbow_abs_quat

        # left_wrist
        offset = self.active_joint_map['left_wrist']
        left_wrist_abs_quat = quaternion_multiply(left_elbow_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_wrist_abs_quat

        # right_collar
        offset = self.active_joint_map['right_shoulder_blade']
        right_shoulder_blade_abs_quat = quaternion_multiply(mid_vertebrae_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_shoulder_blade_abs_quat

        # right_shoulder
        offset = self.active_joint_map['right_shoulder']
        right_shoulder_abs_quat = quaternion_multiply(right_shoulder_blade_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_shoulder_abs_quat

        # right_elbow
        offset = self.active_joint_map['right_elbow']
        right_elbow_abs_quat = quaternion_multiply(right_shoulder_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_elbow_abs_quat

        # right_wrist
        offset = self.active_joint_map['right_wrist']
        right_wrist_abs_quat = quaternion_multiply(right_elbow_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_wrist_abs_quat

        # left_hip
        offset = self.active_joint_map['left_hip']
        left_hip_abs_quat = quaternion_multiply(pelvis_anchor_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_hip_abs_quat

        # left_knee
        offset = self.active_joint_map['left_knee']
        left_knee_abs_quat = quaternion_multiply(left_hip_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_knee_abs_quat

        # left_ankle
        offset = self.active_joint_map['left_ankle']
        left_ankle_abs_quat = quaternion_multiply(left_knee_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = left_ankle_abs_quat

        # right_hip
        offset = self.active_joint_map['right_hip']
        right_hip_abs_quat = quaternion_multiply(pelvis_anchor_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_hip_abs_quat

        # right_knee
        offset = self.active_joint_map['right_knee']
        right_knee_abs_quat = quaternion_multiply(right_hip_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_knee_abs_quat

        # right_ankle
        offset = self.active_joint_map['right_ankle']
        right_ankle_abs_quat = quaternion_multiply(right_knee_abs_quat, active_joints_data[offset])
        self.pose_data[offset] = right_ankle_abs_quat


class GlobalToLocalBodyNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = GlobalToLocalBodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_data = np.zeros((20, 4))
        self.pose_input = self.add_input('absolute pose in', triggers_execution=True)
        self.relative_pose_output = self.add_output('relative pose out')

    def execute(self):
        data = self.pose_input()
        t = type(data)
        if t == torch.Tensor:
            data = tensor_to_array(data)
        self.calc_locals(data)
        self.relative_pose_output.send(self.pose_data)

    def calc_locals(self, active_joints_data):

        # pelvis
        offset = self.active_joint_map['pelvis_anchor']
        pelvis_anchor_rel_quat = active_joints_data[offset]
        self.pose_data[offset] = pelvis_anchor_rel_quat

        # spine1
        offset = self.active_joint_map['spine_pelvis']
        previous_offset = self.active_joint_map['pelvis_anchor']
        spine_pelvis_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = spine_pelvis_rel_quat

        # spine2
        offset = self.active_joint_map['lower_vertebrae']
        previous_offset = self.active_joint_map['spine_pelvis']
        lower_vertebrae_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = lower_vertebrae_rel_quat

        # spine3
        offset = self.active_joint_map['mid_vertebrae']
        previous_offset = self.active_joint_map['lower_vertebrae']
        mid_vertebrae_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = mid_vertebrae_rel_quat

        # neck
        offset = self.active_joint_map['upper_vertebrae']
        previous_offset = self.active_joint_map['mid_vertebrae']
        upper_vertebrae_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = upper_vertebrae_rel_quat

        # head
        offset = self.active_joint_map['base_of_skull']
        previous_offset = self.active_joint_map['upper_vertebrae']
        base_of_skull_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = base_of_skull_rel_quat

        # left_collar
        offset = self.active_joint_map['left_shoulder_blade']
        previous_offset = self.active_joint_map['mid_vertebrae']
        left_shoulder_blade_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_shoulder_blade_rel_quat

        # left_shoulder
        offset = self.active_joint_map['left_shoulder']
        previous_offset = self.active_joint_map['left_shoulder_blade']
        left_shoulder_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_shoulder_rel_quat

        # left_elbow
        offset = self.active_joint_map['left_elbow']
        previous_offset = self.active_joint_map['left_shoulder']
        left_elbow_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_elbow_rel_quat

        # left_wrist
        offset = self.active_joint_map['left_wrist']
        previous_offset = self.active_joint_map['left_elbow']
        left_wrist_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_wrist_rel_quat

        # right_collar
        offset = self.active_joint_map['right_shoulder_blade']
        previous_offset = self.active_joint_map['mid_vertebrae']
        right_shoulder_blade_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_shoulder_blade_rel_quat

        # right_shoulder
        offset = self.active_joint_map['right_shoulder']
        previous_offset = self.active_joint_map['right_shoulder_blade']
        right_shoulder_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_shoulder_rel_quat

        # right_elbow
        offset = self.active_joint_map['right_elbow']
        previous_offset = self.active_joint_map['right_shoulder']
        right_elbow_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_elbow_rel_quat

        # right_wrist
        offset = self.active_joint_map['right_wrist']
        previous_offset = self.active_joint_map['right_elbow']
        right_wrist_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_wrist_rel_quat

        # left_hip
        offset = self.active_joint_map['left_hip']
        previous_offset = self.active_joint_map['pelvis_anchor']
        left_hip_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_hip_rel_quat

        # left_knee
        offset = self.active_joint_map['left_knee']
        previous_offset = self.active_joint_map['left_hip']
        left_knee_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_knee_rel_quat

        # left_ankle
        offset = self.active_joint_map['left_ankle']
        previous_offset = self.active_joint_map['left_knee']
        left_ankle_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = left_ankle_rel_quat

        # right_hip
        offset = self.active_joint_map['right_hip']
        previous_offset = self.active_joint_map['pelvis_anchor']
        right_hip_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_hip_rel_quat

        # right_knee
        offset = self.active_joint_map['right_knee']
        previous_offset = self.active_joint_map['right_hip']
        right_knee_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_knee_rel_quat

        # right_ankle
        offset = self.active_joint_map['right_ankle']
        previous_offset = self.active_joint_map['right_knee']
        right_ankle_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(quaternion_reciprocal_xyzw(active_joints_data[offset]), active_joints_data[previous_offset]))
        self.pose_data[offset] = right_ankle_rel_quat


class ActiveJointsNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ActiveJointsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.full_pose_in = self.add_input('full pose quats in', triggers_execution=True)
        self.active_joints_out = self.add_output('active joint quats out')

    def execute(self):
        incoming = self.full_pose_in()
        t = type(incoming)
        if t == torch.Tensor:
            if incoming.shape[0] == 37:
                active_joints = incoming[self.active_to_shadow_map]
                self.active_joints_out.send(active_joints)
        if t == np.ndarray:
            # work on this!!!
            if incoming.shape[0] == 37:
                active_joints = incoming[self.active_to_shadow_map]
                self.active_joints_out.send(active_joints)



def shadow_service_loop():
    while True:
        was_client = False
        for node in MotionShadowNode.shadow_nodes:
            if node.client is not None:
                node.receive_data()
                was_client = True
        if was_client:
            time.sleep(.001)
        else:
            for node in MotionShadowNode.shadow_nodes:
                if node.client is None:
                    node.frame_task()
            time.sleep(.10)


class MotionShadowNode(MoCapNode):
    shadow_nodes = []
    @staticmethod
    def factory(name, data, args=None):
        node = MotionShadowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.num_bodies = 0
        super().__init__(label, data, args)

        self.launching_shadow = False
        self.__mutex = threading.Lock()
        self.check_for_shadow()
        self.client = None
        if not self.launching_shadow:
            try:
                self.client = shadow.Client("", 32076)
            except Exception as e:
                self.client = None
        self.origin = [0.0, 0.0, 0.0] * 4
        self.positions = np.ndarray((4, 37, 3))
        self.quaternions = np.ndarray((4, 37, 4))

        self.joints_mapped = False
        self.jointMap = [[0, 0]] * 37 * 4

        self.thread = threading.Thread(target=shadow_service_loop)
        self.thread_started = False
        self.direct_out = self.add_property('direct_out', widget_type='checkbox', default_value=False, callback=self.direct_out_changed)
        xml_definition = \
            "<?xml version=\"1.0\"?>" \
            "<configurable inactive=\"1\">" \
            "<Lq/>" \
            "<c/>" \
            "</configurable>"

        if self.client:
            if self.client.writeData(xml_definition):
                print("Sent active channel definition to Configurable service")

        self.body_quat_1 = self.add_output('body 1 quaternions')
        self.body_pos_1 = self.add_output('body 1 positions')
        self.body_quat_2 = self.add_output('body 2 quaternions')
        self.body_pos_2 = self.add_output('body 2 positions')
        self.body_quat_3 = self.add_output('body 3 quaternions')
        self.body_pos_3 = self.add_output('body 3 positions')
        self.body_quat_4 = self.add_output('body 4 quaternions')
        self.body_pos_4 = self.add_output('body 4 positions')
        MotionShadowNode.shadow_nodes.append(self)
        self.new_data = False
        if not self.thread_started:
            self.thread.start()
            self.thread_started = True
        self.add_frame_task()


    def check_for_shadow(self):
        found_shadow = False

        if _platform.system() == 'Windows':
            pass
        elif _platform.system() == 'Darwin':
            if find_process_id('Shadow') is None:
                print("Shadow not found, launching")
                subprocess.Popen( ['open', '/Applications/Shadow.app'] )
                # os.system("open /Applications/Shadow.app")
                self.launching_shadow = True
        elif _platform.system() == 'Linux':
            process = subprocess.Popen(['/opt/Motion/Shadow'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.launching_shadow = True

    def direct_out_changed(self):
        if self.direct_out():
            self.remove_frame_tasks()
        else:
            self.add_frame_task()

    def receive_data(self):
        if self.client:
            data = self.client.readData()
            if data is not None:
                if data.startswith(b'<?xml'):
                    xml_node_list = data
                    name_map = self.parse_name_map(xml_node_list)

                    for it in name_map:
                        thisName = name_map[it]
                        for idx, name_index in enumerate(JointTranslator.shadow_joint_index_to_name):
                            shadow_name = JointTranslator.bmolab_joint_to_shadow_limb[JointTranslator.shadow_joint_index_to_name[name_index]]
                            body = thisName[1]
                            if thisName[0] == shadow_name:
                                self.jointMap[it] = [idx + 1, thisName[1]]
                                break
                    self.joints_mapped = True
                    return

                configData = shadow.Format.Configurable(data)
                if configData is not None:
                    lock = ScopedLock(self.__mutex)
                    for key in configData:
                        master_key = self.jointMap[key][0] - 1          # keys start at 1
                        body_index = self.jointMap[key][1]
                        if master_key >= 0:
                            joint = JointTranslator.shadow_joint_index_to_name[master_key]

                            joint_data = configData[key]

                            # configData[key] is [q0, q1, q2, q3, p0, p1, p2]
                            if joint == 'Hips' and self.origin is None:
                                self.origin[body_index] = [joint_data.value(5) / 100, joint_data.value(6) / 100, joint_data.value(7) / 100]
                            self.positions[body_index, master_key] = [joint_data.value(5) / 100, joint_data.value(6) / 100, joint_data.value(7) / 100]
                            self.quaternions[body_index, master_key] = [joint_data.value(0), joint_data.value(1), joint_data.value(2), joint_data.value(3)]
                    self.new_data = True
                    lock = None
                    if self.direct_out():
                        self.frame_task()
        elif self.launching_shadow:
            if self.direct_out():
                self.frame_task()

    def frame_task(self):
        if self.launching_shadow:
            if _platform.system() in ['Darwin', 'Linux']:
                if find_process_id('Shadow') is not None:
                    try:
                        self.client = shadow.Client("", 32076)
                        xml_definition = \
                            "<?xml version=\"1.0\"?>" \
                            "<configurable inactive=\"1\">" \
                            "<Lq/>" \
                            "<c/>" \
                            "</configurable>"

                        if self.client:
                            print('shadow connected')

                            self.launching_shadow = False

                            if self.client.writeData(xml_definition):
                                print("Sent active channel definition to Configurable service")
                    except Exception as e:
                        self.client = None
                else:
                    return
        if not self.new_data:
            return
        self.new_data = False
        if self.num_bodies > 0:
            self.body_pos_1.send(self.positions[0])
            self.body_quat_1.send(self.quaternions[0])
        if self.num_bodies > 1:
            self.body_pos_2.send(self.positions[1])
            self.body_quat_2.send(self.quaternions[1])
        if self.num_bodies > 2:
            self.body_pos_3.send(self.positions[2])
            self.body_quat_3.send(self.quaternions[2])
        if self.num_bodies > 3:
            self.body_pos_4.send(self.positions[3])
            self.body_quat_4.send(self.quaternions[3])

    def parse_name_map(self, xml_node_list):
        name_map = {}

        tree = XML(xml_node_list)

        # <node key="N" id="Name"> ... </node>
        list = tree.findall(".//node")
        for itr in list:
            node_name = itr.get("id")
            node_local = node_name
            node_body = 0
            for code in JointTranslator.shadow_joint_index_to_name:
                node_code = JointTranslator.bmolab_joint_to_shadow_limb[JointTranslator.shadow_joint_index_to_name[code]]
                if len(node_code) == len(node_name) - 1:
                    if node_name.find(node_code) >= 0:
                        node_local = node_code
                        node_body = int(node_name[-1])
                        break
                elif len(node_code) == len(node_name):
                    if node_name.find(node_code) >= 0:
                        node_local = node_code
                        node_body = 0
                        break

            name_map[int(itr.get("key"))] = [node_local, node_body]
            if node_body >= self.num_bodies:
                self.num_bodies = node_body + 1
        return name_map


class TargetPoseNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TargetPoseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.pose_in = self.add_input('pose in', triggers_execution=True)
        self.capture_in = self.add_input('capture in', widget_type='button', callback=self.capture)
        self.target_pose = np.zeros((20, 4))
        self.target_pose[:, 0] = 1.0
        self.capturing = False
        self.capture_once = False
        self.output = self.add_output('score out')
        self.axis_distances_out = self.add_output('axis distances out')

    def capture(self):
        data = self.capture_in()
        if type(data) is bool:
            self.capturing = data
        else:
            self.capture_once = True

    def execute(self):
        if self.capturing:
            self.target_pose = self.pose_in().copy()
            self.target_pose = self.target_pose / np.linalg.norm(self.target_pose, axis=1, keepdims=True)
        elif self.capture_once:
            self.target_pose = self.pose_in().copy()
            self.target_pose = self.target_pose / np.linalg.norm(self.target_pose, axis=1, keepdims=True)
            self.capture_once = False
        else:
            pose = self.pose_in()
            pose = pose / np.linalg.norm(pose, axis=1, keepdims=True)
            distances = np.zeros(pose.shape[0])
            for index, q in enumerate(pose):
                diff = np.dot(self.target_pose[index], q)
                if diff > 1:
                    diff = 1
                if diff < -1:
                    diff = -1
                distances[index] = math.acos(2 * diff * diff - 1)
            distance = np.sum(distances)
            self.axis_distances_out.send(distances)
            self.output.send(distance)


class AlternateMoCapGLBody(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = AlternateMoCapGLBody(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('pose in', triggers_execution=True)
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.shape_input = self.add_input('shape', widget_type='combo', default_value='humanoid', callback=self.change_shape)
        self.fragment_input = self.add_input('fragment', widget_type='checkbox', default_value=False, callback=self.fragment)
        self.shape_input.widget.combo_items = ['linear', 'humanoid', 'medusa', 'bird', 'branch', 'cocoon', 'moth']
        self.gl_chain_output = self.add_output('gl_chain')

        self.skeleton_only = self.add_option('skeleton_only', widget_type='checkbox', default_value=False)
        self.multi_body_translation_x = self.add_option('multi offset x', widget_type='drag_float', default_value=0.0)
        self.multi_body_translation_y = self.add_option('multi offset y', widget_type='drag_float', default_value=0.0)
        self.multi_body_translation_z = self.add_option('multi offset z', widget_type='drag_float', default_value=0.0)
        self.body = AlternateBodyData()
        self.body.node = self

    def change_shape(self):
        self.body.shape = self.shape_input()
        self.body.connect_limbs()

    def offset_all_children(self, index, offset):
        joint = self.body.joints[index]
        for sub_index in joint.immed_children:
            self.body.limbs[sub_index].offset = offset
            self.offset_all_children(sub_index, offset)


    def fragment(self):
        fragmenter = self.fragment_input()
        if type(fragmenter) is str:
            if fragmenter in self.joint_map:
                index = self.joint_map[fragmenter]
                self.body.limbs[index].offset = np.random.random(3) * .5

        elif type(fragmenter) is list:
            if type(fragmenter[0]) is str:
                fragmenter = fragmenter[0]
                if fragmenter in self.joint_map:
                    index = self.joint_map[fragmenter]
                    offset = np.random.random(3) * .5
                    self.body.limbs[index].offset = offset
                    self.offset_all_children(index, offset)

        else:
            if self.fragment_input():
                self.new_fragmentation = True
                for limb in self.body.limbs:
                    if limb is not None:
                        off = np.random.random(3) * .5
                        limb.offset = off
            else:
                for limb in self.body.limbs:
                    if limb is not None:
                        off = np.zeros(3)
                        limb.offset = off

    def joint_callback(self, index):
        pass

    def process_commands(self, command):
        if type(command[0]) == str:
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in JointTranslator.joint_name_to_bmolab_index:
                    target_joint_index = JointTranslator.joint_name_to_bmolab_index[target_joint]
                    if len(command) >= 2: #lllll error!!!!
                        self.body.joints[target_joint_index].base_length = any_to_float(command[3])
                    if len(command) == 3:
                        self.body.limbs[target_joint_index].dims[1] = any_to_float(command[2])
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[2])

                    if len(command) == 4:
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[3])

                    self.body.joints[target_joint_index].set_matrix()
                    self.body.limbs[target_joint_index] = None

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input()
            t = type(incoming)
            if t == torch.Tensor:
                incoming = tensor_to_array(incoming)
                t = np.ndarray
            if t == np.ndarray:
                if len(incoming.shape) == 1:
                    if incoming.shape[0] == 80:
                        self.body.update_quats(np.reshape(incoming, [20, 4]))
                elif len(incoming.shape) == 2:
                    if incoming.shape[0] == 20:
                        if incoming.shape[1] == 4:
                            self.body.update_quats(incoming)
                    elif incoming.shape[0] == 80:
                        count = incoming.shape[1]
                        self.body.update_quats(np.reshape(incoming, [20, 4, count]))
                elif len(incoming.shape) == 3:
                    if incoming.shape[1] == 20:
                        if incoming.shape[2] == 4:
                            self.body.update_quats(incoming)

            elif t == list:
                self.process_commands(incoming)

        elif self.gl_chain_input.fresh_input:
            translation = [self.multi_body_translation_x(), self.multi_body_translation_y(), self.multi_body_translation_z()]
            self.body.multi_body_translation = translation
            incoming = self.gl_chain_input()
            t = type(incoming)
            if t == str and incoming == 'draw':
                self.body.draw(False, self.skeleton_only())


class LimbSizingNode(MoCapNode):
    @staticmethod
    def factory(name, data, args=None):
        node = LimbSizingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.size_dict = {}
        self.input_dict = {}
        self.in_receive_size = False
        self.size_dict_input = self.add_input('limb_sizes_dict', callback=self.receive_dict)
        self.symmetric_input = self.add_input('symmetric', widget_type='checkbox', default_value=True)

        self.label_map = {
            'head': 'TopOfHead',
            'neck': 'BaseOfSkull',
            'spine4': 'UpperVertebrae',
            'spine3': 'MidVertebrae',
            'spine2': 'LowerVertebrae',
            'spine1': 'SpinePelvis',
            'left_hip': 'LeftHip',
            'left_upper_leg': 'LeftKnee',
            'left_lower_leg': 'LeftAnkle',
            'left_foot': 'LeftBallOfFoot',
            'left_toes': 'LeftToeTip',
            'left_shoulder_blade': 'LeftShoulderBladeBase',
            'left_shoulder': 'LeftShoulder',
            'left_upper_arm': 'LeftElbow',
            'left_lower_arm': 'LeftWrist',
            'left_hand': 'LeftKnuckle',
            'left_fingers': 'LeftFingerTip',

            'right_hip': 'RightHip',
            'right_upper_leg': 'RightKnee',
            'right_lower_leg': 'RightAnkle',
            'right_foot': 'RightBallOfFoot',
            'right_toes': 'RightToeTip',
            'right_shoulder_blade': 'RightShoulderBladeBase',
            'right_shoulder': 'RightShoulder',
            'right_upper_arm': 'RightElbow',
            'right_lower_arm': 'RightWrist',
            'right_hand': 'RightKnuckle',
            'right_fingers': 'RightFingerTip'
        }
        for label in self.label_map:
            code = self.label_map[label]
            self.input_dict[code] = self.add_input(label, widget_type='drag_float', callback=self.sizer)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset)
        self.out = self.add_output('out to gl_body')

    def sizer(self):
        input_name = self.active_input._label
        self.receive_size(self.label_map[input_name])

    def receive_size(self, name):
        if name in self.input_dict and name in self.size_dict:
            data = self.input_dict[name]()
            t = type(data)
            if t is list:
                length = data[0]
                self.input_dict[name].set(length)
            else:
                length = any_to_float(data)

            current = self.size_dict[name]
            if t is list:
                if len(data) >= 3:
                    current[0] = data[0]
                    current[1] = data[1]
                    current[2] = data[2]
            else:
                current[0] = length
            self.size_dict[name] = current
            mess = ['limb_size', name, current[0], current[1], current[2]]
            self.out.send(mess)
            if self.symmetric_input() and not self.in_receive_size:
                self.in_receive_size = True
                if name[:4] == 'Left':
                    alt_name = 'Right' + name[4:]
                    if alt_name in self.input_dict:
                        self.input_dict[alt_name].set(data)
                        self.input_dict[alt_name].widget.value_changed(force=True)
                elif name[:5] == 'Right':
                    alt_name = 'Left' + name[5:]
                    if alt_name in self.input_dict:
                        self.input_dict[alt_name].set(data)
                        self.input_dict[alt_name].widget.value_changed(force=True)
                self.in_receive_size = False

    def receive_dict(self):
        d = self.size_dict_input()
        self.size_dict = copy.deepcopy(d)
        self.default_size_dict = copy.deepcopy(d)
        for limb_name in self.size_dict:
            if limb_name in self.input_dict:
                current = self.size_dict[limb_name]
                self.input_dict[limb_name].set(current[0])

    def reset(self):
        self.size_dict = copy.deepcopy(self.default_size_dict)
        self.in_receive_size = True
        for limb_name in self.size_dict:
            if limb_name in self.input_dict:
                current = self.size_dict[limb_name]
                self.input_dict[limb_name].set(current[0])
                self.receive_size(limb_name)
        self.in_receive_size = False
