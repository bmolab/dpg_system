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
import threading
import time

#
# print('before body_defs')
# import body_defs
# # from body_defs import JointTranslator
# print('after body_defs')
import dpg_system.MotionSDK as shadow
import random
from dpg_system.smpl_nodes import SMPLNode
from dpg_system.interface_nodes import Vector2DNode

def display_file_name(path, max_len=28):
    # Strip the directory and extension, then middle-ellipsis if still too
    # long so both the meaningful prefix and suffix stay visible. Keeps the
    # take node from stretching wide on long file names.
    name = os.path.splitext(os.path.basename(path))[0]
    if len(name) <= max_len:
        return name
    keep = max_len - 3  # room for the '...' ellipsis
    head = (keep + 1) // 2
    tail = keep // 2
    return name[:head] + '...' + name[-tail:]


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
    Node.app.register_node('cadence_filter', CadenceFilterNode.factory)
    Node.app.register_node('json_npz_frame_picker', JsonRandomEventWindowNode.factory)
    Node.app.register_node('tracker_root_inference', TrackerRootInferenceNode.factory)
    Node.app.register_node('sensor_to_root', SensorToRootNode.factory)
    Node.app.register_node('pose_adjust', PoseAdjustmentNode.factory)
    Node.app.register_node('active_pose', ActivePoseNode.factory)
    Node.app.register_node('mag_yaw_correct', MagYawCorrectionNode.factory)
    Node.app.register_node('shadow_arm_correct', ShadowArmCorrectNode.factory)
    Node.app.register_node('shadow_sensor', ShadowSensorNode.factory)
    Node.app.register_node('mag_offset', MagOffsetNode.factory)

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
                self.file_name.set(display_file_name(save_path))
                self.file_name.set_tooltip(save_path)
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
        self.file_name.set(display_file_name(path))
        self.file_name.set_tooltip(path)
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
        self.pending_frame = 0
        self.clip_start = 0
        self.clip_end = -1

        self.streaming = False
        self.paused_outputting = False
        self.take_dict = {}
        self.global_dict = {}
        self.sequence_keys = []
        self.global_keys = []

        self.take_data_in = self.add_input('take data in', callback=self.take_data_received)
        self.global_data_in = self.add_input('global data in', callback=self.receive_globals)

        self.load_button = self.add_input('load', widget_type='button', callback=self.load_take)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_sequence)

        self.file_name = self.add_label('')
        self.record_button = self.add_input('record', widget_type='button', callback=self.record_button_clicked)
        self.record_button.name_archive.append('stop record')
        self.play_pause_button = self.add_input('play', widget_type='button', callback=self.play_button_clicked)
        self.play_pause_button.name_archive.append('pause')
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop_button_clicked)
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=True)
        self.output_when_paused_input = self.add_input('output when paused', widget_type='checkbox', default_value=False)
        # When on, playback is driven by a worker thread paced at the file's
        # mocap framerate (read at load). When off, playback advances on the
        # dpg 60 Hz frame_task as before. Toggling mid-play takes effect on
        # the next play press.
        self.use_file_framerate_input = self.add_input('use file framerate', widget_type='checkbox', default_value=False)
        self.save_temp_input = self.add_input('save temp files', widget_type='checkbox', default_value=False)
        self.add_spacer()
        self.frame_input = self.add_input('frame', widget_type='drag_int', widget_width=50, callback=self.frame_widget_changed)
        self.length_property = self.add_input('length: 0', widget_type='label')
        self.speed = self.add_input('play speed', widget_type='drag_float', widget_width=50, default_value=speed)
        self.file_fps_input = self.add_input('frame rate', widget_type='drag_float', widget_width=50, default_value=60.0, callback=self.file_fps_changed)
        self.add_spacer()

        self.clip_start_input = self.add_input('clip start', widget_type='drag_int', callback=self.clip_changed, widget_width=50, trigger_button=True, trigger_callback=self.clip_start_set)
        self.clip_end_input = self.add_input('clip end', widget_type='drag_int', callback=self.clip_changed, widget_width=50, trigger_button=True, trigger_callback=self.clip_end_set)
        self.save_clip_button = self.add_input('save clip', widget_type='button', callback=self.save_clip)

        self.reset_clip_button = self.add_input('reset clip', widget_type='button', callback=self.reset_clip)

        self.add_spacer()
        self.dump_button = self.add_input('dump', widget_type='button', widget_width=50, callback=self.dump_take)
        self.dump_out = self.add_output('dump')
        self.global_params_out = self.add_output('globals')
        self.take_data_out = self.add_output('take data out')
        self.frame_out = self.add_output('frame')
        self.done_out = self.add_output('done')
        self.path_out = self.add_output('file path')   # emits the loaded file's abs path

        self.load_folder = './dpg_system'
        self.load_folder_option = self.add_option('load folder', widget_type='text_input', width=200, default_value=self.load_folder)
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', width=300, default_value=load_path,
                                         callback=self.load_from_load_path)
        self.temp_save_name = ''
        self.last_frame_out = -1
        self.recording = False
        self.force_frame = False
        self.message_handlers['load'] = self.load_take_message
        self.new_positions = False
        self.load_take_task = -1
        # File framerate (parsed from the npz at load time). 60 is the
        # neutral default when the file does not carry a framerate.
        self.file_fps = 60.0
        # Worker thread for file-rate playback (created on demand).
        self._playback_thread = None
        self._playback_stop_event = threading.Event()
        # True while a worker thread is driving step(). When set, the
        # main-thread frame_task tick reconciles the frame widget instead
        # of advancing playback itself.
        self._driven_by_worker = False

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

    def custom_cleanup(self):
        # Signal any running playback thread to exit. The thread is a
        # daemon, so it won't block process shutdown, but a clean exit
        # avoids spurious activity on a deleted node.
        self.streaming = False
        self._playback_stop_event.set()

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
        self.clip_start = int(self.current_frame)
        self.clip_start_input.set(self.clip_start)

    def clip_end_set(self):
        self.clip_end = int(self.current_frame)
        self.clip_end_input.set(self.clip_end)

    def stop_button_clicked(self):

        if self.streaming or self.paused_outputting or self.play_pause_button.get_label() == 'resume':
            self.streaming = False
            self.paused_outputting = False
            # Signal the worker (if any) and clear the worker-driven flag so
            # the next scrub's frame_task falls through to step() instead of
            # taking the worker-monitoring branch.
            self._playback_stop_event.set()
            self._driven_by_worker = False
            self.remove_frame_tasks()
            self.play_pause_button.set_label('play')
            self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
            self.current_frame = self.clip_start
            self.pending_frame = self.clip_start
        elif self.recording:
            self.record_button_clicked()

    def play_button_clicked(self):
        if not self.streaming and self.frame_count > 0:
            self.last_frame_out = -1
            if self.play_pause_button.get_label() == 'play':
                self.current_frame = self.clip_start - self.speed()
                self.pending_frame = self.current_frame
            if self.paused_outputting:
                self.paused_outputting = False
            else:
                self.start_playing()
            self.streaming = True
            self.play_pause_button.set_label('pause')
            self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)
        else:
            if self.streaming:
                self.streaming = False
                if self.output_when_paused_input():
                    self.paused_outputting = True
                else:
                    self.stop_playing()
                self.play_pause_button.set_label('resume')
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
                self.pending_frame = 0
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
                if self.save_temp_input():
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
        SaveDialog(self, self.save_file_callback, extensions=['.npz'], default_path=self.load_folder_option())

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
                self.file_name.set(display_file_name(save_path))
                self.file_name.set_tooltip(save_path)
                self.update_last_directory(save_path)
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
        SaveDialog(self, self.save_clip_callback, extensions=['.npz'], default_path=self.load_folder_option())

    def save_clip_callback(self, save_path):
        if not self.save_clip_only(save_path):
            print('failed to save clip')

    def save_clip_only(self, save_path):
        if save_path != '':
            clip_dict = {}
            for key in self.sequence_keys:
                if key not in self.take_dict:
                    continue
                value = np.array(self.take_dict[key])
                clip_value = value[self.clip_start:self.clip_end + 1]
                clip_dict[key] = clip_value
            for key, value in self.global_dict.items():
                clip_dict[key] = value
            np.savez(save_path, **clip_dict)
            self.update_last_directory(save_path)
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
                    value = incoming_dict[key]
                    if key in self.take_dict:
                        if isinstance(value, np.ndarray):
                            value = value.copy()
                        self.take_dict[key].append(value)
                    else:
                        if isinstance(value, np.ndarray):
                            value = value.copy()
                        self.take_dict[key] = [value]
                    if len(self.take_dict[key]) > max_len:
                        max_len = len(self.take_dict[key])
                self.frame_count = max_len
            else:
                self.frame_count += 1
            self.frame_input.widget.max = self.frame_count
            self.frame_input.set(self.frame_count, propagate=False)

    def dump_take(self):
        self.dump_out.send(self.take_dict)

    def start_playing(self):
        if self.use_file_framerate_input():
            # Worker-thread playback paced by file_fps. The thread loop
            # exits when either self.streaming flips false or the stop
            # event is set.
            self._playback_stop_event.clear()
            if self._playback_thread is not None and self._playback_thread.is_alive():
                return
            # The caller sets self.streaming = True *after* start_playing()
            # returns; set it here so the thread's first iteration doesn't
            # see streaming=False and exit immediately.
            self.streaming = True
            self._driven_by_worker = True
            self._playback_thread = threading.Thread(
                target=self._playback_loop, daemon=True,
                name=f'OpenTakePlayback-{self.uuid}')
            self._playback_thread.start()
            # Also run the main-thread tick so the frame widget reflects
            # progress — dpg widget mutation isn't safe off-thread.
            self.add_frame_task()
        else:
            self._driven_by_worker = False
            self.add_frame_task()

    def stop_playing(self):
        # Always signal both clock sources — if the user toggled the
        # checkbox mid-play we might have started in one mode and need to
        # stop the other.
        self._playback_stop_event.set()
        # Removing the main frame_task here means the worker-monitoring
        # branch in frame_task can't clear this itself; do it explicitly so
        # the next scrub's frame_task takes the step() path.
        self._driven_by_worker = False
        self.remove_frame_tasks()

    def frame_task(self):
        # When the worker thread drives step() (use_file_framerate on),
        # this main-thread tick only reconciles the frame widget. Calling
        # step() here too would double-advance the clip. Unregister once
        # the worker has fully exited.
        if self._driven_by_worker:
            worker = self._playback_thread
            if not self.force_frame:
                self.frame_input.set(self.current_frame)
            if worker is None or not worker.is_alive():
                self._driven_by_worker = False
                self.remove_frame_tasks()
            return
        # Runs on the main thread at 60 Hz when use_file_framerate is off.
        if self.paused_outputting:
            self.output_current_frame()
            return
        self.step()
        if self.force_frame:
            self.force_frame = False
            self.remove_frame_tasks()
            self.streaming = False

    def _playback_loop(self):
        """Runs on a worker thread when use_file_framerate is on. Drives
        step() at the file's native fps, scaled by the speed slider.

        Uses a deadline-based pacer: each iteration sleeps only the
        remainder of the target period after step() completes, so the
        downstream work time (sends, mailbox writes, etc.) doesn't add
        to the period. If we fall behind, the loop runs back-to-back
        until it catches up (no try-to-catch-up explosion: a single
        missed deadline is recovered next iteration)."""
        fps = self.file_fps if self.file_fps > 0 else 60.0
        speed_abs = abs(self.speed()) or 1.0
        period = 1.0 / (fps * speed_abs)
        next_deadline = time.perf_counter() + period
        while self.streaming and not self._playback_stop_event.is_set():
            if self.paused_outputting:
                self.output_current_frame()
            else:
                self.step()
                if self.force_frame:
                    self.force_frame = False
                    self.streaming = False
                    break
            # Re-read each tick so changes to speed/framerate take effect live.
            fps = self.file_fps if self.file_fps > 0 else 60.0
            speed_abs = abs(self.speed()) or 1.0
            period = 1.0 / (fps * speed_abs)
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                if self._playback_stop_event.wait(timeout=remaining):
                    break
                next_deadline += period
            else:
                # We're behind schedule; skip ahead to the next future
                # deadline rather than letting drift accumulate.
                next_deadline = time.perf_counter() + period

    def step(self):
        # When invoked from the worker thread, skip dpg widget side
        # effects and let the main thread reconcile UI on the next event.
        is_main = (Node.app is not None
                   and threading.get_ident() == getattr(Node.app, 'main_thread_id', threading.get_ident()))
        if self.pending_frame == int(self.current_frame):
            self.current_frame += self.speed()
        else:
            self.current_frame = self.pending_frame
        self.pending_frame = int(self.current_frame)
        if int(self.current_frame) > self.last_frame_out or self.force_frame:
            if self.current_frame > self.clip_end:
                if self.loop_input():
                    self.current_frame = self.clip_start
                    self.pending_frame = self.clip_start
                else:
                    if is_main:
                        self.stop_button_clicked()
                    else:
                        # Off-thread end-of-clip: drop streaming so the
                        # loop exits; widget cleanup waits for next click.
                        self.streaming = False
                self.done_out.send('done')
                if not self.loop_input():
                    return
            if not self.force_frame and is_main:
                self.frame_input.set(self.current_frame)
            frame = int(self.current_frame)

            self.last_frame_out = frame
            self.output_current_frame()

    def output_current_frame(self):
        frame = int(self.current_frame)
        frame_dict = {}
        for key in self.sequence_keys:
            frame_dict[key] = self.take_dict[key][frame]
        self.frame_out.send(frame)
        self.take_data_out.send(frame_dict)

    def load_from_load_path(self):
        path = self.load_path()
        if path != '':
            try:
                self.load_take_from_npz(path)
            except Exception as e:
                print('no take file found:', path)

    def first_frame(self):
        super().first_frame()
        if len(self.global_dict) > 0:
            self.global_params_out.send(self.global_dict)
            self.force_frame = False

    def update_last_directory(self, path):
        dir_path = os.path.dirname(os.path.abspath(path))
        self.load_folder_option.set(dir_path)

    def load_take_from_npz(self, path):
        if self.streaming:
            self.stop_button_clicked()
        take_file = np.load(path, allow_pickle=True)
        self.file_name.set(display_file_name(path))
        self.file_name.set_tooltip(path)
        self.load_path.set(path)
        self.update_last_directory(path)
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
            # Tolerant lookup for the source framerate (matches the key set
            # used by SMPLTorqueNode). When no metadata key is present we
            # leave self.file_fps (and the widget) at their existing values
            # so a user-set rate persists across loads of metadata-less files.
            for k in ('motioncapture_framerate', 'mocap_framerate', 'framerate'):
                if k in self.global_dict:
                    try:
                        self.file_fps = float(self.global_dict[k])
                        self.file_fps_input.set(self.file_fps, propagate=False)
                    except (TypeError, ValueError):
                        pass
                    break
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
        self.pending_frame = self.current_frame
        self.path_out.send(os.path.abspath(path))    # wire to shadow_arm_correct 'take file'

    def frame_widget_changed(self):
        data = self.frame_input()
        if data < 0:
            data = 0
        elif data >= self.frame_count:
            data = self.frame_count - 1
        self.pending_frame = data

        if self.paused_outputting:
            self.current_frame = data
        elif not self.streaming:
            self.streaming = True
            self.force_frame = True
            self.add_frame_task()

    def file_fps_changed(self):
        try:
            v = float(self.file_fps_input())
        except (TypeError, ValueError):
            return
        if v > 0:
            self.file_fps = v

    def load_take_message(self, message='', args=None):
        if args is not None:
            if len(args) > 0:
                path = any_to_string(args[0])
                self.load_take_from_npz(path)
            else:
                self.load_take(args)

    def load_take(self, args=None):
        arg = self.load_button()
        if type(arg) is list:
            arg = ' '.join(any_to_string(a) for a in arg)
        if type(arg) == str and arg != '':
            arg = os.path.expanduser(arg)
            if os.path.exists(arg):
                try:
                    self.load_path.set(arg)
                    self.load_take_from_npz(arg)
                except Exception as e:
                    print('load_npz_callback: error loading take file:', e, arg)
                return
            else:
                print('load_take: path does not exist:', arg)
                return
        LoadDialog(self, self.load_npz_callback, extensions=['.npz'], default_path=self.load_folder_option())

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
        elif joint_index == 29:
            local_index = 28
        elif joint_index == 30:
            local_index = 29

        if local_index > 29:
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
        self.smoothed_quaternions_a = None

    def calc_diff_quaternions(self, incoming_quats):
        if self.smoothed_quaternions_a is None:
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
        except Exception as e:
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

        if not self.events:
            return

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
        self.flush_button = self.add_input('flush', widget_type='button', callback=self.request_flush)
        self.reconnect_button = self.add_input('reconnect', widget_type='button', callback=self.request_reconnect)
        self._flush_requested = False
        self._reconnect_requested = False
        self._xml_definition = \
            "<?xml version=\"1.0\"?>" \
            "<configurable inactive=\"1\">" \
            "<Lq/>" \
            "<c/>" \
            "</configurable>"

        if self.client:
            if self.client.writeData(self._xml_definition):
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

    def request_flush(self):
        # Defer the socket op to the service thread to avoid racing readData().
        self._flush_requested = True

    def request_reconnect(self):
        # Defer the socket op to the service thread to avoid racing readData().
        self._reconnect_requested = True

    def _perform_flush(self):
        if self.client is None:
            return
        try:
            n = self.client.flush()
            print(f"shadow: flushed {n} buffered frames")
        except Exception as e:
            print(f"shadow: flush failed: {e}")

    def _perform_reconnect(self):
        try:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            self.client = shadow.Client("", 32076)
            # Force re-receipt of the joint name map on the next <?xml ...>.
            self.joints_mapped = False
            if self.client.writeData(self._xml_definition):
                print("shadow: reconnected, re-sent active channel definition")
            else:
                print("shadow: reconnected but channel definition write failed")
        except Exception as e:
            print(f"shadow: reconnect failed: {e}")
            self.client = None

    def receive_data(self):
        if self._reconnect_requested:
            self._reconnect_requested = False
            self._perform_reconnect()
            if self.client is None:
                return
        if self._flush_requested:
            self._flush_requested = False
            self._perform_flush()
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


def shadow_sensor_service_loop():
    # Dedicated service loop for ShadowSensorNode instances. Kept separate
    # from shadow_service_loop so the purpose-built capture node and the raw
    # sensor diagnostic node never share a connection or step on each other.
    while True:
        was_client = False
        for node in ShadowSensorNode.sensor_nodes:
            if node.client is not None:
                node.receive_data()
                was_client = True
        if was_client:
            time.sleep(.001)
        else:
            time.sleep(.05)


class ShadowSensorNode(MoCapNode):
    # Raw un-filtered sensor access from the Shadow Configurable service.
    # Where the `shadow` node requests <Lq/><c/> (orientation + position) for
    # motion capture, this node requests the sensor channels:
    #   <a/> accelerometer (g), <m/> magnetometer (uT), <g/> gyroscope (deg/s)
    # giving 9 floats per node: [ax, ay, az, mx, my, mz, gx, gy, gz].
    # It opens its own connection so the capture node stays untouched, and is
    # the foundation for accessing any raw sensor stream from the suit.
    sensor_nodes = []
    _service_thread = None

    @staticmethod
    def factory(name, data, args=None):
        node = ShadowSensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.num_bodies = 1
        super().__init__(label, data, args)

        self.__mutex = threading.Lock()
        self.client = None
        try:
            self.client = shadow.Client("", 32076)
        except Exception:
            self.client = None

        # [body, shadow_joint_index, xyz]
        self.accel = np.zeros((4, 37, 3))
        self.mag = np.zeros((4, 37, 3))
        self.gyro = np.zeros((4, 37, 3))
        # Largest accelerometer magnitude seen per node. A real IMU always
        # reads ~1 g (gravity); derived/virtual joints read ~0. Used to list
        # only physical sensors in the dropdown.
        self.accel_seen = np.zeros((4, 37))

        self.joints_mapped = False
        self.jointMap = [[0, 0]] * 37 * 4

        self._xml_definition = \
            "<?xml version=\"1.0\"?>" \
            "<configurable inactive=\"1\">" \
            "<a/>" \
            "<m/>" \
            "<g/>" \
            "</configurable>"
        self._reconnect_requested = False

        if self.client:
            if self.client.writeData(self._xml_definition):
                print("shadow_sensor: sent sensor channel definition (a, m, g)")

        self.body_index = self.add_property('body', widget_type='input_int', default_value=0, callback=self.selection_changed)
        self.sensor_property = self.add_property('sensor', widget_type='combo', default_value='<waiting for device>', callback=self.selection_changed)
        self.sensor_property.widget.combo_items = ['<waiting for device>']
        self.reconnect_button = self.add_input('reconnect', widget_type='button', callback=self.request_reconnect)

        self.mag_out = self.add_output('magnetometer')
        self.accel_out = self.add_output('accelerometer')
        self.gyro_out = self.add_output('gyroscope')

        self.available_sensors = []
        self._combo_dirty = False
        self.selected_master_key = -1
        self.new_data = False

        ShadowSensorNode.sensor_nodes.append(self)
        if ShadowSensorNode._service_thread is None:
            ShadowSensorNode._service_thread = threading.Thread(target=shadow_sensor_service_loop, daemon=True)
            ShadowSensorNode._service_thread.start()
        self.add_frame_task()

    def selection_changed(self):
        name = self.sensor_property()
        self.selected_master_key = JointTranslator.joint_name_to_shadow_index.get(name, -1)

    def request_reconnect(self):
        # Defer the socket op to the service thread to avoid racing readData().
        self._reconnect_requested = True

    def _perform_reconnect(self):
        try:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            self.client = shadow.Client("", 32076)
            self.joints_mapped = False
            self.jointMap = [[0, 0]] * 37 * 4
            self.accel_seen[:] = 0.0
            if self.client.writeData(self._xml_definition):
                print("shadow_sensor: reconnected, re-sent sensor channel definition")
        except Exception as e:
            print(f"shadow_sensor: reconnect failed: {e}")
            self.client = None

    def receive_data(self):
        if self._reconnect_requested:
            self._reconnect_requested = False
            self._perform_reconnect()
            if self.client is None:
                return
        if self.client:
            data = self.client.readData()
            if data is not None:
                if data.startswith(b'<?xml'):
                    name_map = self.parse_name_map(data)
                    for it in name_map:
                        thisName = name_map[it]
                        for idx, name_index in enumerate(JointTranslator.shadow_joint_index_to_name):
                            shadow_name = JointTranslator.bmolab_joint_to_shadow_limb[JointTranslator.shadow_joint_index_to_name[name_index]]
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
                            joint_data = configData[key]
                            # [a (0-2), m (3-5), g (6-8)] per the XML channel order
                            a = [joint_data.value(0), joint_data.value(1), joint_data.value(2)]
                            self.accel[body_index, master_key] = a
                            self.mag[body_index, master_key] = [joint_data.value(3), joint_data.value(4), joint_data.value(5)]
                            self.gyro[body_index, master_key] = [joint_data.value(6), joint_data.value(7), joint_data.value(8)]
                            a_mag = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
                            if a_mag > self.accel_seen[body_index, master_key]:
                                self.accel_seen[body_index, master_key] = a_mag
                    self.new_data = True
                    lock = None

    def parse_name_map(self, xml_node_list):
        name_map = {}
        tree = XML(xml_node_list)
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

    def _update_available_sensors(self):
        present = []
        for master_key in range(37):
            if np.any(self.accel_seen[:, master_key] > 0.2):
                present.append(JointTranslator.shadow_joint_index_to_name[master_key])
        if present and present != self.available_sensors:
            self.available_sensors = present
            self._combo_dirty = True

    def _refresh_sensor_combo(self):
        items = self.available_sensors if self.available_sensors else ['<waiting for device>']
        self.sensor_property.widget.combo_items = items
        dpg.configure_item(self.sensor_property.widget.uuid, items=items)
        if self.sensor_property() not in items:
            self.sensor_property.set(items[0], propagate=False)
            self.selection_changed()

    def frame_task(self):
        if self.joints_mapped:
            self._update_available_sensors()
        if self._combo_dirty:
            self._refresh_sensor_combo()
            self._combo_dirty = False
        if not self.new_data:
            return
        self.new_data = False
        body = int(self.body_index())
        mk = self.selected_master_key
        if mk < 0 or body < 0 or body >= 4:
            return
        lock = ScopedLock(self.__mutex)
        m = self.mag[body, mk].copy()
        a = self.accel[body, mk].copy()
        g = self.gyro[body, mk].copy()
        lock = None
        self.mag_out.send(m)
        self.accel_out.send(a)
        self.gyro_out.send(g)


class MagOffsetNode(MoCapNode):
    # Accumulates magnetometer samples into a 3D point cloud and fits a sphere
    # to them. As a sensor is rotated through all orientations a clean
    # magnetometer sweeps a sphere centred on the origin; a hard-iron
    # magnetization shifts that sphere off-centre, so the fitted centre IS the
    # offset vector. Outputs feed mgl_point_cloud / mgl_geo_sphere for a 3D
    # view; the 'centered cloud' output lets you verify a calibration recentres
    # the data on the origin.
    @staticmethod
    def factory(name, data, args=None):
        node = MagOffsetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.mag_in = self.add_input('magnetometer', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear)
        self.max_points = self.add_property('max points', widget_type='input_int', default_value=2000)
        self.min_spacing = self.add_option('min point spacing', widget_type='drag_float', default_value=1.0)

        self.offset_display = self.add_property('offset (uT)', widget_type='text_input', width=160, default_value='')
        self.radius_display = self.add_property('radius (uT)', widget_type='text_input', width=160, default_value='')
        self.residual_display = self.add_property('fit residual', widget_type='text_input', width=160, default_value='')
        self.count_display = self.add_property('samples', widget_type='text_input', width=160, default_value='0')

        self.cloud_out = self.add_output('cloud')
        self.centered_out = self.add_output('centered cloud')
        self.center_out = self.add_output('center')
        self.radius_out = self.add_output('radius')
        self.residual_out = self.add_output('residual')

        self.points = np.zeros((0, 3))
        self.center = np.zeros(3)
        self.radius = 0.0
        self.residual = 0.0

    def clear(self):
        self.points = np.zeros((0, 3))
        self.center = np.zeros(3)
        self.radius = 0.0
        self.residual = 0.0
        self._update_displays()
        self.send_outputs()

    def execute(self):
        m = self.mag_in()
        if m is None:
            return
        m = any_to_array(m).reshape(-1)
        if m.size < 3:
            return
        m = m[:3].astype(float)
        if np.linalg.norm(m) < 1e-6:
            return
        spacing = self.min_spacing()
        if self.points.shape[0] > 0 and spacing > 0:
            if np.linalg.norm(self.points[-1] - m) < spacing:
                return
        self.points = np.vstack([self.points, m])
        max_points = int(self.max_points())
        if max_points > 0 and self.points.shape[0] > max_points:
            self.points = self.points[-max_points:]
        self.fit_sphere()
        self._update_displays()
        self.send_outputs()

    def fit_sphere(self):
        # Linear least-squares sphere fit. |p - c|^2 = r^2 expands to
        #   2*c.p + (r^2 - |c|^2) = |p|^2, linear in [cx, cy, cz, k].
        p = self.points
        n = p.shape[0]
        if n < 8:
            return
        A = np.hstack([2.0 * p, np.ones((n, 1))])
        b = np.sum(p * p, axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = sol[:3]
        r2 = sol[3] + c.dot(c)
        if r2 <= 0:
            return
        r = math.sqrt(r2)
        d = np.linalg.norm(p - c, axis=1) - r
        self.center = c
        self.radius = r
        self.residual = float(np.sqrt(np.mean(d * d)))

    def _update_displays(self):
        c = self.center
        self.offset_display.set('%.2f, %.2f, %.2f' % (c[0], c[1], c[2]), propagate=False)
        self.radius_display.set('%.2f' % self.radius, propagate=False)
        self.residual_display.set('%.3f' % self.residual, propagate=False)
        self.count_display.set(str(self.points.shape[0]), propagate=False)

    def send_outputs(self):
        self.residual_out.send(self.residual)
        self.radius_out.send(self.radius)
        self.center_out.send(self.center.copy())
        self.centered_out.send((self.points - self.center).copy())
        self.cloud_out.send(self.points.copy())


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


class TrackerRootInferenceNode(MoCapNode):
    """
    Corrects root (pelvis) position by modeling the actual tracker placement
    on the left thigh and comparing the model's predicted tracker position
    with the tracker's reported position.

    The Shadow mocap system infers root position from a tracker on the left
    thigh but doesn't know the exact mounting position. This causes vertical
    drift when the left leg is raised. This node provides adjustable parameters
    for tracker placement to compute a better root position estimate.

    Inputs:
        positions (37x3): Full Shadow positions array (Y-up, meters)
        pose (20x4 or 37x4): Quaternion pose data (needed for left hip global rotation)

    Parameters:
        tracker_down_thigh: Distance from hip joint down the thigh bone axis (meters)
        tracker_radial_offset: Perpendicular distance from bone axis to tracker (meters)
        tracker_circumference_angle: Rotation around thigh circumference (degrees)
            0 = lateral/outside, 90 = front, 180 = medial/inside, 270 = back
        tracker_index: Which Shadow tracker (0-3)
    """

    # Shadow position array indices
    SHADOW_HIPS_INDEX = 4        # PelvisAnchor in shadow indexing
    SHADOW_LEFT_HIP_INDEX = 14   # LeftHip in shadow indexing
    SHADOW_LEFT_KNEE_INDEX = 12  # LeftKnee in shadow indexing
    SHADOW_RIGHT_HIP_INDEX = 28  # RightHip in shadow indexing
    SHADOW_RIGHT_KNEE_INDEX = 26 # RightKnee in shadow indexing
    SHADOW_TRACKER_INDICES = [33, 34, 35, 36]  # Tracker0-3 in shadow indexing

    # From definition.xml: Hips -> LeftThigh / RightThigh offsets in cm
    HIPS_TO_LEFT_HIP_OFFSET = np.array([8.91, -6.269997, 0.0]) / 100.0    # meters
    HIPS_TO_RIGHT_HIP_OFFSET = np.array([-8.91, -6.269997, 0.0]) / 100.0  # meters

    # From definition.xml: Thigh -> Leg bone vectors (symmetric, same for both sides)
    # translate="0 -44.852665 -1.566289"
    THIGH_BONE_VECTOR = np.array([0.0, -44.852665, -1.566289]) / 100.0  # meters

    # SMPL joint indices for left/right
    SMPL_LEFT_HIP = 1
    SMPL_LEFT_KNEE = 4
    SMPL_RIGHT_HIP = 2
    SMPL_RIGHT_KNEE = 5

    @staticmethod
    def factory(name, data, args=None):
        node = TrackerRootInferenceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Inputs
        self.positions_input = self.add_input('positions', triggers_execution=True)
        self.pose_input = self.add_input('pose')
        self.limb_lengths_input = self.add_input('limb_lengths', callback=self._on_limb_lengths_received)

        # Parameters
        self.thigh_side = self.add_property(
            'thigh_side', widget_type='combo', default_value='right', callback=self._on_side_changed)
        self.thigh_side.widget.combo_items = ['left', 'right']
        self.tracker_down_thigh = self.add_property(
            'tracker_down_thigh', widget_type='drag_float', default_value=0.15)
        self.tracker_radial_offset = self.add_property(
            'tracker_radial_offset', widget_type='drag_float', default_value=0.08)
        self.tracker_circumference_angle = self.add_property(
            'tracker_circumference_angle', widget_type='drag_float', default_value=0.0)
        self.tracker_index = self.add_property(
            'tracker_index', widget_type='drag_int', default_value=0)
        self.enabled = self.add_property(
            'enabled', widget_type='checkbox', default_value=True)

        # Outputs
        self.corrected_root_output = self.add_output('corrected root')
        self.correction_output = self.add_output('correction')
        self.tracker_model_pos_output = self.add_output('tracker model pos')
        self.corrected_positions_output = self.add_output('corrected positions')

        # Current bone vectors (start with definition.xml defaults, updated by limb_lengths)
        # Store both sides; the active side is selected by thigh_side property
        self._hips_to_hip = {
            'left': self.HIPS_TO_LEFT_HIP_OFFSET.copy(),
            'right': self.HIPS_TO_RIGHT_HIP_OFFSET.copy(),
        }
        self._thigh_bone_vecs = {
            'left': self.THIGH_BONE_VECTOR.copy(),
            'right': self.THIGH_BONE_VECTOR.copy(),
        }

    def custom_create(self, from_file):
        self._select_side()

    def _on_side_changed(self):
        """Called when the thigh_side combo changes."""
        self._select_side()

    def _select_side(self):
        """Select active bone vectors for the current thigh side."""
        side = self.thigh_side()
        self._current_hips_to_hip = self._hips_to_hip[side]
        self._thigh_bone_vec = self._thigh_bone_vecs[side]
        # Precompute thigh bone direction (unit vector)
        self._update_bone_frame()

    def _update_bone_frame(self):
        """Recompute bone direction and perpendicular frame from current bone vectors."""
        bone_len = np.linalg.norm(self._thigh_bone_vec)
        if bone_len < 1e-6:
            bone_len = 0.45  # fallback
        self.thigh_bone_direction = self._thigh_bone_vec / bone_len
        self.thigh_bone_length = bone_len
        self._build_thigh_local_frame()

    def _on_limb_lengths_received(self):
        """Handle limb_lengths input from smpl_beta_editor.

        Updates the hips-to-left_hip offset and thigh bone vector
        to match current body proportions.

        Expected format: dict with 'offsets' key containing (30, 3) or (24, 3)
        array in SMPL joint order (pelvis=0, left_hip=1, left_knee=4).
        """
        data = self.limb_lengths_input()
        if data is None:
            return

        if isinstance(data, dict):
            offsets = data.get('offsets', None)
            if offsets is not None:
                offsets = any_to_array(offsets)
                if offsets.ndim == 2 and offsets.shape[1] == 3:
                    # SMPL: left_hip=1, right_hip=2, left_knee=4, right_knee=5
                    if offsets.shape[0] > self.SMPL_LEFT_HIP:
                        self._hips_to_hip['left'] = offsets[self.SMPL_LEFT_HIP].copy()
                    if offsets.shape[0] > self.SMPL_RIGHT_HIP:
                        self._hips_to_hip['right'] = offsets[self.SMPL_RIGHT_HIP].copy()
                    if offsets.shape[0] > self.SMPL_LEFT_KNEE:
                        self._thigh_bone_vecs['left'] = offsets[self.SMPL_LEFT_KNEE].copy()
                    if offsets.shape[0] > self.SMPL_RIGHT_KNEE:
                        self._thigh_bone_vecs['right'] = offsets[self.SMPL_RIGHT_KNEE].copy()
                    self._select_side()
                    return

            # Fallback: use 'lengths' dict if 'offsets' not available
            lengths = data.get('lengths', None)
            if lengths is not None and isinstance(lengths, dict):
                if 'pelvis_width' in lengths:
                    pw = float(lengths['pelvis_width']) * 0.5
                    self._hips_to_hip['left'] = np.array([pw, -0.05, 0.0])
                    self._hips_to_hip['right'] = np.array([-pw, -0.05, 0.0])
                if 'upper_leg' in lengths:
                    ul = float(lengths['upper_leg'])
                    bone_dir = self.thigh_bone_direction
                    self._thigh_bone_vecs['left'] = bone_dir * ul
                    self._thigh_bone_vecs['right'] = bone_dir * ul
                self._select_side()

    def _build_thigh_local_frame(self):
        """Build an orthonormal frame around the thigh bone axis.

        The bone direction is the primary axis. We construct two perpendicular
        vectors that define the plane around the bone for radial offset placement.
        Convention: perp1 = lateral direction, perp2 = forward direction (in rest pose).
        """
        bone_dir = self.thigh_bone_direction

        # Choose a reference vector not parallel to bone direction
        # In Y-up rest pose, bone points mostly in -Y, so X or Z work
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(bone_dir, ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])

        # Gram-Schmidt to get perpendicular vectors
        self.perp1 = ref - np.dot(ref, bone_dir) * bone_dir
        self.perp1 /= np.linalg.norm(self.perp1)

        self.perp2 = np.cross(bone_dir, self.perp1)
        self.perp2 /= np.linalg.norm(self.perp2)

    def _compute_tracker_offset_local(self):
        """Compute the tracker offset from the hip joint in thigh-local (rest pose) coordinates.

        Returns:
            np.ndarray: 3D offset vector from hip joint to tracker in rest-pose coordinates.
        """
        down = self.tracker_down_thigh()
        radial = self.tracker_radial_offset()
        angle_deg = self.tracker_circumference_angle()
        angle_rad = np.radians(angle_deg)

        # Along-bone component
        along_bone = self.thigh_bone_direction * down

        # Radial component: rotate around bone axis by circumference angle
        radial_dir = (self.perp1 * np.cos(angle_rad) +
                      self.perp2 * np.sin(angle_rad))
        radial_vec = radial_dir * radial

        return along_bone + radial_vec

    @staticmethod
    def _quat_rotate_vector(q_wxyz, v):
        """Rotate vector v by quaternion q (w, x, y, z format).

        Uses the formula: v' = q * v * q^-1
        """
        w, x, y, z = q_wxyz
        # Quaternion-vector rotation (optimized)
        t = 2.0 * np.array([
            y * v[2] - z * v[1],
            z * v[0] - x * v[2],
            x * v[1] - y * v[0]
        ])
        return v + w * t + np.cross(np.array([x, y, z]), t)

    def _get_hip_global_quat(self, pose_data):
        """Extract the global (absolute) hip quaternion for the selected thigh side.

        The pose data contains local rotations. To get the global hip rotation,
        we compose: pelvis_global * hip_local.

        Args:
            pose_data: (20, 4) or (37, 4) quaternion array in (w, x, y, z) format

        Returns:
            np.ndarray: global hip quaternion in (w, x, y, z) format, or None
        """
        if pose_data is None:
            return None

        pose = any_to_array(pose_data)

        if pose.ndim == 1:
            if pose.size == 80:
                pose = pose.reshape(20, 4)
            elif pose.size == 148:
                pose = pose.reshape(37, 4)
            else:
                return None

        side = self.thigh_side()

        # Determine which indexing scheme we have
        if pose.shape[0] == 37:
            # Raw shadow data: use shadow indices
            pelvis_idx = self.SHADOW_HIPS_INDEX
            hip_idx = self.SHADOW_LEFT_HIP_INDEX if side == 'left' else self.SHADOW_RIGHT_HIP_INDEX
        elif pose.shape[0] == 20:
            # Active joints: use active joint map
            pelvis_idx = self.active_joint_map['pelvis_anchor']
            hip_key = 'left_hip' if side == 'left' else 'right_hip'
            hip_idx = self.active_joint_map[hip_key]
        else:
            return None

        # Quaternions are already in wxyz format
        pelvis_q = pose[pelvis_idx].copy()
        hip_local_q = pose[hip_idx].copy()

        # Compose: global_hip = pelvis_global * hip_local
        global_hip_q = self._quat_multiply(pelvis_q, hip_local_q)

        return global_hip_q

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions in (w, x, y, z) format."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def execute(self):
        if not self.positions_input.fresh_input:
            return

        positions = any_to_array(self.positions_input())
        if positions is None:
            return

        if positions.ndim != 2 or positions.shape[0] < 37 or positions.shape[1] != 3:
            return

        if not self.enabled():
            # Pass through the original root position
            self.corrected_root_output.send(positions[self.SHADOW_HIPS_INDEX].copy())
            self.corrected_positions_output.send(positions.copy())
            return

        # Get tracker index
        tidx = int(self.tracker_index())
        tidx = max(0, min(3, tidx))
        tracker_shadow_idx = self.SHADOW_TRACKER_INDICES[tidx]

        # Get positions for the selected thigh side
        side = self.thigh_side()
        system_root_pos = positions[self.SHADOW_HIPS_INDEX].copy()
        tracker_pos = positions[tracker_shadow_idx].copy()
        hip_shadow_idx = self.SHADOW_LEFT_HIP_INDEX if side == 'left' else self.SHADOW_RIGHT_HIP_INDEX
        hip_pos = positions[hip_shadow_idx].copy()

        # Get hip global rotation for the selected side
        pose_data = self.pose_input()
        hip_global_q = self._get_hip_global_quat(pose_data)

        if hip_global_q is None:
            # Can't compute correction without pose data, pass through
            self.corrected_root_output.send(system_root_pos)
            self.corrected_positions_output.send(positions.copy())
            return

        # Compute the tracker offset in rest-pose local coordinates
        tracker_offset_local = self._compute_tracker_offset_local()

        # Rotate the offset into world space using the hip's global rotation
        tracker_offset_world = self._quat_rotate_vector(hip_global_q, tracker_offset_local)

        # The modeled tracker position = hip_world_pos + rotated_offset
        tracker_model_pos = hip_pos + tracker_offset_world

        # The correction: the model's predicted tracker position is derived from
        # Shadow's hip_pos (which contains the root positioning error). The
        # difference between model and actual tracker reveals that error.
        correction = tracker_model_pos - tracker_pos

        # Subtract the error from the system root to correct it
        corrected_root = system_root_pos - correction

        # Build corrected positions array
        corrected_positions = positions.copy()
        corrected_positions[self.SHADOW_HIPS_INDEX] = corrected_root

        # Send outputs
        self.corrected_root_output.send(corrected_root)
        self.correction_output.send(correction)
        self.tracker_model_pos_output.send(tracker_model_pos)
        self.corrected_positions_output.send(corrected_positions)


class CadenceFilterNode(MoCapNode):
    """Causal moving-average filter for removing sensor cadence artifacts.
    
    Applies a causal moving average (window size N) to pose and/or trans arrays.
    A window of 3 removes the 33.3 Hz cadence from ~33 Hz sensors upsampled to 100 Hz.
    Uses ring buffers for streaming (single-frame) operation.
    """
    @staticmethod
    def factory(name, data, args=None):
        node = CadenceFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.trans_input = self.add_input('trans in', triggers_execution=True)
        
        self.pose_output = self.add_output('pose out')
        self.trans_output = self.add_output('trans out')
        
        self.enable_prop = self.add_property('enable', widget_type='checkbox', default_value=True)
        self.window_prop = self.add_property('window', widget_type='drag_int', default_value=3)
        
        self._pose_ring = None
        self._trans_ring = None
        self._pose_ring_idx = 0
        self._trans_ring_idx = 0

    def execute(self):
        enabled = self.enable_prop()
        win = self.window_prop()
        
        if self.pose_input.fresh_input:
            pose = self.pose_input()
            if pose is not None:
                pose = np.array(pose, dtype=np.float64)
                if enabled and win >= 2:
                    pose = self._smooth(pose, win, 'pose')
                self.pose_output.send(pose)
        
        if self.trans_input.fresh_input:
            trans = self.trans_input()
            if trans is not None:
                trans = np.array(trans, dtype=np.float64)
                if enabled and win >= 2:
                    trans = self._smooth(trans, win, 'trans')
                self.trans_output.send(trans)

    def _smooth(self, data, win, kind):
        """Apply causal moving average using a ring buffer."""
        flat = data.flatten()
        n = flat.size
        
        ring_attr = f'_{kind}_ring'
        idx_attr = f'_{kind}_ring_idx'
        
        ring = getattr(self, ring_attr, None)
        
        # Re-init ring buffer if size or window changed
        if ring is None or ring.shape[0] != win or ring.shape[1] != n:
            setattr(self, ring_attr, np.tile(flat, (win, 1)))
            setattr(self, idx_attr, 0)
            ring = getattr(self, ring_attr)
        
        idx = getattr(self, idx_attr)
        ring[idx] = flat
        setattr(self, idx_attr, (idx + 1) % win)
        
        smoothed = np.mean(ring, axis=0)
        return smoothed.reshape(data.shape)


class SensorToRootNode(MoCapNode):
    """
    Transforms a lower-back sensor position into the SMPL root (pelvis) position.

    The sensor is affixed to the performer's lower back at approximately belt-line.
    The SMPL root is interior to the body, roughly at the center of the pelvis.
    This node applies a configurable offset (in pelvis-local coordinates) rotated
    by the pelvis orientation to convert sensor position to root position.

    Inputs:
        sensor_pos (3,): Tracker/sensor world position (Y-up, meters)
        pelvis_quat (4,): Pelvis orientation quaternion (w, x, y, z)

    Outputs:
        root_pos (3,): Corrected SMPL root position
    """

    # Shadow indices for convenience
    SHADOW_HIPS_INDEX = 4
    SHADOW_TRACKER_INDICES = [33, 34, 35, 36]

    @staticmethod
    def factory(name, data, args=None):
        node = SensorToRootNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Inputs
        self.sensor_pos_input = self.add_input('sensor pos', triggers_execution=True)
        self.pelvis_quat_input = self.add_input('pelvis quat')
        self.positions_input = self.add_input('positions')  # optional: full Shadow positions array
        self.pose_input = self.add_input('pose')             # optional: full pose array

        # Offset from sensor to root in pelvis-local coordinates (meters)
        # With Y-up: +X = right, +Y = up, +Z = forward
        # Sensor is on the lower back, root is forward and slightly up
        self.offset_x = self.add_property(
            'offset_x', widget_type='drag_float', default_value=0.0)
        self.offset_y = self.add_property(
            'offset_y', widget_type='drag_float', default_value=0.0)
        self.offset_z = self.add_property(
            'offset_z', widget_type='drag_float', default_value=0.12)
        self.tracker_index = self.add_property(
            'tracker_index', widget_type='drag_int', default_value=0)
        self.use_positions = self.add_property(
            'use_positions', widget_type='checkbox', default_value=False)

        # Outputs
        self.root_pos_output = self.add_output('root pos')
        self.corrected_positions_output = self.add_output('corrected positions')

    @staticmethod
    def _quat_rotate_vector(q_wxyz, v):
        """Rotate vector v by quaternion q (w, x, y, z format)."""
        w, x, y, z = q_wxyz
        t = 2.0 * np.array([
            y * v[2] - z * v[1],
            z * v[0] - x * v[2],
            x * v[1] - y * v[0]
        ])
        return v + w * t + np.cross(np.array([x, y, z]), t)

    def execute(self):
        if not self.sensor_pos_input.fresh_input:
            return

        # Get sensor position
        sensor_pos = None
        positions = None

        if self.use_positions():
            # Extract from full positions array
            raw_positions = self.positions_input()
            if raw_positions is not None:
                positions = any_to_array(raw_positions)
                if positions.ndim == 2 and positions.shape[0] >= 37:
                    tidx = max(0, min(3, int(self.tracker_index())))
                    tracker_shadow_idx = self.SHADOW_TRACKER_INDICES[tidx]
                    sensor_pos = positions[tracker_shadow_idx].copy()

        if sensor_pos is None:
            raw = self.sensor_pos_input()
            if raw is None:
                return
            sensor_pos = any_to_array(raw).flatten()
            if sensor_pos.size != 3:
                return

        # Get pelvis quaternion (wxyz)
        pelvis_q = None

        # Try direct pelvis_quat input first
        raw_q = self.pelvis_quat_input()
        if raw_q is not None:
            pelvis_q = any_to_array(raw_q).flatten()
            if pelvis_q.size != 4:
                pelvis_q = None

        # Fall back to extracting from full pose array
        if pelvis_q is None:
            raw_pose = self.pose_input()
            if raw_pose is not None:
                pose = any_to_array(raw_pose)
                if pose.ndim == 1:
                    if pose.size == 80:
                        pose = pose.reshape(20, 4)
                    elif pose.size == 148:
                        pose = pose.reshape(37, 4)
                if pose.ndim == 2:
                    if pose.shape[0] == 37:
                        pelvis_q = pose[self.SHADOW_HIPS_INDEX].copy()
                    elif pose.shape[0] == 20:
                        pelvis_q = pose[self.active_joint_map['pelvis_anchor']].copy()

        if pelvis_q is None:
            # Without orientation, apply offset unrotated (approximate)
            offset = np.array([self.offset_x(), self.offset_y(), self.offset_z()])
            root_pos = sensor_pos + offset
        else:
            # Rotate the offset by the pelvis orientation
            offset_local = np.array([self.offset_x(), self.offset_y(), self.offset_z()])
            offset_world = self._quat_rotate_vector(pelvis_q, offset_local)
            root_pos = sensor_pos + offset_world

        self.root_pos_output.send(root_pos)

        # If we have positions, build a corrected version
        if positions is not None:
            corrected = positions.copy()
            corrected[self.SHADOW_HIPS_INDEX] = root_pos
            self.corrected_positions_output.send(corrected)


class PoseAdjustmentNode(MoCapNode):
    """Applies per-joint quaternion adjustments to a 20-joint active pose.

    By default each non-root joint is pre-multiplied so the correction is in the
    parent bone's frame; the root (pelvis_anchor) is post-multiplied (body-local).
    The 'child frame' checkbox switches all non-root joints to post-multiply,
    applying the correction in the joint's own (child bone) local frame -- useful
    for fixing a sensor mounted on the child side of the joint.
    """

    JOINT_NAMES = [
        'head', 'neck', 'spine3', 'spine2', 'spine1',
        'pelvis', 'l_hip', 'l_knee', 'l_ankle',
        'r_hip', 'r_knee', 'r_ankle',
        'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist',
        'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist'
    ]

    @staticmethod
    def factory(name, data, args=None):
        return PoseAdjustmentNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.child_frame_input = self.add_input('child frame', widget_type='checkbox', default_value=False)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_adjustments)

        # 20 quaternion adjustment widgets (wxyz, identity = [1, 0, 0, 0])
        self.adj_inputs = []
        for i, name in enumerate(self.JOINT_NAMES):
            adj = self.add_input(name, widget_type='drag_float_n',
                                default_value=[1.0, 0.0, 0.0, 0.0], columns=4,
                                widget_width=45)
            self.adj_inputs.append(adj)

        self.pose_output = self.add_output('pose out')

    def reset_adjustments(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        for adj in self.adj_inputs:
            adj.set(identity)

    def get_preset_state(self):
        preset = {}
        values = []
        for adj in self.adj_inputs:
            val = adj()
            if val is None:
                values.append([1.0, 0.0, 0.0, 0.0])
            else:
                values.append(list(any_to_array(val).flatten()))
        preset['values'] = values
        return preset

    def set_preset_state(self, preset):
        if 'values' in preset:
            values = preset['values']
            for i, val in enumerate(values):
                if i < len(self.adj_inputs):
                    self.adj_inputs[i].set(val)

    def execute(self):
        raw = self.pose_input()
        if raw is None:
            return
        pose = any_to_array(raw)
        if pose.ndim == 1 and pose.size == 80:
            pose = pose.reshape(20, 4)
        if pose.ndim != 2 or pose.shape[0] != 20 or pose.shape[1] != 4:
            return

        child_frame = bool(self.child_frame_input())
        result = pose.copy()
        for i in range(20):
            adj_val = self.adj_inputs[i]()
            if adj_val is None:
                continue
            adj_q = any_to_array(adj_val).flatten()
            if adj_q.size != 4:
                continue
            # Skip if identity (optimization)
            if abs(adj_q[0] - 1.0) < 1e-6 and np.linalg.norm(adj_q[1:]) < 1e-6:
                continue
            if i == 5 or child_frame:
                # Post-multiply: adjustment is in the joint's own (child bone)
                # local frame. For the root (pelvis_anchor) this is the body-local
                # frame; for other joints it corrects a child-side sensor offset.
                result[i] = quaternion_multiply_wxyz(result[i], adj_q)
            else:
                # Pre-multiply: adjustment is in the parent bone's frame.
                result[i] = quaternion_multiply_wxyz(adj_q, result[i])

        self.pose_output.send(result.astype(np.float32))


class MagYawCorrectionNode(MoCapNode):
    """Corrects magnetometer-induced yaw errors in IMU motion capture data.

    Provides two independent corrections per sensor:
      - Global yaw (pre-multiply around world Y): ongoing magnetometer error
      - Local yaw (post-multiply in sensor body frame): calibration error baked
        into the T-pose identity quaternion

    The 'sync' checkbox keeps local = global (conjugation, correct when the
    hard-iron bias produces the same yaw error at all orientations).
    The 'symmetric' checkbox mirrors left↔right joint values.
    """

    JOINT_NAMES = [
        'head', 'neck', 'spine3', 'spine2', 'spine1',
        'pelvis', 'l_hip', 'l_knee', 'l_ankle',
        'r_hip', 'r_knee', 'r_ankle',
        'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist',
        'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist'
    ]

    # Parent index for each of the 20 active joints (index into active pose array)
    # -1 means root (no parent). Matches LocalToGlobalBodyNode hierarchy.
    PARENT_INDEX = [
        1,   # 0  base_of_skull   -> upper_vertebrae (1)
        2,   # 1  upper_vertebrae -> mid_vertebrae (2)
        3,   # 2  mid_vertebrae   -> lower_vertebrae (3)
        4,   # 3  lower_vertebrae -> spine_pelvis (4)
        5,   # 4  spine_pelvis    -> pelvis_anchor (5)
        -1,  # 5  pelvis_anchor   -> root
        5,   # 6  left_hip        -> pelvis_anchor (5)
        6,   # 7  left_knee       -> left_hip (6)
        7,   # 8  left_ankle      -> left_knee (7)
        5,   # 9  right_hip       -> pelvis_anchor (5)
        9,   # 10 right_knee      -> right_hip (9)
        10,  # 11 right_ankle     -> right_knee (10)
        2,   # 12 left_shoulder_blade  -> mid_vertebrae (2)
        12,  # 13 left_shoulder   -> left_shoulder_blade (12)
        13,  # 14 left_elbow      -> left_shoulder (13)
        14,  # 15 left_wrist      -> left_elbow (14)
        2,   # 16 right_shoulder_blade -> mid_vertebrae (2)
        16,  # 17 right_shoulder  -> right_shoulder_blade (16)
        17,  # 18 right_elbow     -> right_shoulder (17)
        18,  # 19 right_wrist     -> right_elbow (18)
    ]

    # Topological order for forward kinematics (parents before children)
    TOPO_ORDER = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # Symmetric joint pairs: (left_index, right_index)
    SYMMETRIC_PAIRS = [
        (6, 9),    # hip
        (7, 10),   # knee
        (8, 11),   # ankle
        (12, 16),  # shoulder_blade
        (13, 17),  # shoulder
        (14, 18),  # elbow
        (15, 19),  # wrist
    ]

    @staticmethod
    def factory(name, data, args=None):
        return MagYawCorrectionNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.symmetric_input = self.add_input('symmetric', widget_type='checkbox', default_value=True)
        self.sync_input = self.add_input('sync local/global', widget_type='checkbox', default_value=True)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset_all)

        self._in_propagation = False

        # Global yaw sliders (ongoing magnetometer error, pre-multiply around world Y)
        self.global_inputs = []
        for name in self.JOINT_NAMES:
            inp = self.add_input(name, widget_type='drag_float',
                                default_value=0.0, callback=self.on_global_changed)
            self.global_inputs.append(inp)

        # Local yaw sliders (calibration error, post-multiply in body frame)
        self.local_inputs = []
        for name in self.JOINT_NAMES:
            inp = self.add_input(name + '_cal', widget_type='drag_float',
                                default_value=0.0, callback=self.on_local_changed)
            self.local_inputs.append(inp)

        self.pose_output = self.add_output('pose out')

        # Pre-allocated work arrays
        self._world_quats = np.zeros((20, 4), dtype=np.float64)
        self._world_quats[:, 0] = 1.0

    def custom_create(self, from_file):
        for inp in self.global_inputs:
            inp.widget.set_speed(1.0)
            dpg.set_item_width(inp.widget.uuid, 45)
        for inp in self.local_inputs:
            inp.widget.set_speed(1.0)
            dpg.set_item_width(inp.widget.uuid, 45)

    def reset_all(self):
        self._in_propagation = True
        for inp in self.global_inputs:
            inp.set(0.0)
        for inp in self.local_inputs:
            inp.set(0.0)
        self._in_propagation = False

    def _find_input_index(self, input_list):
        """Find which index in input_list matches self.active_input."""
        for i, inp in enumerate(input_list):
            if inp is self.active_input:
                return i
        return None

    def _propagate_symmetric(self, input_list, changed_idx, value):
        """Copy value to symmetric counterpart in the given input list."""
        for left_idx, right_idx in self.SYMMETRIC_PAIRS:
            if changed_idx == left_idx:
                input_list[right_idx].set(value)
                return
            elif changed_idx == right_idx:
                input_list[left_idx].set(value)
                return

    def on_global_changed(self):
        if self._in_propagation:
            return
        idx = self._find_input_index(self.global_inputs)
        if idx is None:
            return

        self._in_propagation = True
        value = self.global_inputs[idx]()

        # Sync: copy global -> local
        if self.sync_input():
            self.local_inputs[idx].set(value)

        # Symmetric: mirror to other side
        if self.symmetric_input():
            self._propagate_symmetric(self.global_inputs, idx, value)
            if self.sync_input():
                # Also mirror the local side
                self._propagate_symmetric(self.local_inputs, idx, value)

        self._in_propagation = False

    def on_local_changed(self):
        if self._in_propagation:
            return
        idx = self._find_input_index(self.local_inputs)
        if idx is None:
            return

        self._in_propagation = True
        value = self.local_inputs[idx]()

        # Sync: copy local -> global
        if self.sync_input():
            self.global_inputs[idx].set(value)

        # Symmetric: mirror to other side
        if self.symmetric_input():
            self._propagate_symmetric(self.local_inputs, idx, value)
            if self.sync_input():
                # Also mirror the global side
                self._propagate_symmetric(self.global_inputs, idx, value)

        self._in_propagation = False

    def get_preset_state(self):
        return {
            'global_yaw': [g() for g in self.global_inputs],
            'local_cal': [l() for l in self.local_inputs],
        }

    def set_preset_state(self, preset):
        self._in_propagation = True
        if 'global_yaw' in preset:
            for i, val in enumerate(preset['global_yaw']):
                if i < len(self.global_inputs):
                    self.global_inputs[i].set(float(val))
        if 'local_cal' in preset:
            for i, val in enumerate(preset['local_cal']):
                if i < len(self.local_inputs):
                    self.local_inputs[i].set(float(val))
        # Backwards compatibility: old presets with 'yaw_angles'
        if 'yaw_angles' in preset and 'global_yaw' not in preset:
            for i, val in enumerate(preset['yaw_angles']):
                if i < len(self.global_inputs):
                    self.global_inputs[i].set(float(val))
                if i < len(self.local_inputs):
                    self.local_inputs[i].set(float(val))
        self._in_propagation = False

    @staticmethod
    def _qmul(q1, q2):
        """Multiply two quaternions in wxyz format. Returns normalized result."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        out = np.array([w, x, y, z], dtype=np.float64)
        n = np.linalg.norm(out)
        if n > 1e-12:
            out /= n
        return out

    @staticmethod
    def _qinv(q):
        """Inverse of a unit quaternion in wxyz format: conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _yaw_quat(angle_deg):
        """Create a quaternion for rotation around Y axis (up), wxyz format."""
        half = np.radians(angle_deg) * 0.5
        return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)

    def execute(self):
        raw = self.pose_input()
        if raw is None:
            return
        pose = any_to_array(raw)
        if pose.ndim == 1 and pose.size == 80:
            pose = pose.reshape(20, 4)
        if pose.ndim != 2 or pose.shape[0] != 20 or pose.shape[1] != 4:
            return

        local = pose.astype(np.float64)
        world = self._world_quats

        # Read all correction angles
        global_angles = [self.global_inputs[i]() for i in range(20)]
        local_angles = [self.local_inputs[i]() for i in range(20)]
        any_correction = (any(abs(a) > 0.001 for a in global_angles) or
                          any(abs(a) > 0.001 for a in local_angles))

        if not any_correction:
            self.pose_output.send(pose)
            return

        # --- Step 1: Forward kinematics (local -> world) ---
        for i in self.TOPO_ORDER:
            p = self.PARENT_INDEX[i]
            if p == -1:
                world[i] = local[i]
            else:
                world[i] = self._qmul(world[p], local[i])

        # --- Step 2: Apply corrections ---
        # Q_corrected = Q_global(-alpha) * Q_world * Q_local(beta)
        #   global: pre-multiply by yaw(-alpha) around world Y (ongoing error)
        #   local:  post-multiply by yaw(beta) in body frame (calibration error)
        for i in range(20):
            g = global_angles[i]
            l = local_angles[i]
            q = world[i]
            if abs(g) > 0.001:
                q = self._qmul(self._yaw_quat(-g), q)
            if abs(l) > 0.001:
                q = self._qmul(q, self._yaw_quat(l))
            world[i] = q

        # --- Step 3: Inverse kinematics (world -> corrected local) ---
        result = np.zeros((20, 4), dtype=np.float64)
        for i in self.TOPO_ORDER:
            p = self.PARENT_INDEX[i]
            if p == -1:
                result[i] = world[i]
            else:
                result[i] = self._qmul(self._qinv(world[p]), world[i])

        self.pose_output.send(result.astype(np.float32))


class ActivePoseNode(Vector2DNode):
    """Pose editor for 20 bmolab active joints (wxyz quaternions)."""

    JOINT_NAMES = [
        'base_of_skull', 'upper_vertebrae', 'mid_vertebrae', 'lower_vertebrae',
        'spine_pelvis', 'pelvis_anchor', 'left_hip', 'left_knee',
        'left_ankle', 'right_hip', 'right_knee', 'right_ankle',
        'left_shoulder_blade', 'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder_blade', 'right_shoulder', 'right_elbow', 'right_wrist'
    ]

    @staticmethod
    def factory(name, data, args=None):
        return ActivePoseNode(name, data, args)

    def __init__(self, label: str, data, args):
        Vector2DNode.__init__(self, label, data, ['20', '4'])

    def custom_create(self, from_file):
        Vector2DNode.custom_create(self, from_file)
        self.zero_input.set_label('reset')
        for i in range(20):
            inp = self.inputs[i + 1]  # +1 because inputs[0] is 'in'
            dpg.configure_item(inp.widget.uuids[3], label=self.JOINT_NAMES[i])
            for uid in inp.widget.uuids:
                dpg.configure_item(uid, width=45)
            inp.set([1.0, 0.0, 0.0, 0.0])
            self.output_vector[i, 0] = 1.0

    def zero(self):
        """Reset all quaternions to identity [1, 0, 0, 0]."""
        if self.vector_format_input() == 'numpy':
            self.output_vector = np.zeros(self.current_dims)
            self.output_vector[:, 0] = 1.0
        elif self.vector_format_input() == 'torch':
            self.output_vector = torch.zeros(self.current_dims)
            self.output_vector[:, 0] = 1.0
        else:
            self.output_vector = [[1.0, 0.0, 0.0, 0.0]] * self.current_dims[0]
        self.execute()

    def get_preset_state(self):
        preset = {}
        values = []
        for i in range(self.current_dims[0]):
            values.append(list(self.output_vector[i]))
        preset['values'] = values
        return preset

    def set_preset_state(self, preset):
        if 'values' in preset:
            values = preset['values']
            self.input._data = values
            self.input.fresh_input = True
            self.execute()


class ShadowArmCorrectNode(MoCapNode):
    """Live, interactive version of correct_upper_arm_offset.py for the 37-joint
    Shadow pose (quats + positions). Applies the same per-frame correction the
    offline script does — a per-arm fit C plus the anatomical dials (twist /
    abduction / flex / elbow / wrist / hand-twist) — so you tune with sliders and
    watch the SMPL render update, instead of generating files and reloading.

    Accepts the pose as 37-joint Shadow OR 20-joint active (remapped); output matches
    the input layout. Positions are used only to derive the body LATERAL axis (for
    abduction/flex), in the shadow world frame: shadow positions (37 or 20) if given,
    else the chest orientation (MidVertebrae +X). Do NOT feed SMPL joint positions:
    they live in a different post-conversion frame.

    Wire: shadow quats -> 'pose in', shadow positions (optional) -> 'positions in',
    'pose out' -> (shadow_to_smpl / render). Fit button runs the per-arm C from the
    sym/relax anchors; export button writes _armfix.npz for the loaded take file.
    """

    # active(20) joint index -> shadow(37) index, for the joints the math reads/writes.
    _ACT2SHA = {1: 17, 2: 1, 3: 32, 4: 31, 5: 4,
                12: 13, 13: 5, 14: 9, 15: 10, 16: 27, 17: 19, 18: 23, 19: 24}
    # joints the correction modifies (shadow idx -> active idx), for writing back.
    _CORR_SHA2ACT = {5: 13, 9: 14, 10: 15, 19: 17, 23: 18, 24: 19}

    @staticmethod
    def factory(name, data, args=None):
        return ShadowArmCorrectNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        from dpg_system.smpl_utilities import correct_upper_arm_offset as cuo
        self._cuo = cuo
        self._cl = np.zeros(3)   # per-arm upper-arm fit (rotvec); identity until fit/loaded
        self._cr = np.zeros(3)
        self._hy_l = np.zeros(2)  # per-arm upper-arm heading-yaw deviation curve [b, c] (rad)
        self._hy_r = np.zeros(2)

        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.positions_input = self.add_input('positions in')
        # --- fit (batch) controls: load the take, fit C over the anchor frames ---
        self.take_path = self.add_input('take file', widget_type='text_input', default_value='')
        self.sym_field = self.add_input('sym ranges', widget_type='text_input', default_value='')
        self.relax_field = self.add_input('relax', widget_type='text_input', default_value='')
        self.fit_button = self.add_input('fit', widget_type='button', callback=self.do_fit)
        self.fit_path = self.add_input('load fit npz', widget_type='text_input', default_value='',
                                       callback=self.load_fit)
        # symmetric (default on) ties the two abduction sliders so dialing can't break
        # the fit's left/right symmetry; turn off only for a deliberate per-arm offset.
        self.symmetric = self.add_input('symmetric', widget_type='checkbox', default_value=True)
        self.twist = self.add_input('twist', widget_type='drag_float', default_value=0.0)
        self.abduct_l = self.add_input('abduct_l', widget_type='drag_float', default_value=0.0)
        self.abduct_r = self.add_input('abduct_r', widget_type='drag_float', default_value=0.0)
        self.flex = self.add_input('flex', widget_type='drag_float', default_value=0.0)
        self.elbow = self.add_input('elbow', widget_type='drag_float', default_value=0.0)
        self.wrist = self.add_input('wrist', widget_type='drag_float', default_value=0.0)
        self.wtwist = self.add_input('wtwist', widget_type='drag_float', default_value=0.0)
        for w in (self.twist, self.abduct_l, self.abduct_r, self.flex,
                  self.elbow, self.wrist, self.wtwist):
            w.widget.speed = 1.0                       # 1 deg per drag unit
        self.reset_input = self.add_input('reset dials', widget_type='button', callback=self.reset_dials)
        self.export_button = self.add_input('export armfix', widget_type='button', callback=self.do_export)
        self.pose_output = self.add_output('pose out')

    # ---- helpers shared by live execute() and batch export ----
    def _dials(self):
        abl = float(self.abduct_l())
        abr = abl if self.symmetric() else float(self.abduct_r())   # symmetric ties abr=abl
        return dict(twist_deg=float(self.twist()), abduct_l=abl, abduct_r=abr,
                    flex_deg=float(self.flex()), elbow_deg=float(self.elbow()),
                    wrist_deg=float(self.wrist()), wtwist_deg=float(self.wtwist()))

    def _embed37(self, Q):
        """(F, nj, 4) shadow(37) or active(20) -> (F, 37, 4), plus active flag."""
        F, nj = Q.shape[0], Q.shape[1]
        if nj == 37:
            return Q, False
        Q37 = np.tile(np.array([1.0, 0, 0, 0]), (F, 37, 1))
        for a, s in self._ACT2SHA.items():
            Q37[:, s] = Q[:, a]
        return Q37, True

    def _pos37(self, P, F):
        """positions (F, njp, 3) -> (F, 37, 3); zeros (orientation fallback) if unusable."""
        P37 = np.zeros((F, 37, 3))
        if P is None or P.ndim != 3 or P.shape[2] != 3:
            return P37
        if P.shape[1] == 37:
            P37 = P
        elif P.shape[1] == 20:
            for a, s in self._ACT2SHA.items():
                P37[:, s] = P[:, a]
        return P37                                     # else (root only) -> zeros

    def _correct_batch(self, Q, P):
        """Q (F, nj, 4), P (F, njp, 3) or None -> corrected (F, nj, 4) in input layout."""
        cuo = self._cuo
        Q37, active = self._embed37(Q)
        P37 = self._pos37(P, Q.shape[0])
        G = cuo.forward_kinematics(Q37)
        Qc = cuo.apply_correction(Q37, G, P37, self._cl, self._cr,
                                  hy_l=self._hy_l, hy_r=self._hy_r, **self._dials())
        if active:
            out = Q.copy()
            for s, a in self._CORR_SHA2ACT.items():
                out[:, a] = Qc[:, s]
            return out
        return Qc

    def _parse_ranges(self, txt):
        return [tuple(int(v) for v in r.split(':')) for r in txt.split(',') if ':' in r]

    @staticmethod
    def _validate_ranges(ranges, F, kind):
        """Drop empty/reversed ranges, clamp to [0, F); report what was dropped/clamped.
        Returns the cleaned list of (a, b) tuples (possibly empty)."""
        good = []
        for a, b in ranges:
            ca, cb = max(0, min(a, F)), max(0, min(b, F))
            if cb <= ca:
                print(f"shadow_arm_correct: {kind} range {a}:{b} is empty (take has {F} frames); skipped")
                continue
            if (ca, cb) != (a, b):
                print(f"shadow_arm_correct: {kind} range {a}:{b} clamped to {ca}:{cb} (take has {F} frames)")
            good.append((ca, cb))
        return good

    def load_fit(self):
        path = self.fit_path()
        if path:
            try:
                z = np.load(path)
                self._cl, self._cr = np.asarray(z['cl']), np.asarray(z['cr'])
                self._hy_l = np.asarray(z['hy_l']) if 'hy_l' in z.files else np.zeros(2)
                self._hy_r = np.asarray(z['hy_r']) if 'hy_r' in z.files else np.zeros(2)
                print(f"shadow_arm_correct: loaded fit from {path}")
            except Exception as e:
                print(f"shadow_arm_correct: could not load fit '{path}': {e}")

    def do_fit(self):
        """Load the take file, fit the per-arm C over the sym/relax anchor frames."""
        path = self.take_path()
        cuo = self._cuo
        try:
            d = np.load(path, allow_pickle=True)
            Q = d['quats'].astype(np.float64)
            P = d['positions'].astype(np.float64) if 'positions' in d.files else None
        except Exception as e:
            print(f"shadow_arm_correct: could not load take '{path}': {e}")
            return
        sym_raw = self._parse_ranges(self.sym_field())
        rl_raw = [tuple(int(v) for v in self.relax_field().split(':'))] if ':' in self.relax_field() else []
        if not sym_raw or not rl_raw:
            print("shadow_arm_correct: enter sym ranges 'a:b,a:b' and relax 'a:b' before fitting")
            return
        F = Q.shape[0]
        sym = self._validate_ranges(sym_raw, F, 'sym')
        rl = self._validate_ranges(rl_raw, F, 'relax')
        if not sym or not rl:
            print(f"shadow_arm_correct: no valid ranges left after validation (take has {F} frames); "
                  "check sym/relax (each 'a:b' with a<b and both within [0,F))")
            return
        Q37, _ = self._embed37(Q)
        P37 = self._pos37(P, Q.shape[0])
        # positions are optional now (precompute falls back to chest orientation for the lateral
        # axis when shoulder positions are absent), so active-20 + trans-only files can fit too.
        G = cuo.forward_kinematics(Q37)
        pre = cuo.precompute(G, P37, sym, rl[0])
        self._cl, self._cr = cuo.fit_corrections(pre, len(sym))
        self._hy_l, self._hy_r = cuo.fit_heading_yaw(Q37, P37, self._cl, self._cr, sym)
        cuo.report(pre, sym, rl[0], self._cl, self._cr)
        print(f"shadow_arm_correct: fit applied  (heading-yaw amp |L|="
              f"{np.degrees(np.hypot(*self._hy_l)):.0f}  |R|={np.degrees(np.hypot(*self._hy_r)):.0f} deg); "
              "live preview updated")

    def do_export(self):
        """Apply the current fit + dials to the whole take file and save _armfix.npz."""
        path = self.take_path() or self.fit_path()
        try:
            d = np.load(path, allow_pickle=True)
            Q = d['quats'].astype(np.float64)
            P = d['positions'].astype(np.float64) if 'positions' in d.files else None
        except Exception as e:
            print(f"shadow_arm_correct: could not load take to export '{path}': {e}")
            return
        out = self._correct_batch(Q, P)
        save = {k: d[k] for k in d.files}
        save['quats'] = out.astype(np.float32)
        out_path = path.replace('.npz', '_armfix.npz')
        np.savez(out_path, **save)
        print(f"shadow_arm_correct: wrote {out_path}")

    def reset_dials(self):
        for w in (self.twist, self.abduct_l, self.abduct_r, self.flex,
                  self.elbow, self.wrist, self.wtwist):
            w.set(0.0)

    def execute(self):
        raw = self.pose_input()
        if raw is None:
            return
        Q = any_to_array(raw).astype(np.float64)
        if Q.ndim == 1 and Q.size % 4 == 0:
            Q = Q.reshape(-1, 4)
        if Q.ndim != 2 or Q.shape[1] != 4 or Q.shape[0] not in (20, 37):
            return
        pos = self.positions_input()
        P = None
        if pos is not None:
            P = any_to_array(pos).astype(np.float64)
            if P.ndim == 1 and P.size % 3 == 0:
                P = P.reshape(-1, 3)
            if P.ndim == 2 and P.shape[1] == 3:
                P = P[None]
            else:
                P = None
        out = self._correct_batch(Q[None], P)[0]
        self.pose_output.send(out.astype(np.float32))
