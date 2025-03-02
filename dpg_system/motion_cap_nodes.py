
import torch
from dpg_system.body_base import *
from pyquaternion import Quaternion
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML
import scipy

print('about to import shadow')
import dpg_system.MotionSDK as shadow
print('imported shadow')

def register_mocap_nodes():
    Node.app.register_node('gl_body', MoCapGLBody.factory)
    Node.app.register_node('gl_simple_body', SimpleMoCapGLBody.factory)
    Node.app.register_node('gl_alt_body', AlternateMoCapGLBody.factory)
    Node.app.register_node('take', MoCapTakeNode.factory)
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
    shadow_to_active_map = joints_to_input_vector

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

        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.load_button = self.add_property('load', widget_type='button', callback=self.load_take)
        self.dump_button = self.add_property('dump', widget_type='button', callback=self.dump_take)
        self.file_name = self.add_label('')
        self.quaternions_out = self.add_output('quaternions')
        self.positions_out = self.add_output('positions')
        self.labels_out = self.add_output('labels')
        self.dump_out = self.add_output('dump')
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path, callback=self.load_from_load_path)
        self.message_handlers['load'] = self.load_take_message

    def dump_take(self):
        self.dump_out.send(self.position_buffer)

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
        if self.current_frame > self.frames:
            self.current_frame = 0
        self.input.set(self.current_frame)
        frame = int(self.current_frame)
        self.quaternions_out.set_value(self.quat_buffer[frame])
        self.positions_out.set_value(self.position_buffer[frame])
        self.labels_out.set_value(self.label_buffer[frame])
        self.send_all()

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
        data = self.input()
        if data < self.frames:
            self.current_frame = data
            self.quaternions_out.set_value(self.quat_buffer[self.current_frame])
            self.positions_out.set_value(self.position_buffer[self.current_frame])
            self.labels_out.set_value(self.label_buffer[self.current_frame])
            self.send_all()

    def execute(self):
        if self.input.fresh_input:
            data = self.input()

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
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
                             user_data=self, callback=self.load_npz_callback, tag="file_dialog_id"):
            dpg.add_file_extension(".npz")

    def load_npz_callback(self, sender, app_data):
        # print('self=', self, 'sender=', sender, 'app_data=', app_data)
        if 'file_path_name' in app_data:
            load_path = app_data['file_path_name']
            if load_path != '':
                self.load_path.set(load_path)
                self.load_take_from_npz(load_path)
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
                input_ = self.add_input(stripped_key, triggers_execution=True)
                self.joint_inputs.append(input_)
        else:
            for key in self.active_joint_map:
                stripped_key = key.replace('_', ' ')
                input_ = self.add_input(stripped_key, triggers_execution=True)
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
        self.gl_chain_input = self.add_input('gl chain', triggers_execution=True)
        self.gl_chain_output = self.add_output('gl_chain')
        self.capture_pose_input = self.add_input('capture pose', widget_type='button', callback=self.capture_pose)
        self.current_joint_output = self.add_output('current_joint_name')
        self.current_joint_rotation_axis_output = self.add_output('current_joint_quaternion_axis')
        self.current_joint_gl_output = self.add_output('current_joint_gl_chain')

        self.absolute_quats_input = self.add_option('absolute quats', widget_type='checkbox')
        self.calc_diff_quats = self.add_option('calc_diff_quats', widget_type='checkbox', default_value=False, callback=self.set_calc_diff)
        self.skeleton_only = self.add_option('skeleton_only', widget_type='checkbox', default_value=False)
        self.show_joint_spheres = self.add_option('show joint motion', widget_type='checkbox', default_value=self.show_joint_activity)
        self.joint_data_selection = self.add_option('joint data type', widget_type='combo', default_value='diff_axis-angle')
        self.joint_data_selection.widget.combo_items = ['diff_quaternion', 'diff_axis-angle']
        self.joint_motion_scale = self.add_option('joint motion scale', widget_type='drag_float', default_value=5)
        self.diff_quat_smoothing = self.add_option('joint motion smoothing', widget_type='drag_float', default_value=0.8, max=1.0, min=0.0)
        self.joint_disk_alpha = self.add_option('joint motion alpha', widget_type='drag_float', default_value=0.5, max=1.0, min=0.0)
        self.body_color_id = self.add_option('colour id', widget_type='input_int', default_value=0)
        self.body = BodyData()
        self.body.node = self

    def set_calc_diff(self):
        self.body.calc_diff = self.calc_diff_quats()

    def capture_pose(self):
        self.body.capture_current_pose()

    def process_commands(self, command):
        if type(command[0]) == str:
            print(command)
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in joint_name_to_index:
                    target_joint_index = joint_name_to_index[target_joint]
                    self.body.joints[target_joint_index].bone_dim = command[1:]
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
                    if joint_name in joint_name_to_index:
                        joint_index = joint_name_to_index[joint_name]
                        if len(command) > 2:
                            dims = command[2:]
                            if len(dims) == 1:
                                self.body.joints[joint_index].length = dims[0]

    def set_limb_vertices(self, name, vertices):
        self.body.set_limb_vertices(name, vertices)

    def joint_callback(self, joint_index):
        glPushMatrix()
        mode = self.joint_data_selection()
        # joint_name = joint_index_to_name[joint_index]
        self.current_joint_output.send(joint_index)
        if mode == 'diff_axis-angle':
            rotation = np.array(self.body.rotationAxis[joint_index])
            rotation = rotation / (np.linalg.norm(rotation) + 1e-6) * self.body.quaternionDistance[joint_index] * self.joint_motion_scale()
            # self.current_joint_quaternion_output.send(self.body.quaternionDistance[joint_index])
            self.current_joint_rotation_axis_output.send(rotation)
        elif mode == 'diff_quaternion':
            if joint_index in [t_LeftShoulderBladeBase, t_LeftShoulder, t_LeftElbow, t_LeftWrist, t_RightShoulderBladeBase, t_LeftKnuckle, t_RightShoulder, t_RightElbow, t_RightWrist, t_RightKnuckle]:
                glRotate(90, 0.0, 1.0, 0.0)
            else:
                glRotate(90, 1.0, 0.0, 0.0)
            glRotate(90, 0.0, 1.0, 0.0)

            if type(self.body.quaternionDiff[joint_index]) == list:
                self.current_joint_rotation_axis_output.send(self.body.quaternionDiff[joint_index])
            else:
                self.current_joint_rotation_axis_output.send(self.body.quaternionDiff[joint_index].elements)
        self.current_joint_gl_output.send('draw')
        glPopMatrix()

    def execute(self):
        if self.input.fresh_input:
            incoming = self.input()
            t = type(incoming)
            if t == torch.Tensor:
                incoming = tensor_to_array(incoming)
                t = np.ndarray
            if t == np.ndarray:
                if incoming.shape[0] == 80:
                    self.body.update_quats(np.reshape(incoming, [20, 4]))
                elif incoming.shape[0] == 20:
                        if incoming.shape[1] == 4:
                            self.body.update_quats(incoming)
                # work on this!!!
                # if incoming.shape[0] == 37:
                #     for joint_name in self.joint_map:
                #         joint_id = self.joint_map[joint_name]
                #         self.body.update(joint_index=joint_id, quat=incoming[joint_id], label=self.body_color_id())
                # elif incoming.shape[0] == 20:
                #     for index, joint_name in enumerate(self.joint_map):
                #         joint_id = self.joint_map[joint_name]
                #         self.body.update(joint_index=joint_id, quat=incoming[index], label=self.body_color_id())

            elif t == list:
                self.process_commands(incoming)

        elif self.gl_chain_input.fresh_input:
            incoming = self.gl_chain_input()
            t = type(incoming)
            if t == str and incoming == 'draw':
                scale = self.joint_motion_scale()
                smoothing = self.diff_quat_smoothing()
                self.body.joint_motion_scale = scale
                self.body.diffQuatSmoothingA = smoothing
                self.body.joint_disk_alpha = self.joint_disk_alpha()
                if self.absolute_quats_input():
                    self.body.draw_absolute_quats(self.show_joint_spheres(), self.skeleton_only())
                else:
                    self.body.draw(self.show_joint_spheres(), self.skeleton_only())
                self.gl_chain_output.send('draw')


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
            print(command)
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in joint_name_to_index:
                    target_joint_index = joint_name_to_index[target_joint]
                    if len(command) >= 2:
                        # self.body.limbs[target_joint_index].dims[2] = any_to_float(command[1])
                        self.body.joints[target_joint_index].length = any_to_float(command[3])
                    if len(command) == 3:
                        self.body.limbs[target_joint_index].dims[1] = any_to_float(command[2])
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[2])
                        #self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    if len(command) == 4:
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[3])
                        # self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    self.body.joints[target_joint_index].set_matrix()
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
        left_shoulder_blade_rel_quat = quaternion_reciprocal_xyzw(quaternion_multiply(v(active_joints_data[offset]), active_joints_data[previous_offset]))
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
            incoming = tensor_to_array(incoming)
            t = np.ndarray
        if t == np.ndarray:
            # work on this!!!
            if incoming.shape[0] == 37:
                active_joints = incoming[self.active_to_shadow_map]
                self.active_joints_out.send(active_joints)


def shadow_service_loop():
    while True:
        for node in MotionShadowNode.shadow_nodes:
            node.receive_data()
        # time.sleep(.01)


class MotionShadowNode(MoCapNode):
    shadow_nodes = []
    @staticmethod
    def factory(name, data, args=None):
        node = MotionShadowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.num_bodies = 0
        super().__init__(label, data, args)

        self.__mutex = threading.Lock()
        self.client = None
        try:
            self.client = shadow.Client("", 32076)
        except Exception as e:
            self.client = None
        self.origin = [0.0, 0.0, 0.0] * 4
        self.positions = np.ndarray((4, 37, 3))
        self.quaternions = np.ndarray((4, 37, 4))

        self.joints_mapped = False
        self.jointMap = [0, 0] * 37

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
                        for idx, name_index in enumerate(joint_index_to_name):
                            shadow_name = joint_to_shadow_limb[joint_index_to_name[name_index]]
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

                        joint = joint_index_to_name[master_key]

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

    def frame_task(self):
        if not self.new_data:
            return
        self.new_data = False
        if self.num_bodies > 0:
            self.body_quat_1.send(self.quaternions[0])
            self.body_pos_1.send(self.positions[0])
        if self.num_bodies > 1:
            self.body_quat_2.send(self.quaternions[1])
            self.body_pos_2.send(self.positions[1])
        if self.num_bodies > 2:
            self.body_quat_3.send(self.quaternions[2])
            self.body_pos_3.send(self.positions[2])
        if self.num_bodies > 3:
            self.body_quat_4.send(self.quaternions[3])
            self.body_pos_4.send(self.positions[3])

    def parse_name_map(self, xml_node_list):
        name_map = {}

        tree = XML(xml_node_list)

        # <node key="N" id="Name"> ... </node>
        list = tree.findall(".//node")
        for itr in list:
            node_name = itr.get("id")
            node_local = node_name
            node_body = 0
            for code in joint_index_to_name:
                node_code = joint_to_shadow_limb[joint_index_to_name[code]]
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
    #
    def process_commands(self, command):
        if type(command[0]) == str:
            print(command)
            if command[0] in BodyDataBase.smpl_limb_to_joint_dict:
                target_joint = BodyDataBase.smpl_limb_to_joint_dict[command[0]]
                if target_joint in joint_name_to_index:
                    target_joint_index = joint_name_to_index[target_joint]
                    if len(command) >= 2:
                        # self.body.limbs[target_joint_index].dims[2] = any_to_float(command[1])
                        self.body.joints[target_joint_index].length = any_to_float(command[3])
                    if len(command) == 3:
                        self.body.limbs[target_joint_index].dims[1] = any_to_float(command[2])
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[2])
                        #self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    if len(command) == 4:
                        self.body.limbs[target_joint_index].dims[0] = any_to_float(command[3])
                        # self.body.joints[target_joint_index].thickness = (any_to_float(command[2]), any_to_float(command[2]))

                    self.body.joints[target_joint_index].set_matrix()
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





