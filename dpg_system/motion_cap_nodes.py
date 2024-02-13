from dpg_system.body_base import *


def register_mocap_nodes():
    Node.app.register_node('gl_body', MoCapGLBody.factory)
    Node.app.register_node('take', MoCapTakeNode.factory)
    Node.app.register_node('body_to_joints', MoCapBody.factory)
    Node.app.register_node('pose', PoseNode.factory)
    Node.app.register_node('active_joints', ActiveJointsNode.factory)

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

    shadow_to_active_joints_index = [2, 17, 1, 32, 31, 4, 14, 12, 8, 28, 26, 22, 13, 5, 9, 10, 27, 19, 23, 24]

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
        self.file_name = self.add_label('')
        self.quaternions_out = self.add_output('quaternions')
        self.positions_out = self.add_output('positions')
        self.labels_out = self.add_output('labels')
        load_path = ''
        self.load_path = self.add_option('path', widget_type='text_input', default_value=load_path, callback=self.load_from_load_path)
        self.message_handlers['load'] = self.load_take_message

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
            self.load_take_from_npz(path)

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

        self.joint_offsets = []
        for key in self.joint_map:
            index = self.joint_map[key]
            self.joint_offsets.append(index)

        self.joint_inputs = []
        for key in self.joint_map:
            stripped_key = key.replace('_', ' ')
            input = self.add_input(stripped_key, triggers_execution=True)
            self.joint_inputs.append(input)

        self.output = self.add_output('pose out')

    def execute(self):
        # we need to know the size of the pose vector...
        pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 37)
        for index, input in enumerate(self.joint_inputs):
            incoming = input()
            t = type(incoming)

            if t == np.ndarray:
                offset = self.joint_offsets[index]
                pose[offset] = incoming
        self.output.send(pose)


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
            incoming = self.input()
            t = type(incoming)
            if t == np.ndarray:
                for i, index in enumerate(self.joint_offsets):
                    if index < incoming.shape[0]:
                        joint_value = incoming[index]
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()


class MoCapGLBody(MoCapNode):
    smpl_limb_to_joint_dict = {
        'left_hip': 'LeftHip',
        'right_hip': 'RightHip',
        'lower_back': 'SpinePelvis',
        'left_thigh': 'LeftKnee',
        'right_thigh': 'RightKnee',
        'mid_back': 'LowerVertebrae',
        'left_lower_leg': 'LeftFoot',
        'right_lower_leg': 'RightFoot',
        'upper_back': 'MidVertebrae',
        'left_foot': 'LeftBallOfFoot',
        'right_foot': 'RightBallOfFoot',
        'lower_neck': 'UpperVertebrae',
        'left_shoulder_blade': 'LeftShoulderBladeBase',
        'right_shoulder_blade': 'RightShoulderBladeBase',
        'upper_neck': 'BaseOfSkull',
        'left_shoulder': 'LeftShoulder',
        'right_shoulder': 'RightShoulder',
        'left_upper_arm': 'LeftElbow',
        'right_upper_arm': 'RightElbow',
        'left_forearm': 'LeftWrist',
        'right_forearm': 'RightWrist',
        'left_hand': 'LeftKnuckle',
        'right_hand': 'RightKnuckle'
    }

    joint_mapped = {
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
        self.current_joint_output = self.add_output('current_joint_name')
        # self.current_joint_quaternion_output = self.add_output('current_joint_quaternion')
        self.current_joint_rotation_axis_output = self.add_output('current_joint_quaternion_axis')
        self.current_joint_gl_output = self.add_output('current_joint_gl_chain')

        self.skeleton_only = self.add_option('skeleton_only', widget_type='checkbox', default_value=False)
        self.show_joint_spheres = self.add_option('show joint motion', widget_type='checkbox', default_value=self.show_joint_activity)
        self.joint_data_selection = self.add_option('joint data type', widget_type='combo', default_value='diff_axis-angle')
        self.joint_data_selection.widget.combo_items = ['diff_quaternion', 'diff_axis-angle']
        self.joint_motion_scale = self.add_option('joint motion scale', widget_type='drag_float', default_value=5)
        self.diff_quat_smoothing = self.add_option('joint motion smoothing', widget_type='drag_float', default_value=0.8, max=1.0, min=0.0)
        self.joint_disk_alpha = self.add_option('joint motion alpha', widget_type='drag_float', default_value=0.5, max=1.0, min=0.0)
        self.body = BodyData()
        self.body.node = self

    # def joint_callback(self):
    #     self.gl_chain_output.send('draw')
    #
    def process_commands(self, command):
        if type(command[0]) == str:
            print(command)
            if command[0] in self.smpl_limb_to_joint_dict:
                target_joint = self.smpl_limb_to_joint_dict[command[0]]
                if target_joint in joint_name_to_index:
                    target_joint_index = joint_name_to_index[target_joint]
                    self.body.joints[target_joint_index].bone_dim = command[1:]
                    self.body.joints[target_joint_index].set_matrix()
                # self.body.joints[target_joint_index].set_mass()

    def joint_callback(self, joint_index):
        glPushMatrix()
        mode = self.joint_data_selection()
        joint_name = joint_index_to_name[joint_index]
        self.current_joint_output.send(joint_name)
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
            if t == np.ndarray:
                # work on this!!!
                if incoming.shape[0] == 37:
                    for joint_name in self.joint_map:
                        joint_id = self.joint_map[joint_name]
                        self.body.update(joint_index=joint_id, quat=incoming[joint_id])
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
                self.body.draw(self.show_joint_spheres(), self.skeleton_only())


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
        if t == np.ndarray:
            # work on this!!!
            if incoming.shape[0] == 37:
                active_joints = incoming[self.shadow_to_active_joints_index]
                self.active_joints_out.send(active_joints)
