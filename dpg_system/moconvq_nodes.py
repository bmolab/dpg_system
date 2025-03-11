# Adapted code from in MoConVQ/Script/track_something.py from the MoConVQ paper https://moconvq.github.io/
 
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MoConVQ/MoConVQCore'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MoConVQ/ModifyODESrc'))
from dpg_system.body_base import *
from MoConVQCore.Env.vclode_track_env import VCLODETrackEnv
from MoConVQCore.Model.MoConVQ import MoConVQ
from MoConVQCore.Utils.motion_dataset import MotionDataSet
from dpg_system.SMPL_nodes import SMPLShadowTranslator 

import argparse
from MoConVQCore.Utils.misc import *
from MoConVQCore.Utils import pytorch_utils as ptu
from MoConVQCore.Utils.motion_dataset import MotionDataSet, DPGMotionDataset
import psutil
import scipy

def register_moconvq_nodes():
    Node.app.register_node("moconvq_take", MoConVQSMPLTakeNode.factory)
    Node.app.register_node("moconvq_pose_to_joints", MoConVQPoseToJointsNode.factory)
    Node.app.register_node("moconvq_env", MoConVQEnvNode.factory)
    Node.app.register_node("moconvq_storage", MoConVQStorageNode.factory)
    Node.app.register_node("moconvq_torque_to_joints", MoConVQTorqueToJointsNode.factory)
    Node.app.register_node("pose_to_pose_translator", PoseToPoseTranslator.factory)

class MoConVQNode(Node):
    def __init__(self, label: str, data, args):
      super().__init__(label, data, args)
      self.joint_names = ['RootJoint', 'pelvis_lowerback', 'lowerback_torso', 'rHip', 'lHip', 
                          'rKnee', 'lKnee', 'rAnkle', 'lAnkle', 'rToeJoint', 'rToeJoint_end', 
                          'lToeJoint', 'lToeJoint_end', 'torso_head', 'torso_head_end', 'rTorso_Clavicle',
                          'lTorso_Clavicle', 'rShoulder', 'lShoulder', 'rElbow', 'lElbow', 'rWrist',
                          'rWrist_end', 'lWrist', 'lWrist_end'] # class MotionData _skeleton_joints attribute
      self.torque_joints = ['pelvis_lowerback', 'lowerback_torso', 'rHip', 'lHip',
                            'rKnee', 'lKnee', 'rAnkle', 'lAnkle', 'rToeJoint',
                            'lToeJoint', 'torso_head', 'rTorso_Clavicle', 'lTorso_Clavicle',
                            'rShoulder', 'lShoulder', 'rElbow', 'lElbow', 'rWrist', 'lWrist'] # agent.env.sim_character.joints

    @staticmethod
    def create_env_and_agent():
        def build_args(parser:argparse.ArgumentParser, in_args=None):
            # add args for each content 
            parser = VCLODETrackEnv.add_specific_args(parser)
            parser = MoConVQ.add_specific_args(parser)
            args = vars(parser.parse_args(args=in_args))
            # yaml
            config = load_yaml(args['config_file'])
            config = flatten_dict(config)
            args.update(config)
            
            if args['load']:
                import tkinter.filedialog as fd
                config_file = fd.askopenfilename(filetypes=[('YAML','*.yml')])
                data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
                config = load_yaml(config_file)
                config = flatten_dict(config)
                args.update(config)
                args['load'] = True
                args['data_file'] = data_file
                
            #! important!
            seed = args['seed']
            args['seed'] = seed
            VCLODETrackEnv.seed(seed)
            MoConVQ.set_seed(seed)
            return args
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', default='../MoConVQ/Data/Parameters/bigdata.yml', help= 'a yaml file contains the training information')
        parser.add_argument('--seed', type = int, default=0, help='seed for root process')
        parser.add_argument('--experiment_name', type = str, default="debug", help="")
        parser.add_argument('--load', default=False, action='store_true')
        parser.add_argument('--gpu', type = int, default=0, help='gpu id')
        parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
        parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
        parser.add_argument('--using_vanilla', default=False, action='store_true')
        parser.add_argument('--no_train', default=False, action='store_true')
        parser.add_argument('--train_prior', default=False, action='store_true')
        model_args = build_args(parser, in_args=[])    
        parser = argparse.ArgumentParser()
        # parser.add_argument('bvh-file', type=str, nargs='+')
        parser.add_argument('-o', '--output-file', type=str, default='')
        parser.add_argument('--is-bvh-folder', default=False, action='store_true')
        parser.add_argument('--flip-bvh', default=False, action='store_true')
        parser.add_argument('--gpu', type = int, default=0, help='gpu id')
        parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
        parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
        args = vars(parser.parse_args())
        model_args.update(args)
        ptu.init_gpu(True, gpu_id=args['gpu'])
        if args['cpu_e'] !=-1:
            p = psutil.Process()
            cpu_lst = p.cpu_affinity()
            try:
                p.cpu_affinity(range(args['cpu_b'], args['cpu_e']))   
            except:
                pass 

        #build each content
        env = VCLODETrackEnv(**model_args)
        agent = MoConVQ(323, 12, 57, 120, env, training=False, **model_args)
        agent.simple_load(r'../MoConVQ/moconvq_base.data', strict=True)
        agent.eval()
        agent.posterior.limit = False

        from ModifyODESrc import VclSimuBackend
        CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
        saver = CharacterToBVH(agent.env.sim_character, 120) # BVH is 120 fps (Sim is 20fps)
        saver.bvh_hierarchy_no_root()
        return env, agent, saver, args
    
    def make_prediction(self, env, agent, env_observation, info, motion_observation, saver):
        period = 1000000
        info = agent.encode_seq_all(motion_observation, motion_observation,)

        '''
        info:
        'latent_seq': output of the encoder,
        'latent_vq': vector quantization of the latent_seq,
        'latent_dynamic': upsampling of the latent_vq, which is the control signal for the policy,
        'indexs': indexs of the latent_vq in the codebook,
        '''
        observation = env_observation
        # decode the latent_vq with simulator, and save the motion
        seq_latent = info['latent_dynamic']
        physics = {
          'avel':[],
          'vel': [],
          'g_torque':[]
        }
        for i in range(seq_latent.shape[1]):
            obs = observation['observation']
            action, info = agent.act_tracking(
                obs_history = [obs.reshape(1,323)],
                target_latent = seq_latent[:,i%period],
            )
            action = ptu.to_numpy(action).flatten()
            for i in range(6):
                saver.append_no_root_to_buffer()
                if i == 0:
                    step_generator = agent.env.step_core(action, using_yield = True)
                info = next(step_generator)
                physics['avel'].append(env.sim_character.body_info.get_body_ang_velo())
                physics['vel'].append(env.sim_character.body_info.get_body_velo())
            try:
                info_ = next(step_generator)
            except StopIteration as e:
                info_ = e.value
            new_observation, rwd, done, info, global_torques = info_

            # global_torques is (6, num_joints without root, 3)
            for torque in global_torques: 
                g_torque = torque.copy()
                # env.sim_character.joint_info.torque_limit torque limits
                # apply negative joints to parents to get net torque
                for i in range(len(g_torque)): 
                    parent_id = int(env.sim_character.joint_info.parent_body_index[i] - 1) # parent_body_index includes root as 0 index, therefore must shift 1 to the left
                    if parent_id >= 0:
                        g_torque[parent_id] = g_torque[parent_id] - torque[i]

                physics['g_torque'].append(g_torque)
            observation = new_observation

        # if args['output_file'] == '':
        #     import time
        #     motion_name = os.path.join('out', f'track_{time.time()}.bvh')
        # else:
        #     motion_name = args['output_file']
        # saver.to_file(motion_name)

        # returns joint rotations (frames, joints, ), joint translations (frames, 3)
        motion = saver.ret_merge_buf() # savers saves to buffer. do this to get the merged buffer without destroying buffer
        return motion.joint_rotation, motion.joint_translation[:, 0, :], observation, physics

class MoConVQSMPLTakeNode(MoConVQNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoConVQSMPLTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        speed = 1
        self.smpl_data = None
        self.joint_data = None
        self.physics_data = None
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
        self.avel_out = self.add_output('avel')
        self.vel_out = self.add_output('vel')
        self.g_torque_out = self.add_output('g_torque')
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
            self.send_frame(frame)

    def load_smpl_callback(self):
        in_path = self.load_path()
        self.load_smpl(in_path)

    def load_smpl(self, in_path):
        self.on_off.set(False)
        self.start_stop_streaming()
        if os.path.isfile(in_path):            
            self.joint_data, self.root_positions, _, self.physics_data = self.load_and_predict_smpl(in_path)
            if self.joint_data is not None:
                self.file_name.set(in_path.split('/')[-1])
                self.load_path.set(in_path)
                self.frames = self.joint_data.shape[0]
                self.current_frame = 0
                self.input.set(0)


    def frame_widget_changed(self):
        data = self.input()
        if self.joint_data is not None and int(data) < self.frames:
            self.current_frame = int(data)
            frame = int(self.current_frame)
            self.send_frame(frame)
            
    def send_frame(self, frame):
        self.joint_data_out.send(self.joint_data[frame])
        self.root_position_out.send(self.root_positions[frame])
        self.avel_out.send(self.physics_data['avel'][frame])
        self.vel_out.send(self.physics_data['vel'][frame])
        self.g_torque_out.send(self.physics_data['g_torque'][frame])

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
                    self.send_frame(frame)

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

    def load_and_predict_smpl(self, in_path):
        env, agent, saver, args = MoConVQNode.create_env_and_agent()
        motion_data = MotionDataSet(20)
        motion_data.add_bvh_with_character(in_path,  env.sim_character, flip=args['flip_bvh'], smpl_path="./data/smplh/neutral/model.npz")
        observation, info = agent.env.reset(0)
        return self.make_prediction(env, agent, observation, info, motion_data.observation, saver)

class MoConVQPoseToJointsNode(MoConVQNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoConVQPoseToJointsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('pose in', triggers_execution=True)
        self.output_as = self.add_property('output_as', widget_type='combo', default_value='quaternions')
        self.output_as.widget.combo_items = ['quaternions', 'euler angles', 'roll_pitch_yaw']
        self.use_degrees = self.add_property('degrees', widget_type='checkbox', default_value=False)
        self.joint_outputs = []
        self.joint_offsets = []
        for index, key in enumerate(self.joint_names):
            if index < 22:
                self.joint_offsets.append(index)

        for index, key in enumerate(self.joint_names):
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
                        joint_value = incoming[index] # moconvq sents quarts with scalar last
                        if output_quaternions:
                            joint_value = np.roll(joint_value, 1)
                        elif output_rpy:
                            rot = scipy.spatial.transform.Rotation.from_quat(joint_value)
                            joint_value = rot.as_euler('XYZ', degrees=self.use_degrees())
                        self.joint_outputs[i].set_value(joint_value)
                self.send_all()


class MoConVQTorqueToJointsNode(MoConVQNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoConVQTorqueToJointsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('torques in', triggers_execution=True)
        self.joint_outputs = []
        self.joint_offsets = []
        for index, key in enumerate(self.torque_joints):
            self.joint_offsets.append(index)

        for index, key in enumerate(self.torque_joints):
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


class MoConVQEnvNode(MoConVQNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoConVQEnvNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.env, self.agent, self.saver, self.mvq_args = self.create_env_and_agent()
        self.motion_data = DPGMotionDataset(20, self.agent.env.sim_character)
        self.observation, self.info = self.reset_env()
        self.motion_ob_index = 0
        self.rot_in = self.add_input('joint_data', triggers_execution=True)
        self.pos_in = self.add_input('root_position', triggers_execution=True)
        self.reset = self.add_input('reset_env', widget_type='button', callback=self.reset_env)
        self.pos_buf = []
        self.rot_buf = []
        self.joint_data_out = self.add_output('joint_data')
        self.root_position_out = self.add_output('root_position')
        self.avel_out = self.add_output('avel')
        self.vel_out = self.add_output('vel')
        self.g_torque_out = self.add_output('g_torque')

    def execute(self):
        if self.pos_in.fresh_input or self.rot_in.fresh_input:
            if self.pos_in.fresh_input:
                pos_in = self.pos_in() 
                pos_base = np.zeros((len(self.joint_names), 3))
                pos_base[0] = pos_in
                self.pos_buf.append(pos_base)
            else:
                rot_in = self.rot_in()      
                self.rot_buf.append(rot_in)
            if len(self.pos_buf) == len(self.rot_buf) and len(self.pos_buf) >= 24: # ensure we have (pos, rot) with same number of frames and at least 24 frames
                self.motion_data.add_motion_with_character(np.stack(self.pos_buf), np.stack(self.rot_buf))
                quats, pos, obs, physics = self.make_prediction(self.env, self.agent, self.observation, self.info, self.motion_data.observation[self.motion_ob_index:], self.saver)
                self.observation = obs
                self.motion_ob_index = len(self.motion_data.observation)
                for i in range(len(pos)):
                    self.joint_data_out.send(quats[i])
                    self.root_position_out.send(pos[i])
                    self.avel_out.send(physics['avel'][i])
                    self.vel_out.send(physics['vel'][i])
                    self.g_torque_out.send(physics['g_torque'][i])
                                    
                self.pos_buf = []
                self.rot_buf = []
  
    def reset_env(self):
        obs, info = self.agent.env.reset(0)
        self.observation = obs
        self.info = info
        self.pos_buf = []
        self.rot_buf = []
        return obs, info

class MoConVQStorageNode(MoConVQNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MoConVQStorageNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        speed = 1
        self.frames = 0
        self.streaming = False
        self.current_frame = 0
        self.joint_data = []
        self.root_positions = []
        self.physics = {
            'avel':[],
            'vel':[],
            'g_torque':[]
        }
        self.rot_in = self.add_input('joint_data', triggers_execution=True)
        self.pos_in = self.add_input('root_position', triggers_execution=True)
        self.avel_in = self.add_input('avel', triggers_execution=True)
        self.vel_in = self.add_input('vel', triggers_execution=True)
        self.g_torque_in = self.add_input('g_torque', triggers_execution=True)
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.clear = self.add_input('clear', widget_type='button', callback=self.clear_data)
        self.joint_data_out = self.add_output('joint_data')
        self.root_position_out = self.add_output('root_position')
        self.avel_out = self.add_output('avel')
        self.vel_out = self.add_output('vel')
        self.g_torque_out = self.add_output('g_torque')

    def start_stop_streaming(self):
        if self.on_off():
            if not self.streaming:
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
            self.avel_out.send(self.physics['avel'][frame])
            self.vel_out.send(self.physics['vel'][frame])
            self.g_torque_out.send(self.physics['g_torque'][frame])
            

    def frame_widget_changed(self):
        data = self.input()
        if self.joint_data is not None and int(data) < self.frames:
            self.current_frame = int(data)
            frame = int(self.current_frame)
            self.joint_data_out.send(self.joint_data[frame])
            self.root_position_out.send(self.root_positions[frame])
            self.avel_out.send(self.physics['avel'][frame])
            self.vel_out.send(self.physics['vel'][frame])
            self.g_torque_out.send(self.physics['g_torque'][frame])

    def execute(self):
        if self.pos_in.fresh_input:
            pos_in = self.pos_in() 
            self.root_positions.append(pos_in)
        if self.rot_in.fresh_input:
            rot_in = self.rot_in()      
            self.joint_data.append(rot_in)
        if self.avel_in.fresh_input:
            avel_in = self.avel_in() 
            self.physics['avel'].append(avel_in)
        if self.vel_in.fresh_input:
            vel_in = self.vel_in() 
            self.physics['vel'].append(vel_in)
        if self.g_torque_in.fresh_input:
            g_torque_in = self.g_torque_in() 
            self.physics['g_torque'].append(g_torque_in)

        self.frames = min(len(self.root_positions), len(self.joint_data), len(self.physics['avel']), len(self.physics['vel']), len(self.physics['g_torque']))
    
    def clear_data(self):
        self.joint_data = []
        self.root_positions = []
        self.physics = {
            'avel':[],
            'vel':[],
            'g_torque':[]
        }
        self.on_off.set(False)
        self.input.set(0)
        self.start_stop_streaming()
        self.frames = 0
        self.current_frame = 0


class MoConVQTranslator():
    moconvq_joints = {
        'RootJoint': 0,
        'pelvis_lowerback': 1,
        'lowerback_torso': 2,
        'rHip': 3,
        'lHip': 4,
        'rKnee': 5,
        'lKnee': 6,
        'rAnkle': 7,
        'lAnkle': 8,
        'rToeJoint': 9,
        'rToeJoint_end': 10,
        'lToeJoint': 11,
        'lToeJoint_end': 12,
        'torso_head': 13,
        'torso_head_end': 14,
        'rTorso_Clavicle': 15,
        'lTorso_Clavicle': 16,
        'rShoulder': 17,
        'lShoulder': 18,
        'rElbow': 19,
        'lElbow': 20,
        'rWrist': 21,
        'rWrist_end': 22,
        'lWrist': 23,
        'lWrist_end': 24
    }

    moconvq_to_smpl_joint_map = {
        'RootJoint': 'pelvis',
        'pelvis_lowerback': 'spine1',
        'lowerback_torso': 'spine3',
        'rHip': 'right_hip',
        'lHip': 'left_hip',
        'rKnee': 'right_knee',
        'lKnee': 'left_knee',
        'rAnkle': 'right_ankle',
        'lAnkle': 'left_ankle',
        'rToeJoint': 'right_foot',
        # 'rToeJoint_end': 10,
        'lToeJoint': 'left_foot',
        # 'lToeJoint_end': 12,
        'torso_head': 'neck',
        'torso_head_end': 'head',
        'rTorso_Clavicle': 'right_collar',
        'lTorso_Clavicle': 'left_collar',
        'rShoulder': 'right_shoulder',
        'lShoulder': 'left_shoulder',
        'rElbow': 'right_elbow',
        'lElbow': 'left_elbow',
        'rWrist': 'right_wrist',
        # 'rWrist_end': 22,
        'lWrist': 'left_wrist',
        # 'lWrist_end': 24
    }

    moconvq_to_active_joint_map = {
        'RootJoint': 'pelvis_anchor',
        'pelvis_lowerback': 'spine_pelvis',
        'lowerback_torso': 'lower_vertebrae',
        'rHip': 'right_hip',
        'lHip': 'left_hip',
        'rKnee': 'right_knee',
        'lKnee': 'left_knee',
        'rAnkle': 'right_ankle',
        'lAnkle': 'left_ankle',
        # 'rToeJoint': 9,
        # 'rToeJoint_end': 10,
        # 'lToeJoint': 11,
        # 'lToeJoint_end': 12,
        'torso_head': 'upper_vertebrae',
        'torso_head_end': 'base_of_skull',
        'rTorso_Clavicle': 'right_shoulder_blade',
        'lTorso_Clavicle': 'left_shoulder_blade',
        'rShoulder': 'right_shoulder',
        'lShoulder': 'left_shoulder',
        'rElbow': 'right_wrist',
        'lElbow': 'left_elbow',
        'rWrist': 'right_wrist',
        # 'rWrist_end': 22,
        'lWrist': 'left_wrist',
        # 'lWrist_end': 24
    }

    def __init__(self, label, data, args):
        pass


class PoseToPoseTranslator(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PoseToPoseTranslator(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)
        self.input = self.add_input('input pose', triggers_execution=True)
        self.output = self.add_output('output pose')

        self.input_type = self.add_property('input_type', widget_type='combo', default_value='smpl', callback=self.input_type_changed)
        self.input_as = self.add_property('input_as', widget_type='combo', default_value='rotvec')
        self.input_scalar_first = self.add_input('input scalar first', widget_type='checkbox', default_value=False)
        self.input_degrees = self.add_input('input degrees', widget_type='checkbox')

        self.output_type = self.add_property('output_type', widget_type='combo', default_value='glbody', callback=self.output_type_changed)
        self.output_as = self.add_property('output_as', widget_type='combo', default_value='quaternions')
        self.output_degrees = self.add_input('output degrees', widget_type='checkbox')
        self.output_scalar_first = self.add_input('output scalar first', widget_type='checkbox', default_value=True)
        
        self.input_type.widget.combo_items = ['smpl', 'moconvq', 'glbody']
        self.input_as.widget.combo_items = ['quaternions', 'XYZ', 'ZYX', 'rotvec']
        self.output_type.widget.combo_items = ['smpl', 'moconvq', 'glbody']
        self.output_as.widget.combo_items = ['quaternions', 'XYZ', 'ZYX', 'rotvec'] 

        self.joint_order_map = {
            "smpl": SMPLShadowTranslator.smpl_joints,
            'moconvq': MoConVQTranslator.moconvq_joints,
            'glbody': SMPLShadowTranslator.active_joints
        }
        self.io_map = {
            "smpl_glbody": SMPLShadowTranslator.smpl_to_active_joint_map,
            "smpl_moconvq": MoConVQTranslator.moconvq_to_smpl_joint_map,
            "moconvq_glbody": MoConVQTranslator.moconvq_to_active_joint_map
        }

        self.type_params = {
            "moconvq": {
                "scalar_first": False,
                "type": "quaternions",
                "degrees": False
            },
            "smpl": {
                "scalar_first": False,
                "type": "rotvec",
                "degrees": False
            },
            "glbody": {
                "scalar_first": True,
                "type": "quaternions",
                "degrees": False
            }
        }

    def input_type_changed(self):
        input_type = self.input_type()
        self.input_scalar_first.set(self.type_params[input_type]['scalar_first'])
        self.input_degrees.set(self.type_params[input_type]['degrees'])
        self.input_as.set(self.type_params[input_type]['type'])

    def output_type_changed(self):
        output_type = self.output_type()
        self.output_scalar_first.set(self.type_params[output_type]['scalar_first'])
        self.output_degrees.set(self.type_params[output_type]['degrees'])
        self.output_as.set(self.type_params[output_type]['type'])

    def reverse_dict(self, d):
        return {v: k for k, v in d.items()}
    
    def convert(self, joint_data, input_as, output_as):
        if input_as == "quaternions":  
            rot = scipy.spatial.transform.Rotation.from_quat(joint_data, scalar_first=self.input_scalar_first())
        elif input_as == "rotvec":
            rot = scipy.spatial.transform.Rotation.from_rotvec(joint_data, degrees=self.input_degrees())
        else:
            rot = scipy.spatial.transform.Rotation.from_euler(input_as, joint_data, degrees=self.input_degrees())

        if output_as == "quaternions":
            out = rot.as_quat(scalar_first=self.output_scalar_first())
        elif output_as == "rotvec":
            out = rot.as_rotvec(joint_data, degrees=self.output_degrees())
        else:
            out = rot.as_euler(output_as, degrees=self.output_degrees())
        return out

    def execute_equal_type(self, data, input_as, output_as):
        self.output.send([self.convert(j, input_as, output_as) for j in data])

    def execute_diff_type(self, data, input_as, output_as, io_map, in_order_map, out_order_map):
        output_shape = (len(out_order_map), 4) if output_as == "quaternions" else (len(out_order_map), 3)
        active_pose = np.zeros(output_shape)
        if output_as == "quaternions":
            if self.output_scalar_first():
                active_pose[:, 0] = 1.0
            else:
                active_pose[:, 3] = 1.0

        for k, v in out_order_map.items():
            if k not in io_map: # skip (sets as 0)
                continue
            input_key = io_map[k] # look for joint name corresponding to output joint name
            input_index = in_order_map[input_key] # get index of the joint in the input
            active_pose[v] = self.convert(data[input_index], input_as, output_as)
        self.output.send(active_pose)

    def execute(self):
        try:
            pose_in = self.input()
            input_type = self.input_type()
            output_type = self.output_type()
            input_as = self.input_as()
            output_as = self.output_as()

            if input_type == output_type: # simply converting
                self.execute_equal_type(pose_in, input_as, output_as)
            else:
                if f"{output_type}_{input_type}" in self.io_map:
                    io_map = self.io_map[f"{output_type}_{input_type}"] 
                else: # reverse direction
                    io_map = self.reverse_dict(self.io_map[f"{input_type}_{output_type}"])
                in_order_map = self.joint_order_map[input_type]
                out_order_map = self.joint_order_map[output_type]
                self.execute_diff_type(pose_in, input_as, output_as, io_map, in_order_map, out_order_map)
        except:
            print("Ran into errors, make sure configs are correct.")
            
