# Adapted code from in MoConVQ/Script/track_something.py from the MoConVQ paper https://moconvq.github.io/
 
import os
from dpg_system.body_base import *
from MoConVQ.MoConVQCore.Env.vclode_track_env import VCLODETrackEnv
from MoConVQ.MoConVQCore.Model.MoConVQ import MoConVQ
from MoConVQ.MoConVQCore.Utils.motion_dataset import MotionDataSet

import argparse
from MoConVQ.MoConVQCore.Utils.misc import *
from MoConVQ.MoConVQCore.Utils import pytorch_utils as ptu
from MoConVQ.MoConVQCore.Utils.motion_dataset import MotionDataSet, DPGMotionDataset
import psutil
import scipy

def register_moconvq_nodes():
    Node.app.register_node("moconvq_take", MoConVQSMPLTakeNode.factory)
    Node.app.register_node("moconvq_pose_to_joints", MoConVQPoseToJointsNode.factory)
    Node.app.register_node("moconvq_env", MoConVQEnvNode.factory)
    Node.app.register_node("moconvq_storage", MoConVQEnvNode.factory)

class MoConVQNode(Node):
    def __init__(self, label: str, data, args):
      super().__init__(label, data, args)
      self.joint_names = ['RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint',
          'lToeJoint_end', 'rHip', 'rKnee', 'rAnkle', 'rToeJoint',
          'rToeJoint_end', 'pelvis_lowerback', 'lowerback_torso',
          'torso_head', 'torso_head_end', 'lTorso_Clavicle', 'lShoulder',
          'lElbow', 'lWrist', 'lWrist_end', 'rTorso_Clavicle',
          'rShoulder', 'rElbow', 'rWrist', 'rWrist_end']

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
        parser.add_argument('--config_file', default='./MoConVQ/Data/Parameters/bigdata.yml', help= 'a yaml file contains the training information')
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
        agent = MoConVQ(323, 12, 57, 120,env, training=False, **model_args)
        agent.simple_load(r'./MoConVQ/moconvq_base.data', strict=True)
        agent.eval()
        agent.posterior.limit = False

        from MoConVQ.ModifyODESrc import VclSimuBackend
        CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
        saver = CharacterToBVH(agent.env.sim_character, 120)
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
                avel = env.sim_character.body_info.get_body_ang_velo()
            try:
                info_ = next(step_generator)
            except StopIteration as e:
                info_ = e.value
            new_observation, rwd, done, info = info_
            observation = new_observation
        # if args['output_file'] == '':
        #     import time
        #     motion_name = os.path.join('out', f'track_{time.time()}.bvh')
        # else:
        #     motion_name = args['output_file']
        # saver.to_file(motion_name)

        # returns joint rotations (frames, joints, ), joint translations (frames, 3)
        motion = saver.ret_merge_buf() # savers saves to buffer. do this to get the merged buffer without destroying buffer
        return motion.joint_rotation, motion.joint_translation[:, 0, :], observation

# Inputs
# - load amass file
# - on/off for streaming
# - frame for frame number
# Outputs
# - joint_data (N, 25 num joints, 4) quarternions where scalar is last
# - root_position (N, 3) of root position in x y z coordinates
# Usage
# - for reading amass files
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
            self.joint_data, self.root_positions, _ = self.load_and_predict_smpl(in_path)
            if self.joint_data is not None:
                self.file_name.set(in_path.split('/')[-1])
                self.load_path.set(in_path)
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

    def load_and_predict_smpl(self, in_path):
        env, agent, saver, args = MoConVQNode.create_env_and_agent()
        motion_data = MotionDataSet(20)
        motion_data.add_bvh_with_character(in_path,  env.sim_character, flip=args['flip_bvh'], smpl_path="../MoConVQ/smpl/smplh/neutral/model.npz")
        observation, info = agent.env.reset(0)
        return self.make_prediction(env, agent, observation, info, motion_data.observation, saver)

# Inputs
# - Pose (N, 25 num joints, 4) quarternions where scalar is last
# Outputs
# - splitted "MoConVQ joints" where each joint is (N, 4) quarternions where scalar is last
# Usage
# - for splitting joint data into channels
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


# Inputs
# - joint_data (N, 25 num joints, 4) quarternions where scalar is last
# - root_position (N, 3) of root position in x y z coordinates
# Outputs
# - joint_data (N, 25 num joints, 4) quarternions where scalar is last
# - root_position (N, 3) of root position in x y z coordinates
# Usage
# - For streaming and acting in real time
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
            if len(self.pos_buf) == len(self.rot_buf): # ensure we have (pos, rot) with same number of frames
                self.motion_data.add_motion_with_character(np.stack(self.pos_buf), np.stack(self.rot_buf))
                pos, quats, obs = self.make_prediction(self.env, self.agent, self.observation, self.info, self.motion_data.observation[self.motion_ob_index:], self.saver)
                self.observation = obs
                self.motion_ob_index = len(self.motion_data.observation)
                for i in range(len(pos)):
                    self.joint_data_out.send(quats[i])
                    self.root_position_out.send(pos[i])
                self.pos_buf = []
                self.rot_buf = []
  
    def reset_env(self):
        obs, info = self.agent.env.reset(0)
        self.observation = obs
        self.info = info
        self.pos_buf = []
        self.rot_buf = []

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
        self.rot_in = self.add_input('joint_data', triggers_execution=True)
        self.pos_in = self.add_input('root_position', triggers_execution=True)
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.start_stop_streaming)
        self.speed = self.add_input('speed', widget_type='drag_float', default_value=speed)
        self.input = self.add_input('frame', widget_type='drag_int', triggers_execution=True, callback=self.frame_widget_changed)
        self.clear = self.add_input('clear', widget_type='button', callback=self.clear)
        self.joint_data_out = self.add_output('joint_data')
        self.root_position_out = self.add_output('root_position')
      
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

    def frame_widget_changed(self):
        data = self.input()
        if self.joint_data is not None and int(data) < self.frames:
            self.current_frame = int(data)
            frame = int(self.current_frame)
            self.joint_data_out.send(self.joint_data[frame])
            self.root_position_out.send(self.root_positions[frame])

    def execute(self):
        if self.pos_in.fresh_input or self.rot_in.fresh_input:
            if self.pos_in.fresh_input:
                pos_in = self.pos_in() 
                self.root_positions.append(pos_in)
            else:
                rot_in = self.rot_in()      
                self.joint_data.append(rot_in)
