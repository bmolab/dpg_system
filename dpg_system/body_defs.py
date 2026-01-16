print('in')

import numpy as np



# new active joint mode constants

t_NoJoint = -1
t_BaseOfSkull = 0
t_UpperVertebrae = 1
t_MidVertebrae = 2
t_LowerVertebrae = 3
t_SpinePelvis = 4
t_PelvisAnchor = 5
t_LeftHip = 6
t_LeftKnee = 7
t_LeftAnkle = 8
t_RightHip = 9
t_RightKnee = 10
t_RightAnkle = 11
t_LeftShoulderBladeBase = 12
t_LeftShoulder = 13
t_LeftElbow = 14
t_LeftWrist = 15
t_RightShoulderBladeBase = 16
t_RightShoulder = 17
t_RightElbow = 18
t_RightWrist = 19

t_ActiveJointCount = 20
# non - joints

t_TopOfHead = 20
t_LeftBallOfFoot = 21
t_LeftToeTip = 22
t_RightBallOfFoot = 23
t_RightToeTip = 24
t_LeftKnuckle = 25
t_LeftFingerTip = 26
t_RightKnuckle = 27
t_RightFingerTip = 28

t_LeftHeel = 29
t_RightHeel = 30

t_Tracker0 = 31
t_Tracker1 = 32
t_Tracker2 = 33
t_Tracker3 = 34

t_Body = 35
t_Reference = 36












# smpl_joint_names = [
#     'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
#     'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
#     'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
#     'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
#     'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2',
#     'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1',
#     'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3',
#     'right_index1', 'right_index2', 'right_index3', 'right_middle1',
#     'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2',
#     'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
#     'right_thumb1', 'right_thumb2', 'right_thumb3'
# ]

#
# smpl_joints = {
#         'pelvis': 0,
#         'left_hip': 1,
#         'right_hip': 2,
#         'spine1': 3,
#         'left_knee': 4,
#         'right_knee': 5,
#         'spine2': 6,
#         'left_ankle': 7,
#         'right_ankle': 8,
#         'spine3': 9,
#         'left_foot': 10,
#         'right_foot': 11,
#         'neck': 12,
#         'left_collar': 13,
#         'right_collar': 14,
#         'head': 15,
#         'left_shoulder': 16,
#         'right_shoulder': 17,
#         'left_elbow': 18,
#         'right_elbow': 19,
#         'left_wrist': 20,
#         'right_wrist': 21
#     }

class JointTranslator():
    smpl_joints = {
        'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3,
        'left_knee': 4, 'right_knee': 5, 'spine2': 6, 'left_ankle': 7,
        'right_ankle': 8, 'spine3': 9, 'left_foot': 10, 'right_foot': 11,
        'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15,
        'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
        'left_wrist': 20, 'right_wrist': 21, 'left_hand': 22, 'right_hand': 23,
        'left_toe_tip': 24, 'right_toe_tip': 25, 'left_finger_tip': 26, 'right_finger_tip': 27
    }

    bmolab_active_joints = {
        'base_of_skull': 0, 'upper_vertebrae': 1, 'mid_vertebrae': 2, 'lower_vertebrae': 3,
        'spine_pelvis': 4, 'pelvis_anchor': 5, 'left_hip': 6, 'left_knee': 7,
        'left_ankle': 8, 'right_hip': 9, 'right_knee': 10, 'right_ankle': 11,
        'left_shoulder_blade': 12, 'left_shoulder': 13, 'left_elbow': 14, 'left_wrist': 15,
        'right_shoulder_blade': 16, 'right_shoulder': 17, 'right_elbow': 18, 'right_wrist': 19,
        'left_foot': 20, 'right_foot': 21, 'left_hand': 22, 'right_hand': 23,
        'left_toe_tip': 24, 'right_toe_tip': 25, 'left_finger_tip': 26, 'right_finger_tip': 27
    }

    joints_to_input_vector = [-1, 2, 0, -1, 5, 13, -1, -1, 8, 14, 15, -1, 7, 12, 6, -1, -1, 1, -1, 17, -1, -1, 11, 18,
                              19, -1, 10, 16, 9, -1, -1, 4, 3, -1, -1, -1, -1]

    smpl_to_bmolab_active_joint_map = {
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
        'right_wrist': 'right_wrist',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot',
        'left_hand': 'left_hand',
        'right_hand': 'right_hand',
        'left_toe_tip': 'left_toe_tip',
        'right_toe_tip': 'right_toe_tip',
        'left_finger_tip': 'left_finger_tip',
        'right_finger_tip': 'right_finger_tip'
    }

    smpl_to_smpl_active_joint_map = {
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
        'right_wrist': 'right_wrist',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot'
    }

    smpl_from_bmolab_active_joint_map = {
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

    shadow_limb_to_bmolab_joint_dict = {
        'Body': 'Body',
        'Chest': 'MidVertebrae',
        'Head': 'BaseOfSkull',
        'HeadEnd': 'TopOfHead',
        'Hips': 'PelvisAnchor',
        'LeftArm': 'LeftShoulder',
        'LeftFinger': 'LeftKnuckle',
        'LeftFingerEnd': 'LeftFingerTip',
        'LeftFoot': 'LeftAnkle',
        'LeftForearm': 'LeftElbow',
        'LeftHand': 'LeftWrist',
        'LeftHeel': 'LeftHeel',
        'LeftLeg': 'LeftKnee',
        'LeftShoulder': 'LeftShoulderBladeBase',
        'LeftThigh': 'LeftHip',
        'LeftToe': 'LeftBallOfFoot',
        'LeftToeEnd': 'LeftToeTip',
        'Neck': 'UpperVertebrae',
        'Reference': 'Reference',
        'RightArm': 'RightShoulder',
        'RightFinger': 'RightKnuckle',
        'RightFingerEnd': 'RightFingerTip',
        'RightFoot': 'RightAnkle',
        'RightForearm': 'RightElbow',
        'RightHand': 'RightWrist',
        'RightHeel': 'RightHeel',
        'RightLeg': 'RightKnee',
        'RightShoulder': 'RightShoulderBladeBase',
        'RightThigh': 'RightHip',
        'RightToe': 'RightBallOfFoot',
        'RightToeEnd': 'RightToeTip',
        'SpineLow': 'SpinePelvis',
        'SpineMid': 'LowerVertebrae',
        'Tracker0': 'Tracker0',
        'Tracker1': 'Tracker1',
        'Tracker2': 'Tracker2',
        'Tracker3': 'Tracker3'
    }

    joint_name_to_bmolab_index = {
        'BaseOfSkull': 0,
        'UpperVertebrae': 1,
        'MidVertebrae': 2,
        'LowerVertebrae': 3,
        'SpinePelvis': 4,
        'PelvisAnchor': 5,
        'LeftHip': 6,
        'LeftKnee': 7,
        'LeftAnkle': 8,
        'RightHip': 9,
        'RightKnee': 10,
        'RightAnkle': 11,
        'LeftShoulderBladeBase': 12,
        'LeftShoulder': 13,
        'LeftElbow': 14,
        'LeftWrist': 15,
        'RightShoulderBladeBase': 16,
        'RightShoulder': 17,
        'RightElbow': 18,
        'RightWrist': 19,

        # non - joints
        'TopOfHead': 20,
        'LeftBallOfFoot': 21,
        'LeftToeTip': 22,
        'RightBallOfFoot': 23,
        'RightToeTip': 24,
        'LeftKnuckle': 25,
        'LeftFingerTip': 26,
        'RightKnuckle': 27,
        'RightFingerTip': 28,
        'LeftHeel': 29,
        'RightHeel': 30,

        'Tracker0': 31,
        'Tracker1': 32,
        'Tracker2': 33,
        'Tracker3': 34,
        'NoJoint': -1
    }

    bmolab_joint_index_to_name = {
        0: 'BaseOfSkull',
        1: 'UpperVertebrae',
        2: 'MidVertebrae',
        3: 'LowerVertebrae',
        4: 'SpinePelvis',
        5: 'PelvisAnchor',
        6: 'LeftHip',
        7: 'LeftKnee',
        8: 'LeftAnkle',
        9: 'RightHip',
        10: 'RightKnee',
        11: 'RightAnkle',
        12: 'LeftShoulderBladeBase',
        13: 'LeftShoulder',
        14: 'LeftElbow',
        15: 'LeftWrist',
        16: 'RightShoulderBladeBase',
        17: 'RightShoulder',
        18: 'RightElbow',
        19: 'RightWrist',

        # not active joints
        20: 'TopOfHead',
        21: 'LeftBallOfFoot',
        22: 'LeftToeTip',
        23: 'RightBallOfFoot',
        24: 'RightToeTip',
        25: 'LeftKnuckle',
        26: 'LeftFingerTip',
        27: 'RightKnuckle',
        28: 'RightFingerTip',
        29: 'LeftHeel',
        30: 'RightHeel',

        31: 'Tracker0',
        32: 'Tracker1',
        33: 'Tracker2',
        34: 'Tracker3',

        -1: 'NoJoint'
    }

    shadow_joint_index_to_name = {
        0: 'Body',
        1: 'MidVertebrae',
        2: 'BaseOfSkull',
        3: 'TopOfHead',
        4: 'PelvisAnchor',
        5: 'LeftShoulder',
        6: 'LeftKnuckle',
        7: 'LeftFingerTip',
        8: 'LeftAnkle',
        9: 'LeftElbow',
        10: 'LeftWrist',
        11: 'LeftHeel',
        12: 'LeftKnee',
        13: 'LeftShoulderBladeBase',
        14: 'LeftHip',
        15: 'LeftBallOfFoot',
        16: 'LeftToeTip',
        17: 'UpperVertebrae',
        18: 'Reference',
        19: 'RightShoulder',
        20: 'RightKnuckle',
        21: 'RightFingerTip',
        22: 'RightAnkle',
        23: 'RightElbow',
        24: 'RightWrist',
        25: 'RightHeel',
        26: 'RightKnee',
        27: 'RightShoulderBladeBase',
        28: 'RightHip',
        29: 'RightBallOfFoot',
        30: 'RightToeTip',
        31: 'SpinePelvis',
        32: 'LowerVertebrae',
        33: 'Tracker0',
        34: 'Tracker1',
        35: 'Tracker2',
        36: 'Tracker3'
    }

    joint_name_to_shadow_index = {
        'Body': 0,
        'MidVertebrae': 1,
        'BaseOfSkull': 2,
        'TopOfHead': 3,
        'PelvisAnchor': 4,
        'LeftShoulder': 5,
        'LeftKnuckle': 6,
        'LeftFingerTip': 7,
        'LeftAnkle': 8,
        'LeftElbow': 9,
        'LeftWrist': 10,
        'LeftHeel': 11,
        'LeftKnee': 12,
        'LeftShoulderBladeBase': 13,
        'LeftHip': 14,
        'LeftBallOfFoot': 15,
        'LeftToeTip': 16,
        'UpperVertebrae': 17,
        'Reference': 18,
        'RightShoulder': 19,
        'RightKnuckle': 20,
        'RightFingerTip': 21,
        'RightAnkle': 22,
        'RightElbow': 23,
        'RightWrist': 24,
        'RightHeel': 25,
        'RightKnee': 26,
        'RightShoulderBladeBase': 27,
        'RightHip': 28,
        'RightBallOfFoot': 29,
        'RightToeTip': 30,
        'SpinePelvis': 31,
        'LowerVertebrae': 32,
        'Tracker0': 33,
        'Tracker1': 34,
        'Tracker2': 35,
        'Tracker3': 36
    }

    bmolab_joint_to_shadow_limb = {
        'Body': 'Body',
        'MidVertebrae': 'Chest',
        'BaseOfSkull': 'Head',
        'TopOfHead': 'HeadEnd',
        'PelvisAnchor': 'Hips',
        'LeftShoulder': 'LeftArm',
        'LeftKnuckle': 'LeftFinger',
        'LeftFingerTip': 'LeftFingerEnd',
        'LeftAnkle': 'LeftFoot',
        'LeftElbow': 'LeftForearm',
        'LeftWrist': 'LeftHand',
        'LeftHeel': 'LeftHeel',
        'LeftKnee': 'LeftLeg',
        'LeftShoulderBladeBase': 'LeftShoulder',
        'LeftHip': 'LeftThigh',
        'LeftBallOfFoot': 'LeftToe',
        'LeftToeTip': 'LeftToeEnd',
        'UpperVertebrae': 'Neck',
        'Reference': 'Reference',
        'RightShoulder': 'RightArm',
        'RightKnuckle': 'RightFinger',
        'RightFingerTip': 'RightFingerEnd',
        'RightAnkle': 'RightFoot',
        'RightElbow': 'RightForearm',
        'RightWrist': 'RightHand',
        'RightHeel': 'RightHeel',
        'RightKnee': 'RightLeg',
        'RightShoulderBladeBase': 'RightShoulder',
        'RightHip': 'RightThigh',
        'RightBallOfFoot': 'RightToe',
        'RightToeTip': 'RightToeEnd',
        'SpinePelvis': 'SpineLow',
        'LowerVertebrae': 'SpineMid',
        'Tracker0': 'Tracker0',
        'Tracker1': 'Tracker1',
        'Tracker2': 'Tracker2',
        'Tracker3': 'Tracker3'
    }

    @staticmethod
    def shadow_limb_to_bmolab_joint(shadow_name):
        if shadow_name in JointTranslator.shadow_limb_to_bmolab_joint_dict:
            return JointTranslator.shadow_limb_to_bmolab_joint_dict[shadow_name]
        return 'empty'

    @staticmethod
    def bmolab_joint_name_to_index(bmolab_name):
        if bmolab_name in JointTranslator.joint_name_to_bmolab_index:
            return JointTranslator.joint_name_to_bmolab_index[bmolab_name]
        return -1

    @staticmethod
    def shadow_limb_name_to_bmolab_index(shadow_name):
        bmolab_name = JointTranslator.shadow_limb_to_bmolab_joint(shadow_name)
        if bmolab_name != 'empty':
            return JointTranslator.bmolab_joint_name_to_index(bmolab_name)
        return -1

    @staticmethod
    def translate_from_smpl_to_bmolab_active(smpl_pose): #  expects n x 3 in, outputs 20 x 3
        if len(smpl_pose.shape) == 1:
            smpl_pose = np.expand_dims(smpl_pose, axis=1)
        output_size = len(JointTranslator.smpl_to_bmolab_active_joint_map)
        active_pose = np.zeros((output_size, smpl_pose.shape[-1]), dtype=np.float32)

        for smpl_joint in JointTranslator.smpl_to_bmolab_active_joint_map:
            smpl_index = JointTranslator.smpl_joints[smpl_joint]
            active_joint = JointTranslator.smpl_to_bmolab_active_joint_map[smpl_joint]
            active_index = JointTranslator.bmolab_active_joints[active_joint]
            if smpl_index < smpl_pose.shape[0]:
                active_pose[active_index] = smpl_pose[smpl_index]
        return active_pose

    @staticmethod
    def translate_from_smpl_to_smpl_active(smpl_pose): #  expects n x 3 in, outputs 20 x 3
        output_size = len(JointTranslator.smpl_to_smpl_active_joint_map)
        active_pose = np.zeros((output_size, smpl_pose.shape[-1]), dtype=np.float32)

        for smpl_joint in JointTranslator.smpl_to_smpl_active_joint_map:
            smpl_index = JointTranslator.smpl_joints[smpl_joint]
            active_joint = JointTranslator.smpl_to_smpl_active_joint_map[smpl_joint]
            active_index = JointTranslator.bmolab_active_joints[active_joint]
            active_pose[active_index] = smpl_pose[smpl_index]
        return active_pose

    @staticmethod
    def translate_from_bmolab_active_to_smpl(active_pose): #  expects 20 x 3 in, outputs 20 x 3
        output_size = len(JointTranslator.smpl_from_bmolab_active_joint_map)
        smpl_pose = np.zeros((output_size, active_pose.shape[-1]), dtype=np.float32)

        empty = [1.0, 0.0, 0.0, 0.0]
        if active_pose.shape[1] == 3:
            empty = [0.0, 0.0, 0.0]
        elif active_pose.shape[1] == 4:
            empty = [1.0, 0.0, 0.0, 0.0]
        elif active_pose.shape[1] == 2:
            empty = [0.0, 0.0]
        elif active_pose.shape[1] == 2:
            empty = [0.0]

        for smpl_joint in JointTranslator.smpl_from_bmolab_active_joint_map:
            smpl_index = JointTranslator.smpl_joints[smpl_joint]
            active_joint = JointTranslator.smpl_from_bmolab_active_joint_map[smpl_joint]
            if active_joint in JointTranslator.bmolab_active_joints:
                active_index = JointTranslator.bmolab_active_joints[active_joint]
                smpl_pose[smpl_index] = active_pose[active_index]
            else:
                smpl_pose[smpl_index] = empty
        return smpl_pose

    def __init__(self, label, data, args):
        pass

print('body_defs done')