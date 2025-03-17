# t_NoJoint = -1
# t_Body = 0
# t_MidVertebrae = 1
# t_BaseOfSkull = 2
# t_TopOfHead = 3
# t_PelvisAnchor = 4              #  can invert quat *not used*
# t_LeftShoulder = 5
# t_LeftKnuckle = 6
# t_LeftFingerTip = 7
# t_LeftAnkle = 8
# t_LeftElbow = 9                 # close to inversion 0.95
# t_LeftWrist = 10                #  can invert quat 1.99
# t_LeftHeel = 11
# t_LeftKnee = 12
# t_LeftShoulderBladeBase = 13
# t_LeftHip = 14
# t_LeftBallOfFoot = 15
# t_LeftToeTip = 16
# t_UpperVertebrae = 17
# t_Reference = 18
# t_RightShoulder = 19
# t_RightKnuckle = 20
# t_RightFingerTip = 21
# t_RightAnkle = 22
# t_RightElbow = 23               #  can invert quat 1.07
# t_RightWrist = 24               #  can invert quat  1.99
# t_RightHeel = 25
# t_RightKnee = 26                #  .78
# t_RightShoulderBladeBase = 27
# t_RightHip = 28
# t_RightBallOfFoot = 29
# t_RightToeTip = 30
# t_SpinePelvis = 31
# t_LowerVertebrae = 32
# t_Tracker0 = 33
# t_Tracker1 = 34
# t_Tracker2 = 35
# t_Tracker3 = 36

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

#
# t_inputVectorNone = -1
# t_inputVectorBaseOfSkull = 0
# t_inputVectorUpperVertebrae = 1
# t_inputVectorMidVertebrae = 2
# t_inputVectorLowerVertebrae = 3
# t_inputVectorSpinePelvis = 4
# t_inputVectorPelvisAnchor = 5
# t_inputVectorLeftHip = 6
# t_inputVectorLeftKnee = 7
# t_inputVectorLeftAnkle = 8
# t_inputVectorRightHip = 9
# t_inputVectorRightKnee = 10
# t_inputVectorRightAnkle = 11
# t_inputVectorLeftShoulderBlade = 12
# t_inputVectorLeftShoulder = 13
# t_inputVectorLeftElbow = 14
# t_inputVectorLeftWrist = 15
# t_inputVectorRightShoulderBlade = 16
# t_inputVectorRightShoulder = 17
# t_inputVectorRightElbow = 18
# t_inputVectorRightWrist = 19

# actual_joints = {
#     'BaseOfSkull': (t_inputVectorBaseOfSkull, t_BaseOfSkull),
#     'UpperVertebrae': (t_inputVectorUpperVertebrae, t_UpperVertebrae),
#     'MidVertebrae': (t_inputVectorMidVertebrae, t_MidVertebrae),
#     'LowerVertebrae': (t_inputVectorLowerVertebrae, t_LowerVertebrae),
#     'SpinePelvis': (t_inputVectorSpinePelvis, t_SpinePelvis),
#     'PelvisAnchor': (t_inputVectorPelvisAnchor, t_PelvisAnchor),
#     'LeftHip': (t_inputVectorLeftHip, t_LeftHip),
#     'LeftKnee': (t_inputVectorLeftKnee, t_LeftKnee),
#     'LeftAnkle': (t_inputVectorLeftAnkle, t_LeftAnkle),
#     'RightHip': (t_inputVectorRightHip, t_RightHip),
#     'RightKnee': (t_inputVectorRightKnee, t_RightKnee),
#     'RightAnkle': (t_inputVectorRightAnkle, t_RightAnkle),
#     'LeftShoulderBlade': (t_inputVectorLeftShoulderBlade, t_LeftShoulderBladeBase),
#     'LeftShoulder': (t_inputVectorLeftShoulder, t_LeftShoulder),
#     'LeftElbow': (t_inputVectorLeftElbow, t_LeftElbow),
#     'LeftWrist': (t_inputVectorLeftWrist, t_LeftWrist),
#     'RightShoulderBlade': (t_inputVectorRightShoulderBlade, t_RightShoulderBladeBase),
#     'RightShoulder': (t_inputVectorRightShoulder, t_RightShoulder),
#     'RightElbow': (t_inputVectorRightElbow, t_RightElbow),
#     'RightWrist': (t_inputVectorRightWrist, t_RightWrist)
# }

joints_to_input_vector = [-1, 2, 0, -1, 5, 13, -1, -1, 8, 14, 15, -1, 7, 12, 6, -1, -1, 1, -1, 17, -1, -1, 11, 18, 19, -1, 10, 16, 9, -1, -1, 4, 3, -1, -1, -1, -1]

joint_index_to_name = {
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


joint_linear_index_to_name = {
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



joint_name_to_linear_index = {
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

joint_name_to_index = {
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


joint_to_shadow_limb = {
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

shadow_limb_to_joint = {
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
