from pyquaternion import Quaternion
import json
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from dpg_system.body_defs import *
from pylab import *

class Joint:
    def __init__(self, in_body, in_name, in_index):
        self.body = in_body
        self.name = in_name
        self.shadow_name = joint_to_shadow_limb[in_name]
        self.joint_index = in_index
        self.input_vector_index = -1

        self.do_draw = False
        self.children = []
        self.ref_vector = (0.0, 0.0, 1.0)
        self.thickness = (.1, .1)
        self.matrix = None
        self.length = 1.0
        self.mass = [0.0, 0.0, 0.0]
        self.bone_dim = [0.0, 0.0, 1.0]

        self.set_vector_index()
        self.set_thickness()
        self.set_limb_vector()
        self.set_children()
        self.set_draw()

    def set_vector_index(self):
        for idx, actual_joint in enumerate(actual_joints):
            if actual_joint == self.name:
                self.input_vector_index = idx
                break

    def set_thickness(self):
        if self.joint_index == t_MidVertebrae:
            self.thickness = (.25, .12)
        elif self.joint_index == t_BaseOfSkull:
            self.thickness = (.07, .07)
        elif self.joint_index == t_TopOfHead:
            self.thickness = (.1, .12)
        elif self.joint_index in [t_LeftShoulder, t_RightShoulder]:
            self.thickness = (.08, .06)
        elif self.joint_index in [t_LeftKnuckle, t_RightKnuckle]:
            self.thickness = (.08, .04)
        elif self.joint_index in [t_LeftFingerTip, t_RightFingerTip]:
            self.thickness = (.07, .03)
        elif self.joint_index in [t_LeftAnkle, t_RightAnkle]:
            self.thickness = (.06, .08)
        elif self.joint_index in [t_LeftElbow, t_RightElbow]:
            self.thickness = (.08, .05)
        elif self.joint_index in [t_LeftWrist, t_RightWrist]:
            self.thickness = (.06, .04)
        elif self.joint_index in [t_LeftHeel, t_RightHeel]:
            self.thickness = (.06, .06)
        elif self.joint_index in [t_LeftKnee, t_RightKnee]:
            self.thickness = (.08, .1)
        elif self.joint_index in [t_LeftShoulderBladeBase, t_RightShoulderBladeBase]:
            self.thickness = (.02, .02)
        elif self.joint_index in [t_LeftHip, t_RightHip]:
            self.thickness = (.05, .05)
        elif self.joint_index in [t_LeftBallOfFoot, t_RightBallOfFoot]:
            self.thickness = (.07, .03)
        elif self.joint_index in [t_LeftToeTip, t_RightToeTip]:
            self.thickness = (.07, .02)
        elif self.joint_index == t_UpperVertebrae:
            self.thickness = (.28, .15)
        elif self.joint_index == t_SpinePelvis:
            self.thickness = (.21, .11)
        elif self.joint_index == t_LowerVertebrae:
            self.thickness = (.2, .11)

    def set_limb_vector(self):
        if self.joint_index in [t_Body, t_PelvisAnchor, t_Reference, t_Tracker0, t_Tracker1, t_Tracker2, t_Tracker3]:
            self.ref_vector = (1.0, 0.0, 0.0)
        elif self.joint_index in [t_LeftHeel, t_RightHeel]:
            self.ref_vector = (0.0, 1.0, 0.0)
        elif self.joint_index in [t_LeftToeTip, t_RightToeTip]:
            self.ref_vector = (0.0, 0.0001, 1.0)
        elif self.joint_index in [t_LeftFingerTip, t_RightFingerTip]:
            self.ref_vector = (0.0, 0.0, 1.0)

    def set_children(self):
        if self.joint_index == t_MidVertebrae:
            self.children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_UpperVertebrae, t_BaseOfSkull,
                             t_TopOfHead]
        elif self.joint_index == t_BaseOfSkull:
            self.children = [t_TopOfHead]
        elif self.joint_index == t_PelvisAnchor:
            self.children = [t_RightKnee, t_LeftKnee, t_SpinePelvis, t_LowerVertebrae]
        elif self.joint_index == t_LeftShoulder:
            self.children = [t_LeftElbow, t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
        elif self.joint_index == t_LeftKnuckle:
            self.children = [t_LeftFingerTip]
        elif self.joint_index == t_LeftAnkle:
            self.children = [t_LeftBallOfFoot, t_LeftToeTip]
        elif self.joint_index == t_LeftElbow:
            self.children = [t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
        elif self.joint_index == t_LeftWrist:
            self.children = [t_LeftKnuckle, t_LeftFingerTip]
        elif self.joint_index == t_LeftKnee:
            self.children = [t_LeftAnkle, t_LeftBallOfFoot, t_LeftToeTip]
        elif self.joint_index == t_LeftShoulderBladeBase:
            self.children = [t_LeftShoulder, t_LeftElbow, t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
        elif self.joint_index == t_LeftHip:
            self.children = [t_LeftKnee, t_LeftAnkle, t_LeftBallOfFoot, t_LeftToeTip]
        elif self.joint_index == t_LeftBallOfFoot:
            self.children = [t_LeftToeTip]
        elif self.joint_index == t_UpperVertebrae:
            self.children = [t_BaseOfSkull, t_TopOfHead]
        elif self.joint_index == t_RightShoulder:
            self.children = [t_RightElbow, t_RightWrist, t_RightKnuckle, t_RightFingerTip]
        elif self.joint_index == t_RightKnuckle:
            self.children = [t_RightFingerTip]
        elif self.joint_index == t_RightAnkle:
            self.children = [t_RightBallOfFoot, t_RightToeTip]
        elif self.joint_index == t_RightElbow:
            self.children = [t_RightWrist, t_RightKnuckle, t_RightFingerTip]
        elif self.joint_index == t_RightWrist:
            self.children = [t_RightKnuckle, t_RightFingerTip]
        elif self.joint_index == t_RightKnee:
            self.children = [t_RightAnkle, t_RightBallOfFoot, t_RightToeTip]
        elif self.joint_index == t_RightShoulderBladeBase:
            self.children = [t_RightShoulder, t_RightElbow, t_RightWrist, t_RightKnuckle, t_RightFingerTip]
        elif self.joint_index == t_RightHip:
            self.children = [t_RightKnee, t_RightAnkle, t_RightBallOfFoot, t_RightToeTip]
        elif self.joint_index == t_RightBallOfFoot:
            self.children = [t_RightBallOfFoot, t_RightToeTip]
        elif self.joint_index == t_SpinePelvis:
            self.children = [t_LowerVertebrae, t_MidVertebrae, t_LeftShoulderBladeBase, t_RightShoulderBladeBase,
                             t_UpperVertebrae,
                             t_BaseOfSkull, t_TopOfHead]
        elif self.joint_index == t_LowerVertebrae:
            self.children = [t_MidVertebrae, t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_UpperVertebrae,
                             t_BaseOfSkull,
                             t_TopOfHead]

    def set_draw(self):
        if self.joint_index in [t_LeftHeel, t_RightHeel, t_TopOfHead, t_BaseOfSkull, t_UpperVertebrae, t_LeftFingerTip,
                         t_RightFingerTip, t_LeftKnuckle,
                         t_RightKnuckle, t_LeftWrist, t_RightWrist, t_LeftElbow, t_RightElbow, t_LeftShoulder,
                         t_RightShoulder,
                         t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_SpinePelvis, t_LowerVertebrae,
                         t_MidVertebrae, t_LeftHip, t_RightHip,
                         t_LeftKnee, t_LeftAnkle, t_RightKnee, t_RightAnkle, t_LeftBallOfFoot, t_RightBallOfFoot,
                         t_LeftToeTip, t_RightToeTip]:
            self.do_draw = True

    def set_mass(self):
        if self.matrix is not None:
            self.mass = [self.thickness[0], self.length, self.thickness[1]]
            for child_index in self.children:
                child = self.body.joints[child_index]
                self.mass[0] += child.thickness[0]
                self.mass[1] += child.length
                self.mass[2] += child.thickness[1]

    def set_matrix(self):
        base_vector = np.array(self.ref_vector)
        limb_vector = np.array(self.bone_dim)
        print(self.name, base_vector, limb_vector)

        scale = np.linalg.norm(limb_vector)
        limb_vector /= scale
        W = np.cross(limb_vector, base_vector)
        A_ = np.array([limb_vector, W, np.cross(limb_vector, W)])
        A = A_.T
        B_ = np.array([base_vector, W, np.cross(base_vector, W)])
        B = B_.T
        invA = inv(A)
        M = np.dot(B, invA)

        m = []
        m.append(M[0][0])
        m.append(M[0][1])
        m.append(M[0][2])
        m.append(0)
        m.append(M[1][0])
        m.append(M[1][1])
        m.append(M[1][2])
        m.append(0)
        m.append(M[2][0])
        m.append(M[2][1])
        m.append(M[2][2])
        m.append(0)
        m.append(0)
        m.append(0)
        m.append(0)
        m.append(1)

        self.matrix = m
        self.length = scale

        print(self.name, self.matrix)


