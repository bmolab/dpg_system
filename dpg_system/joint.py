from pyquaternion import Quaternion
import json
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from dpg_system.body_defs import *
from pylab import *
from dpg_system.conversion_utils import *


class BaseJoint:
    def __init__(self, in_body, in_name, in_index):
        self.body = in_body
        self.name = in_name
        self.shadow_name = joint_to_shadow_limb[in_name]
        self.joint_index = in_index
        self.input_vector_index = -1

        self.do_draw = False
        self.children = []
        self.immed_children = []
        self.ref_vector = np.array([0.0, 0.0, 1.0])
        self.thickness = (.1, .1)
        self.matrix = None
        self.length = 1.0
        self.mass = [0.0, 0.0, 0.0]
        self.bone_dim = np.array([0.0, 0.0, 1.0])

        self.set_vector_index()
        self.set_thickness()
        self.set_limb_vector()
        self.set_children()
        self.set_immediate_children()
        self.set_draw()

    def set_vector_index(self):
        for idx, actual_joint in enumerate(actual_joints):
            if actual_joint == self.name:
                self.input_vector_index = idx
                break

    def set_thickness(self, dims=None):
        if dims is None:
            self.thickness = (.05, .05)
        else:
            self.thickness = (dims[0], dims[1])

    def set_bone_dim(self, dims):
        self.bone_dim = np.array(dims)

    def set_limb_vector(self, limb_vector=None):
        if limb_vector is None:
            self.ref_vector = np.array([0.0, 0.0, 1.0])
        else:
            self.ref_vector = any_to_array(limb_vector)

    def set_immediate_children(self, kids=None):
        if kids is not None:
            self.immed_children = list(kids)
        else:
            if self.joint_index == t_PelvisAnchor:
                self.immed_children = [t_SpinePelvis, t_RightHip, t_LeftHip]
            elif self.joint_index == t_SpinePelvis:
                self.immed_children = [t_LowerVertebrae]
            elif self.joint_index == t_LowerVertebrae:
                self.immed_children = [t_MidVertebrae]
            elif self.joint_index == t_MidVertebrae:
                self.immed_children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_UpperVertebrae]
            elif self.joint_index == t_UpperVertebrae:
                self.immed_children = [t_BaseOfSkull]
            elif self.joint_index == t_BaseOfSkull:
                self.immed_children = [t_TopOfHead]

            elif self.joint_index == t_LeftHip:
                self.immed_children = [t_LeftKnee]
            elif self.joint_index == t_LeftKnee:
                self.immed_children = [t_LeftAnkle]
            elif self.joint_index == t_LeftAnkle:
                self.immed_children = [t_LeftBallOfFoot]
            elif self.joint_index == t_LeftBallOfFoot:
                self.immed_children = [t_LeftToeTip]

            elif self.joint_index == t_RightHip:
                self.immed_children = [t_RightKnee]
            elif self.joint_index == t_RightKnee:
                self.immed_children = [t_RightAnkle]
            elif self.joint_index == t_RightAnkle:
                self.immed_children = [t_RightBallOfFoot]
            elif self.joint_index == t_RightBallOfFoot:
                self.immed_children = [t_RightToeTip]

            elif self.joint_index == t_LeftShoulderBladeBase:
                self.immed_children = [t_LeftShoulder]
            elif self.joint_index == t_LeftShoulder:
                self.immed_children = [t_LeftElbow]
            elif self.joint_index == t_LeftElbow:
                self.immed_children = [t_LeftWrist]
            elif self.joint_index == t_LeftWrist:
                self.immed_children = [t_LeftKnuckle]
            elif self.joint_index == t_LeftKnuckle:
                self.immed_children = [t_LeftFingerTip]

            elif self.joint_index == t_RightShoulderBladeBase:
                self.immed_children = [t_RightShoulder]
            elif self.joint_index == t_RightShoulder:
                self.immed_children = [t_RightElbow]
            elif self.joint_index == t_RightElbow:
                self.immed_children = [t_RightWrist]
            elif self.joint_index == t_RightWrist:
                self.immed_children = [t_RightKnuckle]
            elif self.joint_index == t_RightKnuckle:
                self.immed_children = [t_RightFingerTip]


    def set_children(self, kids=None):
        if kids is not None:
            self.children = list(kids)
        else:
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
        try:
            base_vector = self.ref_vector  # like an up vector
            limb_vector = self.bone_dim.copy()  # vector defining limb extension from parent joint at T-Pose

            scale = np.linalg.norm(limb_vector)  # limb length
            limb_vector /= scale
            w = np.cross(limb_vector, base_vector)
            a_ = np.stack((limb_vector, w, np.cross(limb_vector, w)), axis=0)
            a = a_.T
            inv_a = np.linalg.pinv(a)
            b_ = np.stack((base_vector, w, np.cross(base_vector, w)), axis=0)
            b = b_.T

            matrix_ = np.dot(b, inv_a)

            joint_matrix = []
            joint_matrix.append(matrix_[0][0])
            joint_matrix.append(matrix_[0][1])
            joint_matrix.append(matrix_[0][2])
            joint_matrix.append(0)
            joint_matrix.append(matrix_[1][0])
            joint_matrix.append(matrix_[1][1])
            joint_matrix.append(matrix_[1][2])
            joint_matrix.append(0)
            joint_matrix.append(matrix_[2][0])
            joint_matrix.append(matrix_[2][1])
            joint_matrix.append(matrix_[2][2])
            joint_matrix.append(0)
            joint_matrix.append(0)
            joint_matrix.append(0)
            joint_matrix.append(0)
            joint_matrix.append(1)

            self.matrix = joint_matrix  # matrix defining rotation at parent joint in T-Pose
            self.length = scale
        except Exception as e:
            print('set_matrix:')
            traceback.print_exception(e)


class Joint(BaseJoint):
    def __init__(self, in_body, in_name, in_index):
        super().__init__(in_body, in_name, in_index)
        # self.body = in_body
        # self.name = in_name
        # self.shadow_name = joint_to_shadow_limb[in_name]
        # self.joint_index = in_index
        # self.input_vector_index = -1
        #
        # self.do_draw = False
        # self.children = []
        # self.ref_vector = np.array([0.0, 0.0, 1.0])
        # self.thickness = (.1, .1)
        # self.matrix = None
        # self.length = 1.0
        # self.mass = [0.0, 0.0, 0.0]
        # self.bone_dim = np.array([0.0, 0.0, 1.0])
        #
        # self.set_vector_index()
        # self.set_thickness()
        # self.set_limb_vector()
        # self.set_children()
        # self.set_draw()

    def set_thickness(self, dims=None):
        if dims is None:
            if self.joint_index == t_MidVertebrae:
                self.thickness = (.25, .12)
            elif self.joint_index == t_BaseOfSkull:
                self.thickness = (.07, .07)
            elif self.joint_index == t_TopOfHead:
                self.thickness = (.1, .12)
            elif self.joint_index in [t_LeftShoulder, t_RightShoulder]:
                self.thickness = (.08, .06)
            elif self.joint_index in [t_LeftKnuckle, t_RightKnuckle]:
                self.thickness = (.07, .025)
            elif self.joint_index in [t_LeftFingerTip, t_RightFingerTip]:
                self.thickness = (.07, .02)
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
                self.thickness = (.15, .10)
            elif self.joint_index in [t_LeftHip, t_RightHip]:
                self.thickness = (.05, .05)
            elif self.joint_index in [t_LeftBallOfFoot, t_RightBallOfFoot]:
                self.thickness = (.07, .03)
            elif self.joint_index in [t_LeftToeTip, t_RightToeTip]:
                self.thickness = (.07, .02)
            elif self.joint_index == t_UpperVertebrae:
                self.thickness = (.07, .07)
                # self.thickness = (.28, .15)
            elif self.joint_index == t_SpinePelvis:
                self.thickness = (.21, .11)
            elif self.joint_index == t_LowerVertebrae:
                self.thickness = (.2, .11)
        else:
            self.thickness = (dims[0], dims[1])

    def set_limb_vector(self, limb_vector=None):
        if limb_vector is None:
            self.ref_vector = np.array([0.0, 0.0, 1.0])
            if self.joint_index in [t_Body, t_PelvisAnchor, t_Reference, t_Tracker0, t_Tracker1, t_Tracker2, t_Tracker3]:
                self.ref_vector = np.array([1.0, 0.0, 0.0])
            elif self.joint_index in [t_LeftHeel, t_RightHeel]:
                self.ref_vector = np.array([0.0, 1.0, 0.0])
            elif self.joint_index in [t_LeftToeTip, t_RightToeTip]:
                self.ref_vector = np.array([0.0, 0.0001, 1.0])
        else:
            self.ref_vector = any_to_array(limb_vector)

    # def set_vector_index(self):
    #     for idx, actual_joint in enumerate(actual_joints):
    #         if actual_joint == self.name:
    #             self.input_vector_index = idx
    #             break

    # def set_bone_dim(self, dims):
    #     self.bone_dim = np.array(dims)

    # def set_limb_vector(self):
    #     self.ref_vector = np.array([0.0, 0.0, 1.0])
    #     if self.joint_index in [t_Body, t_PelvisAnchor, t_Reference, t_Tracker0, t_Tracker1, t_Tracker2, t_Tracker3]:
    #         self.ref_vector = np.array([1.0, 0.0, 0.0])
    #     elif self.joint_index in [t_LeftHeel, t_RightHeel]:
    #         self.ref_vector = np.array([0.0, 1.0, 0.0])
    #     elif self.joint_index in [t_LeftToeTip, t_RightToeTip]:
    #         self.ref_vector = np.array([0.0, 0.0001, 1.0])

    # def set_children(self):
    #     if self.joint_index == t_MidVertebrae:
    #         self.children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_UpperVertebrae, t_BaseOfSkull,
    #                          t_TopOfHead]
    #     elif self.joint_index == t_BaseOfSkull:
    #         self.children = [t_TopOfHead]
    #     elif self.joint_index == t_PelvisAnchor:
    #         self.children = [t_RightKnee, t_LeftKnee, t_SpinePelvis, t_LowerVertebrae]
    #     elif self.joint_index == t_LeftShoulder:
    #         self.children = [t_LeftElbow, t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
    #     elif self.joint_index == t_LeftKnuckle:
    #         self.children = [t_LeftFingerTip]
    #     elif self.joint_index == t_LeftAnkle:
    #         self.children = [t_LeftBallOfFoot, t_LeftToeTip]
    #     elif self.joint_index == t_LeftElbow:
    #         self.children = [t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
    #     elif self.joint_index == t_LeftWrist:
    #         self.children = [t_LeftKnuckle, t_LeftFingerTip]
    #     elif self.joint_index == t_LeftKnee:
    #         self.children = [t_LeftAnkle, t_LeftBallOfFoot, t_LeftToeTip]
    #     elif self.joint_index == t_LeftShoulderBladeBase:
    #         self.children = [t_LeftShoulder, t_LeftElbow, t_LeftWrist, t_LeftKnuckle, t_LeftFingerTip]
    #     elif self.joint_index == t_LeftHip:
    #         self.children = [t_LeftKnee, t_LeftAnkle, t_LeftBallOfFoot, t_LeftToeTip]
    #     elif self.joint_index == t_LeftBallOfFoot:
    #         self.children = [t_LeftToeTip]
    #     elif self.joint_index == t_UpperVertebrae:
    #         self.children = [t_BaseOfSkull, t_TopOfHead]
    #     elif self.joint_index == t_RightShoulder:
    #         self.children = [t_RightElbow, t_RightWrist, t_RightKnuckle, t_RightFingerTip]
    #     elif self.joint_index == t_RightKnuckle:
    #         self.children = [t_RightFingerTip]
    #     elif self.joint_index == t_RightAnkle:
    #         self.children = [t_RightBallOfFoot, t_RightToeTip]
    #     elif self.joint_index == t_RightElbow:
    #         self.children = [t_RightWrist, t_RightKnuckle, t_RightFingerTip]
    #     elif self.joint_index == t_RightWrist:
    #         self.children = [t_RightKnuckle, t_RightFingerTip]
    #     elif self.joint_index == t_RightKnee:
    #         self.children = [t_RightAnkle, t_RightBallOfFoot, t_RightToeTip]
    #     elif self.joint_index == t_RightShoulderBladeBase:
    #         self.children = [t_RightShoulder, t_RightElbow, t_RightWrist, t_RightKnuckle, t_RightFingerTip]
    #     elif self.joint_index == t_RightHip:
    #         self.children = [t_RightKnee, t_RightAnkle, t_RightBallOfFoot, t_RightToeTip]
    #     elif self.joint_index == t_RightBallOfFoot:
    #         self.children = [t_RightBallOfFoot, t_RightToeTip]
    #     elif self.joint_index == t_SpinePelvis:
    #         self.children = [t_LowerVertebrae, t_MidVertebrae, t_LeftShoulderBladeBase, t_RightShoulderBladeBase,
    #                          t_UpperVertebrae,
    #                          t_BaseOfSkull, t_TopOfHead]
    #     elif self.joint_index == t_LowerVertebrae:
    #         self.children = [t_MidVertebrae, t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_UpperVertebrae,
    #                          t_BaseOfSkull,
    #                          t_TopOfHead]

    # def set_draw(self):
    #     if self.joint_index in [t_LeftHeel, t_RightHeel, t_TopOfHead, t_BaseOfSkull, t_UpperVertebrae, t_LeftFingerTip,
    #                      t_RightFingerTip, t_LeftKnuckle,
    #                      t_RightKnuckle, t_LeftWrist, t_RightWrist, t_LeftElbow, t_RightElbow, t_LeftShoulder,
    #                      t_RightShoulder,
    #                      t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_SpinePelvis, t_LowerVertebrae,
    #                      t_MidVertebrae, t_LeftHip, t_RightHip,
    #                      t_LeftKnee, t_LeftAnkle, t_RightKnee, t_RightAnkle, t_LeftBallOfFoot, t_RightBallOfFoot,
    #                      t_LeftToeTip, t_RightToeTip]:
    #         self.do_draw = True

    # def set_mass(self):
    #     if self.matrix is not None:
    #         self.mass = [self.thickness[0], self.length, self.thickness[1]]
    #         for child_index in self.children:
    #             child = self.body.joints[child_index]
    #             self.mass[0] += child.thickness[0]
    #             self.mass[1] += child.length
    #             self.mass[2] += child.thickness[1]

    # def set_matrix(self):
    #     try:
    #         base_vector = self.ref_vector  # like an up vector
    #         limb_vector = self.bone_dim.copy()  # vector defining limb extension from parent joint at T-Pose
    #
    #         scale = np.linalg.norm(limb_vector)  # limb length
    #         limb_vector /= scale
    #         w = np.cross(limb_vector, base_vector)
    #         a_ = np.stack((limb_vector, w, np.cross(limb_vector, w)), axis=0)
    #         a = a_.T
    #         inv_a = np.linalg.pinv(a)
    #         b_ = np.stack((base_vector, w, np.cross(base_vector, w)), axis=0)
    #         b = b_.T
    #
    #         matrix_ = np.dot(b, inv_a)
    #
    #         joint_matrix = []
    #         joint_matrix.append(matrix_[0][0])
    #         joint_matrix.append(matrix_[0][1])
    #         joint_matrix.append(matrix_[0][2])
    #         joint_matrix.append(0)
    #         joint_matrix.append(matrix_[1][0])
    #         joint_matrix.append(matrix_[1][1])
    #         joint_matrix.append(matrix_[1][2])
    #         joint_matrix.append(0)
    #         joint_matrix.append(matrix_[2][0])
    #         joint_matrix.append(matrix_[2][1])
    #         joint_matrix.append(matrix_[2][2])
    #         joint_matrix.append(0)
    #         joint_matrix.append(0)
    #         joint_matrix.append(0)
    #         joint_matrix.append(0)
    #         joint_matrix.append(1)
    #
    #         self.matrix = joint_matrix  # matrix defining rotation at parent joint in T-Pose
    #         self.length = scale
    #     except Exception as e:
    #         print('set_matrix:')
    #         traceback.print_exception(e)


