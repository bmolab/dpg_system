from pylab import *
import numpy as np

from pyquaternion import Quaternion
import json
import json
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import xml.etree.ElementTree as ET
from tempfile import NamedTemporaryFile
from dpg_system.body_defs import *
from dpg_system.joint import *
from dpg_system.node import *
from dpg_system.gl_nodes import GLMaterial

scale = 1.6

joint_quats_np = None


def load_take_from_npz(path):
    take_file = np.load(path)
    quat_np = take_file['quats']
#   note some left wrist quats will be inverted
#   quat_np[:,10,0] - if it is negative, invert quat.
    for idx, quat in enumerate(quat_np):
        if quat[10, 0] < 0:
            quat_np[idx, 10] *= -1
    frames_ = quat_np.shape[0]
    joint_quats_np = np.zeros((frames_, 80))
    for i in range(frames_):
        for index_, key in enumerate(actual_joints):
            data = actual_joints[key]
            joint_quats_np[i, data[0] * 4:data[0] * 4 + 4] = quat_np[i, data[1]]
    return take_file['quats'], take_file['positions'], take_file['labels'], joint_quats_np


def save_take_to_npz(path, take_quats, take_positions, take_labels):
    np.savez(path, quats=take_quats, positions=take_positions, labels=take_labels)

def save_templates_and_distances(path, pose_proximities, templates):
    for idx, template in enumerate(templates):
        if idx == 0:
            temp_np = template.numpy()
        else:
            temp_np = np.stack((temp_np, template.numpy()), axis=0)
    pose_prox = pose_proximities.numpy()
    np.savez(path, templates=temp_np, proximities=pose_prox)

def quaternion_to_R3_rotation(q):
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    aa = a * a
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d

    norme_carre = aa + bb + cc + dd

    result = list()
    for i in range(0, 4):
        for j in range(0, 4):
            if i == j:
                result.append(1)
            else:
                result.append(0)

    if (norme_carre > 1e-6):
        result[0] = (aa + bb - cc - dd) / norme_carre
        result[1] = 2 * (-ad + bc) / norme_carre
        result[2] = 2 * (ac + bd) / norme_carre
        result[4] = 2 * (ad + bc) / norme_carre
        result[5] = (aa - bb + cc - dd) / norme_carre
        result[6] = 2 * (-ab + cd) / norme_carre
        result[8] = 2 * (-ac + bd) / norme_carre
        result[9] = 2 * (ab + cd) / norme_carre
        result[10] = (aa - bb - cc + dd) / norme_carre

    return result


class ScopedLock:
    def __init__(self, mutex):
        self.__mutex = None

        mutex.acquire()
        self.__mutex = mutex

    def __del__(self):
        self.__mutex.release()


class BodyData:
    def __init__(self):
        self.origin = None
        self.positions = []
        self.quaternions = []
        self.quaternionDistance = []
        self.previousQuats = []
        self.rotationAxis = []
        self.diffQuaternionA = []
        self.diffQuaternionB = []
        self.diffDiff = []
        self.diffQuaternionAbsSum = []
        self.diffQuatSmoothingA = 0.8
        self.diffQuatSmoothingB = 0.9
        self.joint_scales = [1.0] * 38
        self.node = None
        self.captured_quaternions = []
        self.label = 0
        self.joint_sphere = gluNewQuadric()
        self.joint_disk = gluNewQuadric()
        self.joint_motion_scale = 7.0
        self.joint_disk_material = GLMaterial()
        self.joint_disk_alpha = 0.5
        self.joint_disk_material.ambient = [0.19125, 0.0735, 0.0225, self.joint_disk_alpha]
        self.joint_disk_material.diffuse = [0.7038, 0.27048, 0.0828, self.joint_disk_alpha]
        self.joint_disk_material.specular = [0.256777, 0.137622, 0.086014, self.joint_disk_alpha]
        self.joint_disk_material.shininess = 12.8


        self.joints = []
        for joint_index in joint_index_to_name:
            name = joint_index_to_name[joint_index]
            new_joint = Joint(self, name, joint_index)
            self.joints.append(new_joint)

        tree = ET.parse('dpg_system/definition.xml')
        root = tree.getroot()
        for node in root.iter('node'):
            if 'translate' in node.attrib:
                trans = node.attrib['translate']
                trans_float = tuple(map(float, trans.split(' ')))
                y = tuple(i / 100.0 for i in trans_float)
                joint_name = shadow_limb_to_joint[node.attrib['id']]
                joint_index = joint_name_to_index[joint_name]
                self.joints[joint_index].bone_dim = y

        for joint in self.joints:
            joint.set_matrix()
            joint.set_mass()

        self.base_material = [0.5, 0.5, 0.5, 1.0]
        self.color = []
        self.create_colours()
        self.capture_next_pose = False
        self.has_captured_pose = False
        self.distance_sense = 1.0
        self.error_band = 0.00
        self.pose_similarity = 0
        self.input_vector = np.zeros(80, dtype=np.float)

        self.__mutex = threading.Lock()
        for joint_index in joint_index_to_name:
            self.quaternions.append([1.0, 0.0, 0.0, 0.0])
            self.captured_quaternions.append([1.0, 0.0, 0.0, 0.0])
            self.positions.append([0.0, 0.0, 0.0])
            self.previousQuats.append([1.0, 0.0, 0.0, 0.0])
            self.quaternionDistance.append(0.0)
            self.rotationAxis.append([0.0, 0.0, 0.0])
            self.diffQuaternionA.append([0.0, 0.0, 0.0, 0.0])
            self.diffQuaternionB.append([0.0, 0.0, 0.0, 0.0])
            self.diffDiff.append([0.0, 0.0, 0.0, 0.0])

            self.diffQuaternionAbsSum.append(0.0)


    def create_colours(self):
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])
        self.color.append([0.5, 0.0, 0.0, 1.0])
        self.color.append([0.0, 0.5, 0.0, 1.0])
        self.color.append([0.2, 0.2, 0.75, 1.0])
        self.color.append([0.5, 0.5, 0.0, 1.0])
        self.color.append([0.5, 0.0, 0.75, 1.0])
        self.color.append([0.0, 0.5, 0.75, 1.0])

    def zero_pelvis(self):
        self.quaternions[t_PelvisAnchor] = [1, 0, 0, 0]

    def set_label(self, in_label, confidence=1.0):
        self.label = in_label
        if in_label >= 0 and in_label < len(self.color):
            self.base_material[0] = self.color[int(in_label)][0] * confidence + 0.5 * (1.0 - confidence)
            self.base_material[1] = self.color[int(in_label)][1] * confidence + 0.5 * (1.0 - confidence)
            self.base_material[2] = self.color[int(in_label)][2] * confidence + 0.5 * (1.0 - confidence)
        else:
            self.base_material[0] = 0.5
            self.base_material[1] = 0.5
            self.base_material[2] = 0.5

    def update(self, joint_index, quat, position=None, label=0, paused=False):
        if position is not None:
            if joint_index == 4 and self.origin is None:
                self.origin = [position[0], position[1], position[2]]
            self.positions[joint_index] = [position[0], position[1], position[2]]

        if not paused:
#            self.previousQuats[joint_index] = self.quaternions[joint_index]
            self.previousQuats[joint_index] = self.diffQuaternionA[joint_index].copy()
        self.quaternions[joint_index] = [quat[0], quat[1], quat[2], quat[3]]
        if not paused:
            self.calc_diff_quaternion(joint_index)

        q1 = Quaternion(self.previousQuats[joint_index])
        q2 = Quaternion(self.diffQuaternionA[joint_index])

        # quaternion smooothing -> slerp on quat and prev or slerp on distance
        # use difference of gaussians trick to get smooth sense of jerk and flex
        if paused == False:
            self.quaternionDistance[joint_index] = Quaternion.sym_distance(q1.unit, q2.unit)
            self.calc_diff_quaternion(joint_index)
        diff = q2 - q1
        self.rotationAxis[joint_index] = list(diff.unit.axis)
        self.set_label(label - 1)

    def calc_diff_quaternion(self, jointIndex):
        w = self.quaternions[jointIndex][0]  # - self.previousQuats[jointIndex][0]
        x = self.quaternions[jointIndex][1]  # - self.previousQuats[jointIndex][1]
        y = self.quaternions[jointIndex][2]  # - self.previousQuats[jointIndex][2]
        z = self.quaternions[jointIndex][3]  # - self.previousQuats[jointIndex][3]
        self.diffQuaternionA[jointIndex][0] = self.diffQuaternionA[jointIndex][0] * self.diffQuatSmoothingA + w * (1.0 - self.diffQuatSmoothingA)
        self.diffQuaternionA[jointIndex][1] = self.diffQuaternionA[jointIndex][1] * self.diffQuatSmoothingA + x * (1.0 - self.diffQuatSmoothingA)
        self.diffQuaternionA[jointIndex][2] = self.diffQuaternionA[jointIndex][2] * self.diffQuatSmoothingA + y * (1.0 - self.diffQuatSmoothingA)
        self.diffQuaternionA[jointIndex][3] = self.diffQuaternionA[jointIndex][3] * self.diffQuatSmoothingA + z * (1.0 - self.diffQuatSmoothingA)
        self.diffQuaternionB[jointIndex][0] = self.diffQuaternionB[jointIndex][0] * self.diffQuatSmoothingB + w * (1.0 - self.diffQuatSmoothingB)
        self.diffQuaternionB[jointIndex][1] = self.diffQuaternionB[jointIndex][1] * self.diffQuatSmoothingB + x * (1.0 - self.diffQuatSmoothingB)
        self.diffQuaternionB[jointIndex][2] = self.diffQuaternionB[jointIndex][2] * self.diffQuatSmoothingB + y * (1.0 - self.diffQuatSmoothingB)
        self.diffQuaternionB[jointIndex][3] = self.diffQuaternionB[jointIndex][3] * self.diffQuatSmoothingB + z * (1.0 - self.diffQuatSmoothingB)
        self.diffDiff[jointIndex][0] = self.diffQuaternionA[jointIndex][0] - self.diffQuaternionB[jointIndex][0]
        self.diffDiff[jointIndex][1] = self.diffQuaternionA[jointIndex][1] - self.diffQuaternionB[jointIndex][1]
        self.diffDiff[jointIndex][2] = self.diffQuaternionA[jointIndex][2] - self.diffQuaternionB[jointIndex][2]
        self.diffDiff[jointIndex][3] = self.diffQuaternionA[jointIndex][3] - self.diffQuaternionB[jointIndex][3]

        self.diffQuaternionAbsSum[jointIndex] = abs(self.diffDiff[jointIndex][0]) + abs(self.diffDiff[jointIndex][1]) + abs(self.diffDiff[jointIndex][2]) + abs(self.diffDiff[jointIndex][3])

    def actual_joint_to_shadow_joint(self, actual_joint):
        data = actual_joints[actual_joint]
        return data[1]

    def shadow_joint_to_actual_joint(self, shadow_joint):
        return joints_to_input_vector[shadow_joint]

    def convert_shadow_quats_to_input_vector(self):
        for i, key in enumerate(actual_joints):
            data = actual_joints[key]
            self.input_vector[data[0] * 4:data[0] * 4 + 4] = self.quaternions[data[1]]

    def convert_input_vector_to_shadow_quats(self, target):
        for i, key in enumerate(actual_joints):
            data = actual_joints[key]
            self.quaternions[data[1]] = target[data[0]].tolist()
            self.quaternions[t_PelvisAnchor] = [1, 0, 0, 0]

    def clear_captured_pose(self):
        self.captured_quaternions = None

    def capture_current_pose(self):
        self.captured_quaternions = self.quaternions.copy()
        self.has_captured_pose = True

    def calc_distance_from_pose(self):
        d = 0
        for j, jointIndex in enumerate(joint_index_to_name):
            if jointIndex == 4:
                continue
            q1 = Quaternion(self.captured_quaternions[jointIndex])
            q2 = Quaternion(self.quaternions[jointIndex])
            joint_distance = Quaternion.distance(q1, q2)
            joint_distance -= self.error_band
            if joint_distance < 0:
                joint_distance = 0
            d += (joint_distance * joint_distance)

        self.pose_similarity = 1.0 - d / self.distance_sense
        if self.pose_similarity > 1.0:
            self.pose_similarity = 1.0
        elif self.pose_similarity < 0.0:
            self.pose_similarity = 0

    def transform_to_opengl(self, transform):
        if transform is not None and len(transform) == 16:
            # Transpose matrix for OpenGL column-major order.
            for i in range(0, 4):
                for j in range((i + 1), 4):
                    temp = transform[4 * i + j]
                    transform[4 * i + j] = transform[4 * j + i]
                    transform[4 * j + i] = temp
        return transform

    def draw(self, show_rotation_spheres=False):
        transform = None
        glPushMatrix()

        glScalef(scale, scale, 1.0)
        glTranslatef(0.0, -1.0, 0.0)

        # UI for showing closeness to captured, but also labels...???
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_material)

        self.move_to(t_PelvisAnchor)

        glPushMatrix()
        self.draw_to(t_LeftHip, False, t_PelvisAnchor)
        self.draw_to(t_LeftKnee, False, t_LeftHip)
        self.draw_to(t_LeftAnkle, False, t_LeftKnee)

        glPushMatrix()
        self.draw_to(t_LeftHeel, False, t_NoJoint)
        glPopMatrix()

        self.draw_to(t_LeftBallOfFoot, False, t_LeftAnkle)
        self.draw_to(t_LeftToeTip, False, t_LeftBallOfFoot)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_RightHip, False, t_PelvisAnchor)
        self.draw_to(t_RightKnee, False, t_RightHip)
        self.draw_to(t_RightAnkle, False, t_RightKnee)

        glPushMatrix()
        self.draw_to(t_RightHeel, False, t_NoJoint)
        glPopMatrix()

        self.draw_to(t_RightBallOfFoot, False, t_RightAnkle)
        self.draw_to(t_RightToeTip, False, t_RightBallOfFoot)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_SpinePelvis, False, t_PelvisAnchor)
        self.draw_to(t_LowerVertebrae, False, t_SpinePelvis)
        self.draw_to(t_MidVertebrae, False, t_LowerVertebrae)

        glPushMatrix()
        self.draw_to(t_LeftShoulderBladeBase, False, t_MidVertebrae)
        self.draw_to(t_LeftShoulder, False, t_LeftShoulderBladeBase)
        self.draw_to(t_LeftElbow, False, t_LeftShoulder)
        self.draw_to(t_LeftWrist, False, t_LeftElbow)
        self.draw_to(t_LeftKnuckle, False, t_LeftWrist)
        self.draw_to(t_LeftFingerTip, False, t_LeftKnuckle)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_RightShoulderBladeBase, False, t_MidVertebrae)
        self.draw_to(t_RightShoulder, False, t_RightShoulderBladeBase)
        self.draw_to(t_RightElbow, False, t_RightShoulder)
        self.draw_to(t_RightWrist, False, t_RightElbow)
        self.draw_to(t_RightKnuckle, False, t_RightWrist)
        self.draw_to(t_RightFingerTip, False, t_RightKnuckle)
        glPopMatrix()

        self.draw_to(t_UpperVertebrae, False, t_MidVertebrae)
        self.draw_to(t_BaseOfSkull, False, t_UpperVertebrae)
        self.draw_to(t_TopOfHead, False, t_BaseOfSkull)

        glPopMatrix()
        glPopMatrix()
        self.draw_joint_spheres(show_rotation_spheres, transform)

    def adjust_clear_colour(self):
        if self.has_captured_pose:
            r = self.pose_similarity * self.color[1][0] + (1 - self.pose_similarity) * self.color[0][0]
            g = self.pose_similarity * self.color[1][1] + (1 - self.pose_similarity) * self.color[0][1]
            b = self.pose_similarity * self.color[1][2] + (1 - self.pose_similarity) * self.color[0][2]
            glClearColor(r, g, b, 1.0)
        else:
            glClearColor(0, 0, 0, 0)

    def test_show_joint_spheres(self, show_rotation_spheres):
        return show_rotation_spheres

    def draw_joint_spheres(self, show_rotation_spheres, transform):
        if self.test_show_joint_spheres(show_rotation_spheres):
            glPushMatrix()
            glScalef(scale, scale, 1.0)
            glTranslatef(0.0, -1.0, 0.0)

            self.move_to(t_PelvisAnchor)

            glPushMatrix()
            self.draw_to(t_LeftHip, True, t_PelvisAnchor)
            self.draw_to(t_LeftKnee, True, t_LeftHip)
            self.draw_to(t_LeftAnkle, True, t_LeftKnee)

            glPushMatrix()
            self.draw_to(t_LeftHeel, True, t_NoJoint)
            glPopMatrix()

            self.draw_to(t_LeftBallOfFoot, True, t_LeftAnkle)
            self.draw_to(t_LeftToeTip, True, t_LeftBallOfFoot)
            glPopMatrix()

            glPushMatrix()
            self.draw_to(t_RightHip, True, t_PelvisAnchor)
            self.draw_to(t_RightKnee, True, t_RightHip)
            self.draw_to(t_RightAnkle, True, t_RightKnee)

            glPushMatrix()
            self.draw_to(t_RightHeel, True, t_NoJoint)
            glPopMatrix()

            self.draw_to(t_RightBallOfFoot, True, t_RightAnkle)
            self.draw_to(t_RightToeTip, True, t_RightBallOfFoot)
            glPopMatrix()

            glPushMatrix()
            self.draw_to(t_SpinePelvis, True, t_PelvisAnchor)
            self.draw_to(t_LowerVertebrae, True, t_SpinePelvis)
            self.draw_to(t_MidVertebrae, True, t_LowerVertebrae)

            glPushMatrix()
            self.draw_to(t_LeftShoulderBladeBase, True, t_MidVertebrae)
            self.draw_to(t_LeftShoulder, True, t_LeftShoulderBladeBase)
            self.draw_to(t_LeftElbow, True, t_LeftShoulder)
            self.draw_to(t_LeftWrist, True, t_LeftElbow)
            self.draw_to(t_LeftKnuckle, True, t_LeftWrist)
            self.draw_to(t_LeftFingerTip, True, t_LeftKnuckle)
            glPopMatrix()

            glPushMatrix()
            self.draw_to(t_RightShoulderBladeBase, True, t_MidVertebrae)
            self.draw_to(t_RightShoulder, True, t_RightShoulderBladeBase)
            self.draw_to(t_RightElbow, True, t_RightShoulder)
            self.draw_to(t_RightWrist, True, t_RightElbow)
            self.draw_to(t_RightKnuckle, True, t_RightWrist)
            self.draw_to(t_RightFingerTip, True, t_RightKnuckle)
            glPopMatrix()

            self.draw_to(t_UpperVertebrae, True, t_MidVertebrae)
            self.draw_to(t_BaseOfSkull, True, t_UpperVertebrae)
            self.draw_to(t_TopOfHead, True, t_BaseOfSkull)

            glPopMatrix()
            glPopMatrix()

    def move_to(self, jointIndex):
        # get quat and convert to matrix
        # apply matrix
        # draw line from 0,0 to limb translation
        # translate along limb
        if jointIndex != t_NoJoint:
            quat = self.quaternions[jointIndex]
            transform = quaternion_to_R3_rotation(quat)
            transform = self.transform_to_opengl(transform)
            glTranslatef(self.joints[jointIndex].bone_dim[0], self.joints[jointIndex].bone_dim[1], self.joints[jointIndex].bone_dim[2])
            glMultMatrixf(transform)

    def draw_to(self, joint_index, orientation=False, next_limb_index=-1):
        quat = self.quaternions[joint_index]
        transform = quaternion_to_R3_rotation(quat)
        transform = self.transform_to_opengl(transform)
        joint_data = self.joints[joint_index]

        m = joint_data.matrix
        length = joint_data.length
        widths = joint_data.thickness

        if joint_data.do_draw:
            if orientation:
                self.show_orientation(joint_index, next_limb_index)
            if self.node and next_limb_index != -1:
                self.node.joint_callback()
            mm = glGetDoublev(GL_MODELVIEW_MATRIX)
            glPushMatrix()
            glMultMatrixf(m)
            glLineWidth(2.0)
            mm = glGetDoublev(GL_MODELVIEW_MATRIX)

            if not orientation:
            #     self.show_orientation(joint_index, next_limb_index)
            # else:
                self.draw_block((widths[0], length, widths[1]))
            # could call out to node...

            glPopMatrix()
            mm = glGetDoublev(GL_MODELVIEW_MATRIX)

        glTranslatef(joint_data.bone_dim[0], joint_data.bone_dim[1], joint_data.bone_dim[2])
        glMultMatrixf(transform)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_material)

    def draw_block(self, dim):
        dim_x = dim[0] / 2
        dim_z = dim[1] - .02
        dim_y = dim[2] / 2

        points = []
        points.append((-dim_x, -dim_y, 0))
        points.append((-dim_x, -dim_y, dim_z))
        points.append((dim_x, -dim_y, 0))
        points.append((dim_x, -dim_y, dim_z))
        points.append((dim_x, dim_y, 0))
        points.append((dim_x, dim_y, dim_z))
        points.append((-dim_x, dim_y, 0))
        points.append((-dim_x, dim_y, dim_z))

        glBegin(GL_TRIANGLE_STRIP)

        normal = self.calc_normal(points[0], points[1], points[2])
        glNormal3fv(normal)

        glVertex3fv(points[0])
        glVertex3fv(points[1])
        glVertex3fv(points[2])
        glVertex3fv(points[3])

        normal = self.calc_normal(points[2], points[3], points[4])
        glNormal3fv(normal)

        glVertex3fv(points[4])
        glVertex3fv(points[5])

        normal = self.calc_normal(points[4], points[5], points[6])
        glNormal3fv(normal)

        glVertex3fv(points[6])
        glVertex3fv(points[7])

        normal = self.calc_normal(points[6], points[7], points[0])
        glNormal3fv(normal)

        glVertex3fv(points[0])
        glVertex3fv(points[1])

        glVertex3fv(points[1])
        glVertex3fv(points[1])

        # DEGENERATE TRIANGLE ISSUE
        normal = self.calc_normal(points[3], points[5], points[7])
        glNormal3fv(normal)

        glVertex3fv(points[1])
        glVertex3fv(points[3])
        glVertex3fv(points[7])
        glVertex3fv(points[5])

        glVertex3fv(points[5])
        glVertex3fv(points[0])

        normal = self.calc_normal(points[2], points[4], points[6])
        glNormal3fv(normal)

        glVertex3fv(points[0])
        glVertex3fv(points[2])
        glVertex3fv(points[6])
        glVertex3fv(points[4])

        glEnd()

    def calc_normal(self, v1, v2, v3):
        v_1 = np.array([v1[0], v1[1], v1[2]])
        v_2 = np.array([v2[0], v2[1], v2[2]])
        v_3 = np.array([v3[0], v3[1], v3[2]])
        temp1 = v_2 - v_1
        temp2 = v_3 - v_1
        normal = np.cross(temp1, temp2)
        normal = normal / np.linalg.norm(normal)
        return (normal[0], normal[1], normal[2])

    def __del__(self):
        if self.__mutex is not None:
            lock = ScopedLock(self.__mutex)
            self.__mutex = None

    def unit_vector(self, data, axis=None, out=None):
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        data /= length
        return data

    def rotationAlign(self, axis, up_vector):
        v = np.cross(axis, up_vector)
        c = np.dot(axis, up_vector)
        k = 1.0 / (1.0 + c)

        alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                  v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                  v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                  0.0, 0.0, 0.0, 1.0])
        alignment_matrix.reshape((4, 4))
        restore_matrix = glGetInteger(GL_MATRIX_MODE)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(self.alignment_matrix)

    def rotation_matrix_from_axis_and_angle(self, direction, angle):
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self.unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array(
            [
                [0.0, -direction[2], direction[1]],
                [direction[2], 0.0, -direction[0]],
                [-direction[1], direction[0], 0.0],
            ]
        )
        M = np.identity(4)
        M[:3, :3] = R
        return M

    def draw_quaternion_distance_sphere(self, limb_index, joint_data):
        if limb_index != -1:
            up_vector = np.array([0.0, 0.0, 1.0])
            d = self.quaternionDistance[limb_index] * self.joint_motion_scale   #self.quaternion_distance_display_scale
            if d > 0.001:
                axis = self.rotationAxis[limb_index]
    #            weight = joint_data.mass[0] * abs(axis[0]) + joint_data.mass[2] * abs(axis[2]) + joint_data.mass[1] * abs(
    #                axis[1])
                self.joint_disk_material.diffuse[3] = self.joint_disk_alpha
                # set colour by orientation
                # blue is axial rotation
                # red is forward / backward
                # yellow is side to side
                # use axis
                # axis_colour = np.array(axis)
                # axis_colour /= axis_colour.sum()
                # self.joint_disk_material.diffuse[0] = axis_colour[0]
                # self.joint_disk_material.diffuse[1] = axis_colour[1]
                # self.joint_disk_material.diffuse[2] = axis_colour[2]
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.joint_disk_material.diffuse)

                v = np.cross(axis, up_vector)
                c = np.dot(axis, up_vector)
                k = 1.0 / (1.0 + c)

                alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                                             v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                                             v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                                             0.0, 0.0, 0.0, 1.0])

                alignment_matrix.reshape((4, 4))
                restore_matrix = glGetInteger(GL_MATRIX_MODE)
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glMultMatrixf(alignment_matrix)
                gluDisk(self.joint_sphere, 0.0, d, 32, 1)
                glPopMatrix()
                glMatrixMode(restore_matrix)

    def show_orientation(self, joint_index, next_limb_index):
        glPushMatrix()
        joint_data = self.joints[joint_index]

        if next_limb_index == -1:
            next_limb_index = joint_index
        self.draw_quaternion_distance_sphere(next_limb_index, joint_data)
        glPopMatrix()
