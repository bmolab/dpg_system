from pylab import *
import numpy as np

from pyquaternion import Quaternion
import quaternion
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
from dpg_system.conversion_utils import *

scale = 1.0

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
        for index in range(t_ActiveJointCount):
            joint_quats_np[i, index * 4:index * 4 + 4] = quat_np[i, index]
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

def quaternion_to_matrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    n = 2.0

    matrix = [1.0 - n * (y2 + z2),
        n * (xy - wz),
        n * (xz + wy),
        0.0,

        n * (xy + wz),
        1.0 - n * (x2 + z2),
        n * (yz - wx),
        0.0,

        n * (xz - wy),
        n * (yz + wx),
        1.0 - n * (x2 + y2),
        0.0,

        0.0,
        0.0,
        0.0,
        1.0
    ]

    return matrix


# input shape is [num_bodies, num_joints, 4]
def quaternions_to_R3_rotation(qs):
    qs = any_to_array(qs)
    qs = np.reshape(qs, (-1, 20, 4))
    quats_squared = qs * qs         #   x^2 y^2 z^2 w^2

    xs = qs[:, :, 0]                #   x
    ys = qs[:, :, 1]                #   y
    zs = qs[:, :, 2]                #   z
    ws = qs[:, :, 3]                #   w

    xy_s = xs * ys                  #   x * y
    xz_s = xs * zs
    xw_s = xs * ws
    yz_s = ys * zs
    yw_s = ys * ws
    zw_s = zs * ws

    norm_carre = quats_squared.sum(axis=2)

    matrices = np.zeros([qs.shape[0], qs.shape[1], 16])
    norm_carre = np.maximum(norm_carre, 1e-6)
    # norm_carre += 1e-6
    matrices[:, :, 15] = norm_carre

    matrices[:, :, 0] = quats_squared[:, :, 0] + quats_squared[:, :, 1] - quats_squared[:, :, 2] - quats_squared[:, :, 3]
    matrices[:, :, 4] = (yz_s[:, :] - xw_s[:, :]) * 2
    matrices[:, :, 8] = (xz_s[:, :] + yw_s[:, :]) * 2

    matrices[:, :, 1] = (xw_s[:, :] + yz_s[:, :]) * 2
    matrices[:, :, 5] = quats_squared[:, :, 0] - quats_squared[:, :, 1] + quats_squared[:, :, 2] - quats_squared[:, :, 3]
    matrices[:, :, 9] = (zw_s[:, :] - xy_s[:, :]) * 2

    matrices[:, :, 2] = (yw_s[:, :] - xz_s[:, :]) * 2
    matrices[:, :, 6] = (xy_s[:, :] + zw_s[:, :]) * 2
    matrices[:, :, 10] = quats_squared[:, :, 0] - quats_squared[:, :, 1] - quats_squared[:, :, 2] + quats_squared[:, :, 3]
    norm_carre = np.expand_dims(norm_carre, axis=2)
    matrices = matrices / norm_carre
    return matrices


def quaternion_to_R3_rotation(q):
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    aa = a * a
    bb = b * b
    cc = c * c
    dd = d * d

    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

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


class BodyDataBase:
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

    def __init__(self):
        self.node = None
        self.skeleton = False
        self.num_bodies = 1
        self.current_body = 0

        self.joint_matrices = None
        self.joint_quats = None
        self.quaternions_np = np.zeros((1, 20, 4), dtype=float)

        self.base_material = [0.5, 0.5, 0.5, 1.0]
        self.multi_body_translation = [0.0, 0.0, 0.0]

        self.color = []
        self.create_colours()

        self.calc_diff = False
        self.limb_vertices = []
        self.create_limbs()

        self.joints = []
        self.limbs = [None] * 36
        # self.joint_mapper = [-1] * 37
        self.joint_display = 'sphere'

        self.create_joints()

        default_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.default_gl_transform = quaternion_to_R3_rotation(default_quat)
        self.default_gl_transform = self.transform_to_opengl(self.default_gl_transform)
        self.__mutex = threading.Lock()


    def draw(self, show_rotation_spheres=False, skeleton=False):
        hold_shade = glGetInteger(GL_SHADE_MODEL)
        glShadeModel(GL_FLAT)
        glPushMatrix()      # 0
        glScalef(scale, scale, 1.0)
        glTranslatef(0.0, -1.0, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_material)

        self.skeleton = skeleton
        # transform = None

        for i in range(self.num_bodies):
            self.current_body = i

            glPushMatrix()  # 1

            self.move_to(t_PelvisAnchor)

            glPushMatrix()  # 2
            self.draw_to(t_LeftHip, t_PelvisAnchor)
            self.draw_to(t_LeftKnee, t_LeftHip)
            self.draw_to(t_LeftAnkle, t_LeftKnee)

            glPushMatrix()  # 3
            self.draw_to(t_LeftHeel, t_NoJoint)
            glPopMatrix()   # 3

            self.draw_to(t_LeftBallOfFoot, t_LeftAnkle)
            self.draw_to(t_LeftToeTip, t_LeftBallOfFoot)
            glPopMatrix()   # 2

            glPushMatrix()  # 3
            self.draw_to(t_RightHip, t_PelvisAnchor)
            self.draw_to(t_RightKnee, t_RightHip)
            self.draw_to(t_RightAnkle, t_RightKnee)

            glPushMatrix()  # 4
            self.draw_to(t_RightHeel, t_NoJoint)
            glPopMatrix()   # 3

            self.draw_to(t_RightBallOfFoot, t_RightAnkle)
            self.draw_to(t_RightToeTip, t_RightBallOfFoot)
            glPopMatrix()   # 2

            glPushMatrix()  # 3
            self.draw_to(t_SpinePelvis, t_PelvisAnchor)
            self.draw_to(t_LowerVertebrae, t_SpinePelvis)
            self.draw_to(t_MidVertebrae, t_LowerVertebrae)

            glPushMatrix()  # 4
            self.draw_to(t_LeftShoulderBladeBase, t_MidVertebrae)
            self.draw_to(t_LeftShoulder, t_LeftShoulderBladeBase)
            self.draw_to(t_LeftElbow, t_LeftShoulder)
            self.draw_to(t_LeftWrist, t_LeftElbow)
            self.draw_to(t_LeftKnuckle, t_LeftWrist)
            self.draw_to(t_LeftFingerTip, t_LeftKnuckle)
            glPopMatrix()   # 3

            glPushMatrix()  # 4
            self.draw_to(t_RightShoulderBladeBase, t_MidVertebrae)
            self.draw_to(t_RightShoulder, t_RightShoulderBladeBase)
            self.draw_to(t_RightElbow, t_RightShoulder)
            self.draw_to(t_RightWrist, t_RightElbow)
            self.draw_to(t_RightKnuckle, t_RightWrist)
            self.draw_to(t_RightFingerTip, t_RightKnuckle)
            glPopMatrix()   # 3

            self.draw_to(t_UpperVertebrae, t_MidVertebrae)
            self.draw_to(t_BaseOfSkull, t_UpperVertebrae)
            self.draw_to(t_TopOfHead, t_BaseOfSkull)

            glPopMatrix()   # 2

            glPopMatrix()   # 1

            self.draw_joint_spheres(show_rotation_spheres)

            glTranslatef(self.multi_body_translation[0], self.multi_body_translation[1], self.multi_body_translation[2])

        glPopMatrix()   # 0
        glShadeModel(hold_shade)

    def update_quats(self, quats):
        if len(quats.shape) > 2:
            self.quaternions_np = quats
            self.num_bodies = quats.shape[0]
        else:
            self.quaternions_np = np.expand_dims(quats, axis=0)
            self.num_bodies = 1
        self.joint_matrices = quaternions_to_R3_rotation(self.quaternions_np)
        self.joint_quats = self.quaternions_np.copy()

    def move_to(self, jointIndex):
        if jointIndex != t_NoJoint:
            transform = self.default_gl_transform
            # linear_index = self.joint_mapper[jointIndex]
            linear_index = jointIndex
            if -1 < linear_index < t_ActiveJointCount:
                if self.joint_matrices is not None:
                    transform = self.joint_matrices[self.current_body, linear_index]
            self.joints[jointIndex].translate_along_bone()
            # glTranslatef(self.joints[jointIndex].bone_translation[0], self.joints[jointIndex].bone_translation[1], self.joints[jointIndex].bone_translation[2])
            glMultMatrixf(transform)

    def draw_to(self, joint_index, prev_limb_index=-1, orientation=False, show_disks=False):
        transform = self.default_gl_transform
        linear_index = -1
        if joint_index != t_NoJoint:
            linear_index = joint_index
            # linear_index = self.joint_mapper[joint_index]
            if -1 < linear_index < t_ActiveJointCount:
                if self.joint_matrices is not None:
                    transform = self.joint_matrices[self.current_body, linear_index]

        joint_data = self.joints[joint_index]

        m = joint_data.matrix

        if joint_data.do_draw:
            if orientation:
                # if show_disks and linear_index != -1:
                if show_disks:
                    self.show_orientation(joint_index, prev_limb_index)
            glPushMatrix()
            glMultMatrixf(m)
            glLineWidth(2.0)

            if not orientation:
                # self.draw_block(joint_index, (widths[0], length, widths[1]))
                self.draw_block(joint_index, joint_data)

            glPopMatrix()

        joint_data.translate_along_bone()

        # glTranslatef(joint_data.bone_translation[0], joint_data.bone_translation[1], joint_data.bone_translation[2])
        if orientation:
            if not show_disks:
                self.node.joint_callback(linear_index)
        glMultMatrixf(transform)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_material)

    def draw_block(self, joint_index, joint_data):  # draw_block could include colours for each end of the block to reflect
        if self.skeleton:
            dim_z = joint_data.dims[0]
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, dim_z)
            glEnd()
        else: #. only called if self.limbs[joint_index] is None!!!!!!
            if self.limbs[joint_index] is None:
                self.limbs[joint_index] = LimbGeometry()
                self.limbs[joint_index].new_shape = True
            if self.limbs[joint_index].new_shape:
                dim_z = joint_data.dims[0] - .02
                dim_x = joint_data.dims[1] / 2
                dim_y = joint_data.dims[2] / 2

                if self.limb_vertices[joint_index] is not None:
                    points = []
                    for vertex in self.limb_vertices[joint_index]:
                        points.append([vertex[0] * dim_x, vertex[1] * dim_y, vertex[2] * dim_z])
                    self.limbs[joint_index].set_points(points)

                else:
                    points = []
                    points.append((-dim_x, -dim_y, 0))
                    points.append((-dim_x, -dim_y, dim_z))
                    points.append((dim_x, -dim_y, 0))
                    points.append((dim_x, -dim_y, dim_z))
                    points.append((dim_x, dim_y, 0))
                    points.append((dim_x, dim_y, dim_z))
                    points.append((-dim_x, dim_y, 0))
                    points.append((-dim_x, dim_y, dim_z))
                    self.limbs[joint_index].set_points(points)
                # set colours - what would determine the colours?
                self.limbs[joint_index].calc_normals()

            self.limbs[joint_index].draw()

    def draw_joint_spheres(self, show_rotation_spheres):
        glPushMatrix()
        glScalef(scale, scale, 1.0)

        self.move_to(t_PelvisAnchor)

        glPushMatrix()
        self.draw_to(t_LeftHip, t_PelvisAnchor, True, show_rotation_spheres)
        self.draw_to(t_LeftKnee, t_LeftHip, True, show_rotation_spheres)
        self.draw_to(t_LeftAnkle, t_LeftKnee, True, show_rotation_spheres)

        glPushMatrix()
        self.draw_to(t_LeftHeel, t_NoJoint, True, show_rotation_spheres)
        glPopMatrix()

        self.draw_to(t_LeftBallOfFoot, t_LeftAnkle, True, show_rotation_spheres)
        self.draw_to(t_LeftToeTip, t_LeftBallOfFoot, True, show_rotation_spheres)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_RightHip, t_PelvisAnchor, True, show_rotation_spheres)
        self.draw_to(t_RightKnee, t_RightHip, True, show_rotation_spheres)
        self.draw_to(t_RightAnkle, t_RightKnee, True, show_rotation_spheres)

        glPushMatrix()
        self.draw_to(t_RightHeel, t_NoJoint, True, show_rotation_spheres)
        glPopMatrix()

        self.draw_to(t_RightBallOfFoot, t_RightAnkle, True, show_rotation_spheres)
        self.draw_to(t_RightToeTip, t_RightBallOfFoot, True, show_rotation_spheres)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_SpinePelvis, t_PelvisAnchor, True, show_rotation_spheres)
        self.draw_to(t_LowerVertebrae, t_SpinePelvis, True, show_rotation_spheres)
        self.draw_to(t_MidVertebrae, t_LowerVertebrae, True, show_rotation_spheres)

        glPushMatrix()
        self.draw_to(t_LeftShoulderBladeBase, t_MidVertebrae, True, show_rotation_spheres)
        self.draw_to(t_LeftShoulder, t_LeftShoulderBladeBase, True, show_rotation_spheres)
        self.draw_to(t_LeftElbow, t_LeftShoulder, True, show_rotation_spheres)
        self.draw_to(t_LeftWrist, t_LeftElbow, True, show_rotation_spheres)
        self.draw_to(t_LeftKnuckle, t_LeftWrist, True, show_rotation_spheres)
        self.draw_to(t_LeftFingerTip, t_LeftKnuckle, True, show_rotation_spheres)
        glPopMatrix()

        glPushMatrix()
        self.draw_to(t_RightShoulderBladeBase, t_MidVertebrae, True, show_rotation_spheres)
        self.draw_to(t_RightShoulder, t_RightShoulderBladeBase, True, show_rotation_spheres)
        self.draw_to(t_RightElbow, t_RightShoulder, True, show_rotation_spheres)
        self.draw_to(t_RightWrist, t_RightElbow, True, show_rotation_spheres)
        self.draw_to(t_RightKnuckle, t_RightWrist, True, show_rotation_spheres)
        self.draw_to(t_RightFingerTip, t_RightKnuckle, True, show_rotation_spheres)
        glPopMatrix()

        self.draw_to(t_UpperVertebrae, t_MidVertebrae, True, show_rotation_spheres)
        self.draw_to(t_BaseOfSkull, t_UpperVertebrae, True, show_rotation_spheres)
        self.draw_to(t_TopOfHead, t_BaseOfSkull, True, show_rotation_spheres)

        glPopMatrix()
        glPopMatrix()


    def create_limbs(self):
        standard_box = []
        standard_box.append((-.5, -.5, 0)) # [[-.5 -.5 0][-1 -1 .5][.5 -.5 0][1 -1 .5][.5 .5 0][1 1 .5][-.5 .5 0][-1 1 .5][-.5 -.5 1][.5 -.5 1][.5 .5 1][-.5 .5 1]]
        standard_box.append((-1.0, -1.0, 0.5))
        standard_box.append((.5, -.5, 0))
        standard_box.append((1.0, -1.0, 0.5))
        standard_box.append((.5, .5, 0))
        standard_box.append((1.0, 1.0, 0.5))
        standard_box.append((-.5, .5, 0))
        standard_box.append((-1.0, 1.0, .5))

        standard_box.append((-.5, -.5, 1.0))
        standard_box.append((.5, -.5, 1.0))
        standard_box.append((.5, .5, 1.0))
        standard_box.append((-.5, .5, 1.0))

        self.limb_vertices = []
        for i in range(37):
            self.limb_vertices.append(standard_box)

        self.limb_vertices[t_TopOfHead] = [[-0.7, -1.0, 0.0], [-1.3, -1.2, 0.5], [0.7, -1.0, 0.0], [1.3, -1.2, 0.5], [0.7, 0.4, 0.0], [1.3, 1.2, 0.5], [-0.7, 0.4, 0.0], [-1.3, 1.2, 0.5], [-0.8, -1.0, 1.0], [0.8, -1.0, 1.0], [0.8, 1.0, 1.0], [-0.8, 1.0, 1.0]]

        self.limb_vertices[t_BaseOfSkull] = self.define_limb_shape(1, .95, .9)
        self.limb_vertices[t_MidVertebrae] = self.define_limb_shape(.7, .8, .9)
        self.limb_vertices[t_LowerVertebrae] = self.define_limb_shape(.9, .9, .9)
        self.limb_vertices[t_SpinePelvis] = [[-0.2, -1.0, -1.5], [-1.0, -1.0, 0.0], [0.2, -1.0, -1.5], [1.0, -1.0, 0.0], [-0.2, 1.0, -1.0], [1.0, 1.0, 0.0], [-0.2, 1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]
        self.limb_vertices[t_RightKnee] = self.define_limb_shape(1.3, 1.2, .9)
        self.limb_vertices[t_LeftKnee] = self.define_limb_shape(1.3, 1.2, .9)
        self.limb_vertices[t_RightAnkle] = self.define_limb_shape(1.1, 1, .7)
        self.limb_vertices[t_LeftAnkle] = self.define_limb_shape(1.1, 1, .7)
        self.limb_vertices[t_RightElbow] = self.define_limb_shape(1., 1.1, .9)
        self.limb_vertices[t_LeftElbow] = self.define_limb_shape(1., 1.1, .9)
        self.limb_vertices[t_RightWrist] = self.define_limb_shape(1., 1, .7)
        self.limb_vertices[t_LeftWrist] = self.define_limb_shape(1., 1, .7)
        self.limb_vertices[t_RightHeel] = self.define_limb_shape(.8, .8, .8)
        self.limb_vertices[t_LeftHeel] = self.define_limb_shape(.8, .8, .8)

        self.limb_vertices[t_RightKnuckle] = self.define_limb_shape(.7, 1, .9)
        self.limb_vertices[t_LeftKnuckle] = self.define_limb_shape(.7, 1, .9)
        self.limb_vertices[t_RightFingerTip] = self.define_limb_shape(.9, .75, .6)
        self.limb_vertices[t_LeftFingerTip] = self.define_limb_shape(.9, .75, .6)
        self.limb_vertices[t_RightBallOfFoot] = self.define_limb_shape(.8, .9, 1.0)
        self.limb_vertices[t_LeftBallOfFoot] = self.define_limb_shape(.8, .9, 1.0)
        self.limb_vertices[t_RightToeTip] = self.define_limb_shape(1, .9, .6)
        self.limb_vertices[t_LeftToeTip] = self.define_limb_shape(1, .9, .6)

        self.limb_vertices[t_RightHip] = self.define_limb_shape(.7, 1.5, .7)
        self.limb_vertices[t_LeftHip] = self.define_limb_shape(.7, 1.5, .7)

        self.limb_vertices[t_LeftShoulderBladeBase] = [[-0.18, -0.9, 0.06], [-0.37, -0.97, 0.16], [0.91, -1.96, 0.31], [0.93, -1.96, 0.67], [1.27, 0.12, 0.29], [1.5, 0.38, 0.5], [-0.09, 0.18, -0.05], [0.34, 1.07, 0.21], [-0.85, -0.72, 0.91], [0.62, -1.29, 1.33], [1.3, 1.0, 1.26], [-0.56, 1.2, 1.11]]
        self.limb_vertices[t_RightShoulderBladeBase] = [[-0.91, -1.85, 0.3], [-0.93, -1.96, 0.67], [0.18, -0.9, 0.06], [0.37, -0.97, 0.16], [-0.09, 0.18, -0.06], [-0.34, 1.07, 0.21], [-1.27, -0.12, 0.29], [-1.49, 0.38, 0.5], [-0.62, -1.29, 1.33], [0.85, -0.72, 0.91], [0.56, 1.2, 1.11], [-1.33, 1.0, 1.26]]

    def __del__(self):
        if self.__mutex is not None:
            lock = ScopedLock(self.__mutex)
            self.__mutex = None
        for limb in self.limbs:
            delete(limb)

    def create_joints(self):
        for joint_index in joint_linear_index_to_name:
            if joint_index != -1:
                name = joint_linear_index_to_name[joint_index]
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
                if joint_name in joint_name_to_linear_index:
                    joint_index = joint_name_to_linear_index[joint_name]
                    self.joints[joint_index].set_bone_translation(y)

        # limb_sizes = {}
        for joint in self.joints:
            joint.set_matrix()
            joint.set_mass()

    def set_limb_dims(self, joint_index, dims):
        if 0 <= joint_index < len(self.joints):
            if len(dims) == 3:
                self.joints[joint_index].set_limb_length(dims[0])
                self.joints[joint_index].set_thickness([dims[1], dims[2]])
            elif len(dims) == 1:
                self.joints[joint_index].set_limb_length(dims[0])
            elif len(dims) == 2:
                self.joints[joint_index].set_thickness([dims[0], dims[1]])
            self.limbs[joint_index].new_shape = True

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

    def define_limb_shape(self, start_thickness, middle_thickness, end_thickness):
        shape = []
        shape.append((-start_thickness, -start_thickness, 0))
        shape.append((-middle_thickness, -middle_thickness, 0.5))
        shape.append((start_thickness, -start_thickness, 0))
        shape.append((middle_thickness, -middle_thickness, 0.5))
        shape.append((start_thickness, start_thickness, 0))
        shape.append((middle_thickness, middle_thickness, 0.5))
        shape.append((-start_thickness, start_thickness, 0))
        shape.append((-middle_thickness, middle_thickness, .5))

        shape.append((-end_thickness, -end_thickness, 1.0))
        shape.append((end_thickness, -end_thickness, 1.0))
        shape.append((end_thickness, end_thickness, 1.0))
        shape.append((-end_thickness, end_thickness, 1.0))

        return shape

    def transform_to_opengl(self, transform):
        if transform is not None and len(transform) == 16:
            # Transpose matrix for OpenGL column-major order.
            for i in range(0, 4):
                for j in range((i + 1), 4):
                    temp = transform[4 * i + j]
                    transform[4 * i + j] = transform[4 * j + i]
                    transform[4 * j + i] = temp
        return transform

    def show_orientation(self, joint_index, next_limb_index):
        glPushMatrix()
        joint_data = self.joints[joint_index]

        linear_index = next_limb_index
        if -1 < linear_index < t_ActiveJointCount:
            if self.joint_display == 'disk':
                self.draw_rotation_disk(linear_index, joint_data)
            else:
                self.draw_quaternion_distance_sphere(linear_index, joint_data)
        glPopMatrix()

    def draw_quaternion_distance_sphere(self, limb_index, joint_data):
        pass

    def draw_rotation_disk(self, limb_index, joint_data):
        pass

class BodyData(BodyDataBase):
    def __init__(self):
        from dpg_system.gl_nodes import GLMaterial
        super().__init__()
        self.origin = None
        self.diff_quats = None
        self.axes = None
        self.magnitudes = None
        self.normalized_axes = None
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
        self.quaternionDiff = []
        # self.joint_scales = [1.0] * 38
        self.calc_diff = False
        self.captured_quaternions = []
        self.label = 0
        self.joint_sphere = gluNewQuadric()
        self.joint_disk = gluNewQuadric()
        self.joint_motion_scale = 1.0
        self.joint_disk_material = GLMaterial()
        self.joint_disk_alpha = 0.5
        self.joint_disk_material.ambient = [0.19125, 0.0735, 0.0225, self.joint_disk_alpha]
        self.joint_disk_material.diffuse = [0.7038, 0.27048, 0.0828, self.joint_disk_alpha]
        self.joint_disk_material.specular = [0.256777, 0.137622, 0.086014, self.joint_disk_alpha]
        self.joint_disk_material.shininess = 12.8

        self.smoothed_quaternions_a = np.zeros((1, 20, 4), dtype=float)
        self.smoothed_quaternions_b = np.zeros((1, 20, 4), dtype=float)
        self.current_body = 0
        self.capture_next_pose = False
        self.has_captured_pose = False
        self.distance_sense = 1.0
        self.error_band = 0.00
        self.pose_similarity = 0
        self.input_vector = np.zeros(80, dtype=float)

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
            self.quaternionDiff.append([0.0, 0.0, 0.0, 0.0])
            self.diffQuaternionAbsSum.append(0.0)

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

    def update_quats(self, quats):
        super().update_quats(quats)
        if self.calc_diff:
            self.calc_diff_quaternions()

    def update(self, joint_index, quat, position=None, label=0, paused=False):
        if position is not None:
            if joint_index == 4 and self.origin is None:
                self.origin = [position[0], position[1], position[2]]
            self.positions[joint_index] = [position[0], position[1], position[2]]

        if not paused:
            self.previousQuats[joint_index] = self.diffQuaternionA[joint_index].copy()
        self.quaternions[joint_index] = [quat[0], quat[1], quat[2], quat[3]]
        # if not paused:
        #     self.calc_diff_quaternion(joint_index)

        q1 = Quaternion(self.previousQuats[joint_index])
        q2 = Quaternion(self.diffQuaternionA[joint_index])

        # quaternion smooothing -> slerp on quat and prev or slerp on distance
        # use difference of gaussians trick to get smooth sense of jerk and flex
        if not paused:
            self.quaternionDistance[joint_index] = Quaternion.sym_distance(q1.unit, q2.unit)
            self.calc_diff_quaternion(joint_index)

        self.quaternionDiff[joint_index] = q2 - q1
        self.rotationAxis[joint_index] = list(self.quaternionDiff[joint_index].unit.axis)
        self.set_label(label - 1)

    def quaternion_distances(self, q1, q2):
        q1_ = q1 / np.expand_dims(np.linalg.norm(q1, axis=-1), -1)
        q2_ = q2 / np.expand_dims(np.linalg.norm(q2, axis=-1), -1)
        diff = np.sum(np.multiply(q1_[:, :], q2_[:, :]), axis=2)
        diff = np.clip(diff, a_min=-1, a_max=1)
        distances = np.arccos(2 * diff[:, :] * diff[:, :] - 1)
        return distances

    def calc_diff_quaternions(self):
        self.smoothed_quaternions_a = self.smoothed_quaternions_a * self.diffQuatSmoothingA + self.quaternions_np * (1.0 - self.diffQuatSmoothingA)
        self.smoothed_quaternions_b = self.smoothed_quaternions_b * self.diffQuatSmoothingB + self.quaternions_np * (1.0 - self.diffQuatSmoothingB)
        # we want axis, magnitude from diff_quat

        a = quaternion.as_quat_array(self.smoothed_quaternions_a)
        b = quaternion.as_quat_array(self.smoothed_quaternions_b)
        self.diff_quats = a - b
        self.axes = quaternion.as_rotation_vector(self.diff_quats)
        angles = np.linalg.norm(self.axes, axis=2)
        self.magnitudes = self.quaternion_distances(self.smoothed_quaternions_a, self.smoothed_quaternions_b)
        self.normalized_axes = self.axes / np.expand_dims(angles, axis=-1)

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

    def adjust_clear_colour(self):
        if self.has_captured_pose:
            r = self.pose_similarity * self.color[1][0] + (1 - self.pose_similarity) * self.color[0][0]
            g = self.pose_similarity * self.color[1][1] + (1 - self.pose_similarity) * self.color[0][1]
            b = self.pose_similarity * self.color[1][2] + (1 - self.pose_similarity) * self.color[0][2]
            glClearColor(r, g, b, 1.0)
        else:
            glClearColor(0, 0, 0, 0)

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

    def draw_quaternion_distance_sphere(self, limb_index, joint_data):
        if limb_index != -1 and self.diff_quats is not None:
            up_vector = np.array([0.0, 0.0, 1.0])
            d = self.magnitudes[self.current_body, limb_index] * self.joint_motion_scale   #self.quaternion_distance_display_scale
            if d > 0.00001:
                axis = self.normalized_axes[self.current_body, limb_index]
    #            weight = joint_data.mass[0] * abs(axis[0]) + joint_data.mass[2] * abs(axis[2]) + joint_data.mass[1] * abs(
    #                axis[1])
                self.joint_disk_material.diffuse[3] = self.joint_disk_alpha
                # set colour by orientation
                # blue is axial rotation
                # red is forward / backward
                # yellow is side to side
                # use axis
                # axis_colour = np.array(axis)
                # # axis_colour /= axis_colour.sum()
                # self.joint_disk_material.diffuse[0] = axis_colour[0] + 1 / 2.0
                # self.joint_disk_material.diffuse[1] = axis_colour[1] + 1 / 2.0
                # self.joint_disk_material.diffuse[2] = axis_colour[2] + 1 / 2.0
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.joint_disk_material.diffuse)

                # v = np.cross(axis, up_vector)
                # c = np.dot(axis, up_vector)
                # k = 1.0 / (1.0 + c)
                #
                # alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                #                              v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                #                              v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                #                              0.0, 0.0, 0.0, 1.0])
                #
                # alignment_matrix.reshape((4, 4))
                # restore_matrix = glGetInteger(GL_MATRIX_MODE)
                # glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                # glMultMatrixf(alignment_matrix)
                gluSphere(self.joint_sphere, d, 32, 32)
                # gluDisk(self.joint_sphere, 0.0, d, 32, 1)
                glPopMatrix()
                # glMatrixMode(restore_matrix)

    def draw_rotation_disk(self, limb_index, joint_data):
        if limb_index != -1 and self.diff_quats is not None:
            up_vector = np.array([0.0, 0.0, 1.0])
            d = self.magnitudes[self.current_body, limb_index] * self.joint_motion_scale   #self.quaternion_distance_display_scale
            if d > 0.00001:
                axis = self.normalized_axes[self.current_body, limb_index]
    #            weight = joint_data.mass[0] * abs(axis[0]) + joint_data.mass[2] * abs(axis[2]) + joint_data.mass[1] * abs(
    #                axis[1])
                self.joint_disk_material.diffuse[3] = self.joint_disk_alpha
                # set colour by orientation
                # blue is axial rotation
                # red is forward / backward
                # yellow is side to side
                # use axis
                # axis_colour = np.array(axis)
                # # axis_colour /= axis_colour.sum()
                # self.joint_disk_material.diffuse[0] = axis_colour[0] + 1 / 2.0
                # self.joint_disk_material.diffuse[1] = axis_colour[1] + 1 / 2.0
                # self.joint_disk_material.diffuse[2] = axis_colour[2] + 1 / 2.0
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
                # gluSphere(self.joint_sphere, d, 32, 32)
                gluDisk(self.joint_sphere, 0.0, d, 32, 1)
                glPopMatrix()
                glMatrixMode(restore_matrix)

class LimbGeometry:
    def __init__(self):
        self.points = []
        self.normals = [None] * 6
        self.list_index = -1
        self.new_shape = False
        self.dims = np.array([1.0, 1.0, 1.0])
        self.offset = np.array([0.0, 0.0, 0.0])

    def set_points(self, points):
        self.points = np.ndarray([len(points), 3])
        for index, point in enumerate(points):
            self.points[index] = point
        # if self.list_index != -1:
        #     glDeleteLists(self.list_index, 1)
        self.new_shape = True

    def calc_normals(self):
        if len(self.points) == 8:
            self.normals = np.ndarray([6, 3])
            self.normals[0] = self.calc_normal(self.points[0], self.points[1], self.points[2])
            self.normals[1] = self.calc_normal(self.points[2], self.points[3], self.points[4])
            self.normals[2] = self.calc_normal(self.points[4], self.points[5], self.points[6])
            self.normals[3] = self.calc_normal(self.points[6], self.points[7], self.points[0])
            self.normals[4] = self.calc_normal(self.points[3], self.points[5], self.points[7])
            self.normals[5] = self.calc_normal(self.points[2], self.points[4], self.points[6])
        else:
            self.normals = np.ndarray([10, 3])
            self.normals[0] = self.calc_normal(self.points[0], self.points[1], self.points[2])
            self.normals[1] = self.calc_normal(self.points[2], self.points[3], self.points[4])
            self.normals[2] = self.calc_normal(self.points[4], self.points[5], self.points[6])
            self.normals[3] = self.calc_normal(self.points[6], self.points[7], self.points[0])

            self.normals[4] = self.calc_normal(self.points[1], self.points[8], self.points[3])
            self.normals[5] = self.calc_normal(self.points[3], self.points[9], self.points[5])
            self.normals[6] = self.calc_normal(self.points[5], self.points[10], self.points[7])
            self.normals[7] = self.calc_normal(self.points[7], self.points[11], self.points[1])

            self.normals[8] = self.calc_normal(self.points[9], self.points[10], self.points[11])
            self.normals[9] = self.calc_normal(self.points[2], self.points[4], self.points[6])

    def draw(self):  #  limb structure could include colors and if so, do not use call_list... do vertex colors and manipulate GL_COLOR_MATERIAL
        if self.new_shape:
            self.new_shape = False
            if self.list_index != -1:
                glDeleteLists(self.list_index, 1)
            self.list_index = -1

        glTranslate(self.offset[0], self.offset[1], self.offset[2])

        if self.list_index == -1:
            self.list_index = glGenLists(1)
            glNewList(self.list_index, GL_COMPILE)

            glBegin(GL_TRIANGLE_STRIP)

            if len(self.points) == 8:

                glNormal3fv(self.normals[0])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[2] * self.dims)
                glVertex3fv(self.points[3] * self.dims)

                glNormal3fv(self.normals[1])
                glVertex3fv(self.points[4] * self.dims)
                glVertex3fv(self.points[5] * self.dims)

                glNormal3fv(self.normals[2])
                glVertex3fv(self.points[6] * self.dims)
                glVertex3fv(self.points[7] * self.dims)

                glNormal3fv(self.normals[3])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[1] * self.dims)

                glNormal3fv(self.normals[4])
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[3] * self.dims)
                glVertex3fv(self.points[7] * self.dims)
                glVertex3fv(self.points[5] * self.dims)

                glVertex3fv(self.points[5] * self.dims)
                glVertex3fv(self.points[0] * self.dims)

                glNormal3fv(self.normals[5])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[2] * self.dims)
                glVertex3fv(self.points[6] * self.dims)
                glVertex3fv(self.points[4] * self.dims)
            else:
                glNormal3fv(self.normals[0])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[2] * self.dims)
                glVertex3fv(self.points[3] * self.dims)

                glNormal3fv(self.normals[1])
                glVertex3fv(self.points[4] * self.dims)
                glVertex3fv(self.points[5] * self.dims)

                glNormal3fv(self.normals[2])
                glVertex3fv(self.points[6] * self.dims)
                glVertex3fv(self.points[7] * self.dims)

                glNormal3fv(self.normals[3])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[1] * self.dims)

                glNormal3fv(self.normals[4])
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[8] * self.dims)
                glVertex3fv(self.points[3] * self.dims)
                glVertex3fv(self.points[9] * self.dims)

                glNormal3fv(self.normals[5])
                glVertex3fv(self.points[5] * self.dims)
                glVertex3fv(self.points[10] * self.dims)

                glNormal3fv(self.normals[6])
                glVertex3fv(self.points[7] * self.dims)
                glVertex3fv(self.points[11] * self.dims)

                glNormal3fv(self.normals[7])
                glVertex3fv(self.points[1] * self.dims)
                glVertex3fv(self.points[8] * self.dims)
                glVertex3fv(self.points[8] * self.dims)
                glVertex3fv(self.points[8] * self.dims)

                glNormal3fv(self.normals[8])
                glVertex3fv(self.points[8] * self.dims)
                glVertex3fv(self.points[9] * self.dims)
                glVertex3fv(self.points[11] * self.dims)
                glVertex3fv(self.points[10] * self.dims)

                glVertex3fv(self.points[10])
                glVertex3fv(self.points[0] * self.dims)

                glNormal3fv(self.normals[9])
                glVertex3fv(self.points[0] * self.dims)
                glVertex3fv(self.points[2] * self.dims)
                glVertex3fv(self.points[6] * self.dims)
                glVertex3fv(self.points[4] * self.dims)
            glEnd()
            glEndList()
        if self.list_index != -1:
            glCallList(self.list_index)

    def calc_normal(self, v1, v2, v3):
        v_1 = np.array([v1[0], v1[1], v1[2]])
        v_2 = np.array([v2[0], v2[1], v2[2]])
        v_3 = np.array([v3[0], v3[1], v3[2]])
        temp1 = v_2 - v_1
        temp2 = v_3 - v_1
        normal = np.cross(temp1, temp2)
        if normal.any() == 0:
            for i in range(3):
                if normal[i] == 0:
                    normal[i] = 1e-6
        dist = np.linalg.norm(normal)
        normal = normal / dist
        return (normal[0], normal[1], normal[2])

    def __del__(self):
        if self.list_index != -1:
            glDeleteLists(self.list_index, 1)


class SimpleBodyData(BodyDataBase):
    def __init__(self):
        super().__init__()


class AlternateBodyData(BodyDataBase):
    def __init__(self):
        self.shape = 'humanoid'
        self.humanoid_bone_dims = np.zeros([37,3])
        super().__init__()

    def draw(self, show_rotation_spheres=False, skeleton=False):
        hold_shade = glGetInteger(GL_SHADE_MODEL)
        glShadeModel(GL_FLAT)
        glPushMatrix()      # 0
        glScalef(scale, scale, 1.0)
        glTranslatef(0.0, -1.0, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_material)

        self.skeleton = skeleton

        # transform = None

        for i in range(self.num_bodies):
            self.current_body = i

            glPushMatrix()  # 1
            self.move_to(t_PelvisAnchor)
            self.draw_heirarchy(t_PelvisAnchor)
            glPopMatrix()   # 1

            # self.draw_joint_spheres(show_rotation_spheres)
            glTranslatef(self.multi_body_translation[0], self.multi_body_translation[1], self.multi_body_translation[2])

        glPopMatrix()   # 0
        glShadeModel(hold_shade)

    def draw_heirarchy(self, root_joint):
        joint = self.joints[root_joint]
        connected_joints = joint.immed_children
        child_count = len(connected_joints)
        for sub_joint in connected_joints:
            # if child_count > 1:
            glPushMatrix()
            self.draw_to(sub_joint, root_joint)
            self.draw_heirarchy(sub_joint)
            # if child_count > 1:
            glPopMatrix()

    def create_joints(self):
        for joint_index in joint_index_to_name:
            name = joint_index_to_name[joint_index]
            new_joint = BaseJoint(self, name, joint_index)
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
                self.humanoid_bone_dims[joint_index] = np.array(y)
                self.joints[joint_index].set_bone_translation(y)

        self.connect_limbs()
        for joint in self.joints:
            joint.set_matrix()
            joint.set_mass()

    def create_limbs(self, start=0.75, mid=1.0, end=0.75):
        standard_box = []
        standard_box.append((-.5, -.5, 0)) # [[-.5 -.5 0][-1 -1 .5][.5 -.5 0][1 -1 .5][.5 .5 0][1 1 .5][-.5 .5 0][-1 1 .5][-.5 -.5 1][.5 -.5 1][.5 .5 1][-.5 .5 1]]
        standard_box.append((-1.0, -1.0, 0.5))
        standard_box.append((.5, -.5, 0))
        standard_box.append((1.0, -1.0, 0.5))
        standard_box.append((.5, .5, 0))
        standard_box.append((1.0, 1.0, 0.5))
        standard_box.append((-.5, .5, 0))
        standard_box.append((-1.0, 1.0, .5))

        standard_box.append((-.5, -.5, 1.0))
        standard_box.append((.5, -.5, 1.0))
        standard_box.append((.5, .5, 1.0))
        standard_box.append((-.5, .5, 1.0))

        self.limb_vertices = []
        for i in range(37):
            self.limb_vertices.append(standard_box)

        self.limb_vertices[t_TopOfHead] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_BaseOfSkull] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_UpperVertebrae] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_MidVertebrae] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LowerVertebrae] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_SpinePelvis] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightKnee] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftKnee] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightAnkle] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftAnkle] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightShoulder] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftShoulder] = self.define_limb_shape(start, mid, end)

        self.limb_vertices[t_RightElbow] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftElbow] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightWrist] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftWrist] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightHeel] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftHeel] = self.define_limb_shape(start, mid, end)

        self.limb_vertices[t_RightKnuckle] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftKnuckle] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightFingerTip] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftFingerTip] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightBallOfFoot] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftBallOfFoot] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightToeTip] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftToeTip] = self.define_limb_shape(start, mid, end)

        self.limb_vertices[t_RightHip] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_LeftHip] = self.define_limb_shape(start, mid, end)

        self.limb_vertices[t_LeftShoulderBladeBase] = self.define_limb_shape(start, mid, end)
        self.limb_vertices[t_RightShoulderBladeBase] = self.define_limb_shape(start, mid, end)

    def connect_limbs(self):
        if self.shape == 'linear':
            self.connect_linear()
        elif self.shape == 'humanoid':
            self.connect_humanoid()
        elif self.shape == 'medusa':
            self.connect_medusa()
        elif self.shape == 'bird':
            self.connect_bird()
        elif self.shape == 'branch':
            self.connect_branch()
        elif self.shape == 'cocoon':
            self.connect_cocoon()
        elif self.shape == 'moth':
            self.connect_moth()

    def connect_linear(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for joint in self.joints:
            joint.set_thickness([.05, .05])
            joint.immed_children = []
            joint.set_bone_translation((0.3, 0.0, 0.0))
            joint.set_limb_vector((0.0, 0.0, 1.0))
            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_PelvisAnchor].immed_children = [t_SpinePelvis]
        self.joints[t_SpinePelvis].immed_children = [t_LowerVertebrae]
        self.joints[t_LowerVertebrae].immed_children = [t_MidVertebrae]
        self.joints[t_MidVertebrae].immed_children = [t_UpperVertebrae]
        self.joints[t_UpperVertebrae].immed_children = [t_BaseOfSkull]
        self.joints[t_BaseOfSkull].immed_children = [t_TopOfHead]
        self.joints[t_TopOfHead].immed_children = [t_RightHip]
        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftHip]
        self.joints[t_LeftHip].immed_children = [t_LeftShoulderBladeBase]
        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_RightWrist]
        self.joints[t_RightWrist].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightShoulderBladeBase]
        self.joints[t_RightShoulder].immed_children = []

    def connect_branch(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for joint in self.joints:
            joint.immed_children = []
            joint.set_bone_translation((0.15, 0.00, 0.0))
            joint.set_limb_vector((0.00001, .00001, 1.0))
            if joint.joint_index == t_SpinePelvis:
                joint.set_thickness([.075, .075])
            elif joint.joint_index == t_LowerVertebrae:
                joint.set_thickness([.07, .07])
            elif joint.joint_index == t_MidVertebrae:
                joint.set_thickness([.065, .065])
            elif joint.joint_index == t_UpperVertebrae:
                joint.set_thickness([.06, .06])
            elif joint.joint_index == t_BaseOfSkull:
                joint.set_thickness([.055, .055])
            elif joint.joint_index == t_TopOfHead:
                joint.set_thickness([.05, .05])
            elif joint.joint_index == t_RightHip:
                joint.set_thickness([.045, .045])
            elif joint.joint_index == t_RightKnee:
                joint.set_thickness([.035, .035])
            elif joint.joint_index == t_RightAnkle:
                joint.set_thickness([.025, .025])
            elif joint.joint_index == t_LeftAnkle:
                joint.set_thickness([.015, .015])
            elif joint.joint_index == t_LeftKnee:
                joint.set_thickness([.01, .01])
            elif joint.joint_index == t_LeftHip:
                joint.set_thickness([.005, .005])
            elif joint.joint_index == t_RightWrist:
                joint.set_thickness([.045, .045])
            elif joint.joint_index == t_RightElbow:
                joint.set_thickness([.04, .04])
            elif joint.joint_index == t_RightShoulder:
                joint.set_thickness([.035, .035])
            elif joint.joint_index == t_RightShoulderBladeBase:
                joint.set_thickness([.03, .03])
            elif joint.joint_index == t_LeftShoulderBladeBase:
                joint.set_thickness([.025, .025])
            elif joint.joint_index == t_LeftShoulder:
                joint.set_thickness([.02, .02])
            elif joint.joint_index == t_LeftElbow:
                joint.set_thickness([.015, .015])
            elif joint.joint_index == t_LeftWrist:
                joint.set_thickness([.01, .01])
            elif joint.joint_index == t_LeftKnuckle:
                joint.set_thickness([.005, .005])

            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_PelvisAnchor].immed_children = [t_SpinePelvis]
        self.joints[t_SpinePelvis].immed_children = [t_LowerVertebrae]
        self.joints[t_LowerVertebrae].immed_children = [t_MidVertebrae]
        self.joints[t_MidVertebrae].immed_children = [t_UpperVertebrae]
        self.joints[t_UpperVertebrae].immed_children = [t_BaseOfSkull]
        self.joints[t_BaseOfSkull].immed_children = [t_TopOfHead]
        self.joints[t_TopOfHead].immed_children = [t_RightHip, t_RightWrist]
        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftHip]
        self.joints[t_RightWrist].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightShoulderBladeBase]
        self.joints[t_RightShoulderBladeBase].immed_children = [t_LeftShoulderBladeBase]
        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_LeftKnuckle]

    def connect_humanoid(self):
        self.create_limbs()
        for index, joint in enumerate(self.joints):
            joint.set_thickness([.05, .05])
            joint.immed_children = []
            joint.set_bone_translation(self.humanoid_bone_dims[index])
            joint.ref_vector = np.array([0.0, 0.0, 1.0])
            if index in [t_Body, t_PelvisAnchor, t_Reference, t_Tracker0, t_Tracker1, t_Tracker2,
                                    t_Tracker3]:
                joint.ref_vector = np.array([1.0, 0.0, 0.0])
            elif index in [t_LeftHeel, t_RightHeel]:
                joint.ref_vector = np.array([0.0, 1.0, 0.0])
            elif index in [t_LeftToeTip, t_RightToeTip]:
                joint.ref_vector = np.array([0.0, 0.0001, 1.0])
            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_PelvisAnchor].immed_children = [t_SpinePelvis, t_LeftHip, t_RightHip]
        self.joints[t_SpinePelvis].immed_children = [t_LowerVertebrae]
        self.joints[t_LowerVertebrae].immed_children = [t_MidVertebrae]
        self.joints[t_MidVertebrae].immed_children = [t_UpperVertebrae, t_LeftShoulderBladeBase, t_RightShoulderBladeBase]
        self.joints[t_UpperVertebrae].immed_children = [t_BaseOfSkull]
        self.joints[t_BaseOfSkull].immed_children = [t_TopOfHead]
        self.joints[t_TopOfHead].immed_children = []
        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_RightBallOfFoot]
        self.joints[t_RightBallOfFoot].immed_children = [t_RightToeTip]
        self.joints[t_RightToeTip].immed_children = []

        self.joints[t_LeftHip].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftBallOfFoot]
        self.joints[t_LeftBallOfFoot].immed_children = [t_LeftToeTip]
        self.joints[t_LeftToeTip].immed_children = []

        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_LeftKnuckle]
        self.joints[t_LeftKnuckle].immed_children = [t_LeftFingerTip]
        self.joints[t_LeftFingerTip].immed_children = []

        self.joints[t_RightShoulderBladeBase].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightWrist]
        self.joints[t_RightWrist].immed_children = [t_RightKnuckle]
        self.joints[t_RightKnuckle].immed_children = [t_RightFingerTip]
        self.joints[t_RightFingerTip].immed_children = []

    def connect_medusa(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for joint in self.joints:
            joint.set_thickness([.05, .05])
            joint.immed_children = []
            joint.set_bone_translation((0.3, 0.0, 0.0))
            joint.set_limb_vector((0.0, 0.0, 1.0))
            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_PelvisAnchor].immed_children = [
            t_SpinePelvis, t_LeftHip, t_RightHip, t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_LeftWrist
        ]
        self.joints[t_SpinePelvis].immed_children = [t_LowerVertebrae]
        self.joints[t_LowerVertebrae].immed_children = [t_MidVertebrae]
        self.joints[t_MidVertebrae].immed_children = [t_UpperVertebrae]

        self.joints[t_LeftHip].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftBallOfFoot]

        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_RightBallOfFoot]

        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]

        self.joints[t_RightShoulderBladeBase].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightWrist]

    def connect_bird(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for index, joint in enumerate(self.joints):
            joint.set_thickness([.05, .05])

            joint.immed_children = []
            if index in [t_Body, t_PelvisAnchor, t_Reference, t_Tracker0, t_Tracker1, t_Tracker2,
                                    t_Tracker3]:
                joint.ref_vector = np.array([1.0, 0.0, 0.0])
            elif index == t_LeftShoulderBladeBase:
                joint.set_bone_translation((0.075, 0.0, 0.0))  # bone_dim is key!!!!
                joint.ref_vector = np.array([0.0, 0.0, 1.0])
                joint.set_thickness([.1, .03])
            elif index == t_RightShoulderBladeBase:
                joint.set_bone_translation((-0.075, 0.0, 0.0))
                joint.ref_vector = np.array([0.0, 0.0, 1.0])
                joint.set_thickness([.1, .03])
            elif index in [t_SpinePelvis, t_LowerVertebrae]:
                joint.set_bone_translation((0.0, 0.0, -0.2))
                joint.ref_vector = np.array([0.000001, 0.0, 1.0])
            elif index in [t_MidVertebrae, t_UpperVertebrae]:
                joint.set_bone_translation((0.0, 0.0, 0.2))
                joint.ref_vector = np.array([0.000001, 0.0, 1.0])
            elif index in [t_RightHip, t_RightKnee, t_RightAnkle]:
                joint.set_bone_translation((-0.075, 0.0, 0.0))
                joint.ref_vector = np.array([0.000001, 0.000001, 1.0])
                joint.set_thickness([.1, .03])
            elif index in [t_RightWrist, t_RightElbow, t_RightShoulder]:
                joint.set_bone_translation((-0.075, 0.0, 0.0))
                joint.ref_vector = np.array([0.0, 0.0, 1.0])
                joint.set_thickness([.1, .03])
            else:
                joint.set_bone_translation((0.075, 0.0, 0.0))
                joint.ref_vector = np.array([0.0, 0.0, 1.0])
                joint.set_thickness([.1, .03])

            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None
        # self.joints[t_LeftHip].do_draw = True
        # self.joints[t_RightHip].do_draw = True

        self.limb_vertices[t_MidVertebrae] = self.define_limb_shape(1.0, .7, .1)
        self.limb_vertices[t_SpinePelvis] = self.define_limb_shape(1.0, .7, .1)


        self.joints[t_PelvisAnchor].immed_children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_SpinePelvis, t_MidVertebrae]

        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_LeftHip]
        self.joints[t_LeftHip].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftAnkle]

        self.joints[t_RightShoulderBladeBase].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightWrist]
        self.joints[t_RightWrist].immed_children = [t_RightHip]
        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]


        # self.joints[t_SpinePelvis].immed_children = [t_LowerVertebrae]
        # self.joints[t_MidVertebrae].immed_children = [t_UpperVertebrae]

    def connect_cocoon(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for joint in self.joints:
            joint.set_thickness([.05, .05])

            joint.immed_children = []
            joint.set_bone_translation((0.3, 0.0, 0.0))
            joint.set_limb_vector((0.0, 0.0, 1.0))
            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_RightShoulderBladeBase].set_bone_translation((0.1, 0.1, 0.0))
        self.joints[t_RightShoulderBladeBase].set_matrix()
        self.joints[t_RightShoulder].set_bone_translation((0.2, 0.08, 0.0))
        self.joints[t_RightShoulder].set_matrix()
        self.joints[t_RightElbow].set_bone_translation((0.2, 0.0, 0.0))
        self.joints[t_RightElbow].set_matrix()
        self.joints[t_RightWrist].set_bone_translation((0.2, -0.08, 0.0))
        self.joints[t_RightWrist].set_matrix()
        self.joints[t_RightKnuckle].set_bone_translation((0.1, -0.1, 0.0))
        self.joints[t_RightKnuckle].set_matrix()

        self.joints[t_LeftShoulderBladeBase].set_bone_translation((0.1, -0.1, 0.0))
        self.joints[t_LeftShoulderBladeBase].set_matrix()
        self.joints[t_LeftShoulder].set_bone_translation((0.2, -0.08, 0.0))
        self.joints[t_LeftShoulder].set_matrix()
        self.joints[t_LeftElbow].set_bone_translation((0.2, 0.0, 0.0))
        self.joints[t_LeftElbow].set_matrix()
        self.joints[t_LeftWrist].set_bone_translation((0.2, 0.08, 0.0))
        self.joints[t_LeftWrist].set_matrix()
        self.joints[t_LeftKnuckle].set_bone_translation((0.1, 0.1, 0.0))
        self.joints[t_LeftKnuckle].set_matrix()

        self.joints[t_RightHip].set_bone_translation((0.1, 0.0, 0.1))
        self.joints[t_RightHip].set_matrix()
        self.joints[t_RightKnee].set_bone_translation((0.2, 0.0, 0.08))
        self.joints[t_RightKnee].set_matrix()
        self.joints[t_RightAnkle].set_bone_translation((0.2, 0.0, 0.0))
        self.joints[t_RightAnkle].set_matrix()
        self.joints[t_RightBallOfFoot].set_bone_translation((0.2, 0.0, -0.08))
        self.joints[t_RightBallOfFoot].set_matrix()

        self.joints[t_LeftHip].set_bone_translation((0.1, 0.0, -0.1))
        self.joints[t_LeftHip].set_matrix()
        self.joints[t_LeftKnee].set_bone_translation((0.2, 0.0, -0.08))
        self.joints[t_LeftKnee].set_matrix()
        self.joints[t_LeftAnkle].set_bone_translation((0.2, 0.0, 0.00))
        self.joints[t_LeftAnkle].set_matrix()
        self.joints[t_LeftBallOfFoot].set_bone_translation((0.2, 0.0, 0.08))
        self.joints[t_LeftBallOfFoot].set_matrix()

        self.joints[t_PelvisAnchor].immed_children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_LeftHip, t_RightHip]

        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_LeftKnuckle]

        self.joints[t_RightShoulderBladeBase].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightWrist]
        self.joints[t_RightWrist].immed_children = [t_RightKnuckle]

        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_RightBallOfFoot]

        self.joints[t_LeftHip].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftBallOfFoot]


    def connect_moth(self):
        self.create_limbs(1.0, 1.0, 1.0)
        for joint in self.joints:
            joint.set_thickness([.025, .025])

            joint.immed_children = []
            joint.set_bone_translation((0.3, 0.0, 0.0))
            joint.set_limb_vector((0.0, 0.0, 1.0))
            joint.set_matrix()
            joint.set_mass()

        for index, limb in enumerate(self.limbs):
            self.limbs[index] = None

        self.joints[t_RightShoulderBladeBase].set_bone_translation((0.04, 0.04, 0.0))
        self.joints[t_RightShoulderBladeBase].set_matrix()
        self.joints[t_RightShoulder].set_bone_translation((0.4, 0.4, 0.0))
        self.joints[t_RightShoulder].set_matrix()
        self.joints[t_RightElbow].set_bone_translation((0.2, -0.4, 0.0))
        self.joints[t_RightElbow].set_matrix()
        self.joints[t_RightWrist].set_bone_translation((-0.4, 0.0, 0.0))
        self.joints[t_RightWrist].set_matrix()
        self.joints[t_RightKnuckle].set_bone_translation((-0.2, 0.0, 0.0))
        self.joints[t_RightKnuckle].set_matrix()

        self.joints[t_LeftShoulderBladeBase].set_bone_translation((-0.04, 0.04, 0.0))
        self.joints[t_LeftShoulderBladeBase].set_matrix()
        self.joints[t_LeftShoulder].set_bone_translation((-0.4, 0.4, 0.0))
        self.joints[t_LeftShoulder].set_matrix()
        self.joints[t_LeftElbow].set_bone_translation((-0.2, -0.4, 0.0))
        self.joints[t_LeftElbow].set_matrix()
        self.joints[t_LeftWrist].set_bone_translation((0.4, 0.0, 0.0))
        self.joints[t_LeftWrist].set_matrix()
        self.joints[t_LeftKnuckle].set_bone_translation((0.2, 0.0, 0.0))
        self.joints[t_LeftKnuckle].set_matrix()

        self.joints[t_RightHip].set_bone_translation((0.04, -0.04, 0.0))
        self.joints[t_RightHip].set_matrix()
        self.joints[t_RightKnee].set_bone_translation((0.4, -0.1, 0.0))
        self.joints[t_RightKnee].set_matrix()
        self.joints[t_RightAnkle].set_bone_translation((-0.2, -0.24, 0.0))
        self.joints[t_RightAnkle].set_matrix()
        self.joints[t_RightBallOfFoot].set_bone_translation((-0.24, 0.38, 0.0))
        self.joints[t_RightBallOfFoot].set_matrix()

        self.joints[t_LeftHip].set_bone_translation((-0.04, -0.04, 0.0))
        self.joints[t_LeftHip].set_matrix()
        self.joints[t_LeftKnee].set_bone_translation((-0.4, -0.1, 0.0))
        self.joints[t_LeftKnee].set_matrix()
        self.joints[t_LeftAnkle].set_bone_translation((0.2, -0.24, 0.0))
        self.joints[t_LeftAnkle].set_matrix()
        self.joints[t_LeftBallOfFoot].set_bone_translation((0.24, 0.38, 0.0))
        self.joints[t_LeftBallOfFoot].set_matrix()

        self.joints[t_PelvisAnchor].immed_children = [t_LeftShoulderBladeBase, t_RightShoulderBladeBase, t_RightHip, t_LeftHip]

        self.joints[t_LeftShoulderBladeBase].immed_children = [t_LeftShoulder]
        self.joints[t_LeftShoulder].immed_children = [t_LeftElbow]
        self.joints[t_LeftElbow].immed_children = [t_LeftWrist]
        self.joints[t_LeftWrist].immed_children = [t_LeftKnuckle]

        self.joints[t_RightShoulderBladeBase].immed_children = [t_RightShoulder]
        self.joints[t_RightShoulder].immed_children = [t_RightElbow]
        self.joints[t_RightElbow].immed_children = [t_RightWrist]
        self.joints[t_RightWrist].immed_children = [t_RightKnuckle]

        self.joints[t_RightHip].immed_children = [t_RightKnee]
        self.joints[t_RightKnee].immed_children = [t_RightAnkle]
        self.joints[t_RightAnkle].immed_children = [t_RightBallOfFoot]

        self.joints[t_LeftHip].immed_children = [t_LeftKnee]
        self.joints[t_LeftKnee].immed_children = [t_LeftAnkle]
        self.joints[t_LeftAnkle].immed_children = [t_LeftBallOfFoot]

