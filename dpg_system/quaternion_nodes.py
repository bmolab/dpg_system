import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import scipy

import torch.nn.functional as F

def register_quaternion_nodes():
    Node.app.register_node('quaternion_to_euler', QuaternionToEulerNode.factory)
    Node.app.register_node('quaternion_to_rotvec', QuaternionToRotVecNode.factory)
    Node.app.register_node('euler_to_quaternion', EulerToQuaternionNode.factory)
    Node.app.register_node('rotvec_to_quaternion', RotVecToQuaternionNode.factory)
    Node.app.register_node('quaternion_to_matrix', QuaternionToRotationMatrixNode.factory)
    Node.app.register_node('quaternion_distance', QuaternionDistanceNode.factory)
    Node.app.register_node('rotvec_to_quaternion', RotVecToQuaternionNode.factory)
    Node.app.register_node('matrix_to_6d', RotationMatrixTo6DNode.factory)
    Node.app.register_node('quaternion_to_6d', QuaternionTo6DNode.factory)
    Node.app.register_node('6d_to_matrix', SixDToRotationMatrixNode.factory)


class QuaternionToEulerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionToEulerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.output = self.add_output('euler angles')
        self.x_offset = self.add_option('offset x', widget_type='drag_int', default_value=0)
        self.y_offset = self.add_option('offset y', widget_type='drag_int', default_value=0)
        self.z_offset = self.add_option('offset x', widget_type='drag_int', default_value=0)
        self.degrees = self.add_option('degrees', widget_type='checkbox', default_value=True)

    def execute(self):
        offset = np.array([any_to_float(self.x_offset()), any_to_float(self.y_offset()), any_to_float(self.z_offset())], dtype=float)

        if self.input.fresh_input:
            data = any_to_array(self.input())
            if data.shape[-1] % 4 == 0:
                rot = scipy.spatial.transform.Rotation.from_quat(data, scalar_first=True)
                euler = rot.as_euler('xyz')

                if self.degrees():
                    euler *= self.degree_factor
                euler += offset
                self.output.send(euler)
            else:
                if self.app.verbose:
                    print('quaternion_to_euler received improperly formatted input')


class QuaternionToRotVecNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionToRotVecNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.output = self.add_output('euler angles')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if data.shape[-1] % 4 == 0:
                q = quaternion.as_quat_array(data)
                rot_vec = quaternion.as_rotation_vector(q)
                self.output.send(rot_vec)
            else:
                if self.app.verbose:
                    print('quaternion_to_rotvec received improperly formatted input')


class RotVecToQuaternionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RotVecToQuaternionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('rotation vector', triggers_execution=True)
        self.output = self.add_output('quaternion')

    def my_quaternion_from_rotvec(self, rot_vec):
        rot = scipy.spatial.transform.Rotation.from_rotvec(rot_vec)
        q = rot.as_quat(scalar_first=True)
        return q

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if data.shape[-1] % 3 == 0:
                quat = self.my_quaternion_from_rotvec(data)
                self.output.send(quat)
            else:
                if self.app.verbose:
                    print('rot_vec_to_quaternion received improperly formatted input')


class EulerToQuaternionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = EulerToQuaternionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi

        self.input = self.add_input('xyz rotation', triggers_execution=True)
        self.degrees = self.add_property('degrees', widget_type='checkbox', default_value=True)
        self.output = self.add_output('quaternion rotation')
        self.order = self.add_property('order', widget_type='combo', default_value='xyz')
        self.order.widget.combo_items = ['xyz', 'zyx', 'xzy', 'zxy', 'yxz', 'yzx']

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)
            if data.shape[-1] % 3 == 0:
                # if self.degrees():
                #     data /= self.degree_factor
                q = self.my_quaternion_from_euler(data)
                self.output.send(q)
            else:
                if self.app.verbose:
                    print('euler_to_quaternion received improperly formatted input')

    def my_quaternion_from_euler(self, eulers):
        rot = scipy.spatial.transform.Rotation.from_euler(self.order(), eulers, degrees=self.degrees())
        q = rot.as_quat(scalar_first=True)
        return q


class QuaternionToRotationMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionToRotationMatrixNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.output = self.add_output('rotation matrix')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)
            if data.shape[-1] % 4 == 0:
                q = quaternion.as_quat_array(data)
                rotation_matrix = quaternion.as_rotation_matrix(q)
                self.output.send(rotation_matrix)
            else:
                if self.app.verbose:
                    print('quaternion_to_matrix received improperly formatted input')


class QuaternionDistanceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi
        self.reference = np.array([1.0, 0.0, 0.0, 0.0])

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.reference_input = self.add_input('reference')
        self.freeze = self.add_input('freeze ref', widget_type='checkbox', default_value=False)
        self.axis = self.add_property('##distanceAxis', widget_type='combo', default_value='all axes')
        self.axis.widget.combo_items = ['x axis', 'y axis', 'z axis', 'w axis', 'all axes']
        self.output = self.add_output('distance')
        self.distance_squared = self.add_option('distance squared', widget_type='checkbox', default_value=False)

    def quaternion_distance(self, q1, q2):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        diff = np.dot(q1, q2)
        if diff > 1:
            diff = 1
        if diff < -1:
            diff = -1
        distance = math.acos(2 * diff * diff - 1)
        return distance

    def execute(self):
        freeze = self.freeze()

        if self.reference_input.fresh_input:
            data = any_to_array(self.reference_input())
            if data.shape[-1] % 4 == 0:
                self.reference = data
                self.freeze.set(True)
                freeze = True
            else:
                if self.app.verbose:
                    print('quaternion_distance received improperly formatted reference')

        if self.input.fresh_input:
            distance = 0
            data = any_to_array(self.input())
            if data.shape[-1] % 4 == 0:
                if self.reference is not None:
                    axis = self.axis()
                    if axis == 'all axes':
                        distance = self.quaternion_distance(data, self.reference)
                    if axis == 'x axis':
                        distance = data[0] - self.reference[0]
                    elif axis == 'y axis':
                        distance = data[1] - self.reference[1]
                    elif axis == 'z axis':
                        distance = data[2] - self.reference[2]
                    elif axis == 'w axis':
                        distance = data[3] - self.reference[3]

                    if self.distance_squared():
                        distance *= distance
                if not freeze or self.reference is None:
                    self.reference = data
                self.output.send(distance)
            else:
                if self.app.verbose:
                    print('quaternion_distance received improperly formatted input')

def torch_matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def numpy_matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if len(matrix.shape) == 2:
        matrix = np.expand_dims(matrix, 0)
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))


class RotationMatrixTo6DNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RotationMatrixTo6DNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('rotation matrix', triggers_execution=True)
        self.output = self.add_output('6D rotation')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            if type(data) not in [np.ndarray, torch.Tensor]:
                data = any_to_array(data)
            if type(data) is np.ndarray:
                rot6d = numpy_matrix_to_rotation_6d(data)
                self.output.send(rot6d)
            elif type(data) is torch.Tensor:
                rot6d = torch_matrix_to_rotation_6d(data)
                self.output.send(rot6d)


class QuaternionTo6DNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionTo6DNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.output = self.add_output('rotation matrix')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)
            if data.shape[-1] % 4 == 0:
                q = quaternion.as_quat_array(data)
                rotation_matrix = quaternion.as_rotation_matrix(q)
                rot6d = numpy_matrix_to_rotation_6d(rotation_matrix)
                self.output.send(rot6d)
            else:
                if self.app.verbose:
                    print('quaternion_to_matrix received improperly formatted input')


def torch_rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if len(d6.shape) == 2:
        d6 = d6.unsqueeze(0)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class SixDToRotationMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SixDToRotationMatrixNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('6d rotation', triggers_execution=True)
        self.output = self.add_output('rotation matrix')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_tensor(self.input())
            # if type(data) not in [np.ndarray, torch.Tensor]:
            #     data = any_to_array(data)
            # if type(data) is np.ndarray:
            #     rot6d = numpy_matrix_to_rotation_6d(data)
            #     self.output.send(rot6d)
            # elif type(data) is torch.Tensor:
            mat = torch_rotation_6d_to_matrix(data)
            self.output.send(mat)

# def numpy_rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
#     """
#     Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
#     using Gram--Schmidt orthogonalization per Section B of [1].
#     Args:
#         d6: 6D rotation representation, of size (*, 6)
#
#     Returns:
#         batch of rotation matrices of size (*, 3, 3)
#
#     [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
#     On the Continuity of Rotation Representations in Neural Networks.
#     IEEE Conference on Computer Vision and Pattern Recognition, 2019.
#     Retrieved from http://arxiv.org/abs/1812.07035
#     """
#
#     a1, a2 = d6[..., :3], d6[..., 3:]
#     b1 = F.normalize(a1, dim=-1)
#     b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#     b2 = F.normalize(b2, dim=-1)
#     b3 = torch.cross(b1, b2, dim=-1)
#     return torch.stack((b1, b2, b3), dim=-2)