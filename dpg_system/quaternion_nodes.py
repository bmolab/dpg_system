import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import scipy
import platform

import torch.nn.functional as F
from kornia.geometry.conversions import quaternion_to_axis_angle

def register_quaternion_nodes():
    Node.app.register_node('quaternion_to_euler', QuaternionToEulerNode.factory)
    Node.app.register_node('quaternion_to_rotvec', QuaternionToRotVecNode.factory)
    Node.app.register_node('quaternion_to_axis_angle', QuaternionToRotVecNode.factory)
    Node.app.register_node('euler_to_quaternion', EulerToQuaternionNode.factory)
    Node.app.register_node('rotvec_to_quaternion', RotVecToQuaternionNode.factory)
    Node.app.register_node('axis_angle_to_quaternion', RotVecToQuaternionNode.factory)
    Node.app.register_node('quaternion_to_matrix', QuaternionToRotationMatrixNode.factory)
    Node.app.register_node('quaternion_distance', QuaternionDistanceNode.factory)
    Node.app.register_node('matrix_to_6d', RotationMatrixTo6DNode.factory)
    Node.app.register_node('quaternion_to_6d', QuaternionTo6DNode.factory)
    Node.app.register_node('6d_to_matrix', SixDToRotationMatrixNode.factory)
    Node.app.register_node('6d_to_rotvec', SixDToAxisAngleNode.factory)
    Node.app.register_node('6d_to_axis_angle', SixDToAxisAngleNode.factory)
    Node.app.register_node('matrix_to_quaternion', MatrixToQuaternionNode.factory)
    Node.app.register_node('quaternion_diff', QuaternionDiffNode.factory)
    Node.app.register_node('rotation_matrix_diff', RotationMatrixDiffNode.factory)
    Node.app.register_node('matrix_to_axis_angle', MatrixToAxisAngleNode.factory)
    Node.app.register_node('matrix_to_rotvec', MatrixToAxisAngleNode.factory)


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

def torch_quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    added_dim = False
    if len(quaternions.shape) == 1:
        added_dim = True
        quaternions = quaternions.unsqueeze(0)
    # out = quaternion_to_axis_angle(quaternions)
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    if platform.system() == 'Darwin':
        mask = half_angles == 0.0
        half_angles[mask] = 1.0
        sin_half_angles_over_angles = 0.5 * torch.sin(half_angles) / half_angles
        sin_half_angles_over_angles[mask] = 0.5
    else:
        sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)
    # angles/2 are between [-pi/2, pi/2], thus sin_half_angles_over_angles
    # can't be zero
    out = quaternions[..., 1:] / sin_half_angles_over_angles
    if added_dim:
        return out.squeeze()
    return out


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
            data = self.input()
            if type(data) not in [torch.Tensor, np.ndarray]:
                data = any_to_tensor(data)
            if type(data) == torch.Tensor:
                if data.shape[-1] % 4 == 0:
                    rot_vec = torch_quaternion_to_axis_angle(data)
                    # q = quaternion.as_quat_array(data)
                    # rot_vec = quaternion.as_rotation_vector(q)
                    self.output.send(rot_vec)
                else:
                    if self.app.verbose:
                        print('quaternion_to_rotvec received improperly formatted input')

def torch_axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    added_dim = False
    if len(axis_angle.shape) == 1:
        added_dim = True
        axis_angle = axis_angle.unsqueeze(0)
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    if added_dim:
        return torch.cat([torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1).squeeze(0)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1)


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
            data = self.input()
            if type(data) not in [torch.Tensor, np.ndarray]:
                data = any_to_tensor(data)
            if data.shape[-1] % 3 == 0:
                quat = torch_axis_angle_to_quaternion(data)
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
            if type(data) not in [torch.Tensor, np.ndarray]:
                data = any_to_tensor(data)
            if type(data) is torch.Tensor:
                if data.shape[-1] % 4 == 0:
                    rotation_matrix = torch_quaternion_to_matrix(data)
                    # q = quaternion.as_quat_array(data)
                    # rotation_matrix = quaternion.as_rotation_matrix(q)
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
    added_dim = False
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)
        added_dim = True
    batch_dim = matrix.size()[:-2]
    if added_dim:
        return matrix[..., :2, :].clone().reshape((6,))
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
    added_dim = False
    if len(matrix.shape) == 2:
        matrix = np.expand_dims(matrix, 0)
        added_dim = True
    batch_dim = matrix.shape[:-2]
    if added_dim:
        return matrix[..., :2, :].copy().reshape((6,))
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    added_dim = False
    if len(matrix.shape) == 2:
        added_dim = True
        matrix = matrix.unsqueeze(0)
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    if added_dim:
        return standardize_quaternion(out).squeeze(0)
    else:
        return standardize_quaternion(out)

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
            if type(data) not in [np.ndarray, torch.Tensor]:
                data = any_to_array(data)
            if type(data) is np.ndarray:
                if data.shape[-1] % 4 == 0:
                    q = quaternion.as_quat_array(data)
                    rotation_matrix = quaternion.as_rotation_matrix(q)
                    rot6d = numpy_matrix_to_rotation_6d(rotation_matrix)
                    self.output.send(rot6d)
                else:
                    if self.app.verbose:
                        print('quaternion_to_matrix received improperly formatted input')
            elif type(data) is torch.Tensor:
                if data.shape[-1] % 4 == 0:
                    matrix = torch_quaternion_to_matrix(data)
                    rot6d = torch_matrix_to_rotation_6d(matrix)
                    # q = quaternion.as_quat_array(data)
                    # rotation_matrix = quaternion.as_rotation_matrix(q)
                    # rot6d = numpy_matrix_to_rotation_6d(rotation_matrix)
                    self.output.send(rot6d)
                else:
                    if self.app.verbose:
                        print('quaternion_to_matrix received improperly formatted input')

def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    added_dim = False
    if len(quaternions.shape) == 1:
        added_dim = True
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    if added_dim:
        return o.reshape((3, 3))
    return o.reshape(quaternions.shape[:-1] + (3, 3))

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
    added_dim = False
    if len(d6.shape) == 1:
        d6 = d6.unsqueeze(0)
        added_dim = True
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    if added_dim:
        return torch.stack((b1, b2, b3), dim=-2).squeeze(0)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_axis_angle(matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = (
        0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles


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
        data = self.input()
        if type(data) is not torch.Tensor:
            data = any_to_tensor(self.input())
        # if type(data) not in [np.ndarray, torch.Tensor]:
        #     data = any_to_array(data)
        # if type(data) is np.ndarray:
        #     rot6d = numpy_matrix_to_rotation_6d(data)
        #     self.output.send(rot6d)
        # elif type(data) is torch.Tensor:
        mat = torch_rotation_6d_to_matrix(data)
        self.output.send(mat)

class SixDToAxisAngleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SixDToAxisAngleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('6d rotation', triggers_execution=True)
        self.output = self.add_output('axis angle')

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
            aa = matrix_to_axis_angle(mat)
            self.output.send(aa)


def torch_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    added_dim = False
    if len(matrix.shape) == 2:
        added_dim = True
        matrix = matrix.unsqueeze(0)

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = (
        0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)
    if added_dim:
        return axis_angles.squeeze(0)

    return axis_angles


class MatrixToAxisAngleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MatrixToAxisAngleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('rotation matrix', triggers_execution=True)
        self.output = self.add_output('axis angle')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            if type(data) is not torch.Tensor:
                data = any_to_tensor(data)
            if data.size(-1) != 3 or data.size(-2) != 3:
                print('bad format for input to matrix_to_axis_angle')
                return
            aa = torch_matrix_to_axis_angle(data)
            self.output.send(aa)


class MatrixToQuaternionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MatrixToQuaternionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('rotation matrix', triggers_execution=True)
        self.output = self.add_output('quaternion')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            if type(data) is not torch.Tensor:
                data = any_to_tensor(data)
            # data = any_to_tensor(self.input())
            # if type(data) not in [np.ndarray, torch.Tensor]:
            #     data = any_to_array(data)
            # if type(data) is np.ndarray:
            #     rot6d = numpy_matrix_to_rotation_6d(data)
            #     self.output.send(rot6d)
            # elif type(data) is torch.Tensor:
            added_dim = False
            if len(data.shape) == 2:
                added_dim = True
                data = data.unsqueeze(0)
            quat = matrix_to_quaternion(data)
            if added_dim:
                self.output.send(quat.squeeze(0))
            else:
                self.output.send(quat)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def relative_rotation_matrix(R1, R2):
    """
    Compute the relative rotation matrix R_rel = R2 * R1^T
    where:
        R1: (..., 3, 3) tensor (initial orientation)
        R2: (..., 3, 3) tensor (target orientation)
    Returns:
        R_rel: (..., 3, 3) tensor representing the rotation from R1 to R2
    """
    return torch.matmul(R2, R1.transpose(-2, -1))

class RotationMatrixDiffNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RotationMatrixDiffNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('rotation matrix', triggers_execution=True)
        self.previous = None
        self.output = self.add_output('rotation matrix difference')

    def execute(self):
        data = self.input()
        if type(data) is not torch.Tensor:
            data = any_to_tensor(data)
        if self.previous is None:
            self.previous = data.clone()
        else:
            # note... because there are two identical rotations expressed by a quaternion and its negation
            # we might need to determine the closer
            if self.previous.device != data.device:
                self.previous = self.previous.to(data.device)
            diff = relative_rotation_matrix(data, self.previous)
            self.output.send(diff)
            self.previous = data.clone()


class QuaternionDiffNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionDiffNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.previous = None
        self.output = self.add_output('quaternion difference')

    def execute(self):
        data = self.input()
        if type(data) is not torch.Tensor:
            data = any_to_tensor(data)
        if self.previous is None:
            self.previous = data.clone()
        else:
            # note... because there are two identical rotations expressed by a quaternion and its negation
            # we might need to determine the closer
            if self.previous.device != data.device:
                self.previous = self.previous.to(data.device)
            scaling = torch.tensor([1, -1, -1, -1], device=data.device)
            inverse = self.previous * scaling
            diff = quaternion_multiply(data, inverse)
            diff = F.normalize(diff, p=2, dim=-1)
            self.output.send(diff)
            self.previous = data.clone()


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