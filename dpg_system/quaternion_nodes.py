import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import scipy
import platform

import torch.nn.functional as F

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
    Node.app.register_node('quaternion_norm', NormalizeQuaternionNode.factory)
    Node.app.register_node('quaternion_relative', QuaternionRelativeNode.factory)
    Node.app.register_node('tracker_align', TrackerAlignNode.factory)
    Node.app.register_node('swing_twist', SwingTwistNode.factory)


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


def numpy_quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as numpy array of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a numpy array
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    quaternions = np.array(quaternions, copy=False)
    added_dim = False
    if quaternions.ndim == 1:
        added_dim = True
        quaternions = quaternions[np.newaxis, :]

    # Compute the magnitude of the imaginary (vector) part
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)

    # Compute the half-angle theta/2
    # atan2(norm(v), w)
    half_angles = np.arctan2(norms, quaternions[..., :1])

    # Compute 0.5 * sin(theta/2) / (theta/2)
    # np.sinc(x) computes sin(pi*x)/(pi*x)
    # So np.sinc(half_angles / np.pi) computes sin(half_angles)/half_angles
    # This handles the limit as half_angles -> 0 correctly (returns 1.0)
    sin_half_angles_over_angles = 0.5 * np.sinc(half_angles / np.pi)

    # Compute the result v / (sin(theta/2) / theta) = v * theta / sin(theta/2)
    out = quaternions[..., 1:] / sin_half_angles_over_angles

    if added_dim:
        return out.squeeze(axis=0)
    return out

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
        sin_half_angles_over_angles = 0.5 * mps_sinc(half_angles / torch.pi)
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
        self.output = self.add_output('axis angles')

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
            elif type(data) == np.ndarray:
                if data.shape[-1] % 4 == 0:
                    rot_vec = numpy_quaternion_to_axis_angle(data)
                    # q = quaternion.as_quat_array(data)
                    # rot_vec = quaternion.as_rotation_vector(q)
                    self.output.send(rot_vec)
                else:
                    if self.app.verbose:
                        print('quaternion_to_rotvec received improperly formatted input')


def mps_sinc(x: torch.Tensor) -> torch.Tensor:
    """
    An MPS-compatible implementation of the normalized sinc function.
    The PyTorch definition is sin(pi * x) / (pi * x), with sinc(0) = 1.
    """
    # Where x is 0, the value is 1. Otherwise, it's sin(pi*x)/(pi*x).
    # torch.where is perfect for this and is MPS-compatible.
    if x.is_mps:
        return torch.where(
            x == 0,
            torch.ones_like(x),
            torch.sin(torch.pi * x) / (torch.pi * x)
        )
    else:
        return torch.sinc(x)

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
    sin_half_angles_over_angles = 0.5 * mps_sinc(angles * 0.5 / torch.pi)
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
            # Convert anything that is not already a tensor -- including a
            # NumPy array. The old guard let ndarray through unconverted and
            # the branch below only handled torch.Tensor, so a NumPy quaternion
            # (which is what euler_to_quaternion emits) was silently dropped.
            if type(data) is not torch.Tensor:
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


class NormalizeQuaternionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NormalizeQuaternionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('quaternions', triggers_execution=True)
        self.output = self.add_output('normalized')

    def execute(self):
        expanded = False
        quats = self.input()
        if isinstance(quats, list):
            quats = any_to_array(quats)
        if isinstance(quats, np.ndarray):
            if len(quats.shape) == 1 and quats.shape[0] == 4:
                quats = np.expand_dims(quats, 0)
                expanded = True
            if len(quats.shape) == 2 and quats.shape[1] == 4:
                num_quats = quats.shape[0]
                magnitudes = np.linalg.norm(quats, axis=1, keepdims=True)
                norm_quats = quats / magnitudes
                if expanded:
                    norm_quats = np.squeeze(norm_quats, 0)
                self.output.send(norm_quats)
        elif isinstance(quats, torch.Tensor):
            if len(quats.shape) == 1 and quats.shape[0] == 4:
                quats = torch.unsqueeze(quats, 0)
                expanded = True
            if len(quats.shape) == 2 and quats.shape[1] == 4:
                num_quats = quats.shape[0]
                magnitudes = torch.linalg.norm(quats, axis=1, keepdims=True)
                norm_quats = quats / magnitudes
                if expanded:
                    norm_quats = torch.squeeze(norm_quats, 0)
                self.output.send(norm_quats)




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
        0.5 * omegas[~near_pi] / mps_sinc(angles[~near_pi] / torch.pi)
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
        0.5 * omegas[~near_pi] / mps_sinc(angles[~near_pi] / torch.pi)
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


class QuaternionRelativeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionRelativeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.q1_input = self.add_input('q1', triggers_execution=True)
        self.q2_input = self.add_input('q2')
        self.output = self.add_output('q_diff')

    def execute(self):
        q1 = self.q1_input()
        q2 = self.q2_input()
        if q1 is None or q2 is None:
            return

        if type(q1) is not torch.Tensor:
            q1 = any_to_tensor(q1)
        if type(q2) is not torch.Tensor:
            q2 = any_to_tensor(q2)

        if q1.device != q2.device:
            q2 = q2.to(q1.device)

        if q1.shape[-1] != 4 or q2.shape[-1] != 4:
            return

        # conjugate of q1: negate imaginary part [w, -x, -y, -z]
        scaling = torch.tensor([1, -1, -1, -1], dtype=q1.dtype, device=q1.device)
        q1_inv = q1 * scaling

        # q_diff = q2 * q1_inv (rotation from q1 to q2)
        q_diff = quaternion_multiply(q2, q1_inv)
        q_diff = F.normalize(q_diff, p=2, dim=-1)
        self.output.send(q_diff)


class TrackerAlignNode(Node):
    # Root (pelvis_anchor) quaternion index by pose size, mirroring
    # MoCapNode.joint_map / active_joint_map['pelvis_anchor']:
    #   single root quat (1) -> 0, full Shadow pose (37) -> 4, active set (20) -> 5
    root_index_by_count = {1: 0, 20: 5, 37: 4}

    @staticmethod
    def factory(name, data, args=None):
        return TrackerAlignNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.imu_root_input = self.add_input('imu root quat', triggers_execution=True)  # root quat [w,x,y,z], or a full Shadow (37) / active (20) pose
        self.tracker_pos_input = self.add_input('tracker pos')  # [x, y, z]
        self.tracker_quat_input = self.add_input('tracker quat')  # [w, x, y, z] scalar first
        self.body_offset_input = self.add_input('body offset')  # [x, y, z] offset from IMU root to tracker in body-local coords
        self.calibrate_prop = self.add_property('calibrate', widget_type='button', callback=self.do_calibrate)
        self.continuous_opt = self.add_option('continuous', widget_type='checkbox', default_value=True)
        self.smoothing_opt = self.add_option('smoothing', widget_type='drag_float', default_value=0.88, min_value=0.0, max_value=0.999)

        self.corrected_pos_output = self.add_output('corrected pos')
        self.correction_quat_output = self.add_output('correction quat')
        self.yaw_offset_output = self.add_output('yaw offset')

        self.smoothed_yaw = 0.0
        self.calibrated_yaw = 0.0
        self.calibrated = False
        self.initialized = False

    def do_calibrate(self):
        self.calibrated_yaw = self.smoothed_yaw
        self.calibrated = True
        self.continuous_opt.set(False)

    @staticmethod
    def quat_rotate_vector(q_wxyz, vec):
        """Rotate a 3D vector by quaternion [w, x, y, z]."""
        w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ], dtype=np.float64)
        return R @ vec

    @staticmethod
    def quat_conjugate(q):
        """Conjugate of quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_multiply(a, b):
        """Hamilton product of two quaternions [w, x, y, z]."""
        aw, ax, ay, az = a[0], a[1], a[2], a[3]
        bw, bx, by, bz = b[0], b[1], b[2], b[3]
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw
        ])

    @staticmethod
    def extract_twist_around_y(q):
        """Swing-twist decomposition: extract the twist (rotation around Y) from quaternion [w, x, y, z].
        This is stable regardless of tilt around X or Z axes."""
        # Project quaternion onto the Y twist axis: keep only w and y components
        twist = np.array([q[0], 0.0, q[2], 0.0])
        n = np.linalg.norm(twist)
        if n < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])  # identity
        twist /= n
        # Standardize so w >= 0
        if twist[0] < 0:
            twist = -twist
        return twist

    @staticmethod
    def twist_to_angle(twist_quat):
        """Convert a Y-axis twist quaternion to a signed angle in radians."""
        return 2.0 * np.arctan2(twist_quat[2], twist_quat[0])

    @staticmethod
    def wrap_angle(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def execute(self):
        pos = self.tracker_pos_input()
        tracker_quat = self.tracker_quat_input()
        imu_quat = self.imu_root_input()

        if pos is None or tracker_quat is None or imu_quat is None:
            return

        pos = any_to_array(pos).astype(np.float64).flatten()[:3]
        tracker_quat = any_to_array(tracker_quat).astype(np.float64).flatten()
        imu_quat = any_to_array(imu_quat).astype(np.float64).flatten()

        if tracker_quat.size < 4 or imu_quat.size < 4:
            return

        # Tracker quat is already [w, x, y, z] scalar-first (matches the rest of the graph)
        tracker_wxyz = tracker_quat[:4].copy()

        # The root input may be a single root quaternion (4 values) or a full
        # pose; pick the root (pelvis_anchor) row based on the quaternion count.
        n_imu_quats = imu_quat.size // 4
        root_index = self.root_index_by_count.get(n_imu_quats, 0)
        imu_wxyz = imu_quat[root_index * 4: root_index * 4 + 4].copy()

        # Normalize
        tracker_wxyz /= np.linalg.norm(tracker_wxyz) + 1e-10
        imu_wxyz /= np.linalg.norm(imu_wxyz) + 1e-10

        # The yaw between the two world frames is the relative rotation
        #   q_rel = q_tracker * conj(q_imu)
        # The body's tilt/lean is shared by both sensors and cancels in this
        # product, leaving (ideally) a pure rotation about world Y, so twist
        # extraction is stable under body tilt. Differencing each sensor's
        # independent Y-twist does NOT cancel the lean -- that leakage (which
        # differs between the two sensors) is what made the old approach
        # erratic during walking.
        # NOTE: order assumes q_imu/q_tracker are body-to-world rotations
        # (v_world = q * v_body * conj(q)). If your convention is mirrored,
        # swap the two arguments below to conj(imu) * tracker.
        q_rel = self.quat_multiply(tracker_wxyz, self.quat_conjugate(imu_wxyz))
        # Extract yaw the same way the quaternion_relative -> quaternion_to_euler
        # path does (scipy 'xyz' intrinsic decomposition; Y/yaw is index 1).
        # The swing-twist projection used previously reads 2*atan2(y, w), which
        # diverges and jitters whenever q_rel carries an off-axis (mounting)
        # component -- this matches the stable +-8 deg reference instead.
        rel_euler = scipy.spatial.transform.Rotation.from_quat(q_rel, scalar_first=True).as_euler('xyz')
        current_offset = self.wrap_angle(rel_euler[1])

        # Smooth the offset (handles angle wrapping via angular difference)
        if not self.initialized:
            self.smoothed_yaw = current_offset
            self.initialized = True
        else:
            alpha = self.smoothing_opt()
            diff = self.wrap_angle(current_offset - self.smoothed_yaw)
            self.smoothed_yaw = self.wrap_angle(self.smoothed_yaw + (1.0 - alpha) * diff)

        # Determine which offset to use
        if self.continuous_opt():
            active_offset = self.smoothed_yaw
        elif self.calibrated:
            active_offset = self.calibrated_yaw
        else:
            active_offset = self.smoothed_yaw

        # Build correction quaternion: rotate around Y by -offset
        cos_half = np.cos(-active_offset / 2.0)
        sin_half = np.sin(-active_offset / 2.0)
        q_correction = np.array([cos_half, 0.0, sin_half, 0.0])

        # Apply yaw correction to tracker position
        corrected_pos = self.quat_rotate_vector(q_correction, pos)

        # Subtract body offset rotated by the IMU root orientation.
        # Both corrected_pos and R(imu)*offset are in the IMU world frame.
        # Using just imu_wxyz (not corrected_imu) aligns the offset axes with the
        # body's visual orientation in the scene.
        body_offset_data = self.body_offset_input()
        if body_offset_data is not None:
            body_offset = any_to_array(body_offset_data).astype(np.float64).flatten()[:3]
            world_offset = self.quat_rotate_vector(imu_wxyz, body_offset)
            corrected_pos = corrected_pos - world_offset

        self.corrected_pos_output.send(corrected_pos.astype(np.float32))
        self.correction_quat_output.send(q_correction.astype(np.float32))
        self.yaw_offset_output.send(float(np.degrees(active_offset)))


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

class SwingTwistNode(Node):
    """Split an orientation into where the device points (swing) and how far it
    is turned about its own pointing axis (twist).

    A yaw/pitch/roll triple measures two of its angles about axes that do not
    move with the device, so once the device is tilted those angles stop
    matching what a hand feels. The twist here is measured about a body axis,
    so rolling the device reads as roll however it is held. The swing is what
    is left: the rotation that carries the twist axis to where it now points.
    q = swing * twist, both scalar-first [w, x, y, z].

    Two references for zero twist:
      level         top of the device up - the aircraft's bank. Undefined
                    pointing straight up or down, so it blends into the
                    shortest-arc twist within a few degrees of vertical.
      shortest arc  twist beyond the shortest rotation that carries the twist
                    axis to its current direction. Well defined everywhere
                    except pointing exactly backwards, but its zero rolls
                    with direction: at azimuth 90 and elevation 45 a device
                    with its top up reads 45 degrees of twist."""

    axis_vectors = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0),
                    '-x': (-1.0, 0.0, 0.0), '-y': (0.0, -1.0, 0.0), '-z': (0.0, 0.0, -1.0)}
    up_index = {'x': 0, 'y': 1, 'z': 2}
    vertical_blend = math.sin(math.radians(12.0))   # horizontal reach below which level bank fades out

    @staticmethod
    def factory(name, data, args=None):
        return SwingTwistNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.degree_factor = 180.0 / math.pi

        # swing_twist [twist axis] [up axis]   e.g.  swing_twist x z  for a Pipo
        twist_default = 'x'
        up_default = 'y'
        if args is not None:
            axes = [str(a) for a in args if str(a) in self.axis_vectors]
            if len(axes) > 0:
                twist_default = axes[0]
            if len(axes) > 1 and axes[1] in self.up_index:
                up_default = axes[1]

        self.neutral = None          # captured reference, scalar-first, or None
        self.last_q = None

        self.input = self.add_input('quaternion', triggers_execution=True)
        self.twist_axis = self.add_property('twist axis', widget_type='combo', default_value=twist_default)
        self.twist_axis.widget.combo_items = ['x', 'y', 'z', '-x', '-y', '-z']
        self.up_axis = self.add_property('up axis', widget_type='combo', default_value=up_default)
        self.up_axis.widget.combo_items = ['x', 'y', 'z']
        self.set_neutral_button = self.add_property('set neutral', widget_type='button', callback=self.set_neutral)
        self.clear_neutral_button = self.add_property('clear neutral', widget_type='button', callback=self.clear_neutral)
        self.twist_reference = self.add_option('twist reference', widget_type='combo', default_value='level')
        self.twist_reference.widget.combo_items = ['level', 'shortest arc']
        self.degrees = self.add_option('degrees', widget_type='checkbox', default_value=True)

        self.direction_output = self.add_output('direction')
        self.azimuth_output = self.add_output('azimuth')
        self.elevation_output = self.add_output('elevation')
        self.twist_output = self.add_output('twist')
        self.swing_quat_output = self.add_output('swing quaternion')
        self.twist_quat_output = self.add_output('twist quaternion')

    def set_neutral(self):
        """Take the pose being held now as zero: no swing, no twist. Everything
        after is measured from it, in that pose's own frame, so 'up axis' means
        the device's up when it was held here."""
        if self.last_q is not None:
            q = np.asarray(self.last_q, dtype=np.float64)
            if q.ndim > 1:
                q = q.reshape(-1, 4)[0]
            self.neutral = q / (np.linalg.norm(q) + 1e-12)

    def clear_neutral(self):
        self.neutral = None

    @staticmethod
    def decompose(q, axis):
        """Shortest-arc swing-twist. q: (..., 4) scalar-first, axis: (3,) unit body axis.
        Returns swing (..., 4), twist (..., 4), twist angle (...) in radians, direction (..., 3)."""
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        w = q[..., 0]
        v = q[..., 1:]
        p = v @ axis                                    # component of the vector part along the twist axis
        twist = np.concatenate([w[..., None], p[..., None] * axis], axis=-1)
        n = np.linalg.norm(twist, axis=-1, keepdims=True)
        degenerate = n[..., 0] < 1e-10                  # a half turn exactly across the axis: twist undefined
        twist = np.where(degenerate[..., None], np.array([1.0, 0.0, 0.0, 0.0]), twist / np.where(degenerate[..., None], 1.0, n))
        twist = np.where((twist[..., 0] < 0)[..., None], -twist, twist)   # shortest representation, angle in (-pi, pi]
        angle = 2.0 * np.arctan2(twist[..., 1:] @ axis, twist[..., 0])
        conj = twist * np.array([1.0, -1.0, -1.0, -1.0])
        swing = quaternion_multiply_wxyz_np(q, conj)
        direction = rotate_vector_wxyz_np(q, axis)
        return swing, twist, angle, direction

    @staticmethod
    def level_bank(q, axis, up, direction):
        """Twist measured against 'top up': the angle from the horizontal
        sideways axis at the current direction to where the device's own
        sideways axis has gone. Returns bank (...) in radians and the
        horizontal reach (...) of the direction, 0 when vertical."""
        side_neutral = np.cross(up, axis)               # the device's sideways axis when it sits at identity
        side_now = rotate_vector_wxyz_np(q, side_neutral)
        side_level = np.cross(up, direction)            # sideways at the current direction, kept horizontal
        reach = np.linalg.norm(side_level, axis=-1)
        side_level = side_level / (reach[..., None] + 1e-12)
        bank = np.arctan2(np.sum(np.cross(side_level, side_now) * direction, axis=-1),
                          np.sum(side_level * side_now, axis=-1))
        return bank, reach

    @staticmethod
    def azimuth_from(axis, up, direction):
        """Bearing of the direction about up, zero where the twist axis points at
        identity, positive turning right - clockwise seen from above, as a compass reads."""
        reference = axis - (axis @ up) * up
        if np.linalg.norm(reference) < 1e-6:            # twist axis is the up axis: fall back to the next axis round
            reference = np.roll(up, 1)
        reference = reference / np.linalg.norm(reference)
        return np.arctan2(np.sum(np.cross(direction, reference) * up, axis=-1), direction @ reference)

    def execute(self):
        if not self.input.fresh_input:
            return
        q = any_to_array(self.input()).astype(np.float64)
        if q.shape[-1] != 4:
            if self.app.verbose:
                print('swing_twist expects quaternions of 4 values, scalar first')
            return
        self.last_q = q
        neutral = self.neutral
        if neutral is not None:
            # rotation from the neutral pose to now, expressed in the neutral pose's frame
            q = quaternion_multiply_wxyz_np(neutral * np.array([1.0, -1.0, -1.0, -1.0]), q)
        axis = np.array(self.axis_vectors[self.twist_axis()], dtype=np.float64)
        up = np.zeros(3)
        up[self.up_index[self.up_axis()]] = 1.0
        swing, twist, angle, direction = self.decompose(q, axis)

        if self.twist_reference() == 'level':
            bank, reach = self.level_bank(q, axis, up, direction)
            # fade from bank to shortest-arc twist as the direction nears vertical, on the circle
            weight = np.clip(reach / self.vertical_blend, 0.0, 1.0) ** 2
            angle = np.arctan2(weight * np.sin(bank) + (1.0 - weight) * np.sin(angle),
                               weight * np.cos(bank) + (1.0 - weight) * np.cos(angle))
            half = 0.5 * angle
            twist = np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axis], axis=-1)
            swing = quaternion_multiply_wxyz_np(q, twist * np.array([1.0, -1.0, -1.0, -1.0]))

        elevation = np.arcsin(np.clip(direction @ up, -1.0, 1.0))
        azimuth = self.azimuth_from(axis, up, direction)
        if self.degrees():
            angle = angle * self.degree_factor
            elevation = elevation * self.degree_factor
            azimuth = azimuth * self.degree_factor

        self.twist_quat_output.send(swing_squeeze(twist))
        self.swing_quat_output.send(swing_squeeze(swing))
        self.twist_output.send(swing_squeeze(angle))
        self.elevation_output.send(swing_squeeze(elevation))
        self.azimuth_output.send(swing_squeeze(azimuth))
        self.direction_output.send(swing_squeeze(direction))


def swing_squeeze(a):
    """A lone quaternion in gives plain values out; batches stay arrays."""
    a = np.asarray(a)
    if a.ndim == 0:
        return float(a)
    return a


def quaternion_multiply_wxyz_np(a, b):
    """Hamilton product of scalar-first quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rotate_vector_wxyz_np(q, vec):
    """Rotate a body vector into the world by scalar-first q: v' = q v q*, broadcasting over leading dims."""
    w = q[..., 0:1]
    u = q[..., 1:]
    vec = np.broadcast_to(vec, u.shape)
    t = 2.0 * np.cross(u, vec)
    return vec + w * t + np.cross(u, t)
