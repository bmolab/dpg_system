import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import scipy

def register_quaternion_nodes():
    Node.app.register_node('quaternion_to_euler', QuaternionToEulerNode.factory)
    Node.app.register_node('euler_to_quaternion', EulerToQuaternionNode.factory)
    Node.app.register_node('quaternion_to_matrix', QuaternionToRotationMatrixNode.factory)
    Node.app.register_node('quaternion_distance', QuaternionDistanceNode.factory)


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
                q = quaternion.as_quat_array(data)
                euler = quaternion.as_euler_angles(q)
                if self.degrees():
                    euler *= self.degree_factor
                euler += offset
                self.output.send(euler)
            else:
                if self.app.verbose:
                    print('quaternion_to_euler received improperly formatted input')


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
        rot = scipy.spatial.transform.Rotation.from_euler(self.order(), any_to_list(eulers), degrees=self.degrees())
        q = rot.as_quat()
        qq = np.array([q[3], q[0], q[1], q[2]])
        return qq


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






