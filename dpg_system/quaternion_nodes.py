import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion

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

        self.input = self.add_input("quaternion", triggers_execution=True)
        self.output = self.add_output("euler angles")
        self.x_offset_option = self.add_option('offset x', widget_type='drag_int', default_value=0)
        self.y_offset_option = self.add_option('offset y', widget_type='drag_int', default_value=0)
        self.z_offset_option = self.add_option('offset x', widget_type='drag_int', default_value=0)
        self.degrees_option = self.add_option('degrees', widget_type='checkbox', default_value=True)

    def execute(self):
        x_offset = self.x_offset_option.get_widget_value()
        y_offset = self.y_offset_option.get_widget_value()
        z_offset = self.z_offset_option.get_widget_value()
        offset = np.array([x_offset, y_offset, z_offset], dtype=float)
        degrees = self.degrees_option.get_widget_value()

        if self.input.fresh_input:
            data = self.input.get_received_data()
            data = any_to_array(data)
            if data.shape[-1] % 4 == 0:
                q = quaternion.as_quat_array(data)
                euler = quaternion.as_euler_angles(q)
                if degrees:
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

        self.input = self.add_input("xyz rotation", triggers_execution=True)
        self.output = self.add_output("quaternion rotation")
        self.degrees_option = self.add_option('degrees', widget_type='checkbox', default_value=True)

    def execute(self):
        degrees = self.degrees_option.get_widget_value()

        if self.input.fresh_input:
            data = self.input.get_received_data()
            data = any_to_array(data)
            if data.shape[-1] % 3 == 0:
                if degrees:
                    data /= self.degree_factor

                q = quaternion.from_euler_angles(alpha_beta_gamma=data)
                self.output.send(quaternion.as_float_array(q))
            else:
                if self.app.verbose:
                    print('euler_to_quaternion received improperly formatted input')


class QuaternionToRotationMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = QuaternionToRotationMatrixNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree_factor = 180.0 / math.pi

        self.input = self.add_input("quaternion", triggers_execution=True)
        self.output = self.add_output("rotation matrix")

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
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

        self.input = self.add_input("quaternion", triggers_execution=True)
        self.reference_input = self.add_input("reference")
        self.freeze_input = self.add_input('freeze ref', widget_type='checkbox', default_value=False)
        self.distance_axis_property = self.add_property('##distanceAxis', widget_type='combo', default_value='all axes')
        self.distance_axis_property.widget.combo_items = ['x axis', 'y axis', 'z axis', 'w axis', 'all axes']
        self.output = self.add_output("distance")
        self.distance_squared_property = self.add_option('distance squared', widget_type='checkbox', default_value=False)

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
        axis = self.distance_axis_property.get_widget_value()
        squared = self.distance_squared_property.get_widget_value()
        freeze = self.freeze_input.get_widget_value()

        if self.reference_input.fresh_input:
            data = self.reference_input.get_received_data()
            data = any_to_array(data)
            if data.shape[-1] % 4 == 0:
                self.reference = data
                self.freeze_input.set(True)
                freeze = True
            else:
                if self.app.verbose:
                    print('quaternion_distance received improperly formatted reference')

        if self.input.fresh_input:
            distance = 0
            data = self.input.get_received_data()
            data = any_to_array(data)
            if data.shape[-1] % 4 == 0:
                if self.reference is not None:
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

                    if squared:
                        distance *= distance
                if not freeze or self.reference is None:
                    self.reference = data
                self.output.send(distance)
            else:
                if self.app.verbose:
                    print('quaternion_distance received improperly formatted input')






