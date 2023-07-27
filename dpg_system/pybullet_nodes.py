
from dpg_system.conversion_utils import *
from dpg_system.node import Node
import threading
import pybullet as p
import pybullet_data
# NOTE changing target name changed, changing target port crashed


def register_pybullet_nodes():
    Node.app.register_node('pybullet_body', PyBulletBodyNode.factory)

# class PyBulletNode(Node):
#     pybullet = None
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#         if self.pybullet is not None:
#             self.pybullet =

class PyBulletBodyNode(Node):
    joint_index = {
        'left_hip': 0,
        'left_knee': 1,
        'left_ankle': 2,
        'right_hip': 3,
        'right_knee': 4,
        'right_ankle': 5,
        'lower_back': 6,
        'upper_back': 7,
        'chest': 8,
        'lower_neck': 9,
        'upper_neck': 10,
        'left_clavicle': 11,
        'left_shoulder': 12,
        'left_elbow': 13,
        'left wrist': 14,
        'right_clavicle': 15,
        'right_shoulder': 16,
        'right_elbow': 17,
        'right_wrist': 18
    }

    @staticmethod
    def factory(name, data, args=None):
        node = PyBulletBodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF('plane_implicit.urdf', [0, 0, 0], useMaximalCoordinates=True)
        self.body_id = p.loadURDF(fileName='dpg_system/humanoid.urdf', basePosition=[0, 0, 0], globalScaling=1.0, useFixedBase=False, flags=p.URDF_MAINTAIN_LINK_ORDER)
        self.movable_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        for joint_idx in self.movable_indices:  # (joint position, joint velocity, (force along X, force along Y, force along Z, torque around X, torque around Y, torque around Z), applied joint motor torque)
            p.enableJointForceTorqueSensor(self.body_id, joint_idx, True)

        p.setGravity(0, 0, -10.0)

        # disable motors

        for j in range(p.getNumJoints(self.body_id)):
            ji = p.getJointInfo(self.body_id, j)
            targetPosition = [0]
            jointType = ji[2]

            if jointType == p.JOINT_SPHERICAL:
                targetPosition = [0, 0, 0, 1]
                p.setJointMotorControlMultiDof(self.body_id, j, p.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0],
                                               positionGain=0, velocityGain=1, force=[0, 0, 0])
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(self.body_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        self.joint_orientations = []
        self.joint_forces = []

        self.trigger_input = self.add_input('step', triggers_execution=True)
        self.pelvis_position = self.add_input('pelvis_position')
        self.pelvis_orientation = self.add_input('pelvis_orientation')
        keys = list(self.joint_index.keys())
        for joint in keys:
            self.joint_orientations.append(self.add_input(joint))
        # we need inputs for each joint (quaternion)
        for joint in keys:
            self.joint_forces.append(self.add_output(joint))

    def disable_motors(self):
        for j in range(p.getNumJoints(self.body_id)):
            ji = p.getJointInfo(self.body_id, j)
            targetPosition = [0]
            jointType = ji[2]

            if jointType == p.JOINT_SPHERICAL:
                targetPosition = [0, 0, 0, 1]
                p.setJointMotorControlMultiDof(self.body_id, j, p.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0],
                                               positionGain=0, velocityGain=1, force=[0, 0, 0])
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(self.body_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    def execute(self):
        # get joint positions
        if p.isConnected():
            joint_orientations = np.ndarray((len(self.movable_indices), 4))
            for index in self.movable_indices:
                joint_orientations[index] = np.roll(self.joint_orientations[index](), -1)

            p.resetBasePositionAndOrientation(self.body_id, self.pelvis_position(), np.roll(self.pelvis_orientation(), -1))
            p.resetJointStatesMultiDof(self.body_id, self.movable_indices, joint_orientations)

            p.stepSimulation()

            for joint_idx in self.movable_indices:
                joint_state_dof = p.getJointStateMultiDof(self.body_id, joint_idx)
                self.joint_forces[joint_idx].send(list(joint_state_dof[2]))
