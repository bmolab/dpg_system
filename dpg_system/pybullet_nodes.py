import os

from dpg_system.conversion_utils import *
from dpg_system.node import Node
import pybullet as p
import pybullet_data
# NOTE changing target name changed, changing target port crashed
# NOTE pybullet's p.* calls use the most-recent connected client unless
# physicsClientId= is passed. Multiple PyBulletBodyNode instances will
# therefore step on each other; today this code assumes a single node.


_HUMANOID_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'humanoid.urdf')


def _as_float_seq(value, length):
    """Coerce a widget value to a list of *length* floats, or return None."""
    if value is None:
        return None
    try:
        seq = list(value)
    except TypeError:
        return None
    if len(seq) != length:
        return None
    try:
        return [float(v) for v in seq]
    except (TypeError, ValueError):
        return None


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

    joint_reverse_index = {
        0: 'left_hip',
        1: 'left_knee',
        2: 'left_ankle',
        3: 'right_hip',
        4: 'right_knee',
        5: 'right_ankle',
        6: 'lower_back',
        7: 'upper_back',
        8: 'chest',
        9: 'lower_neck',
        10: 'upper_neck',
        11: 'left_clavicle',
        12: 'left_shoulder',
        13: 'left_elbow',
        14: 'left_wrist',
        15: 'right_clavicle',
        16: 'right_shoulder',
        17: 'right_elbow',
        18: 'right_wrist'
    }

    joint_to_shadow_index = {
        'left_hip': 14,
        'left_knee': 12,
        'left_ankle': 8,
        'right_hip': 28,
        'right_knee': 26,
        'right_ankle': 22,
        'lower_back': 31,
        'upper_back': 32,
        'chest': 1,
        'lower_neck': 17,
        'upper_neck': 2,
        'left_clavicle': 13,
        'left_shoulder': 5,
        'left_elbow': 9,
        'left wrist': 10,
        'right_clavicle': 27,
        'right_shoulder': 19,
        'right_elbow': 23,
        'right_wrist': 24
    }

    @staticmethod
    def factory(name, data, args=None):
        node = PyBulletBodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.client_id = -1
        self.plane_id = -1
        self.body_id = -1
        self.movable_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

        try:
            self.client_id = p.connect(p.DIRECT)
        except Exception as e:
            print('pybullet_body: connect failed:', e)

        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        except Exception as e:
            print('pybullet_body: setAdditionalSearchPath failed:', e)

        try:
            self.plane_id = p.loadURDF('plane_implicit.urdf', [0, 0, 0], useMaximalCoordinates=True)
        except Exception as e:
            print('pybullet_body: loadURDF(plane) failed:', e)

        urdf_flags = (
            p.URDF_MAINTAIN_LINK_ORDER
            | p.URDF_USE_SELF_COLLISION
            | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        )
        try:
            self.body_id = p.loadURDF(
                fileName=_HUMANOID_URDF,
                basePosition=[0, 0, 0],
                globalScaling=1.0,
                useFixedBase=False,
                flags=urdf_flags,
            )
        except Exception as e:
            print('pybullet_body: loadURDF(humanoid) failed for', _HUMANOID_URDF, ':', e)
            self.body_id = -1

        if self.body_id >= 0:
            for joint_idx in self.movable_indices:  # (joint position, joint velocity, (force along X, force along Y, force along Z, torque around X, torque around Y, torque around Z), applied joint motor torque)
                try:
                    p.enableJointForceTorqueSensor(self.body_id, joint_idx, True)
                except Exception as e:
                    print('pybullet_body: enableJointForceTorqueSensor failed for joint', joint_idx, ':', e)

            try:
                p.setGravity(0, 0, -10.0)
                p.setTimeStep(1.0 / 60.0)
            except Exception as e:
                print('pybullet_body: setGravity/setTimeStep failed:', e)
            # disable motors

            try:
                num_joints = p.getNumJoints(self.body_id)
            except Exception as e:
                print('pybullet_body: getNumJoints failed:', e)
                num_joints = 0

            for j in range(num_joints):
                try:
                    ji = p.getJointInfo(self.body_id, j)
                except Exception as e:
                    print('pybullet_body: getJointInfo failed for joint', j, ':', e)
                    continue
                targetPosition = [0]
                jointType = ji[2]

                try:
                    if jointType == p.JOINT_SPHERICAL:
                        targetPosition = [0, 0, 0, 1]
                        p.setJointMotorControlMultiDof(self.body_id, j, p.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1, force=[0, 0, 0])
                    if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                        p.setJointMotorControl2(self.body_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                except Exception as e:
                    print('pybullet_body: motor disable failed for joint', j, ':', e)

        self.joint_orientations = []
        self.joint_forces = []
        self.out_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 37)
        self.trigger_input = self.add_input('step', triggers_execution=True)
        self.pelvis_position = self.add_input('pelvis_position')
        self.pelvis_orientation = self.add_input('pelvis_orientation')
        keys = list(self.joint_index.keys())
        for joint in keys:
            self.joint_orientations.append(self.add_input(joint))

        # we need inputs for each joint (quaternion)
        for joint in keys:
            self.joint_forces.append(self.add_output(joint))
        self.pose_out = self.add_output('pose_out')
        self.position_out = self.add_output('position_out')

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
        if not p.isConnected():
            return
        if self.body_id < 0:
            return

        try:
            if self.pelvis_orientation.fresh_input and self.pelvis_position.fresh_input:
                pelvis_pos = _as_float_seq(self.pelvis_position(), 3)
                pelvis_quat = _as_float_seq(self.pelvis_orientation(), 4)
                if pelvis_pos is not None and pelvis_quat is not None:
                    p.resetBasePositionAndOrientation(
                        self.body_id, pelvis_pos, np.roll(pelvis_quat, -1)
                    )

            # Default every joint to identity quaternion so stale/non-fresh
            # slots don't pass uninitialized memory into pybullet.
            joint_orientations = np.zeros((len(self.movable_indices), 4))
            joint_orientations[:, 3] = 1.0
            for index in self.movable_indices:
                if self.joint_orientations[index].fresh_input:
                    quat = _as_float_seq(self.joint_orientations[index](), 4)
                    if quat is not None:
                        joint_orientations[index] = np.roll(quat, -1)
                p.resetJointStateMultiDof(self.body_id, index, joint_orientations[index])

            p.stepSimulation()
            # built shadow pose
            for joint_idx in self.movable_indices:
                if joint_idx in self.joint_reverse_index:
                    joint = self.joint_reverse_index[joint_idx]
                    if joint in self.joint_to_shadow_index:
                        joint_state_dof = p.getJointStateMultiDof(self.body_id, joint_idx)
                        if joint_state_dof and len(joint_state_dof) > 0:
                            self.out_pose[self.joint_to_shadow_index[joint]] = np.roll(joint_state_dof[0], 1)
            pos, base_quat = p.getBasePositionAndOrientation(self.body_id)
            self.out_pose[4] = np.roll(base_quat, 1)

            for joint_idx in self.movable_indices:
                joint_state_dof = p.getJointStateMultiDof(self.body_id, joint_idx)
                if joint_state_dof and len(joint_state_dof) > 2:
                    self.joint_forces[joint_idx].send(list(joint_state_dof[2]))

            self.pose_out.send(self.out_pose)
            self.position_out.send(list(pos))
        except Exception as e:
            print('pybullet_body: step failed:', e)