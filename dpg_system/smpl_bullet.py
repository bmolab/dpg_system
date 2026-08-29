"""
SMPL body as a pybullet multibody: the physics core behind smpl_ragdoll.

A body is generated as a URDF from SMPLProcessor's skeleton -- bone offsets,
segment masses and lengths from the betas -- one link per SMPL joint, joined
by spherical joints.  Each link's frame sits at its joint with no rotation
offset, so a joint's quaternion *is* the SMPL local rotation and reads straight
back into a pose; the segment is a capsule tilted along the bone in the
collision origin, with the inertial origin at mid-segment so gravity torques
are right (a hanging arm whose mass sat at the shoulder would not hang).

The body lives in the processor's internal frame, +Y up, with gravity set
accordingly; pybullet has no opinion about which way is up.
"""

import math
import os
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:      # the native core stays available without it
    p = None
    PYBULLET_AVAILABLE = False


JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',
]

# Capsule radius per segment.  Not anatomical: the largest that leave the body
# clear of itself in ordinary poses (the thighs pass within 0.119 m of each
# other standing, the spine axis 0.103 m from the thigh line).
SEG_RADIUS = {
    0: 0.10, 3: 0.10, 6: 0.11, 9: 0.11,          # pelvis, spine
    12: 0.05, 15: 0.09,                           # neck, head
    13: 0.045, 14: 0.045,                         # collars
    16: 0.055, 17: 0.055, 18: 0.045, 19: 0.045,   # upper arms, forearms
    20: 0.03, 21: 0.03, 22: 0.028, 23: 0.028,     # wrists (hand segment), hands
    1: 0.055, 2: 0.055, 4: 0.055, 5: 0.055,       # thighs, shins
    7: 0.04, 8: 0.04, 10: 0.03, 11: 0.03,         # ankles (foot segment), feet
}

# Link pairs that overlap at rest and must not collide, beyond the
# parent-child pairs Bullet excludes on its own.
SELF_COLLISION_EXCLUDE = [
    (3, 1), (3, 2),      # lower spine against the thighs
    (0, 4), (0, 5),      # pelvis against the shins (they sit under it, seated)
    (9, 13), (9, 14),    # upper spine against the collars
    (12, 13), (12, 14),  # neck against the collars
    (15, 13), (15, 14),  # head against the collars
]


def _rpy_from_z_to(direction):
    """URDF roll-pitch-yaw (fixed-axis xyz) taking +Z onto a direction."""
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(z @ d, -1.0, 1.0))
    if c > 1.0 - 1e-9:
        rot = R.identity()
    elif c < -1.0 + 1e-9:
        rot = R.from_rotvec([math.pi, 0.0, 0.0])
    else:
        axis = np.cross(z, d)
        axis /= np.linalg.norm(axis)
        rot = R.from_rotvec(axis * math.acos(c))
    return tuple(float(v) for v in rot.as_euler('xyz'))


def build_urdf(processor, name='smpl'):
    """The body as URDF text, from the processor's skeleton."""
    parents = list(processor._get_hierarchy())
    offsets = np.asarray(processor.skeleton_offsets)          # (30, 3) parent-relative
    seg_mass = np.asarray(processor._seg_mass)
    seg_len = np.asarray(processor._seg_length)
    children = {j: [] for j in range(24)}
    for j in range(1, 24):
        children[parents[j]].append(j)

    # The processor's virtual tips: toe, finger tip, heel, by parent joint.
    tips = {}
    for v in range(24, min(30, offsets.shape[0])):
        if v < len(parents):
            tips.setdefault(parents[v], []).append(v)

    def segment(j):
        """(direction unit vector, length) of joint j's segment in its own frame."""
        kids = children[j]
        if kids:
            end = np.mean([offsets[c] for c in kids], axis=0)
            n = np.linalg.norm(end)
            if n > 1e-6:
                return end / n, float(n)
        # Leaf: toward its virtual tip where the processor has one (a toe, a
        # finger tip).  Continuing the parent's bone instead pointed the toe
        # capsule forward-and-down into the floor, and a driven foot standing
        # on it carried fifteen kilonewtons of contact.
        for v in tips.get(j, []):
            if v in (24, 25, 26, 27):
                d = offsets[v]
                n = np.linalg.norm(d)
                if n > 1e-6:
                    return d / n, float(max(n, 0.03))
        d = offsets[j] if j > 0 else np.array([0.0, 1.0, 0.0])
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        return d, float(max(seg_len[j], 0.05))

    out = ['<?xml version="1.0"?>', '<robot name="%s">' % name]
    for j in range(24):
        d, L = segment(j)
        r = SEG_RADIUS.get(j, 0.04)
        m = float(max(seg_mass[j], 0.05))
        mid = d * (0.5 * L)
        rpy = _rpy_from_z_to(d)
        # Rod-with-radius inertia about its own centre, Z along the bone --
        # then rotated into the link frame and written as the full tensor.
        # The inertial origin carries no rotation: pybullet's importer
        # re-centres links on their inertial frames internally, and with a
        # rotated inertial frame it placed every chain wrongly from the rest
        # pose on (the feet 1.6 m off).  A plain tensor in the link frame
        # gives it nothing to get wrong.
        i_ax = 0.5 * m * r * r
        i_pe = m * (L * L / 12.0 + r * r / 4.0)
        rot = R.from_euler('xyz', rpy).as_matrix()
        tensor = rot @ np.diag([i_pe, i_pe, i_ax]) @ rot.T
        cyl = max(L - 2.0 * r, 0.01)
        out.append('  <link name="%s">' % JOINT_NAMES[j])
        out.append('    <inertial><origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>'
                   '<mass value="%.5f"/><inertia ixx="%.7f" ixy="%.7f" ixz="%.7f" iyy="%.7f" iyz="%.7f" izz="%.7f"/></inertial>'
                   % (*mid, m, tensor[0, 0], tensor[0, 1], tensor[0, 2],
                      tensor[1, 1], tensor[1, 2], tensor[2, 2]))
        out.append('    <collision><origin xyz="%.6f %.6f %.6f" rpy="%.6f %.6f %.6f"/>'
                   '<geometry><capsule radius="%.4f" length="%.4f"/></geometry></collision>'
                   % (*mid, *rpy, r, cyl))
        out.append('  </link>')
        if j > 0:
            out.append('  <joint name="%s" type="spherical">' % JOINT_NAMES[j])
            out.append('    <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % tuple(offsets[j]))
            out.append('    <parent link="%s"/><child link="%s"/><axis xyz="1 0 0"/>'
                       % (JOINT_NAMES[parents[j]], JOINT_NAMES[j]))
            out.append('  </joint>')
    out.append('</robot>')
    return '\n'.join(out)


class BulletBody:
    """The multibody in its own physics client, with the frame bookkeeping."""

    def __init__(self, processor, floor_height=0.0, friction=0.8, floor=True, self_collision=True):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError('pybullet is not installed')
        self.processor = processor
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0.0, -9.81, 0.0, physicsClientId=self.cid)

        self.urdf_path = os.path.join(tempfile.gettempdir(), 'smpl_ragdoll_%d.urdf' % os.getpid())
        with open(self.urdf_path, 'w') as f:
            f.write(build_urdf(processor))
        flags = p.URDF_MAINTAIN_LINK_ORDER | (p.URDF_USE_SELF_COLLISION if self_collision else 0)
        self.self_collision = self_collision
        self.body = p.loadURDF(self.urdf_path, [0, 0, 0], [0, 0, 0, 1],
                               flags=flags, useFixedBase=False, physicsClientId=self.cid)

        # Joint j is link j - 1; check rather than assume.
        self.link_of = {}
        for li in range(p.getNumJoints(self.body, physicsClientId=self.cid)):
            info = p.getJointInfo(self.body, li, physicsClientId=self.cid)
            self.link_of[JOINT_NAMES.index(info[1].decode())] = li
        assert all(self.link_of[j] == j - 1 for j in range(1, 24)), self.link_of

        # pybullet places and reports the base by its centre of mass, not its
        # link frame; the pelvis joint sits one inertial offset away.
        self.base_offset = np.asarray(p.getDynamicsInfo(self.body, -1, physicsClientId=self.cid)[3])

        # Self-collision pairs.  Bullet excludes parent-child on its own, but
        # along the spine the segments are short and the capsules fat, so
        # grandparent pairs overlap at rest (spine1 against spine3, spine3
        # against the head) and fight -- a free joint given one radian a
        # second ran away to seven in a tenth of a second with nothing else
        # acting.  Rather than a hand-kept list, measure the rest pose and
        # filter every pair that overlaps.
        self.excluded_pairs = self._exclude_overlapping_pairs(processor) if self_collision else []

        # No floor means no plane: Bullet's static plane is infinite and its
        # placement is not to be relied on -- one put a kilometre down still
        # gripped the body (a free joint's velocity killed in one step).
        self.plane = None
        self.floor_y = floor_height
        if floor:
            plane = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0],
                                           physicsClientId=self.cid)
            self.plane = p.createMultiBody(0, plane, basePosition=[0, floor_height, 0],
                                           physicsClientId=self.cid)
            # the floor meets free links (group 1) and driven links on a free
            # root (group 2); see set_link_collisions
            p.setCollisionFilterGroupMask(self.plane, -1, 1, 3, physicsClientId=self.cid)
        self.set_friction(friction)
        # Penetration is resolved as a position correction, not a velocity: a
        # link released while overlapping the floor or another link is eased
        # out rather than fired.
        p.setPhysicsEngineParameter(useSplitImpulse=1, splitImpulsePenetrationThreshold=-0.01,
                                    physicsClientId=self.cid)
        self.collide = np.ones(24, dtype=bool)
        # No bounce (a body lands, it does not rebound), no air drag, and no
        # sleeping: Bullet deactivates a body at rest, and a driven body that
        # is placed each frame with its joints still reads as at rest -- a
        # free arm on it then never falls, its joint simply not integrated.
        for li in range(-1, 23):
            p.changeDynamics(self.body, li, restitution=0.0, linearDamping=0.0,
                             angularDamping=0.0,
                             activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
                             physicsClientId=self.cid)

    def _exclude_overlapping_pairs(self, processor, margin=0.01):
        parents = list(processor._get_hierarchy())
        offsets = np.asarray(processor.skeleton_offsets)
        children = {j: [] for j in range(24)}
        for j in range(1, 24):
            children[parents[j]].append(j)
        # Rest-pose joint positions and capsule end points (world = rest frame).
        pos = np.zeros((24, 3))
        for j in range(1, 24):
            pos[j] = pos[parents[j]] + offsets[j]
        ends = np.zeros((24, 3))
        for j in range(24):
            kids = children[j]
            if kids:
                ends[j] = np.mean([pos[c] for c in kids], axis=0)
            else:
                tip = [v for v in range(24, min(28, offsets.shape[0])) if v < len(parents) and parents[v] == j]
                if tip:
                    ends[j] = pos[j] + offsets[tip[0]]
                else:
                    d = offsets[j] if j > 0 else np.array([0.0, 1.0, 0.0])
                    n = np.linalg.norm(d)
                    d = d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
                    ends[j] = pos[j] + d * max(float(processor._seg_length[j]), 0.05)

        def seg_dist(a0, a1, b0, b1):
            d1, d2, r = a1 - a0, b1 - b0, a0 - b0
            A, E, F, C, B = d1 @ d1, d2 @ d2, d2 @ r, d1 @ r, d1 @ d2
            den = A * E - B * B
            s_ = float(np.clip((B * F - C * E) / den, 0, 1)) if den > 1e-12 else 0.0
            t_ = (B * s_ + F) / E if E > 1e-12 else 0.0
            t_ = float(np.clip(t_, 0, 1))
            s_ = float(np.clip((B * t_ - C) / A, 0, 1)) if A > 1e-12 else 0.0
            return float(np.linalg.norm((a0 + d1 * s_) - (b0 + d2 * t_)))

        excluded = []
        for a in range(24):
            for b in range(a + 1, 24):
                if parents[b] == a or parents[a] == b:
                    continue                      # Bullet already excludes these
                reach = SEG_RADIUS.get(a, 0.04) + SEG_RADIUS.get(b, 0.04) + margin
                if seg_dist(pos[a], ends[a], pos[b], ends[b]) < reach:
                    p.setCollisionFilterPair(self.body, self.body, a - 1, b - 1, 0,
                                             physicsClientId=self.cid)
                    excluded.append((JOINT_NAMES[a], JOINT_NAMES[b]))
        return excluded

    def set_floor(self, height):
        if self.plane is not None and abs(height - self.floor_y) > 1e-6:
            p.resetBasePositionAndOrientation(self.plane, [0, height, 0], [0, 0, 0, 1],
                                              physicsClientId=self.cid)
            self.floor_y = height

    def lowest_point(self):
        """The lowest capsule surface in the body, in world -- what would touch
        a floor first.  Approximated as the lowest link frame or virtual tip
        minus its radius."""
        jp = self.joint_positions()
        low = np.inf
        for j in range(24):
            low = min(low, jp[j, 1] - SEG_RADIUS.get(j, 0.04))
        # Toes and finger tips, from the processor's virtual offsets.
        offsets = np.asarray(self.processor.skeleton_offsets)
        parents = list(self.processor._get_hierarchy())
        states = p.getLinkStates(self.body, list(range(23)), computeForwardKinematics=1,
                                 physicsClientId=self.cid)
        # Toes, finger tips and heels: the heel is what stands on the ground.
        for v in (24, 25, 26, 27, 28, 29):
            if v < len(parents) and v < offsets.shape[0]:
                pa = parents[v]
                orn = states[pa - 1][5] if pa > 0 else p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)[1]
                tip = jp[pa] + R.from_quat(orn).apply(offsets[v])
                low = min(low, tip[1] - 0.02)
        return float(low)

    def set_link_collisions(self, enabled):
        """Which links collide, indexed by joint (0 = pelvis/base).  A driven
        link is kinematic and must not: held against the floor or another link
        by the root constraint and the drive motors, the contact solver fought
        back with fifteen kilonewtons on a foot, all of it released as a kick
        the moment the joint went free."""
        # 0: collides with nothing.  1: free link -- collides with the floor
        # (group 1), other free links and driven links.  2: driven link on a
        # free root -- a motor-held part of a floating body, it rests on the
        # floor and meets free links, but not the other driven links its
        # motors already hold in place.
        for j in range(24):
            want = int(enabled[j])
            if want != self.collide[j]:
                group, mask = ((0, 0), (1, 3), (2, 1))[want]
                p.setCollisionFilterGroupMask(self.body, j - 1, group, mask, physicsClientId=self.cid)
                self.collide[j] = want

    def set_friction(self, mu):
        if self.plane is not None:
            p.changeDynamics(self.plane, -1, lateralFriction=mu, physicsClientId=self.cid)
        for li in range(-1, 23):
            p.changeDynamics(self.body, li, lateralFriction=mu, physicsClientId=self.cid)

    def set_pose(self, aa24, trans, base_vel=None, base_ang_vel=None, joint_vel=None):
        """Place the body: root rotation and translation, and every joint's
        local rotation, with optional velocities (world linear and angular for
        the base; per joint, the spherical joint's own angular velocity)."""
        root_r = R.from_rotvec(aa24[0])
        root = root_r.as_quat()
        p.resetBasePositionAndOrientation(self.body, list(np.asarray(trans) + root_r.apply(self.base_offset)),
                                          list(root), physicsClientId=self.cid)
        if base_vel is not None:
            p.resetBaseVelocity(self.body, list(base_vel),
                                list(base_ang_vel if base_ang_vel is not None else (0, 0, 0)),
                                physicsClientId=self.cid)
        for j in range(1, 24):
            q = R.from_rotvec(aa24[j]).as_quat()
            if joint_vel is not None and j in joint_vel:
                p.resetJointStateMultiDof(self.body, j - 1, list(q), list(joint_vel[j]),
                                          physicsClientId=self.cid)
            else:
                p.resetJointStateMultiDof(self.body, j - 1, list(q), physicsClientId=self.cid)

    def joint_positions(self):
        """World position of every joint (link frame origins), (24, 3)."""
        out = np.zeros((24, 3))
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        out[0] = np.asarray(pos) - R.from_quat(orn).apply(self.base_offset)
        states = p.getLinkStates(self.body, list(range(23)), computeForwardKinematics=1,
                                 physicsClientId=self.cid)
        for j in range(1, 24):
            out[j] = states[j - 1][4]        # world link frame position
        return out

    def read_pose(self):
        """(aa24, trans) from the engine's state."""
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        aa = np.zeros((24, 3))
        aa[0] = R.from_quat(orn).as_rotvec()
        pos = np.asarray(pos) - R.from_quat(orn).apply(self.base_offset)
        states = p.getJointStatesMultiDof(self.body, list(range(23)), physicsClientId=self.cid)
        for j in range(1, 24):
            aa[j] = R.from_quat(states[j - 1][0]).as_rotvec()
        return aa, np.asarray(pos, dtype=float)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass


# ----------------------------------------------------------------------
# The simulation
# ----------------------------------------------------------------------

# Two pybullet conventions not taken on trust; a probe sets them.
JOINT_VEL_IN_CHILD_FRAME = True     # spherical joint velocity: child link frame?
# Probed on this pybullet (3.25): WORLD_FRAME means world.  (Older versions
# had the flags swapped, issue #1949; the probe in the scratch tests settles it.)
TORQUE_FLAG_WORLD = p.WORLD_FRAME if PYBULLET_AVAILABLE else None


def _log_map(mat):
    """Rotation vector of a rotation matrix (small-angle safe)."""
    c = 0.5 * (mat[0, 0] + mat[1, 1] + mat[2, 2] - 1.0)
    c = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)
    theta = math.acos(c)
    sn = math.sin(theta)
    if theta < 1e-7:
        return np.zeros(3)
    if sn < 1e-6:
        return R.from_matrix(mat).as_rotvec()
    f = theta / (2.0 * sn)
    return np.array(((mat[2, 1] - mat[1, 2]) * f,
                     (mat[0, 2] - mat[2, 0]) * f,
                     (mat[1, 0] - mat[0, 1]) * f))


class _Seed:
    """Smoothed velocity estimate with the averaging lag cancelled, for the
    momentum a release inherits.  Same design as the native core's."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.prev_raw = None
        self.value = np.zeros(3)

    def push(self, raw, dt):
        a = self.alpha
        self.vel = self.vel + a * (raw - self.vel)
        if self.prev_raw is not None:
            self.acc = self.acc + a * ((raw - self.prev_raw) / dt - self.acc)
        self.prev_raw = raw
        self.value = self.vel + self.acc * ((1.0 - a) / a * dt)
        return self.value


class BulletRagdollSim:
    """Interface-compatible with the native SMPLRagdollSim: the node does not
    know which core it is driving.

    While a joint is driven (weight 1) it is placed every frame at the captured
    rotation *with its captured angular velocity*, so the engine's state is
    always "here, moving like this" -- and release is nothing but ceasing to
    place it.  That one detail is the whole difference from the earlier
    attempt, which placed poses and never velocities, so a body released in
    mid-air stopped dead.  Below weight 1 a joint is a motor whose force limit
    is the weight times the joint's strength: weight 0 is free, 0.3 is a limb
    that tries and sags.  The root, while driven, is placed the same way; below
    weight 1 a spring on the base pulls it toward the capture in proportion.
    """

    def __init__(self, processor):
        from dpg_system.smpl_ragdoll import RAGDOLL_JOINT_LIMITS   # lazy: circular import
        self.processor = processor
        self.free_indices = []
        self.root_free = False
        self.body = None
        self.floor_height = 0.0
        self.friction = 0.8
        self.parents = list(processor._get_hierarchy())
        self.joint_names = JOINT_NAMES
        self.max_torque = np.asarray(processor.max_torque_array)[:24]
        self.link_masses = np.maximum(np.asarray(processor._seg_mass, dtype=float)[:24], 0.05)
        self.mass = float(self.link_masses.sum())
        self._body_weight = self.mass * 9.81

        # Per-axis limit boxes, Euler order per joint (tightest axis in the middle).
        self._lim_min = np.zeros((24, 3)); self._lim_max = np.zeros((24, 3))
        self._lim_k = np.zeros(24); self._lim_active = np.zeros(24, dtype=bool)
        self._lim_order = [(0, 1, 2)] * 24
        for j in range(1, 22):
            lim = RAGDOLL_JOINT_LIMITS.get(JOINT_NAMES[j])
            if lim is None:
                continue
            self._lim_min[j] = lim[0]; self._lim_max[j] = lim[1]
            self._lim_k[j] = max(4.0 * float(np.mean(self.max_torque[j])), 30.0)
            self._lim_active[j] = True
            span = np.asarray(lim[1]) - np.asarray(lim[0])
            first, mid = int(np.argmax(span)), int(np.argmin(span))
            if first == mid:
                first, mid = 0, 1
            self._lim_order[j] = (first, mid, 3 - first - mid)
        # Swing-twist axes: the coordinate axis nearest the bone is the twist
        # axis, the other two are the swing.  From the skeleton's segment
        # directions (mean of the children's offsets; a leaf continues its
        # parent's bone).
        offsets = np.asarray(processor.skeleton_offsets)
        kids = {j: [] for j in range(24)}
        for j in range(1, 24):
            kids[self.parents[j]].append(j)
        self._twist_axis = np.zeros(24, dtype=int)
        self._twist_sign = np.ones(24)
        for j in range(1, 22):
            if kids[j]:
                d = np.mean([offsets[c] for c in kids[j]], axis=0)
            else:
                d = offsets[j]
            k = int(np.argmax(np.abs(d)))
            self._twist_axis[j] = k
            self._twist_sign[j] = 1.0 if d[k] >= 0 else -1.0
        self.reset()

    # -- configuration -------------------------------------------------

    def set_free_joints(self, indices):
        # Every joint is always simulated; "free" only decides whose weight
        # matters.  Changing the set does not reset the simulation.
        self.free_indices = sorted({int(i) for i in indices if 1 <= int(i) < 22})

    def set_root_free(self, free):
        self.root_free = bool(free)

    def reset(self):
        self.seeded = False
        if getattr(self, 'root_constraint', None) is not None and self.body is not None:
            try:
                p.removeConstraint(self.root_constraint, physicsClientId=self.body.cid)
            except Exception:
                pass
        self.root_constraint = None
        self.reel = None
        self.reel_d0 = 1.0
        self.prev_cap = None
        self.joint_reel = {}          # j -> [target orientation on its way to the capture, rad/s]
        self.prev_w = np.zeros(24)
        self.was_free = np.zeros(24, dtype=bool)
        self.floor_est = None
        self.floor_level = 0.0
        self.lim_margin = [None] * 24
        self.lim_margin0 = [None] * 24
        self.base_seed = _Seed(); self.base_ang_seed = _Seed()
        self.joint_seed = {j: _Seed() for j in range(1, 24)}
        self.joint_speed_ema = np.full(24, 1.0)
        self.prev_root = None
        self.prev_joint = {}
        self.was_prescribed = np.ones(24, dtype=bool)
        self.last_torque = np.zeros((22, 3))
        self.last_inertia = np.zeros(22)
        self.last_contact_force = np.zeros((24, 3))
        self.last_energy_injected = 0.0
        self.last_support = 1.0
        self.com = None; self.com_vel = np.zeros(3)
        self.root_ang_vel = np.zeros(3); self.ang_momentum = np.zeros(3)
        self.root_rot = None; self.trans = None

    def _ensure_body(self, p_):
        key = (bool(p_.floor_enable), bool(p_.self_collision))
        if self.body is None or self.floor_height != key:
            if self.body is not None:
                self.body.close()
            self.floor_height = key
            self.body = BulletBody(self.processor, floor_height=p_.floor_height,
                                   friction=p_.friction, floor=bool(p_.floor_enable),
                                   self_collision=bool(p_.self_collision))
            self.root_constraint = None
            self.seeded = False
            self.friction = p_.friction
            # The root constraint's error reduction.  changeConstraint's own
            # erp argument has no effect (measured: a 6.7 mm lag at every
            # value), and at Bullet's default of 0.2 the driven root trails a
            # moving capture by four substeps -- one frame -- which is what a
            # release then inherits.  The engine-wide parameter does work: at
            # 1.0 the lag is zero with no overshoot.
            p.setPhysicsEngineParameter(numSolverIterations=int(p_.solver_iterations), erp=p_.root_erp,
                                        physicsClientId=self.body.cid)
        elif self.friction != p_.friction:
            self.body.set_friction(p_.friction); self.friction = p_.friction

    # -- limits --------------------------------------------------------

    _EVEN = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}

    def _limit_angles(self, j, mat):
        """Euler angles per axis in the joint's order (true angles about the
        original axes), from the joint's rotation matrix."""
        i, k, m = self._lim_order[j]
        sb = mat[i, m]; sb = 1.0 if sb > 1.0 else (-1.0 if sb < -1.0 else sb)
        beta = math.asin(sb)
        alpha = math.atan2(-mat[k, m], mat[m, m])
        gamma = math.atan2(-mat[i, k], mat[i, i])
        if (i, k, m) not in self._EVEN:
            alpha, beta, gamma = -alpha, -beta, -gamma
        ang = np.zeros(3); ang[i], ang[k], ang[m] = alpha, beta, gamma
        return ang

    def _swing_twist(self, j, quat):
        """Decompose the joint's rotation (child relative to parent, xyzw)
        into a twist about the bone axis and a swing of the bone.  Returns
        (angles indexed by table axis: twist on the bone axis, the two swing
        components on the others; the twist quaternion; the swing quaternion),
        with q = swing * twist.
        """
        k = self._twist_axis[j]
        x, y, z, w = quat
        v = np.array([x, y, z])
        proj = np.zeros(3); proj[k] = v[k]
        n = math.sqrt(proj[k] * proj[k] + w * w)
        if n < 1e-12:
            q_twist = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            q_twist = np.array([proj[0] / n, proj[1] / n, proj[2] / n, w / n])
        r_twist = R.from_quat(q_twist)
        r_swing = R.from_quat(quat) * r_twist.inv()
        ang = np.zeros(3)
        ang[k] = 2.0 * math.atan2(q_twist[k], q_twist[3])
        if ang[k] > math.pi:
            ang[k] -= 2.0 * math.pi
        elif ang[k] < -math.pi:
            ang[k] += 2.0 * math.pi
        sw = r_swing.as_rotvec()
        for ax in range(3):
            if ax != k:
                ang[ax] = sw[ax]
        return ang, r_twist, r_swing

    def _limit_target(self, j, mat, margin=None):
        """If the joint is outside its limits, the nearest allowed orientation
        as a quaternion; None if it is inside.  `margin` widens the limits per
        axis -- the soft entry for a joint that went free outside them.

        Limits are swing-twist, not Euler.  In Euler angles the twist about
        the bone is entangled with the swing, and a limp arm hanging under
        gravity could come to rest against a *twist* stop that had nothing to
        do with where the bone pointed -- held up, twisted, in a pose no joint
        would hold.  Here the twist about the bone and the swing of the bone
        are limited independently; gravity acts only on the swing, so no
        twist stop can ever support the arm.

        Enforced through Bullet's own joint motor, commanded toward this
        target with a bounded force: the motor is solved implicitly and is
        stable at any stiffness, where an explicit stop torque at a 240 Hz
        step exceeded the stability bound ninefold on a hand.
        """
        quat = R.from_matrix(mat).as_quat()
        ang, r_twist, r_swing = self._swing_twist(j, quat)
        lo, hi = self._lim_min[j], self._lim_max[j]
        if margin is not None:
            lo = lo - margin
            hi = hi + margin
        clamped = np.minimum(np.maximum(ang, lo), hi)
        if np.all(clamped == ang):
            return None
        k = self._twist_axis[j]
        tw = np.zeros(3); tw[k] = clamped[k]
        sw = np.zeros(3)
        for ax in range(3):
            if ax != k:
                sw[ax] = clamped[ax]
        return (R.from_rotvec(sw) * R.from_rotvec(tw)).as_quat()

    # -- the frame ---------------------------------------------------------

    def advance(self, mocap_aa, mocap_trans, weights, p_):
        self._ensure_body(p_)
        body = self.body; cid = body.cid
        dt = p_.dt
        mocap_aa = np.asarray(mocap_aa, dtype=float).reshape(-1, 3)[:24].copy()
        mocap_trans = np.asarray(mocap_trans, dtype=float).reshape(3)
        w = np.ones(24); w[:22] = np.clip(weights[:22], 0.0, 1.0)
        if not self.root_free:
            w[0] = 1.0
        for j in range(1, 22):
            if j not in self.free_indices:
                w[j] = 1.0
        prescribed = w >= 1.0 - 1e-6
        n_sub = max(1, min(int(p_.substeps), int(round(p_.substep_rate * dt)) or 1))
        h = dt / n_sub

        # -- velocity estimates from the capture, for every joint and the base
        root_rot = R.from_rotvec(mocap_aa[0])
        prev_root, prev_trans = self.prev_root, getattr(self, 'prev_trans', None)
        if prev_root is None:
            base_v = np.zeros(3); base_w = np.zeros(3)
        else:
            base_v = self.base_seed.push((mocap_trans - prev_trans) / dt, dt)
            base_w = self.base_ang_seed.push(
                _log_map((root_rot * prev_root.inv()).as_matrix()) / dt, dt)
        self.prev_root = root_rot; self.prev_trans = mocap_trans.copy()
        joint_w = {}
        for j in range(1, 24):
            q = R.from_rotvec(mocap_aa[j])
            prev = self.prev_joint.get(j)
            if prev is not None:
                # child-frame rate: q_prev^-1 q ; parent-frame: q q_prev^-1
                rel = (prev.inv() * q) if JOINT_VEL_IN_CHILD_FRAME else (q * prev.inv())
                raw = _log_map(rel.as_matrix()) / dt
                # Capture glitches arrive as single-frame spikes of twenty
                # radians a second on a two radian a second mean; a motor fed
                # that as a target rate whips the limb.  Clip to a few times
                # the joint's running speed, never below a floor.
                cap = max(p_.spike_ratio * self.joint_speed_ema[j], p_.spike_floor)
                n_ = float(np.linalg.norm(raw))
                if n_ > cap:
                    raw = raw * (cap / n_)
                self.joint_speed_ema[j] += 0.1 * (min(n_, cap) - self.joint_speed_ema[j])
                joint_w[j] = self.joint_seed[j].push(raw, dt)
            else:
                joint_w[j] = np.zeros(3)
            self.prev_joint[j] = q

        # -- the root: a fixed constraint to the world, retargeted through the
        # frame.  Not a reset: a multibody whose base is reset every frame
        # stops applying gravity to its free joints unless something else
        # touches it (a free arm on a driven pelvis never fell, floor off),
        # and a reset carries no velocity into a release.  The constraint
        # solver moves the base, so its velocity state is real -- provided the
        # target keeps moving until the last substep: a constraint closes its
        # error at a rate, and a target held fixed for the frame leaves the
        # base nearly stopped by the end of it (release velocity came out at
        # 83 percent of the capture's).  So the target walks the path from
        # the previous pose to the current one, substep by substep.
        #
        # A catch reels: the target moves from where the body *is* toward the
        # capture at a bounded speed, and exact tracking resumes once it has
        # caught up.  A fixed constraint limited only by force snapped a
        # fallen body back at thirty metres per second squared.
        cap_pos = mocap_trans + root_rot.apply(body.base_offset)
        cap_rot = root_rot
        if not self.seeded:
            body.set_pose(mocap_aa, mocap_trans, base_v, base_w, joint_w)
            self.seeded = True
            self.prev_cap = (cap_pos.copy(), cap_rot)
            self.reel = None
        prev_cap_pos, prev_cap_rot = self.prev_cap
        root_targets = None
        if w[0] > 1e-6:
            pos_now, orn_now = p.getBasePositionAndOrientation(body.body, physicsClientId=cid)
            if self.root_constraint is None:
                self.root_constraint = p.createConstraint(
                    body.body, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                    list(pos_now), childFrameOrientation=list(orn_now), physicsClientId=cid)
                # A catch completes over the ramp, however far the body has
                # gone: the reel's speed is the distance over the ramp time,
                # never below a floor, capped so a body a long way off is
                # brought back quickly rather than teleported.
                d0 = float(np.linalg.norm(cap_pos - np.asarray(pos_now)))
                a0 = float((cap_rot * R.from_quat(orn_now).inv()).magnitude())
                ramp = max(p_.ramp_s, 1e-3)
                self.reel_speed = min(max(d0 / ramp, p_.root_catch_speed), 15.0)
                self.reel_rate = min(max(a0 / ramp, p_.root_catch_rate), 30.0)
                self.reel = (np.asarray(pos_now), R.from_quat(orn_now))
                self.reel_d0 = max(d0, 1e-3)
            if self.reel is not None:
                # Reeling in: is the target close enough to the capture yet?
                d = float(np.linalg.norm(cap_pos - self.reel[0]))
                a = float((cap_rot * self.reel[1].inv()).magnitude())
                if d < 0.02 and a < 0.05:
                    self.reel = None
            force = (p_.root_hold_force if prescribed[0]
                     else max(float(w[0]) * p_.root_tether, 0.002) * p_.root_hold_force)
            root_targets = []
            for k in range(1, n_sub + 1):
                if self.reel is None:
                    f_ = k / n_sub
                    pos_k = prev_cap_pos + (cap_pos - prev_cap_pos) * f_
                    rot_k = R.from_rotvec(
                        (cap_rot * prev_cap_rot.inv()).as_rotvec() * f_) * prev_cap_rot
                else:
                    rp, rr = self.reel
                    step = cap_pos - rp
                    dn = float(np.linalg.norm(step))
                    lim = self.reel_speed * h
                    if dn > lim:
                        step = step * (lim / dn)
                    rp = rp + step
                    rv = (cap_rot * rr.inv()).as_rotvec()
                    an = float(np.linalg.norm(rv))
                    alim = self.reel_rate * h
                    if an > alim:
                        rv = rv * (alim / an)
                    rr = R.from_rotvec(rv) * rr
                    self.reel = (rp, rr)
                    pos_k, rot_k = rp, rr
                root_targets.append((pos_k, rot_k, force))
        elif self.root_constraint is not None:
            p.removeConstraint(self.root_constraint, physicsClientId=cid)
            self.root_constraint = None
            self.reel = None
            # Release: the one moment a velocity is set outright, from the
            # smoothed estimate with the averaging lag cancelled.
            p.resetBaseVelocity(body.body, list(base_v), list(base_w), physicsClientId=cid)
        self.prev_cap = (cap_pos.copy(), cap_rot)

        # While the root is being reeled in, the joints come back with it:
        # their motor strength scales with the reel's progress, so the whole
        # body reassembles over the catch.  Snapped to pose by full-strength
        # motors the instant the ramp ended, the limbs of a body reeled from
        # 1.4 metres away were still flailing at seven metres a second two
        # frames later.
        reel_progress = 1.0
        if self.reel is not None:
            reel_progress = float(np.clip(
                1.0 - np.linalg.norm(cap_pos - self.reel[0]) / self.reel_d0, 0.05, 1.0))

        # The floor follows the capture while the root is driven: the lowest
        # capsule bottom in the body is where the ground is -- a capture whose
        # feet sat four centimetres under y=0 released a foot from inside the
        # floor, and it was kicked out at thirty radians a second.  Instant to
        # drop, slow to rise, so a jump does not pull the floor up after it.
        if body.plane is not None:
            if p_.floor_auto and prescribed[0]:
                low = body.lowest_point()
                if self.floor_est is None or low < self.floor_est:
                    self.floor_est = low
                else:
                    self.floor_est += p_.floor_rise * dt * (low - self.floor_est)
                body.set_floor(self.floor_est)
            elif not p_.floor_auto:
                body.set_floor(p_.floor_height)
        self.floor_level = body.floor_y

        # Driven links on a driven root do not collide (kinematic, held
        # against the floor by the root constraint); on a free root they
        # collide with the floor and the free links, or the pelvis is the only
        # thing that meets the ground and the rest of the body goes through it.
        collide = np.ones(24, dtype=int)
        collide[0] = 0 if prescribed[0] else 1
        for j in range(1, 22):
            collide[j] = 1 if not prescribed[j] else (0 if prescribed[0] else 2)
        collide[22] = collide[20]
        collide[23] = collide[21]
        body.set_link_collisions(collide)

        # -- motors: strong for the driven, scaled for the partially driven,
        #    none for the free --------------------------------------------------
        for j in range(1, 22):
            li = j - 1
            if w[j] <= 1e-6:
                self.joint_reel.pop(j, None)
                p.setJointMotorControlMultiDof(body.body, li, p.POSITION_CONTROL,
                                               targetPosition=[0, 0, 0, 1], force=[0, 0, 0],
                                               physicsClientId=cid)
                continue
            # A joint taking hold is reeled like the root: its target travels
            # from where the joint is to the capture at a bounded rate, so the
            # motor error stays small.  Aimed straight at the capture with only
            # its force scaled, a caught joint sat 2.8 radians off when the
            # weight reached one and the full-strength motor slammed it at a
            # hundred and fifty radians a second, ringing for most of a second
            # -- and a release in that window inherited the ringing as a flick.
            q_cap = R.from_rotvec(mocap_aa[j])
            if self.prev_w[j] <= 1e-6:
                q_now = R.from_quat(p.getJointStateMultiDof(body.body, li, physicsClientId=cid)[0])
                a0 = float((q_cap * q_now.inv()).magnitude())
                self.joint_reel[j] = [q_now, min(max(a0 / max(p_.ramp_s, 1e-3), p_.root_catch_rate), 30.0)]
            ff = joint_w[j]
            if j in self.joint_reel:
                q_r, rate = self.joint_reel[j]
                rv = (q_cap * q_r.inv()).as_rotvec()
                an = float(np.linalg.norm(rv))
                if an <= rate * dt + 0.05:
                    self.joint_reel.pop(j)
                else:
                    q_r = R.from_rotvec(rv * (rate * dt / an)) * q_r
                    self.joint_reel[j][0] = q_r
                    q_cap = q_r
                    ff = np.zeros(3)
            # Full strength from the start of a catch: the root is reeled in at
            # metres a second, and joints weakened for the reel could not
            # follow the pelvis -- the limbs whipped at a hundred and forty
            # radians a second.  The rate-bounded target above is what keeps
            # the joint from snapping, not a weak motor.
            if prescribed[j]:
                p.setJointMotorControlMultiDof(
                    body.body, li, p.POSITION_CONTROL,
                    targetPosition=list(q_cap.as_quat()),
                    targetVelocity=list(ff),
                    positionGain=p_.drive_kp, velocityGain=p_.drive_kd,
                    force=[p_.drive_force] * 3, physicsClientId=cid)
            else:
                f = float(w[j]) * self.max_torque[j] * p_.motor_strength
                p.setJointMotorControlMultiDof(
                    body.body, li, p.POSITION_CONTROL,
                    targetPosition=list(q_cap.as_quat()),
                    targetVelocity=list(ff),
                    positionGain=p_.motor_kp, velocityGain=p_.motor_kd,
                    force=[float(v) for v in f], physicsClientId=cid)
        self.prev_w = w.copy()
        for j in range(22, 24):        # hands: always driven (massless)
            p.setJointMotorControlMultiDof(
                body.body, j - 1, p.POSITION_CONTROL,
                targetPosition=list(R.from_rotvec(mocap_aa[j]).as_quat()),
                targetVelocity=list(joint_w[j]), positionGain=p_.drive_kp,
                velocityGain=p_.drive_kd, force=[p_.drive_force] * 3, physicsClientId=cid)

        # -- step, with the limit motors on the free joints -------------------
        p.setTimeStep(h, physicsClientId=cid)
        self.last_torque[:] = 0.0
        free_joints = [j for j in range(1, 22) if not prescribed[j] and self._lim_active[j]
                       and w[j] <= 1e-6]
        # Soft entry.  A capture is feasible for the performer by definition;
        # a limit tighter than what they did is my error, not theirs, and a
        # joint that goes free while outside its box must be eased in, never
        # snapped -- released outside its box, a spine was slammed back at 66
        # radians a second by a stop at full stiffness.  The box is widened by
        # however far outside the joint is when it goes free, and that margin
        # shrinks to nothing over `limit_entry_s`.
        for j in free_joints:
            if not self.was_free[j]:
                st = p.getJointStateMultiDof(body.body, j - 1, physicsClientId=cid)
                ang, _t, _s = self._swing_twist(j, np.asarray(st[0]))
                self.lim_margin[j] = (np.maximum(self._lim_min[j] - ang, 0.0)
                                      + np.maximum(ang - self._lim_max[j], 0.0)) + 0.02
            elif self.lim_margin[j] is not None:
                self.lim_margin[j] = np.maximum(
                    self.lim_margin[j] - dt / max(p_.limit_entry_s, 1e-3) * self.lim_margin0[j], 0.0)
                if not np.any(self.lim_margin[j] > 0.0):
                    self.lim_margin[j] = None
            if not self.was_free[j]:
                self.lim_margin0[j] = self.lim_margin[j].copy()
        self.was_free[:] = False
        for j in free_joints:
            self.was_free[j] = True
        for k_sub in range(n_sub):
            if root_targets is not None:
                pos_k, rot_k, force = root_targets[k_sub]
                p.changeConstraint(self.root_constraint, jointChildPivot=list(pos_k),
                                   jointChildFrameOrientation=list(rot_k.as_quat()),
                                   maxForce=force, physicsClientId=cid)
            if free_joints:
                states = p.getJointStatesMultiDof(body.body, [j - 1 for j in free_joints],
                                                  physicsClientId=cid)
                for j, st in zip(free_joints, states):
                    mat = R.from_quat(st[0]).as_matrix()
                    target = self._limit_target(j, mat, self.lim_margin[j])
                    k_ = self._lim_k[j] * p_.limit_stiffness
                    if target is None:
                        p.setJointMotorControlMultiDof(
                            body.body, j - 1, p.POSITION_CONTROL, targetPosition=list(st[0]),
                            targetVelocity=[0, 0, 0], positionGain=0.0,
                            velocityGain=p_.joint_damping_gain,
                            force=[p_.joint_damping_fraction * float(np.mean(self.max_torque[j]))] * 3,
                            physicsClientId=cid)
                        self.last_torque[j] = 0.0
                    else:
                        p.setJointMotorControlMultiDof(
                            body.body, j - 1, p.POSITION_CONTROL, targetPosition=list(target),
                            targetVelocity=[0, 0, 0], positionGain=p_.limit_gain,
                            velocityGain=p_.limit_damping_gain,
                            force=[float(k_)] * 3, physicsClientId=cid)
                        self.last_torque[j] = 1.0
            p.stepSimulation(physicsClientId=cid)

        # -- read back ---------------------------------------------------------
        aa, trans = body.read_pose()
        pos, orn = p.getBasePositionAndOrientation(body.body, physicsClientId=cid)
        lv, av = p.getBaseVelocity(body.body, physicsClientId=cid)
        self.root_rot = R.from_quat(orn); self.trans = np.asarray(trans)
        self.root_ang_vel = np.asarray(av)
        # The body's centre of mass and its velocity, mass-weighted over the
        # links' own centres -- not the pelvis, which orbits it as the body
        # turns.
        states = p.getLinkStates(body.body, list(range(23)), computeLinkVelocity=1,
                                 physicsClientId=cid)
        com = self.link_masses[0] * np.asarray(pos)
        vel = self.link_masses[0] * np.asarray(lv)
        for li, st in enumerate(states):
            com += self.link_masses[li + 1] * np.asarray(st[0])
            vel += self.link_masses[li + 1] * np.asarray(st[6])
        self.com = com / self.mass
        self.com_vel = vel / self.mass
        self.was_prescribed = prescribed

        # contact, per link, for the outputs and the support measure
        self.last_contact_force[:] = 0.0
        supplied = 0.0
        contacts = (p.getContactPoints(bodyA=body.body, bodyB=body.plane, physicsClientId=cid)
                    if body.plane is not None else [])
        for c in contacts:
            li = c[3]; fn = c[9]; nrm = np.asarray(c[7])
            self.last_contact_force[li + 1] += -nrm * fn
            supplied += max(fn, 0.0)
        # Support: what the captured motion needs against what is actually
        # holding the body up.  A driven link at floor level supplies it in
        # full -- it is kinematic and does not collide, so it shows no contact
        # force, but it holds whatever is asked of it.  A released limb counts
        # only for the contact force it really produces, which drops as its
        # knee buckles; and a ballistic capture asks for nothing, so flight is
        # exempt.  This is what lets the root go when the leg under it has
        # been released: continuing the captured translation on a limb that
        # no longer holds makes no physical sense.
        needed = self.mass * (self.base_seed.acc[1] + 9.81 * p_.gravity)
        if needed > 0.05 * self._body_weight:
            held = float(np.clip(supplied / needed, 0.0, 1.0))
            if body.plane is not None:
                held = max(held, self._capture_support(prescribed, mocap_aa, mocap_trans, p_))
            self.last_support = held
        else:
            self.last_support = 1.0

        result = {j: aa[j] for j in range(1, 22) if not prescribed[j]}
        if prescribed[0]:
            return result, root_rot, mocap_trans
        return result, self.root_rot, self.trans

    def _capture_lows(self, mocap_aa, mocap_trans):
        """Forward kinematics of the captured pose: the lowest point of every
        link's capsule (joints 0-21) and of the toe / finger / heel tips,
        keyed by the link that carries them.  Captured, not simulated: a
        released leg's simulated copy is limp on the floor and says nothing
        about where the capture is standing."""
        offsets = np.asarray(self.processor.skeleton_offsets)
        n = min(len(self.parents), offsets.shape[0])
        pos = np.zeros((n, 3))
        rot = [None] * n
        pos[0] = np.asarray(mocap_trans, dtype=float)
        rot[0] = R.from_rotvec(mocap_aa[0])
        lows = {0: pos[0, 1] - SEG_RADIUS.get(0, 0.05)}
        low_xz = {0: pos[0, [0, 2]]}
        for j in range(1, n):
            pa = self.parents[j]
            pos[j] = pos[pa] + rot[pa].apply(offsets[j])
            if j < 24:
                rot[j] = rot[pa] * R.from_rotvec(mocap_aa[j])
            # hands (22, 23) and the virtual tips belong to the link that
            # carries them: the wrist, the foot, the ankle
            link = j
            while link >= 22:
                link = self.parents[link]
            r = SEG_RADIUS.get(j, 0.04) if j < 22 else 0.02
            if pos[j, 1] - r < lows.get(link, np.inf):
                lows[link] = pos[j, 1] - r
                low_xz[link] = pos[j, [0, 2]]
        self._last_lows = lows
        self._last_fk = pos
        self._low_xz = low_xz
        return lows

    def _capture_support(self, prescribed, mocap_aa, mocap_trans, p_):
        """How much of the captured pose's weight the driven links can be
        carrying, 0..1.  The capture's ground-level points are those within
        support_tolerance of its lowest point; the share falls to the driven
        ones by how much nearer the pelvis stands over them than over the
        released ones.  Height alone cannot tell -- in a walk the two feet are
        within centimetres of each other for most of the cycle -- and no floor
        plane is involved: capture foot heights wander, and a floor estimate
        that keeps the deepest heel strike sits below where the foot is now."""
        lows = self._capture_lows(mocap_aa, mocap_trans)
        floor = min(lows.values())
        cands = [j for j, v in lows.items() if v <= floor + p_.support_tolerance]
        driven = [j for j in cands if prescribed[j]]
        free = [j for j in cands if not prescribed[j]]
        if not driven:
            return 0.0
        if not free:
            return 1.0
        hip = self._last_fk[0, [0, 2]]
        d_d = min(np.linalg.norm(self._low_xz[j] - hip) for j in driven)
        d_f = min(np.linalg.norm(self._low_xz[j] - hip) for j in free)
        return float(d_f ** 2 / (d_d ** 2 + d_f ** 2 + 1e-9))

    def close(self):
        if self.body is not None:
            self.body.close(); self.body = None
