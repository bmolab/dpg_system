"""Per-segment limb scaling for SMPL / SMPL-H meshes.

The scaling happens in the REST pose, before posing, so it never shears a child
limb: the rest joints are rebuilt hierarchically with each segment's length
factor, and every rest vertex is scaled about the joint it is skinned to, in
that bone's frame (length along the bone, width across it in the body's
lateral/vertical plane, depth front-to-back), blended by the skinning weights.
Pose blend shapes, the rigid transform chain and skinning then run unchanged on
the scaled rest mesh.

SMPLLimbScaler wraps an smplx model and reproduces its forward pass. With every
factor at 1.0 the vertices it returns match the model's own to float precision.
"""
import numpy as np

try:
    import torch
    from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

from dpg_system.limb_scale import LimbScaleSet

# The joint each SMPL joint's bone points at, for the bone axis in the rest pose.
# Joints not listed (feet, head, wrists, fingers) continue their parent's bone.
_PRIMARY_CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12,
                  12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}

# The rest pose faces +Z with Y up; depth is front-to-back.
_FRONT = np.array([0.0, 0.0, 1.0])


def _descendants(parents, root):
    out = []
    frontier = [j for j in range(len(parents)) if parents[j] == root]
    while frontier:
        j = frontier.pop()
        out.append(j)
        frontier.extend(c for c in range(len(parents)) if parents[c] == j)
    return sorted(out)


def build_segment_map(parents):
    """segment name -> (length joints: offsets scaled, skin joints: vertices scaled).

    A segment ending at joint j scales j's offset from its parent for length,
    and the vertices skinned to j's parent for shape. Hips and shoulder blades
    share their parent's skin with the spine, so they are length-only. Toes and
    the head have no joint beyond them, so they are shape-only. Heels have no
    SMPL joint at all and are ignored.
    """
    parents = [int(p) for p in parents]
    n = len(parents)
    seg = {
        'spine_lower':   ([3], [0]),
        'spine_mid':     ([6], [3]),
        'spine_upper':   ([9], [6]),
        'spine_to_neck': ([12], [9]),
        'neck':          ([15], [12]),
        'head':          ([], [15]),
        'left_hip':      ([1], []),      'right_hip':      ([2], []),
        'left_upper_leg': ([4], [1]),    'right_upper_leg': ([5], [2]),
        'left_lower_leg': ([7], [4]),    'right_lower_leg': ([8], [5]),
        'left_foot':     ([10], [7]),    'right_foot':     ([11], [8]),
        'left_toes':     ([], [10]),     'right_toes':     ([], [11]),
        'left_heel':     ([], []),       'right_heel':     ([], []),
        'left_shoulder_blade': ([13], []), 'right_shoulder_blade': ([14], []),
        'left_collar':   ([16], [13]),   'right_collar':   ([17], [14]),
        'left_upper_arm': ([18], [16]),  'right_upper_arm': ([19], [17]),
        'left_lower_arm': ([20], [18]),  'right_lower_arm': ([21], [19]),
    }
    for side, wrist in (('left', 20), ('right', 21)):
        knuckles = [j for j in range(n) if parents[j] == wrist]
        below = _descendants(parents, wrist)
        seg[side + '_hand'] = (knuckles, [wrist])
        seg[side + '_fingers'] = ([j for j in below if j not in knuckles], below)
    return {k: ([j for j in L if j < n], [j for j in K if j < n]) for k, (L, K) in seg.items()}


class SMPLLimbScaler:
    def __init__(self, model):
        self.model = model
        self.parents = model.parents.detach().cpu().numpy().astype(int)
        self.n_joints = len(self.parents)
        self.segment_map = build_segment_map(self.parents)
        self.W = model.lbs_weights.detach()                      # (V, J)
        self.W_np = self.W.cpu().numpy()
        self.pose_mean = getattr(model, 'pose_mean', None)
        self.scales = LimbScaleSet()
        self.betas = None
        self._rest_cache = None
        self.set_betas(None)

    # --- inputs ---
    def set_betas(self, betas_tensor):
        if betas_tensor is None:
            betas_tensor = torch.zeros(1, self.model.shapedirs.shape[-1], dtype=torch.float32)
        self.betas = betas_tensor
        with torch.no_grad():
            v_shaped = self.model.v_template.unsqueeze(0) + blend_shapes(self.betas, self.model.shapedirs)
            J = vertices2joints(self.model.J_regressor, v_shaped)
        self.v_shaped = v_shaped                                  # (1, V, 3)
        self.J = J                                                # (1, J, 3)
        self.v_shaped_np = v_shaped[0].cpu().numpy().astype(np.float64)
        self.J_np = J[0].cpu().numpy().astype(np.float64)
        self._rest_cache = None

    def receive_limb_scale(self, message):
        if self.scales.apply_message(message):
            self._rest_cache = None
            return True
        return False

    # --- rest pose ---
    def _joint_factors(self):
        length = np.ones(self.n_joints)
        shape = {}                                                 # skin joint -> [l, w, d]
        for name, (L, K) in self.segment_map.items():
            s = self.scales.get(name)
            if s == [1.0, 1.0, 1.0]:
                continue
            for j in L:
                length[j] = s[0]
            for k in K:
                shape[k] = s
        return length, shape

    def _bone_axis(self, k):
        J = self.J_np
        c = _PRIMARY_CHILD.get(k)
        if c is not None and c < self.n_joints:
            u = J[c] - J[k]
        elif self.parents[k] >= 0:
            u = J[k] - J[self.parents[k]]
        else:
            u = np.array([0.0, 1.0, 0.0])
        n = np.linalg.norm(u)
        return u / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

    def _bone_scale_matrix(self, k, factors):
        u = self._bone_axis(k)
        w = np.cross(_FRONT, u)
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(np.array([1.0, 0.0, 0.0]), u)
        w /= np.linalg.norm(w)
        d = np.cross(u, w)
        U = np.stack([u, w, d])                                    # rows: bone frame
        S = np.diag(factors[:3])
        return U.T @ S @ U

    def rest(self):
        """(v_rest (1,V,3) tensor, J_rest (1,J,3) tensor) with the scale layer applied."""
        if self._rest_cache is not None:
            return self._rest_cache
        if self.scales.is_identity():
            self._rest_cache = (self.v_shaped, self.J)
            return self._rest_cache

        length, shape = self._joint_factors()
        J = self.J_np
        J_new = J.copy()
        for j in range(self.n_joints):                             # parents precede children
            p = self.parents[j]
            if p >= 0:
                J_new[j] = J_new[p] + length[j] * (J[j] - J[p])

        v = self.v_shaped_np
        v_new = v.copy()
        I3 = np.eye(3)
        for k in range(self.n_joints):
            M = self._bone_scale_matrix(k, shape[k]) if k in shape else I3
            dJ = J_new[k] - J[k]
            if k not in shape and np.abs(dJ).max() < 1e-12:
                continue
            wk = self.W_np[:, k]
            idx = np.nonzero(wk > 1e-7)[0]
            if len(idx) == 0:
                continue
            delta = dJ + (v[idx] - J[k]) @ (M - I3).T
            v_new[idx] += wk[idx, None] * delta

        self._rest_cache = (torch.tensor(v_new, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(J_new, dtype=torch.float32).unsqueeze(0))
        return self._rest_cache

    def rest_vertices(self):
        return self.rest()[0][0].cpu().numpy()

    # --- forward ---
    def forward(self, global_orient, body_pose, transl):
        """Pose the scaled rest mesh. Returns (vertices (V,3), joints (J,3)) as float32
        numpy, with the pelvis placed exactly at transl (matching smpl_processor)."""
        go = torch.as_tensor(np.asarray(global_orient, dtype=np.float32)).reshape(1, 3)
        bp = torch.as_tensor(np.asarray(body_pose, dtype=np.float32)).reshape(1, -1)
        tr = np.asarray(transl, dtype=np.float32).reshape(3)
        n_pose = self.n_joints * 3
        with torch.no_grad():
            full_pose = torch.zeros(1, n_pose, dtype=torch.float32)
            full_pose[0, :3] = go[0]
            n_body = min(bp.shape[1], n_pose - 3)
            full_pose[0, 3:3 + n_body] = bp[0, :n_body]
            if self.pose_mean is not None:
                full_pose = full_pose + self.pose_mean
            v_rest, J_rest = self.rest()
            rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
            ident = torch.eye(3, dtype=torch.float32)
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
            pose_offsets = torch.matmul(pose_feature, self.model.posedirs).view(1, -1, 3)
            v_posed = pose_offsets + v_rest
            J_transformed, A = batch_rigid_transform(rot_mats, J_rest, self.model.parents, dtype=torch.float32)
            T = torch.matmul(self.W.unsqueeze(0), A.view(1, self.n_joints, 16)).view(1, -1, 4, 4)
            ones = torch.ones(1, v_posed.shape[1], 1, dtype=torch.float32)
            v_homo = torch.matmul(T, torch.cat([v_posed, ones], dim=2).unsqueeze(-1))
            verts = v_homo[0, :, :3, 0].cpu().numpy()
            joints = J_transformed[0].cpu().numpy()
        # The root rotates about the rest pelvis, so the posed pelvis sits at
        # J_rest[0]; shift so it sits at transl instead.
        pelvis = J_rest[0, 0].cpu().numpy()
        verts = verts - pelvis + tr
        joints = joints - pelvis + tr
        return verts.astype(np.float32), joints.astype(np.float32)
