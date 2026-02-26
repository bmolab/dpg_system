import os
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
import moderngl

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False


def register_mgl_smpl_heatmap_nodes():
    Node.app.register_node('mgl_smpl_heatmap', MGLSMPLHeatmapNode.factory)


def _heatmap_color(t):
    """Map a normalized value [0,1] to a blue→cyan→green→yellow→red heatmap color (RGB)."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        s = t / 0.25
        return (0.0, s, 1.0)           # blue → cyan
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (0.0, 1.0, 1.0 - s)     # cyan → green
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (s, 1.0, 0.0)           # green → yellow
    else:
        s = (t - 0.75) / 0.25
        return (1.0, 1.0 - s, 0.0)     # yellow → red


# SMPL parent joint indices (child -> parent)
SMPL_PARENT = [
    -1,  # 0  pelvis (root)
     0,  # 1  L_Hip
     0,  # 2  R_Hip
     0,  # 3  Spine1
     1,  # 4  L_Knee
     2,  # 5  R_Knee
     3,  # 6  Spine2
     4,  # 7  L_Ankle
     5,  # 8  R_Ankle
     6,  # 9  Spine3
     7,  # 10 L_Foot
     8,  # 11 R_Foot
     9,  # 12 Neck
     9,  # 13 L_Collar
     9,  # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
    20,  # 22 L_Hand
    21,  # 23 R_Hand
]


class MGLSMPLHeatmapNode(Node):
    """
    Renders the SMPL mesh as a translucent heatmap overlay colored by torque magnitude.

    Uses SMPL skinning weights to map per-joint torque magnitudes to per-vertex
    colors using a blue→red heatmap. Renders with alpha blending as a translucent
    overlay that can be placed on top of an mgl_smpl_mesh node.
    """

    @staticmethod
    def factory(name, data, args=None):
        return MGLSMPLHeatmapNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

        self.smpl_model = None
        self.faces_np = None
        self.n_verts = 0
        self.skinning_weights = None  # (V, 24) numpy
        self.betas_tensor = None
        self.current_gender = None

        self.last_pose = None
        self.last_trans = None
        self.last_vertices = None
        self.last_joint_positions = None  # (24, 3) from forward pass
        self.torques_data = None

        self.ctx = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.heatmap_shader = None
        self.initialize(args)

    def initialize(self, args):
        # super().initialize(args)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.torques_input = self.add_input('torques')
        self.config_input = self.add_input('config', triggers_execution=True)

        self.max_torque_prop = self.add_option('max torque', widget_type='drag_float',
                                                default_value=50.0, speed=1.0)
        self.opacity_prop = self.add_option('opacity', widget_type='drag_float',
                                             default_value=0.5, speed=0.01)
        self.min_opacity_prop = self.add_option('min opacity', widget_type='drag_float',
                                                 default_value=0.15, speed=0.01)
        self.weight_mode_prop = self.add_option('weight mode', widget_type='combo', default_value='directional')
        self.weight_mode_prop.widget.combo_items = ['directional', 'proximity', 'skinning']
        self.spread_prop = self.add_option('spread', widget_type='drag_float',
                                           default_value=0.08, speed=0.005)
        self.dir_bias_prop = self.add_option('dir bias', widget_type='drag_float',
                                              default_value=0.7, speed=0.01)
        self.gender_prop = self.add_property('gender', widget_type='combo', default_value='male')
        self.gender_prop.widget.combo_items = ['male', 'female']
        self.model_path_prop = self.add_property('model_path', widget_type='text_input',
                                                  default_value='.')
        self.up_axis_prop = self.add_property('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']

        self.gl_output = self.add_output('gl chain out')

    def _load_model(self):
        if not SMPLX_AVAILABLE:
            print("MGLSMPLHeatmapNode: smplx/torch not available")
            return False

        gender_map = {'male': 'MALE', 'female': 'FEMALE'}
        g_tag = gender_map.get(self.gender_prop(), 'MALE')
        model_path = self.model_path_prop() or '.'

        try:
            self.smpl_model = smplx.create(
                model_path=model_path,
                model_type='smplh',
                gender=g_tag,
                num_betas=10,
                ext='pkl'
            )
            self.smpl_model.eval()

            self.faces_np = np.array(self.smpl_model.faces, dtype=np.int32)

            with torch.no_grad():
                output = self.smpl_model()
                verts = output.vertices[0].cpu().numpy()
                self.n_verts = len(verts)

            # Extract skinning weights for body joints only (first 24 of 52)
            weights_full = self.smpl_model.lbs_weights.cpu().numpy()  # (V, 52)
            self.skinning_weights = weights_full[:, :24].copy()  # (V, 24)

            self.current_gender = self.gender_prop()
            return True

        except Exception as e:
            print(f"MGLSMPLHeatmapNode: Failed to load model: {e}")
            return False

    def _get_heatmap_shader(self):
        if self.heatmap_shader is None:
            ctx = MGLContext.get_instance().ctx
            self.heatmap_shader = ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;

                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec4 in_color;

                    out vec3 v_normal;
                    out vec3 v_frag_pos;
                    out vec4 v_color;

                    void main() {
                        vec4 world_pos = M * vec4(in_position, 1.0);
                        gl_Position = P * V * world_pos;
                        v_frag_pos = world_pos.xyz;
                        v_normal = normalize(mat3(M) * in_normal);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec3 light_dir;

                    in vec3 v_normal;
                    in vec3 v_frag_pos;
                    in vec4 v_color;

                    out vec4 f_color;

                    void main() {
                        // Simple directional diffuse lighting
                        vec3 norm = normalize(v_normal);
                        float diff = max(dot(norm, normalize(light_dir)), 0.0);
                        float ambient = 0.3;
                        float lighting = ambient + 0.7 * diff;

                        vec3 lit_color = v_color.rgb * lighting;
                        f_color = vec4(lit_color, v_color.a);
                    }
                '''
            )
        return self.heatmap_shader

    def _run_forward(self, pose_data, trans_data):
        if self.smpl_model is None:
            return None

        pose = any_to_array(pose_data)
        if pose is None:
            return None
        pose = pose.flatten().astype(np.float32)
        if pose.shape[0] < 66:
            return None

        global_orient = torch.tensor(pose[:3], dtype=torch.float32).unsqueeze(0)
        body_pose = torch.tensor(pose[3:66], dtype=torch.float32).unsqueeze(0)

        trans = any_to_array(trans_data)
        if trans is not None:
            trans = trans.flatten().astype(np.float32)[:3]
            transl = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
        else:
            transl = torch.zeros(1, 3, dtype=torch.float32)

        with torch.no_grad():
            fwd_kwargs = dict(global_orient=global_orient, body_pose=body_pose, transl=transl)
            if self.betas_tensor is not None:
                fwd_kwargs['betas'] = self.betas_tensor
            output = self.smpl_model(**fwd_kwargs)
            vertices = output.vertices[0].cpu().numpy()
            joint_positions = output.joints[0, :24].cpu().numpy()  # (24, 3)

        if self.up_axis_prop() == 'Y':
            out = np.empty_like(vertices)
            out[:, 0] = vertices[:, 0]
            out[:, 1] = vertices[:, 2]
            out[:, 2] = -vertices[:, 1]
            vertices = out
            # Same transform for joint positions
            jp_out = np.empty_like(joint_positions)
            jp_out[:, 0] = joint_positions[:, 0]
            jp_out[:, 1] = joint_positions[:, 2]
            jp_out[:, 2] = -joint_positions[:, 1]
            joint_positions = jp_out

        return vertices, joint_positions

    def _compute_normals(self, vertices, faces):
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals, faces[:, 0], face_normals)
        np.add.at(normals, faces[:, 1], face_normals)
        np.add.at(normals, faces[:, 2], face_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-8)
        return normals

    def _compute_proximity_weights(self, vertices, joint_positions):
        """Compute per-vertex weights based on proximity to joints using Gaussian falloff."""
        sigma = max(self.spread_prop(), 0.01)
        n_joints = joint_positions.shape[0]

        # (V, 1, 3) - (1, J, 3) -> (V, J, 3) -> (V, J)
        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)  # (V, J)

        weights = np.exp(-dists_sq / (2.0 * sigma * sigma))  # (V, J)

        # Normalize per vertex
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)

        return weights

    def _compute_directional_weights(self, vertices, joint_positions, torques):
        """Compute proximity weights biased toward agonist muscle direction.

        For each joint, computes muscle_dir = cross(torque_axis, bone_dir).
        This naturally points toward the muscles producing the torque:
        - Knee flexion torque -> hamstrings (posterior)
        - Knee extension torque -> quadriceps (anterior)
        """
        sigma = max(self.spread_prop(), 0.01)
        dir_bias = self.dir_bias_prop()
        n_joints = min(len(joint_positions), len(torques), 24)

        # Compute bone directions: parent -> child
        bone_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
            if 0 <= parent_idx < n_joints:
                d = joint_positions[j] - joint_positions[parent_idx]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    bone_dirs[j] = d / length
                else:
                    bone_dirs[j] = np.array([0, 1, 0])  # fallback
            else:
                bone_dirs[j] = np.array([0, 1, 0])  # root: default up

        # Compute muscle direction for each joint: cross(torque_axis, bone_dir)
        muscle_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            tau_mag = np.linalg.norm(torques[j])
            if tau_mag > 1e-6:
                tau_axis = torques[j] / tau_mag
                md = np.cross(tau_axis, bone_dirs[j])
                md_len = np.linalg.norm(md)
                if md_len > 1e-6:
                    muscle_dirs[j] = md / md_len

        # Vectorized proximity: (V, J)
        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :n_joints, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        proximity = np.exp(-dists_sq / (2.0 * sigma * sigma))

        # Directional bias: cosine similarity of (v - j) with muscle_dir
        dists = np.sqrt(dists_sq + 1e-10)
        dir_dots = np.sum(diffs * muscle_dirs[np.newaxis, :, :], axis=2)  # (V, J)
        dir_cos = dir_dots / dists  # cosine similarity [-1, 1]

        # Bias: ramp from (1-dir_bias) on antagonist side to (1+dir_bias) on agonist side
        directional_bias = 1.0 + dir_bias * dir_cos
        directional_bias = np.maximum(directional_bias, 0.05)  # don't fully suppress

        weights = proximity * directional_bias

        # Normalize per vertex
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)

        return weights

    def _compute_vertex_colors(self, torques):
        """Compute per-vertex RGBA heatmap colors from joint torque vectors.

        Args:
            torques: (J, 3) array of torque vectors

        Returns:
            (V, 4) float32 array of RGBA colors
        """
        mode = self.weight_mode_prop()

        if mode == 'directional' and self.last_vertices is not None and self.last_joint_positions is not None:
            n_joints = min(torques.shape[0], self.last_joint_positions.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_directional_weights(
                self.last_vertices, self.last_joint_positions[:n_joints], torques[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif mode == 'proximity' and self.last_vertices is not None and self.last_joint_positions is not None:
            n_joints = min(torques.shape[0], self.last_joint_positions.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_proximity_weights(self.last_vertices, self.last_joint_positions[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif self.skinning_weights is not None:
            n_joints = min(torques.shape[0], self.skinning_weights.shape[1])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self.skinning_weights[:, :n_joints]
            vert_magnitudes = weights @ magnitudes
        else:
            return None

        # Normalize by max_torque (fixed scale)
        max_t = max(self.max_torque_prop(), 0.01)
        t = np.clip(vert_magnitudes / max_t, 0.0, 1.0)  # (V,)

        # Vectorized heatmap: blue→cyan→green→yellow→red
        r = np.zeros(self.n_verts, dtype=np.float32)
        g = np.zeros(self.n_verts, dtype=np.float32)
        b = np.zeros(self.n_verts, dtype=np.float32)

        # Segment 1: [0, 0.25) blue→cyan
        mask = t < 0.25
        s = t[mask] / 0.25
        r[mask] = 0.0; g[mask] = s; b[mask] = 1.0

        # Segment 2: [0.25, 0.5) cyan→green
        mask = (t >= 0.25) & (t < 0.5)
        s = (t[mask] - 0.25) / 0.25
        r[mask] = 0.0; g[mask] = 1.0; b[mask] = 1.0 - s

        # Segment 3: [0.5, 0.75) green→yellow
        mask = (t >= 0.5) & (t < 0.75)
        s = (t[mask] - 0.5) / 0.25
        r[mask] = s; g[mask] = 1.0; b[mask] = 0.0

        # Segment 4: [0.75, 1.0] yellow→red
        mask = t >= 0.75
        s = (t[mask] - 0.75) / 0.25
        r[mask] = 1.0; g[mask] = 1.0 - s; b[mask] = 0.0

        # Alpha: ramp from min_opacity to full opacity based on torque magnitude
        opacity = self.opacity_prop()
        min_opacity = self.min_opacity_prop()
        alpha = min_opacity + (opacity - min_opacity) * np.clip(t * 2.0, 0.0, 1.0)

        colors = np.stack([r, g, b, alpha], axis=1).astype(np.float32)
        return colors

    def _build_vbo_data(self, vertices, normals, colors):
        """Build interleaved VBO: pos(3) + normal(3) + color(4) = 10 floats per vertex."""
        n = len(vertices)
        data = np.zeros((n, 10), dtype=np.float32)
        data[:, 0:3] = vertices
        data[:, 3:6] = normals
        data[:, 6:10] = colors
        return data

    def _apply_config(self, config):
        needs_reload = False

        gender = config.get('gender', None)
        if gender is not None:
            if isinstance(gender, np.ndarray):
                gender = str(gender)
            gender = gender.lower().strip()
            if gender in ('male', 'female') and gender != self.current_gender:
                self.gender_prop.set(gender)
                needs_reload = True

        betas = config.get('betas', None)
        if betas is not None:
            betas = np.array(betas, dtype=np.float32).flatten()
            if len(betas) > 10:
                betas = betas[:10]
            bt = torch.zeros(1, 10, dtype=torch.float32)
            bt[0, :len(betas)] = torch.tensor(betas)
            self.betas_tensor = bt
            needs_reload = True

        if needs_reload:
            if self._load_model():
                self.vao = None
                self.vbo = None

    def execute(self):
        if self.config_input.fresh_input:
            config = self.config_input()
            if isinstance(config, dict):
                self._apply_config(config)

        if self.torques_input.fresh_input:
            data = self.torques_input()
            if data is not None:
                data = any_to_array(data)
                if data is not None and data.ndim == 2 and data.shape[1] == 3:
                    self.torques_data = data.astype(np.float32)

        if self.pose_input.fresh_input:
            pose = self.pose_input()
            trans = self.trans_input()
            self.last_pose = pose
            self.last_trans = trans

            result = self._run_forward(pose, trans)
            if result is not None:
                self.last_vertices, self.last_joint_positions = result

        # Handle gl chain 'draw' message
        if self.gl_input.fresh_input:
            msg = self.gl_input()
            do_draw = False
            if isinstance(msg, str) and msg == 'draw':
                do_draw = True
            elif isinstance(msg, list) and len(msg) > 0 and msg[0] == 'draw':
                do_draw = True
            if do_draw:
                self.draw()
                self.gl_output.send('draw')

    def draw(self):
        # Get context from MGLContext singleton
        mgl_ctx = MGLContext.get_instance()
        if mgl_ctx is None or mgl_ctx.ctx is None:
            return
        self.ctx = mgl_ctx

        if self.smpl_model is None:
            if not self._load_model():
                return

        if self.last_vertices is None or self.torques_data is None:
            return

        inner_ctx = self.ctx.ctx
        prog = self._get_heatmap_shader()

        vertices = self.last_vertices
        normals = self._compute_normals(vertices, self.faces_np)
        colors = self._compute_vertex_colors(self.torques_data)
        if colors is None:
            return

        vbo_data = self._build_vbo_data(vertices, normals, colors)

        # Create or update buffers
        vbo_bytes = vbo_data.astype('f4').tobytes()
        if self.vbo is None or self.vbo.size != len(vbo_bytes):
            if self.vbo is not None:
                self.vbo.release()
            self.vbo = inner_ctx.buffer(vbo_bytes)
            # Create IBO
            if self.ibo is not None:
                self.ibo.release()
            idx_data = self.faces_np.flatten().astype(np.int32)
            self.ibo = inner_ctx.buffer(idx_data.tobytes())
            # Create VAO
            if self.vao is not None:
                self.vao.release()
            self.vao = inner_ctx.vertex_array(
                prog,
                [(self.vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')],
                self.ibo
            )
        else:
            self.vbo.write(vbo_bytes)

        # Set uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())
        if 'light_dir' in prog:
            prog['light_dir'].value = (0.3, 1.0, 0.5)

        # Enable alpha blending for translucent overlay
        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        inner_ctx.disable(moderngl.CULL_FACE)

        # Render with slight polygon offset to avoid z-fighting with base mesh
        inner_ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render()

        inner_ctx.disable(moderngl.BLEND)
