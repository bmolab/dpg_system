"""
mgl_body_orientation_node.py

Combined body + orientation disks node.  Subclasses MGLBodyNode so all
body rendering, skeleton modes, limb lengths, etc. are inherited.  On top
of that it renders orientation disks at the first 20 joints in a single
instanced draw call — no per-joint callback overhead.
"""

import numpy as np
import math
import moderngl
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
from dpg_system.mgl_body_node import MGLBodyNode
from dpg_system.body_base import rotation_matrix_from_vectors, t_PelvisAnchor, t_ActiveJointCount
from dpg_system.body_defs import *


class MGLBodyOrientationNode(MGLBodyNode):
    """MGLBodyNode with built-in orientation disk rendering.

    Renders the body (limbs, skeleton, spheres) exactly like mgl_body,
    then additionally renders orientation disks at the first 20 joints
    in a single instanced draw call.
    """

    _default_disk_colors = [
        [0.0, 0.4, 1.0, 0.5],
        [0.0, 1.0, 1.0, 0.5],
        [0.0, 1.0, 0.2, 0.5],
        [1.0, 1.0, 0.0, 0.5],
        [1.0, 0.5, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.5],
        [0.9, 0.0, 1.0, 0.5],
    ]

    _default_disk_scales = [2.25, 2.25, 3.375, 5.0625, 7.59375, 11.390625, 17.0859375]

    @staticmethod
    def factory(name, data, args=None):
        return MGLBodyOrientationNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

        # Disk count per joint (from constructor arg or default)
        self.disk_count = 7
        if args and len(args) > 0:
            try:
                self.disk_count = int(args[0])
            except (ValueError, TypeError):
                pass

        # Disk inputs
        self.show_disks_input = self.add_input('show_disks', widget_type='checkbox', default_value=True)
        self.disk_scale_input = self.add_input('disk_scale', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.disk_slices_input = self.add_input('disk_slices', widget_type='drag_int', default_value=32,
                                                 min_value=6, max_value=128, callback=self._invalidate_disk_ring)
        self.disk_ring_width_input = self.add_input('disk_ring_width', widget_type='drag_float', default_value=0.1,
                                                     speed=0.01, callback=self._invalidate_disk_ring)

        # Per-disk scale factors
        self.disk_per_scales = np.array(
            [self._default_disk_scales[i % len(self._default_disk_scales)] for i in range(self.disk_count)],
            dtype=np.float32
        )
        self.disk_per_scales_input = self.add_input('disk_scales', callback=self._disk_per_scales_changed)

        # Per-disk-index colors (same color pattern across all joints)
        self.disk_color_inputs = []
        self.disk_colors = np.zeros((self.disk_count, 4), dtype=np.float32)
        for i in range(self.disk_count):
            default_c = self._default_disk_colors[i % len(self._default_disk_colors)]
            self.disk_colors[i] = np.array(default_c, dtype=np.float32)
            self.disk_color_inputs.append(
                self.add_input('disk color ' + str(i), callback=self._disk_colors_changed)
            )

        # Disk orientation data: shape (20, disk_count, 3) for axis-angle
        #                        or    (20, disk_count, 3, 3) for 3×3 rotation matrices
        self.disk_axis_angle_input = self.add_input('disk_orientations', triggers_execution=True)
        self.disk_orientations = None  # set during execute
        self._orientations_are_matrices = False  # auto-detected from shape

        # GPU resources for disks
        self._disk_shader = None
        self._disk_ring_vbo = None
        self._disk_ring_ibo = None
        self._disk_inst_vbo = None
        self._disk_vao = None
        self._disk_ring_index_count = 0
        self._disk_built_slices = 0
        self._disk_built_width = 0.0
        self._disk_inst_capacity = 0

    def _disk_per_scales_changed(self):
        val = self.disk_per_scales_input()
        if val is not None:
            val = any_to_array(val).astype(np.float32).flatten()
            for i in range(min(self.disk_count, len(val))):
                self.disk_per_scales[i] = val[i]

    def _disk_colors_changed(self):
        for i in range(self.disk_count):
            val = self.disk_color_inputs[i]()
            if val is not None:
                val = any_to_array(val)
                if val.ndim == 1 and val.shape[0] == 4:
                    self.disk_colors[i] = val.astype(np.float32)

    def _invalidate_disk_ring(self):
        self._disk_vao = None
        if self._disk_ring_vbo is not None:
            self._disk_ring_vbo.release()
            self._disk_ring_vbo = None
        if self._disk_ring_ibo is not None:
            self._disk_ring_ibo.release()
            self._disk_ring_ibo = None

    # ------------------------------------------------------------------
    #  Disk shader (emissive / unlit, instanced)
    # ------------------------------------------------------------------
    def _get_disk_shader(self):
        if self._disk_shader is not None:
            return self._disk_shader
        ctx = MGLContext.get_instance().ctx
        self._disk_shader = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 V;
                uniform mat4 P;
                in vec3 in_position;
                in vec4 inst_m0;
                in vec4 inst_m1;
                in vec4 inst_m2;
                in vec4 inst_m3;
                in vec4 inst_color;
                out vec4 v_color;
                void main() {
                    mat4 M = mat4(inst_m0, inst_m1, inst_m2, inst_m3);
                    gl_Position = P * V * M * vec4(in_position, 1.0);
                    v_color = inst_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    f_color = v_color;
                }
            '''
        )
        return self._disk_shader

    # ------------------------------------------------------------------
    #  Unit ring geometry (built once, cached)
    # ------------------------------------------------------------------
    def _build_unit_ring(self, slices, ring_width_frac):
        inner_r = max(0.0, 1.0 - ring_width_frac)
        outer_r = 1.0

        angles = np.linspace(0, 2.0 * math.pi, slices + 1, dtype=np.float32)
        cos_a = np.cos(angles[:slices])
        sin_a = np.sin(angles[:slices])

        inner_verts = np.stack([inner_r * cos_a, inner_r * sin_a, np.zeros(slices, dtype=np.float32)], axis=1)
        outer_verts = np.stack([outer_r * cos_a, outer_r * sin_a, np.zeros(slices, dtype=np.float32)], axis=1)

        verts = np.empty((slices * 2, 3), dtype=np.float32)
        verts[0::2] = inner_verts
        verts[1::2] = outer_verts

        s = np.arange(slices, dtype=np.int32)
        s_next = (s + 1) % slices
        inner_idx = s * 2
        outer_idx = s * 2 + 1
        next_inner = s_next * 2
        next_outer = s_next * 2 + 1

        front = np.column_stack([inner_idx, next_inner, outer_idx,
                                  outer_idx, next_inner, next_outer]).reshape(-1)
        back_offset = slices * 2
        back = np.column_stack([inner_idx + back_offset, outer_idx + back_offset, next_inner + back_offset,
                                 outer_idx + back_offset, next_outer + back_offset, next_inner + back_offset]).reshape(-1)

        indices = np.concatenate([front, back]).astype(np.int32)
        all_verts = np.vstack([verts, verts])  # double-sided

        self._disk_ring_index_count = len(indices)
        self._disk_built_slices = slices
        self._disk_built_width = ring_width_frac
        return all_verts, indices

    def _ensure_disk_ring(self, inner_ctx, slices, ring_width_frac):
        if (self._disk_ring_vbo is not None and
                self._disk_built_slices == slices and
                abs(self._disk_built_width - ring_width_frac) < 1e-6):
            return
        if self._disk_ring_vbo is not None:
            self._disk_ring_vbo.release()
        if self._disk_ring_ibo is not None:
            self._disk_ring_ibo.release()
        self._disk_vao = None
        verts, indices = self._build_unit_ring(slices, ring_width_frac)
        self._disk_ring_vbo = inner_ctx.buffer(verts.tobytes())
        self._disk_ring_ibo = inner_ctx.buffer(indices.tobytes())

    def _ensure_disk_vao(self, inner_ctx, prog, num_instances):
        if self._disk_inst_vbo is None or self._disk_inst_capacity < num_instances:
            if self._disk_inst_vbo is not None:
                self._disk_inst_vbo.release()
            capacity = max(num_instances, 128)
            self._disk_inst_vbo = inner_ctx.buffer(reserve=capacity * 20 * 4)
            self._disk_inst_capacity = capacity
            self._disk_vao = None

        if self._disk_vao is None:
            self._disk_vao = inner_ctx.vertex_array(prog, [
                (self._disk_ring_vbo, '3f', 'in_position'),
                (self._disk_inst_vbo, '4f 4f 4f 4f 4f/i', 'inst_m0', 'inst_m1', 'inst_m2', 'inst_m3', 'inst_color'),
            ], self._disk_ring_ibo)

    # ------------------------------------------------------------------
    #  execute override — also pull disk orientation data
    # ------------------------------------------------------------------
    def execute(self):
        if self.disk_axis_angle_input.fresh_input:
            data = self.disk_axis_angle_input()
            if data is not None:
                arr = any_to_array(data).astype(np.float32)
                # Auto-detect: [20, 7, 3, 3] = rotation matrices, [20, 7, 3] = axis-angle
                if arr.ndim == 4 and arr.shape[2] == 3 and arr.shape[3] == 3:
                    self._orientations_are_matrices = True
                else:
                    self._orientations_are_matrices = False
                self.disk_orientations = arr

        super().execute()

    # ------------------------------------------------------------------
    #  draw override — parent draw + orientation disks
    # ------------------------------------------------------------------
    def draw(self):
        # Run full parent draw (body, limbs, spheres, callbacks)
        super().draw()

        # Now render orientation disks
        if not self.show_disks_input():
            return
        if self.disk_orientations is None:
            return
        if not self.ctx or not self.ctx.ctx:
            return
        if not hasattr(self, 'joint_tip_matrices'):
            return

        inner_ctx = self.ctx.ctx
        prog = self._get_disk_shader()

        scale = self.disk_scale_input()
        slices = max(6, self.disk_slices_input())
        ring_width_frac = max(0.01, min(self.disk_ring_width_input(), 1.0))

        self._ensure_disk_ring(inner_ctx, slices, ring_width_frac)

        up_vector = np.array([0.0, 0.0, 1.0])

        # Pre-allocate for max instances: 20 joints × disk_count disks
        max_instances = t_ActiveJointCount * self.disk_count
        inst_data = np.empty((max_instances, 20), dtype=np.float32)
        num_visible = 0

        orient_data = self.disk_orientations
        use_matrices = self._orientations_are_matrices

        for joint_idx in range(t_ActiveJointCount):
            if joint_idx == t_PelvisAnchor:
                continue

            # Get joint world position matrix
            tip_mat = self.joint_tip_matrices.get(joint_idx)
            if tip_mat is None:
                continue

            if joint_idx >= orient_data.shape[0]:
                continue

            joint_data = orient_data[joint_idx]  # (7, 3, 3) or (7, 3) or (7, 4)

            for d in range(min(self.disk_count, joint_data.shape[0])):
                if use_matrices:
                    # --- 3×3 rotation matrix path ---
                    rot3 = joint_data[d]  # (3, 3)

                    # Extract rotation angle from matrix trace: angle = arccos((trace-1)/2)
                    trace_val = rot3[0, 0] + rot3[1, 1] + rot3[2, 2]
                    cos_angle = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    size = angle * scale * self.disk_per_scales[d]
                    if size < 1e-6:
                        continue

                    # Extract rotation axis from 3×3 matrix
                    # axis = [R32-R23, R13-R31, R21-R12] (unnormalized)
                    ax = np.array([
                        rot3[2, 1] - rot3[1, 2],
                        rot3[0, 2] - rot3[2, 0],
                        rot3[1, 0] - rot3[0, 1]
                    ], dtype=np.float32)
                    ax_norm = np.linalg.norm(ax)
                    if ax_norm < 1e-6:
                        continue
                    direction = ax / ax_norm

                    # Orient disk perpendicular to rotation axis
                    orient_flat = rotation_matrix_from_vectors(up_vector, direction)
                    orient_4x4 = np.array(orient_flat, dtype=np.float32).reshape(4, 4)

                    scale_mat = np.diag([size, size, size, 1.0]).astype(np.float32)
                    combined = tip_mat @ orient_4x4 @ scale_mat
                else:
                    # --- Axis-angle path ---
                    axis = joint_data[d]  # (3,) or (4,)

                    if axis.shape[0] == 3:
                        angle = np.linalg.norm(axis)
                        if angle < 1e-6:
                            continue
                        direction = axis / angle
                        size = angle * scale * self.disk_per_scales[d]
                    elif axis.shape[0] == 4:
                        direction = axis[:3]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-6:
                            continue
                        direction = direction / norm
                        size = axis[3] * scale * self.disk_per_scales[d]
                    else:
                        continue

                    if size < 1e-6:
                        continue

                    orient_flat = rotation_matrix_from_vectors(up_vector, direction)
                    orient_4x4 = np.array(orient_flat, dtype=np.float32).reshape(4, 4)
                    scale_mat = np.diag([size, size, size, 1.0]).astype(np.float32)
                    combined = tip_mat @ orient_4x4 @ scale_mat

                inst_data[num_visible, :16] = combined.T.flatten()
                inst_data[num_visible, 16:20] = self.disk_colors[d % self.disk_count]
                num_visible += 1

        if num_visible == 0:
            return

        self._ensure_disk_vao(inner_ctx, prog, num_visible)
        self._disk_inst_vbo.write(inst_data[:num_visible].tobytes())

        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.disable(moderngl.CULL_FACE)

        self._disk_vao.render(moderngl.TRIANGLES, instances=num_visible)

        inner_ctx.enable(moderngl.CULL_FACE)

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------
    def custom_cleanup(self):
        super().custom_cleanup() if hasattr(super(), 'custom_cleanup') else None
        for buf in (self._disk_ring_vbo, self._disk_ring_ibo, self._disk_inst_vbo):
            if buf is not None:
                buf.release()
        if self._disk_vao is not None:
            self._disk_vao.release()

    # ------------------------------------------------------------------
    #  Persistence for disk colors
    # ------------------------------------------------------------------
    def save_custom(self, container):
        if hasattr(super(), 'save_custom'):
            super().save_custom(container)
        for i in range(self.disk_count):
            container['disk_color_' + str(i)] = list(self.disk_colors[i].tolist())

    def load_custom(self, container):
        if hasattr(super(), 'load_custom'):
            super().load_custom(container)
        for i in range(self.disk_count):
            name = 'disk_color_' + str(i)
            if name in container:
                self.disk_colors[i] = np.array(container[name], dtype=np.float32)
                self.disk_color_inputs[i].set(self.disk_colors[i])
        self._disk_colors_changed()
