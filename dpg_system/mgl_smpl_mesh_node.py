import os
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
import moderngl
from dpg_system.moderngl_nodes import MGLShapeNode

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False


def register_mgl_smpl_mesh_nodes():
    Node.app.register_node('mgl_smpl_mesh', MGLSMPLMeshNode.factory)


class MGLSMPLMeshNode(MGLShapeNode):
    """
    Renders the SMPL skin mesh as a deformable triangle mesh.
    
    Takes pose (axis-angle) and translation inputs, runs the SMPL forward
    pass to get deformed vertices, computes per-vertex normals, and renders
    using the standard MGL lighting/material pipeline.
    """
    
    @staticmethod
    def factory(name, data, args=None):
        return MGLSMPLMeshNode(name, data, args)
    
    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.smpl_model = None
        self.faces_np = None
        self.n_verts = 0
        self.n_faces = 0
        self.prev_vbo_data = None
        self.betas_tensor = None
        self.current_gender = None
        self.last_pose = None
        self.last_trans = None
    
    def initialize(self, args):
        super().initialize(args)
        
        # Pose input: (22*3,) or (24*3,) axis-angle, or (22,3) / (24,3)
        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans', widget_type='drag_float_n',
                                          default_value=[0.0, 0.0, 0.0], columns=3)
        
        # Properties
        self.gender_prop = self.add_property('gender', widget_type='combo', default_value='male')
        self.gender_prop.widget.combo_items = ['male', 'female']
        self.opacity_prop = self.add_property('opacity', widget_type='drag_float',
                                              default_value=1.0)
        self.model_path_prop = self.add_property('model_path', widget_type='text_input',
                                                  default_value='.')
        self.up_axis_prop = self.add_property('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']
        # Config input: dict with {gender, betas, mocap_framerate} from NPZ
        self.config_input = self.add_input('config', triggers_execution=True)
        
        self.end_initialization()
    
    def _load_model(self):
        """Load the SMPL-H model."""
        if not SMPLX_AVAILABLE:
            print("MGLSMPLMeshNode: smplx/torch not available")
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
            
            # Extract faces (static)
            self.faces_np = np.array(self.smpl_model.faces, dtype=np.int32)
            self.n_faces = len(self.faces_np)
            
            # Get vertex count from T-pose
            with torch.no_grad():
                output = self.smpl_model()
                verts = output.vertices[0].cpu().numpy()
                self.n_verts = len(verts)
            
            # print(f"MGLSMPLMeshNode: Loaded SMPL-H ({g_tag}), {self.n_verts} verts, {self.n_faces} faces")
            self.current_gender = self.gender_prop()
            return True
            
        except Exception as e:
            print(f"MGLSMPLMeshNode: Failed to load model: {e}")
            return False
    
    def _compute_normals(self, vertices, faces):
        """Compute per-vertex normals from face normals (area-weighted)."""
        normals = np.zeros_like(vertices)
        
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Face normals (not normalized — magnitude = 2× face area)
        face_normals = np.cross(v1 - v0, v2 - v0)
        
        # Accumulate face normals to vertices (area-weighted)
        np.add.at(normals, faces[:, 0], face_normals)
        np.add.at(normals, faces[:, 1], face_normals)
        np.add.at(normals, faces[:, 2], face_normals)
        
        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        normals /= lengths
        
        return normals
    
    def _run_forward(self, pose_data, trans_data):
        """Run SMPL forward pass and return deformed vertices."""
        if self.smpl_model is None:
            return None
        
        # Parse pose
        pose = any_to_array(pose_data)
        if pose is None:
            return None
        
        pose = pose.flatten().astype(np.float32)
        
        # Expect at least 22*3 = 66 values
        if pose.shape[0] < 66:
            return None
        
        # Split into global orient (3,) and body pose
        global_orient = torch.tensor(pose[:3], dtype=torch.float32).unsqueeze(0)
        
        # SMPL-H body_pose is 21 joints × 3 = 63 (joints 1-21)
        body_pose_vals = pose[3:66]  # joints 1-21
        body_pose = torch.tensor(body_pose_vals, dtype=torch.float32).unsqueeze(0)
        
        # Translation
        trans = any_to_array(trans_data)
        if trans is not None:
            trans = trans.flatten().astype(np.float32)[:3]
            transl = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
        else:
            transl = torch.zeros(1, 3, dtype=torch.float32)
        
        with torch.no_grad():
            fwd_kwargs = dict(
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl
            )
            if self.betas_tensor is not None:
                fwd_kwargs['betas'] = self.betas_tensor
            output = self.smpl_model(**fwd_kwargs)
            vertices = output.vertices[0].cpu().numpy()
        
        # Reorient if needed (SMPL is Z-up, convert to Y-up)
        if self.up_axis_prop() == 'Y':
            vertices = self._z_to_y_up(vertices)
        
        return vertices
    
    def _z_to_y_up(self, verts):
        """Convert Z-up coords to Y-up: (x,y,z) -> (x,z,-y)"""
        out = np.empty_like(verts)
        out[:, 0] = verts[:, 0]   # X stays
        out[:, 1] = verts[:, 2]   # Y <- Z
        out[:, 2] = -verts[:, 1]  # Z <- -Y
        return out
    
    def _build_vbo_data(self, vertices, normals):
        """Build interleaved VBO data: pos(3) + normal(3) + uv(2)."""
        n = len(vertices)
        data = np.zeros((n, 8), dtype=np.float32)
        data[:, 0:3] = vertices
        data[:, 3:6] = normals
        # UV stays 0 (no texture mapping for now)
        return data
    
    def create_geometry(self):
        """Override: create initial geometry from T-pose."""
        if self.smpl_model is None:
            if not self._load_model():
                return None, None
        
        # T-pose vertices
        with torch.no_grad():
            output = self.smpl_model()
            vertices = output.vertices[0].cpu().numpy()
        
        normals = self._compute_normals(vertices, self.faces_np)
        
        # Reorient T-pose if Y-up
        if self.up_axis_prop() == 'Y':
            vertices = self._z_to_y_up(vertices)
            normals = self._z_to_y_up(normals)
        
        vbo_data = self._build_vbo_data(vertices, normals)
        
        vertices_list = vbo_data.flatten().tolist()
        indices_list = self.faces_np.flatten().tolist()
        
        return vertices_list, indices_list
    
    def execute(self):
        """Handle pose/config input updates, then delegate to base for draw dispatch."""
        # Handle config dict (gender, betas, etc.)
        if self.config_input.fresh_input:
            config = self.config_input()
            if isinstance(config, dict):
                self._apply_config(config)
        
        if self.pose_input.fresh_input:
            pose = self.pose_input()
            trans = self.trans_input()
            self.last_pose = pose
            self.last_trans = trans
            
            vertices = self._run_forward(pose, trans)
            if vertices is not None:
                normals = self._compute_normals(vertices, self.faces_np)
                self.prev_vbo_data = self._build_vbo_data(vertices, normals)
        
        # Let base MGLNode.execute() handle 'draw' messages and chain propagation
        super().execute()
    
    def _apply_config(self, config):
        """Apply config dict from NPZ playback (gender, betas, mocap_framerate)."""
        needs_reload = False
        
        # Gender
        gender = config.get('gender', None)
        if gender is not None:
            if isinstance(gender, np.ndarray):
                gender = str(gender)
            gender = gender.lower().strip()
            if gender in ('male', 'female') and gender != self.current_gender:
                self.gender_prop.set(gender)
                needs_reload = True
        
        # Betas
        betas = config.get('betas', None)
        if betas is not None:
            betas = np.array(betas, dtype=np.float32).flatten()
            if len(betas) > 10:
                betas = betas[:10]
            bt = torch.zeros(1, 10, dtype=torch.float32)
            bt[0, :len(betas)] = torch.tensor(betas)
            self.betas_tensor = bt
            needs_reload = True  # Shape changed, need to rebuild geometry
        
        if needs_reload:
            if self._load_model():
                # Force geometry rebuild on next draw
                self.vao = None
                self.vbo = None
                # Re-run forward pass with last pose so we don't flash T-pose
                if self.last_pose is not None:
                    vertices = self._run_forward(self.last_pose, self.last_trans)
                    if vertices is not None:
                        normals = self._compute_normals(vertices, self.faces_np)
                        self.prev_vbo_data = self._build_vbo_data(vertices, normals)
                else:
                    self.prev_vbo_data = None
                # print(f"MGLSMPLMeshNode: Config applied — gender={self.current_gender}, betas={'set' if self.betas_tensor is not None else 'none'}")
    
    def draw(self):
        if self.ctx is None:
            return
        
        inner_ctx = self.ctx.ctx
        
        # Ensure geometry is initialized
        if self.vao is None:
            vertices, indices = self.create_geometry()
            if vertices is None:
                return
            self.render_geometry(vertices, indices)
        
        # Update VBO with deformed vertices if available
        if self.prev_vbo_data is not None and self.vbo is not None:
            data = self.prev_vbo_data.astype('f4')
            self.vbo.write(data.tobytes())
        
        if self.prog is None:
            return
        
        # Set uniforms
        self.handle_shape_params()
        if 'V' in self.prog:
            self.prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in self.prog:
            self.prog['P'].write(self.ctx.projection_matrix.tobytes())
        if 'color' in self.prog:
            c = self.ctx.current_color
            opacity = self.opacity_prop()
            self.prog['color'].value = (c[0], c[1], c[2], opacity)
        
        # Lights and material
        self.ctx.update_lights(self.prog)
        self.ctx.update_material(self.prog)
        
        # Render mode
        mode = self.mode_input()
        cull = self.cull_input()
        
        if 'point_culling' in self.prog:
            self.prog['point_culling'].value = (mode == 'points' and cull)
        if 'round_points' in self.prog:
            self.prog['round_points'].value = (mode == 'points' and self.round_input())
        
        # Texturing
        tex = self.texture_input()
        if tex is not None and isinstance(tex, moderngl.Texture):
            tex.use(location=0)
            if 'diffuse_map' in self.prog:
                self.prog['diffuse_map'].value = 0
            if 'has_texture' in self.prog:
                self.prog['has_texture'].value = True
        else:
            if 'has_texture' in self.prog:
                self.prog['has_texture'].value = False
        
        if cull:
            inner_ctx.enable(moderngl.CULL_FACE)
        else:
            inner_ctx.disable(moderngl.CULL_FACE)
        
        # Alpha blending for opacity
        opacity = self.opacity_prop()
        if opacity < 1.0:
            inner_ctx.enable(moderngl.BLEND)
            inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        if mode == 'wireframe':
            inner_ctx.wireframe = True
            self.vao.render()
            inner_ctx.wireframe = False
        elif mode == 'points':
            if 'point_size' in self.prog:
                self.prog['point_size'].value = self.point_size_input()
            self.vao.render(mode=moderngl.POINTS)
        else:
            self.vao.render()
        
        if opacity < 1.0:
            inner_ctx.disable(moderngl.BLEND)
    
    def handle_shape_params(self):
        """Set model matrix (identity — mesh is already in world space)."""
        if self.ctx is not None and self.prog is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                self.prog['M'].write(model.astype('f4').T.tobytes())
