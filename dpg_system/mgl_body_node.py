import os
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
import moderngl
import math
from dpg_system.matrix_utils import *
from dpg_system.body_base import BodyData, t_PelvisAnchor, t_ActiveJointCount
from dpg_system.body_defs import *
from dpg_system.moderngl_nodes import MGLNode


class MGLBodyNode(MGLNode):
    # Mapping from SMPL limb length keys to body_base joint indices.
    # Each joint's bone_translation is the offset FROM ITS PARENT to that joint,
    # so e.g. 'upper_leg' (hip→knee) maps to the Knee joints, not the Hip joints.
    SMPL_TO_BODY_MAP = {
        'upper_leg':     [t_LeftKnee, t_RightKnee],              # hip → knee
        'lower_leg':     [t_LeftAnkle, t_RightAnkle],            # knee → ankle
        'upper_arm':     [t_LeftElbow, t_RightElbow],            # shoulder → elbow
        'lower_arm':     [t_LeftWrist, t_RightWrist],            # elbow → wrist
        'collar':        [t_LeftShoulder, t_RightShoulder],      # shoulder_blade → shoulder
        'spine_lower':   [t_SpinePelvis],                        # pelvis → spine1
        'spine_mid':     [t_LowerVertebrae],                     # spine1 → spine2
        'spine_upper':   [t_MidVertebrae],                       # spine2 → spine3
        'spine_to_neck': [t_UpperVertebrae],                     # spine3 → neck
        'neck':          [t_BaseOfSkull],                        # upper_vertebrae → base_of_skull
        'head':          [t_TopOfHead],                          # base_of_skull → top_of_head
        'foot':          [t_LeftBallOfFoot, t_RightBallOfFoot],  # ankle → ball_of_foot
        'hand':          [t_LeftKnuckle, t_RightKnuckle],        # wrist → knuckle
    }
    # pelvis_width and shoulder_width need special ratio-based handling
    # because the translations go at angles from center, not purely lateral.

    @staticmethod
    def factory(name, data, args=None):
        return MGLBodyNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.body = BodyData()
        self.body.node = self
        
        # Manually initialize joint hierarchy because standard BodyData lacks connect_limbs
        self.build_hierarchy(self.body.joints)
        self.body.create_limbs()

        self.pose_input = self.add_input('pose', triggers_execution=True)

        self.enable_callbacks_input = self.add_input('enable_callbacks', widget_type='checkbox', default_value=True)
        self.draw_spheres_input = self.add_input('draw_spheres', widget_type='checkbox', default_value=True)
        self.display_mode_input = self.add_input('display_mode', widget_type='combo', default_value='lines')
        self.display_mode_input.widget.combo_items = ['lines', 'solid']
        
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0)
        self.joint_radius_input = self.add_input('joint_radius', widget_type='drag_float', default_value=0.1)
        self.joint_data_input = self.add_input('joint_data')
        self.limb_lengths_input = self.add_input('limb_lengths')
        self.skeleton_mode_input = self.add_input('skeleton_mode', widget_type='combo', default_value='shadow', callback=self._on_skeleton_mode_changed)
        self.skeleton_mode_input.widget.combo_items = ['shadow', 'smpl']
        self.color_input = self.add_input('color', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0])
        self.instanced_display_mode_input = self.add_input('instanced_mode', widget_type='combo', default_value='solid')
        self.instanced_display_mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        
        self.joint_callback = self.add_output('joint_callback')
        self.joint_data_out = self.add_output('joint_data')
        self.joint_id_out = self.add_output('joint_id')

        # Original bone translations from definition.xml (immutable reference)
        self._original_translations = {}
        for joint_idx, joint in enumerate(self.body.joints):
            if joint is not None:
                self._original_translations[joint_idx] = joint.bone_translation.copy()

        # Baseline translations used for Shadow mode scaling (captured from originals)
        self._baseline_translations = None
        self._gl_dirty = False
        self._last_limb_data = None  # Store last limb data for mode switching
        
        # Mapping for Active Joint Data (User provides data in t_* order, separate from internal list order)
        self.joint_name_to_pose_index = {
            'BaseOfSkull': t_BaseOfSkull,
            'UpperVertebrae': t_UpperVertebrae,
            'MidVertebrae': t_MidVertebrae,
            'LowerVertebrae': t_LowerVertebrae,
            'SpinePelvis': t_SpinePelvis,
            'PelvisAnchor': t_PelvisAnchor,
            'LeftHip': t_LeftHip,
            'LeftKnee': t_LeftKnee,
            'LeftAnkle': t_LeftAnkle,
            'RightHip': t_RightHip,
            'RightKnee': t_RightKnee,
            'RightAnkle': t_RightAnkle,
            'LeftShoulderBladeBase': t_LeftShoulderBladeBase,
            'LeftShoulder': t_LeftShoulder,
            'LeftElbow': t_LeftElbow,
            'LeftWrist': t_LeftWrist,
            'RightShoulderBladeBase': t_RightShoulderBladeBase,
            'RightShoulder': t_RightShoulder,
            'RightElbow': t_RightElbow,
            'RightWrist': t_RightWrist,
            'TopOfHead': t_TopOfHead,
            'LeftBallOfFoot': t_LeftBallOfFoot,
            'LeftToeTip': t_LeftToeTip,
            'RightBallOfFoot': t_RightBallOfFoot,
            'RightToeTip': t_RightToeTip,
            'LeftKnuckle': t_LeftKnuckle,
            'LeftFingerTip': t_LeftFingerTip,
            'RightKnuckle': t_RightKnuckle,
            'RightFingerTip': t_RightFingerTip,
            'LeftHeel': t_LeftHeel,
            'RightHeel': t_RightHeel
        }

    def initialize_instanced_gl(self):
        ctx = self.ctx.ctx
        # 1. Geodesic Sphere Geometry (Unit Icosphere)
        subdivisions = 2
        
        t = (1.0 + math.sqrt(5.0)) / 2.0
        
        # Base icosahedron vertices (normalized to unit sphere)
        ico_verts = [
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ]
        
        def normalize_v(v):
            l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            return [v[0]/l, v[1]/l, v[2]/l]
        
        ico_verts = [normalize_v(v) for v in ico_verts]
        
        # Base icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        # Subdivide
        midpoint_cache = {}
        
        def get_midpoint(i1, i2):
            key = tuple(sorted((i1, i2)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            v1 = ico_verts[i1]
            v2 = ico_verts[i2]
            mid = normalize_v([v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]])
            ico_verts.append(mid)
            idx = len(ico_verts) - 1
            midpoint_cache[key] = idx
            return idx
        
        for _ in range(subdivisions):
            new_faces = []
            for tri in faces:
                v1, v2, v3 = tri
                a = get_midpoint(v1, v2)
                b = get_midpoint(v2, v3)
                c = get_midpoint(v3, v1)
                new_faces.extend([[v1,a,c], [v2,b,a], [v3,c,b], [a,b,c]])
            faces = new_faces
            midpoint_cache.clear()
        
        # Build vertex and index buffers (normals == positions for unit sphere)
        verts = []
        norms = []
        for v in ico_verts:
            verts.extend(v)
            norms.extend(v)
        
        indices = []
        for tri in faces:
            indices.extend(tri)
        
        self.instanced_sphere_vbo = ctx.buffer(np.array(verts, dtype='f4').tobytes())
        self.instanced_sphere_ibo = ctx.buffer(np.array(indices, dtype='i4').tobytes())
        self.instanced_sphere_norms = ctx.buffer(np.array(norms, dtype='f4').tobytes())
        
        # 2. Instance Data Buffers (Dynamic)
        # We need to pre-calculate the "valid" bone indices to skip duplicates
        valid_indices = []
        for i, joint in enumerate(self.body.joints):
             if i in [t_PelvisAnchor]:
                continue
             if joint is None:
                continue
             valid_indices.append(i)
        
        self.instanced_bone_indices = np.array(valid_indices, dtype='i4')
        # Buffer for bone indices (static enough unless body changes)
        self.instanced_bone_idx_buffer = ctx.buffer(self.instanced_bone_indices.tobytes())
        
        # Buffer for radii (dynamic) - Initial size matching valid count
        self.instanced_radius_buffer = ctx.buffer(reserve=len(valid_indices) * 4) # float32
        
        # 3. VAO
        # Attributes: in_pos, in_norm, in_radius, in_bone_idx
        # Radius and BoneIdx are per-instance (divisor=1)
        
        self.instanced_vao = ctx.vertex_array(self.ctx.mgl_instanced_joint_shader, [
            (self.instanced_sphere_vbo, '3f', 'in_pos'),
            (self.instanced_sphere_norms, '3f', 'in_norm'),
            (self.instanced_radius_buffer, '1f/i', 'in_radius'),
            (self.instanced_bone_idx_buffer, '1i/i', 'in_bone_idx'),
        ], index_buffer=self.instanced_sphere_ibo)
        
    def draw_instanced(self, joint_data):
        # DEBUG PRINTS FOR USER
        if not hasattr(self, 'instanced_vao'):
            # print("MGLBodyNode: No instanced_vao in draw_instanced - skipping")
            return
            
        ctx = self.ctx.ctx
        num_joints = len(self.body.joints)
        
        # Robust Data Handling
        if not isinstance(joint_data, np.ndarray):
            joint_data = np.array(joint_data, dtype=np.float32)

        # Handle Multi-dimensional Data (e.g. Vectors)
        # Use L2 Norm if shape is (N, M) where M > 1
        if joint_data.ndim > 1:
            if joint_data.shape[-1] > 1:
                # Compute Magnitude (L2 Norm) along the last axis
                joint_data = np.linalg.norm(joint_data, axis=-1)
            else:
                # Just flatten if (N, 1) or similar
                joint_data = joint_data.flatten()
        else:
            joint_data = joint_data.flatten()
        
        # Data arrives in body t_ index order (smpl_to_active outputs proper indices)
        full_data = np.zeros(num_joints, dtype=np.float32)
        for i in range(min(num_joints, len(joint_data))):
            full_data[i] = joint_data[i]

        # Filter to Valid Indices
        # self.instanced_bone_indices contains indices of joints we actually draw
        # We need to extract data for these indices
        valid_data = full_data[self.instanced_bone_indices]
        
        # Apply base radius multiplier
        base_r = self.joint_radius_input()
        if base_r is None: base_r = 0.1 # Default or prevent crash
        
        # Multiply valid data by base radius
        scaled_data = valid_data * base_r
        
        self.instanced_radius_buffer.write(scaled_data.tobytes())
        
        # Setup Shader Uniforms (VP)
        if 'VP' in self.ctx.mgl_instanced_joint_shader:
            # Match MGLBodyNode.draw logic: View @ Projection likely
            # Though standard is Projection @ View, it depends on conventions
            # Let's align with working limb code
            vp = self.ctx.view_matrix @ self.ctx.projection_matrix
            self.ctx.mgl_instanced_joint_shader['VP'].write(vp.flatten().tobytes())
            
        # Upload Bones to Shader (use joint_tip_matrices for actual joint positions)
        max_bones = 50
        bones_bytes = bytearray(max_bones * 64) 
        
        for i in range(min(num_joints, max_bones)):
            if hasattr(self, 'joint_tip_matrices') and i in self.joint_tip_matrices:
                mat = self.joint_tip_matrices[i]
                # Transpose for OpenGL (Column-Major)
                bones_bytes[i*64 : (i+1)*64] = mat.T.astype('f4').flatten().tobytes()
            elif i in self.global_matrices:
                mat = self.global_matrices[i]
                bones_bytes[i*64 : (i+1)*64] = mat.T.astype('f4').flatten().tobytes()
        
        if 'bones' in self.ctx.mgl_instanced_joint_shader:
            self.ctx.mgl_instanced_joint_shader['bones'].write(bones_bytes)
            
        # Note: Shader does NOT have base_radius uniform, so we handle it on CPU above.
        
        
        # Set Color
        # Use node input if available, otherwise context
        sphere_color = self.color_input()
        if sphere_color is None:
             sphere_color = self.ctx.current_color
             
        # Normalize colors (handle 0-255 vs 0-1)
        # Identical logic to MGLMaterialNode.norm(c)
        def norm(c):
             if c is None or len(c) == 0: return [1.0, 1.0, 1.0, 1.0] # Fallback
             
             # Handle list/tuple to list
             c_list = list(c)
             
             if max(c_list) > 1.0:
                 c_list = [val / 255.0 for val in c_list]
                 
             # Ensure 4 components (defaults to alpha 1.0 if missing)
             if len(c_list) == 3:
                 return c_list + [1.0]
             return c_list
             
        final_color = norm(sphere_color)
        
        if 'color' in self.ctx.mgl_instanced_joint_shader:
            self.ctx.mgl_instanced_joint_shader['color'].value = tuple(final_color)
             
        # Handle Display Mode
        mode = self.instanced_display_mode_input()
        
        # Enable Standard Alpha Blending
        self.ctx.ctx.enable(moderngl.BLEND)
        self.ctx.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        count = len(scaled_data)
        
        if mode == 'wireframe':
            self.ctx.ctx.disable(moderngl.CULL_FACE)
            self.ctx.ctx.wireframe = True
            self.instanced_vao.render(instances=count)
            self.ctx.ctx.wireframe = False
            self.ctx.ctx.enable(moderngl.CULL_FACE)
        elif mode == 'points':
            self.ctx.ctx.disable(moderngl.CULL_FACE)
            self.instanced_vao.render(instances=count, mode=moderngl.POINTS)
            self.ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            # Solid: enable culling to prevent back-face checkering
            self.ctx.ctx.enable(moderngl.CULL_FACE)
            self.instanced_vao.render(instances=count)
            
        self.ctx.ctx.disable(moderngl.BLEND)
        
    def build_hierarchy(self, joints):
        # Manually connect limbs (Hierarchy for traversal)
        # Based on AlternateBodyData.connect_humanoid
        
        # Initialize empty lists first
        for joint in joints:
            joint.immed_children = []
            
        # Helper to safely append
        def add_child(parent_idx, child_idx):
            if 0 <= parent_idx < len(joints) and 0 <= child_idx < len(joints):
                joints[parent_idx].immed_children.append(child_idx)

        # Spine
        add_child(t_PelvisAnchor, t_SpinePelvis)
        add_child(t_SpinePelvis, t_LowerVertebrae)
        add_child(t_LowerVertebrae, t_MidVertebrae)
        add_child(t_MidVertebrae, t_UpperVertebrae)
        add_child(t_UpperVertebrae, t_BaseOfSkull)
        # Connect extremities (Indices 20+)
        add_child(t_BaseOfSkull, t_TopOfHead) 
        
        # Legs
        add_child(t_PelvisAnchor, t_LeftHip)
        add_child(t_PelvisAnchor, t_RightHip)
        
        add_child(t_LeftHip, t_LeftKnee)
        add_child(t_LeftKnee, t_LeftAnkle)
        add_child(t_LeftAnkle, t_LeftBallOfFoot)
        if t_LeftHeel < len(joints):
            add_child(t_LeftAnkle, t_LeftHeel)
        if t_LeftBallOfFoot < len(joints) and t_LeftToeTip < len(joints):
             add_child(t_LeftBallOfFoot, t_LeftToeTip)
        
        add_child(t_RightHip, t_RightKnee)
        add_child(t_RightKnee, t_RightAnkle)
        add_child(t_RightAnkle, t_RightBallOfFoot)
        if t_RightHeel < len(joints):
            add_child(t_RightAnkle, t_RightHeel)
        if t_RightBallOfFoot < len(joints) and t_RightToeTip < len(joints):
             add_child(t_RightBallOfFoot, t_RightToeTip)
        
        # Arms (Collar/ShoulderBlade attached to MidVertebrae usually, or Spine?)
        # AlternateBodyData: MidVertebrae -> ShoulderBlade -> Shoulder
        add_child(t_MidVertebrae, t_LeftShoulderBladeBase)
        add_child(t_LeftShoulderBladeBase, t_LeftShoulder)
        add_child(t_LeftShoulder, t_LeftElbow)
        add_child(t_LeftElbow, t_LeftWrist)
        add_child(t_LeftWrist, t_LeftKnuckle)
        if t_LeftKnuckle < len(joints) and t_LeftFingerTip < len(joints):
             add_child(t_LeftKnuckle, t_LeftFingerTip)
        
        add_child(t_MidVertebrae, t_RightShoulderBladeBase)
        add_child(t_RightShoulderBladeBase, t_RightShoulder)
        add_child(t_RightShoulder, t_RightElbow)
        add_child(t_RightElbow, t_RightWrist)
        add_child(t_RightWrist, t_RightKnuckle)
        if t_RightKnuckle < len(joints) and t_RightFingerTip < len(joints):
             add_child(t_RightKnuckle, t_RightFingerTip)
        
        self.cube_vbo = None
        self.cube_vao = None
        self.prog = None
        self.instance_vbo = None
        self.max_limbs = 64

    def initialize_gl(self):
        if self.ctx.ctx is None: return
        
        # Simple Cube Geometry (centered at 0,0,0, size 1x1x1)
        # But body_base cubes are complex. Let's use simple unit cube valid for scaling.
        vertices = np.array([
            # Front
            -0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
             0.5, -0.5, 0.5, 0, 0, 1, 1, 0,
             0.5,  0.5, 0.5, 0, 0, 1, 1, 1,
            -0.5,  0.5, 0.5, 0, 0, 1, 0, 1,
            # Back
            -0.5, -0.5, -0.5, 0, 0, -1, 0, 0,
             0.5, -0.5, -0.5, 0, 0, -1, 1, 0,
             0.5,  0.5, -0.5, 0, 0, -1, 1, 1,
            -0.5,  0.5, -0.5, 0, 0, -1, 0, 1,
            # Top
            -0.5,  0.5, -0.5, 0, 1, 0, 0, 0,
            -0.5,  0.5,  0.5, 0, 1, 0, 0, 1,
             0.5,  0.5,  0.5, 0, 1, 0, 1, 1,
             0.5,  0.5, -0.5, 0, 1, 0, 1, 0,
            # Bottom
            -0.5, -0.5, -0.5, 0, -1, 0, 0, 0,
             0.5, -0.5, -0.5, 0, -1, 0, 1, 0,
             0.5, -0.5,  0.5, 0, -1, 0, 1, 1,
            -0.5, -0.5,  0.5, 0, -1, 0, 0, 1,
            # Right
             0.5, -0.5, -0.5, 1, 0, 0, 0, 0,
             0.5,  0.5, -0.5, 1, 0, 0, 1, 0,
             0.5,  0.5,  0.5, 1, 0, 0, 1, 1,
             0.5, -0.5,  0.5, 1, 0, 0, 0, 1,
            # Left
            -0.5, -0.5, -0.5, -1, 0, 0, 0, 0,
            -0.5, -0.5,  0.5, -1, 0, 0, 1, 0,
            -0.5,  0.5,  0.5, -1, 0, 0, 1, 1,
            -0.5,  0.5, -0.5, -1, 0, 0, 0, 1,
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 7, 6, 6, 5, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ], dtype='i4')

        ctx = self.ctx.ctx
        self.cube_vbo = ctx.buffer(vertices)
        self.cube_ibo = ctx.buffer(indices)
        
        # Shader for Skinned Rendering (Batched Limbs)
        self.prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 VP; // ViewProjection
                
                #define MAX_BONES 50
                uniform mat4 bones[MAX_BONES];
                
                in vec3 in_pos;
                in vec3 in_norm;
                in float in_bone_index;
                
                // Material Color
                uniform vec4 color;
                
                out vec3 v_norm;
                out vec3 v_pos;
                out vec4 v_color;
                
                void main() {
                    int b_idx = int(in_bone_index);
                    mat4 bone_mat = bones[b_idx];
                    
                    vec4 world_pos = bone_mat * vec4(in_pos, 1.0);
                    gl_Position = VP * world_pos;
                    v_pos = world_pos.xyz;
                    
                    // Normal Matrix
                    mat3 normal_matrix = transpose(inverse(mat3(bone_mat)));
                    v_norm = normal_matrix * in_norm;
                    
                    v_color = color; 
                }
            ''',
            fragment_shader='''
                #version 330
                #define MAX_LIGHTS 8
                
                in vec3 v_norm;
                in vec3 v_pos;
                in vec4 v_color;
                
                uniform vec3 view_pos;
                
                // Lighting Uniforms
                struct Light {
                    vec3 pos;
                    vec3 ambient;
                    vec3 diffuse;
                    vec3 specular;
                    float intensity;
                };
                
                uniform int num_lights;
                uniform Light lights[MAX_LIGHTS];
                
                // Material Uniforms
                uniform vec4 material_ambient;
                uniform vec4 material_diffuse;
                uniform vec4 material_specular;
                uniform float material_shininess;
                
                out vec4 f_color;
                
                void main() {
                    vec3 norm = normalize(v_norm);
                    vec3 viewDir = normalize(view_pos - v_pos);
                    vec3 result = vec3(0.0);
                    
                    // If no lights, use simple ambient fallback based on material
                    if (num_lights == 0) {
                        result = material_ambient.rgb * v_color.rgb + material_diffuse.rgb * v_color.rgb * 0.5;
                    } else {
                        vec3 total_ambient = vec3(0.0);
                        vec3 total_diffuse = vec3(0.0);
                        vec3 total_specular = vec3(0.0);
                        
                        for(int i = 0; i < MAX_LIGHTS; i++) {
                            if (i >= num_lights) break;
                            
                            // Ambient
                            total_ambient += lights[i].ambient * material_ambient.rgb;
                            
                            // Diffuse
                            vec3 lightDir = normalize(lights[i].pos - v_pos);
                            float diff = max(dot(norm, lightDir), 0.0);
                            
                            total_diffuse += diff * lights[i].diffuse * material_diffuse.rgb * lights[i].intensity;
                            
                            // Specular (Blinn-Phong)
                            vec3 halfwayDir = normalize(lightDir + viewDir);
                            float spec = pow(max(dot(norm, halfwayDir), 0.0), material_shininess);
                            total_specular += spec * lights[i].specular * material_specular.rgb * lights[i].intensity;
                        }
                        
                        // Premultiplied Alpha Output
                        // Body Color = (Ambient + Diffuse) * Alpha
                        // Specular = Specular (Additive)
                        float alpha = v_color.a * material_diffuse.a * material_ambient.a;
                        result = (total_ambient + total_diffuse) * v_color.rgb * alpha + total_specular;
                        
                        f_color = vec4(result, alpha);
                    }
                }
            '''
        )
        
        # Build Limb Geometry
        all_verts = []
        all_norms = []
        all_indices = []
        
        # Skeleton Lines
        line_verts = []
        line_norms = []
        line_indices = []
        line_bone_indices = []
        line_vert_count = 0
        
        def_points = [
            (-.5, -.5, 0), (-1.0, -1.0, 0.5), (.5, -.5, 0), (1.0, -1.0, 0.5),
            (.5, .5, 0), (1.0, 1.0, 0.5), (-.5, .5, 0),(-1.0, 1.0, .5),
            (-.5, -.5, 1.0), (.5, -.5, 1.0), (.5, .5, 1.0), (-.5, .5, 1.0)
        ]

        for i in range(len(self.body.joints)):
            if i == t_PelvisAnchor: continue
            if self.body.joints[i] is None or not self.body.joints[i].do_draw:
                continue
                
            pts = def_points
            if i < len(self.body.limb_vertices) and self.body.limb_vertices[i] is not None:
                pts = self.body.limb_vertices[i]
            
            # Simple scaling logic based on dims or defaults
            scale_vec = np.array([0.05, 0.05, 1.0])
            if i < len(self.body.joints) and self.body.joints[i] is not None:
                j = self.body.joints[i]
                if hasattr(j, 'dims'):
                     dz = j.dims[0]
                     dx = j.dims[1] / 2.0
                     dy = j.dims[2] / 2.0
                     scale_vec = np.array([dx, dy, dz])


            def add_tri(p1, p2, p3, n):
                all_verts.extend(p1); all_norms.extend(n); all_indices.append(float(i))
                all_verts.extend(p2); all_norms.extend(n); all_indices.append(float(i))
                all_verts.extend(p3); all_norms.extend(n); all_indices.append(float(i))

            def calc_n(p1, p2, p3):
                v1 = np.array(p2) - np.array(p1)
                v2 = np.array(p3) - np.array(p1)
                c = np.cross(v2, v1)
                l = np.linalg.norm(c)
                return c / l if l > 0 else c
                
            # Add Skeleton Line Segment (0,0,0 -> 0,0,dz)
            l_start = [0.0, 0.0, 0.0]
            l_end = [0.0, 0.0, scale_vec[2]]
            line_verts.extend(l_start); line_verts.extend(l_end)
            line_norms.extend([0,0,1]); line_norms.extend([0,0,1]) # Dummy normals
            line_bone_indices.append(float(i)); line_bone_indices.append(float(i))
            line_indices.append(line_vert_count); line_indices.append(line_vert_count+1)
            line_vert_count += 2
            
            # Apply scale FIRST
            pts_np = [np.array(p) * scale_vec for p in pts]
            
            if len(pts) == 8:
                n0 = calc_n(pts_np[0], pts_np[1], pts_np[2])
                add_tri(pts_np[0], pts_np[1], pts_np[2], n0)
                add_tri(pts_np[2], pts_np[1], pts_np[3], n0)
                n1 = calc_n(pts_np[2], pts_np[3], pts_np[4])
                add_tri(pts_np[2], pts_np[3], pts_np[4], n1)
                add_tri(pts_np[4], pts_np[3], pts_np[5], n1)
                n2 = calc_n(pts_np[4], pts_np[5], pts_np[6])
                add_tri(pts_np[4], pts_np[5], pts_np[6], n2)
                add_tri(pts_np[6], pts_np[5], pts_np[7], n2)
                n3 = calc_n(pts_np[6], pts_np[7], pts_np[0])
                add_tri(pts_np[6], pts_np[7], pts_np[0], n3)
                add_tri(pts_np[0], pts_np[7], pts_np[1], n3)
                n4 = calc_n(pts_np[1], pts_np[3], pts_np[7])
                add_tri(pts_np[1], pts_np[3], pts_np[7], n4); add_tri(pts_np[7], pts_np[3], pts_np[5], n4)
                n5 = calc_n(pts_np[0], pts_np[2], pts_np[6])
                add_tri(pts_np[0], pts_np[2], pts_np[6], n5); add_tri(pts_np[6], pts_np[2], pts_np[4], n5)
            else:
                n = calc_n(pts_np[0], pts_np[1], pts_np[2])
                add_tri(pts_np[0], pts_np[1], pts_np[2], n); add_tri(pts_np[2], pts_np[1], pts_np[3], n)
                n = calc_n(pts_np[2], pts_np[3], pts_np[4])
                add_tri(pts_np[2], pts_np[3], pts_np[4], n); add_tri(pts_np[4], pts_np[3], pts_np[5], n)
                n = calc_n(pts_np[4], pts_np[5], pts_np[6])
                add_tri(pts_np[4], pts_np[5], pts_np[6], n); add_tri(pts_np[6], pts_np[5], pts_np[7], n)
                n = calc_n(pts_np[6], pts_np[7], pts_np[0])
                add_tri(pts_np[6], pts_np[7], pts_np[0], n); add_tri(pts_np[0], pts_np[7], pts_np[1], n)
                n = calc_n(pts_np[1], pts_np[8], pts_np[3])
                add_tri(pts_np[1], pts_np[8], pts_np[3], n); add_tri(pts_np[3], pts_np[8], pts_np[9], n)
                n = calc_n(pts_np[3], pts_np[9], pts_np[5])
                add_tri(pts_np[3], pts_np[9], pts_np[5], n); add_tri(pts_np[5], pts_np[9], pts_np[10], n)
                n = calc_n(pts_np[5], pts_np[10], pts_np[7])
                add_tri(pts_np[5], pts_np[10], pts_np[7], n); add_tri(pts_np[7], pts_np[10], pts_np[11], n)
                n = calc_n(pts_np[7], pts_np[11], pts_np[1])
                add_tri(pts_np[7], pts_np[11], pts_np[1], n); add_tri(pts_np[1], pts_np[11], pts_np[8], n)
                # TOP CAP Normal (should be +Z/Outward, but standard calc gives -Z/Inward)
                # So we swap arguments to flip it back
                n = calc_n(pts_np[8], pts_np[11], pts_np[9])
                add_tri(pts_np[8], pts_np[9], pts_np[11], n); add_tri(pts_np[11], pts_np[9], pts_np[10], n)
                n = calc_n(pts_np[0], pts_np[2], pts_np[6])
                add_tri(pts_np[0], pts_np[2], pts_np[6], n); add_tri(pts_np[6], pts_np[2], pts_np[4], n)

        
        v_data = np.array(all_verts, dtype='f4')
        n_data = np.array(all_norms, dtype='f4')
        i_data = np.array(all_indices, dtype='f4')
        
        self.limb_top_fix_vbo = ctx.buffer(np.hstack([v_data.reshape(-1,3), n_data.reshape(-1,3), i_data.reshape(-1,1)]).flatten().tobytes())
        self.limb_top_fix_vao = ctx.vertex_array(self.prog, [
            (self.limb_top_fix_vbo, '3f 3f 1f', 'in_pos', 'in_norm', 'in_bone_index')
        ])

        # Create Skeleton VBO/VAO
        l_v_data = np.array(line_verts, dtype='f4')
        l_n_data = np.array(line_norms, dtype='f4')
        l_b_data = np.array(line_bone_indices, dtype='f4')
        l_i_data = np.array(line_indices, dtype='i4') # Indices are int
        
        # Interleave
        # buffer data: pos(3), norm(3), bone(1)
        self.skeleton_vbo = ctx.buffer(np.hstack([l_v_data.reshape(-1,3), l_n_data.reshape(-1,3), l_b_data.reshape(-1,1)]).flatten().tobytes())
        self.skeleton_index_buffer = ctx.buffer(l_i_data.tobytes())
        
        self.skeleton_vao = ctx.vertex_array(self.prog, [
            (self.skeleton_vbo, '3f 3f 1f', 'in_pos', 'in_norm', 'in_bone_index')
        ], index_buffer=self.skeleton_index_buffer)

        # Unlit Shader for Optimized Lines
        self.unlit_prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 VP;
                #define MAX_BONES 50
                uniform mat4 bones[MAX_BONES];
                in vec3 in_pos;
                // Skipped Normals
                in float in_bone_index;
                uniform vec4 color;
                out vec4 v_color;
                void main() {
                    int b_idx = int(in_bone_index);
                    mat4 bone_mat = bones[b_idx];
                    vec4 world_pos = bone_mat * vec4(in_pos, 1.0);
                    gl_Position = VP * world_pos;
                    v_color = color; 
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
        self.skeleton_unlit_vao = ctx.vertex_array(self.unlit_prog, [
            (self.skeleton_vbo, '3f 12x 1f', 'in_pos', 'in_bone_index')
        ], index_buffer=self.skeleton_index_buffer)

        # Initialize Instanced Geometry (Spheres)
        self.initialize_instanced_gl()

    # Explicit mapping: SMPL joint index → body joint array index (t_* constants)
    # NOTE: bmolab_active_joints indices diverge from body joint array indices
    # for non-active joints (>=20), so we must use t_* constants directly.
    SMPL_JOINT_TO_BODY_JOINT = {
        0: t_PelvisAnchor,              # pelvis
        1: t_LeftHip,                   # left_hip
        2: t_RightHip,                  # right_hip
        3: t_SpinePelvis,               # spine1
        4: t_LeftKnee,                  # left_knee
        5: t_RightKnee,                 # right_knee
        6: t_LowerVertebrae,            # spine2
        7: t_LeftAnkle,                 # left_ankle
        8: t_RightAnkle,                # right_ankle
        9: t_MidVertebrae,              # spine3
        10: t_LeftBallOfFoot,           # left_foot
        11: t_RightBallOfFoot,          # right_foot
        12: t_UpperVertebrae,           # neck
        13: t_LeftShoulderBladeBase,     # left_collar
        14: t_RightShoulderBladeBase,    # right_collar
        15: t_BaseOfSkull,              # head
        16: t_LeftShoulder,             # left_shoulder
        17: t_RightShoulder,            # right_shoulder
        18: t_LeftElbow,                # left_elbow
        19: t_RightElbow,               # right_elbow
        20: t_LeftWrist,                # left_wrist
        21: t_RightWrist,               # right_wrist
        22: t_LeftKnuckle,              # left_hand
        23: t_RightKnuckle,             # right_hand
        24: t_LeftToeTip,               # left_toe_tip
        25: t_RightToeTip,              # right_toe_tip
        26: t_LeftFingerTip,            # left_finger_tip
        27: t_RightFingerTip,           # right_finger_tip
        28: t_LeftHeel,                 # left_heel
        29: t_RightHeel,                # right_heel
    }

    def _on_skeleton_mode_changed(self):
        """Re-apply limb data when skeleton mode is toggled."""
        # Restore original Shadow bone translations before re-applying
        for joint_idx, orig_bt in self._original_translations.items():
            if joint_idx < len(self.body.joints) and self.body.joints[joint_idx] is not None:
                self.body.joints[joint_idx].bone_translation = orig_bt.copy()
        self._baseline_translations = None

        mode = self.skeleton_mode_input()
        if mode == 'smpl':
            # Use existing limb data if available, otherwise compute defaults
            if self._last_limb_data is not None and isinstance(self._last_limb_data, dict) and 'offsets' in self._last_limb_data:
                self._apply_limb_lengths(self._last_limb_data)
            else:
                offsets = self._get_default_smpl_offsets()
                if offsets is not None:
                    self._apply_smpl_offsets(offsets, {})
                    self._gl_dirty = True
                    print('mgl_body: applied default SMPL skeleton offsets')
                else:
                    print('mgl_body: smplx not available, cannot switch to SMPL skeleton')
        elif self._last_limb_data is not None:
            self._apply_limb_lengths(self._last_limb_data)
        self._gl_dirty = True

    def _apply_limb_lengths(self, limb_data):
        """Apply SMPL limb lengths to body joints by scaling bone_translation vectors."""
        lengths = limb_data.get('lengths', limb_data) if isinstance(limb_data, dict) else {}
        if not lengths:
            return

        # Store baseline translations on first use
        if self._baseline_translations is None:
            self._baseline_translations = {}
            for joint_idx, joint in enumerate(self.body.joints):
                if joint is not None:
                    self._baseline_translations[joint_idx] = joint.bone_translation.copy()

        skeleton_mode = self.skeleton_mode_input()

        # --- SMPL skeleton mode: use actual SMPL offset vectors ---
        if skeleton_mode == 'smpl':
            offsets = limb_data.get('offsets') if isinstance(limb_data, dict) else None
            if offsets is not None:
                self._apply_smpl_offsets(offsets, lengths)
                self._gl_dirty = True
                return
            # Fall through to shadow mode if no offsets available

        # --- Shadow skeleton mode: scale existing directions ---
        bl = self._baseline_translations

        # Joints with asymmetric custom geometry that extends well below z=0
        # (e.g. pelvis bowl, shoulder blade shapes). Only update bone_translation
        # for correct positioning, not dims[0] which would deform the shape.
        SKIP_DIMS_JOINTS = {t_SpinePelvis, t_LeftShoulderBladeBase, t_RightShoulderBladeBase}

        # Standard limb lengths: scale bone_translation direction-preserving
        for smpl_key, joint_indices in self.SMPL_TO_BODY_MAP.items():
            if smpl_key not in lengths:
                continue
            new_length = lengths[smpl_key]
            for ji in joint_indices:
                if ji >= len(self.body.joints) or self.body.joints[ji] is None:
                    continue
                if ji not in bl:
                    continue
                baseline = bl[ji]
                baseline_len = np.linalg.norm(baseline)
                if baseline_len < 1e-6:
                    continue
                direction = baseline / baseline_len
                self.body.joints[ji].bone_translation = direction * new_length
                if ji not in SKIP_DIMS_JOINTS:
                    self.body.joints[ji].dims[0] = new_length
                    self.body.joints[ji].base_dims[0] = new_length

        # Special handling: pelvis_width (ratio-based, translations at angles)
        if 'pelvis_width' in lengths and t_LeftHip in bl and t_RightHip in bl:
            baseline_pw = np.linalg.norm(bl[t_LeftHip] - bl[t_RightHip])
            if baseline_pw > 1e-6:
                ratio = lengths['pelvis_width'] / baseline_pw
                for ji in [t_LeftHip, t_RightHip]:
                    new_bt = bl[ji] * ratio
                    self.body.joints[ji].bone_translation = new_bt
                    base_dims = self.body.joints[ji].base_dims
                    self.body.joints[ji].dims[1] = base_dims[1] * ratio
                    self.body.joints[ji].dims[2] = base_dims[2] * ratio

        # Special handling: shoulder_width (ratio-based)
        if 'shoulder_width' in lengths:
            l_blade, r_blade = t_LeftShoulderBladeBase, t_RightShoulderBladeBase
            l_sh, r_sh = t_LeftShoulder, t_RightShoulder
            if all(j in bl for j in [l_blade, r_blade, l_sh, r_sh]):
                l_pos = bl[l_blade] + bl[l_sh]
                r_pos = bl[r_blade] + bl[r_sh]
                baseline_sw = np.linalg.norm(l_pos - r_pos)
                if baseline_sw > 1e-6:
                    ratio = lengths['shoulder_width'] / baseline_sw
                    for ji in [l_blade, r_blade]:
                        new_bt = bl[ji] * ratio
                        self.body.joints[ji].bone_translation = new_bt
                        base_dims = self.body.joints[ji].base_dims
                        self.body.joints[ji].dims[1] = base_dims[1] * ratio
                        self.body.joints[ji].dims[2] = base_dims[2] * ratio

        self._gl_dirty = True

    def _get_default_smpl_offsets(self):
        """Compute default SMPL offsets from smplx model (betas=0, neutral)."""
        if hasattr(self, '_default_smpl_offsets') and self._default_smpl_offsets is not None:
            return self._default_smpl_offsets

        try:
            import torch
            import smplx
        except ImportError:
            return None

        model_path = os.path.dirname(os.path.abspath(__file__))
        try:
            model = smplx.create(model_path=model_path, model_type='smplh',
                                 gender='MALE', num_betas=10, ext='pkl')
            output = model(betas=torch.zeros(1, 10))
            joints = output.joints[0].detach().cpu().numpy()

            print(f"mgl_body: SMPL-H joints shape: {joints.shape}")

            # SMPL parent indices
            smpl_parents = [
                -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
                7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
                18, 19, 20, 21,
            ]

            offsets = np.zeros((30, 3))
            for i in range(1, 24):
                child_idx = i
                # Handle SMPL-H hand joint remapping
                if joints.shape[0] > 24:
                    if i == 22:
                        child_idx = 22  # Left hand base in SMPL-H
                    if i == 23:
                        child_idx = 37  # Right hand base in SMPL-H
                offsets[i] = joints[child_idx] - joints[smpl_parents[i]]

            # Diagnostic: print hand offset distances
            l_hand_dist = np.linalg.norm(offsets[22])
            r_hand_dist = np.linalg.norm(offsets[23])
            l_forearm_dist = np.linalg.norm(offsets[20])
            r_forearm_dist = np.linalg.norm(offsets[21])
            print(f"mgl_body: SMPL offset distances:")
            print(f"  left forearm (20): {l_forearm_dist:.4f}m")
            print(f"  right forearm (21): {r_forearm_dist:.4f}m")
            print(f"  left hand (22, using joints[22]): {l_hand_dist:.4f}m")
            print(f"  right hand (23, using joints[37]): {r_hand_dist:.4f}m")

            # If left hand offset is suspiciously large (>0.15m = 15cm),
            # joints[22] is probably not the left hand base.
            # Try alternative indices.
            if l_hand_dist > 0.15 and joints.shape[0] > 24:
                print(f"  WARNING: left hand offset {l_hand_dist:.4f}m seems too large!")
                # Scan nearby indices for the actual left hand base
                best_idx = 22
                best_dist = l_hand_dist
                for try_idx in range(22, min(37, joints.shape[0])):
                    d = np.linalg.norm(joints[try_idx] - joints[20])
                    if 0.01 < d < best_dist:
                        best_dist = d
                        best_idx = try_idx
                if best_idx != 22:
                    print(f"  Using joints[{best_idx}] instead (dist={best_dist:.4f}m)")
                    offsets[22] = joints[best_idx] - joints[20]

            # Virtual extensions matching smpl_processor
            foot_vec = np.array([0.0, 0.0, 1.0])
            offsets[24] = foot_vec * 0.15  # L_Toe
            offsets[25] = foot_vec * 0.15  # R_Toe

            # Fingertips: extend hand direction
            l_dir = offsets[22] / max(np.linalg.norm(offsets[22]), 1e-6)
            r_dir = offsets[23] / max(np.linalg.norm(offsets[23]), 1e-6)
            offsets[26] = l_dir * 0.08
            offsets[27] = r_dir * 0.08

            # Heels
            offsets[28] = np.array([0.0, -0.03, -0.07])
            offsets[29] = np.array([0.0, -0.03, -0.07])

            self._default_smpl_offsets = offsets
            return offsets
        except Exception as e:
            print(f"mgl_body: error computing default SMPL offsets: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_smpl_offsets(self, offsets, lengths):
        """Replace body joint bone translations with SMPL model offset vectors."""
        idx_map = MGLBodyNode.SMPL_JOINT_TO_BODY_JOINT

        # Joints with custom geometry that should NOT have dims[0] scaled
        SKIP_DIMS_JOINTS = {t_SpinePelvis, t_LeftShoulderBladeBase, t_RightShoulderBladeBase}

        # Print diagnostics for key joints on first call
        if not hasattr(self, '_smpl_offsets_debug_done'):
            self._smpl_offsets_debug_done = True
            print(f"_apply_smpl_offsets: offsets shape={offsets.shape}")
            for si in [0, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
                if si < offsets.shape[0]:
                    v = offsets[si]
                    bi = idx_map.get(si, -1)
                    print(f"  SMPL {si} -> body {bi}: offset={v}, norm={np.linalg.norm(v):.4f}m")
            # Check fingertip original translations
            for bi in [t_LeftFingerTip, t_RightFingerTip, t_LeftKnuckle, t_RightKnuckle]:
                orig = self._original_translations.get(bi)
                if orig is not None:
                    print(f"  body {bi}: original bone_translation={orig}, norm={np.linalg.norm(orig):.4f}m")
                else:
                    print(f"  body {bi}: NOT in _original_translations")

        # Virtual extension indices — preserve Shadow geometry for these
        # (toes and heels only; fingertips get SMPL offsets applied directly)
        VIRTUAL_INDICES = {24, 25, 28, 29}

        # offsets is (30, 3) in SMPL joint order
        for smpl_i in range(min(offsets.shape[0], 30)):
            body_i = idx_map.get(smpl_i, -1)
            if body_i < 0 or body_i >= len(self.body.joints):
                continue
            joint = self.body.joints[body_i]
            if joint is None:
                continue

            offset_vec = offsets[smpl_i]
            length = np.linalg.norm(offset_vec)

            # Skip virtual extensions — preserve Shadow geometry for these
            if smpl_i in VIRTUAL_INDICES:
                continue

            joint.bone_translation = offset_vec.copy()
            if length > 1e-6 and body_i not in SKIP_DIMS_JOINTS:
                joint.dims[0] = length
                joint.base_dims[0] = length

    def execute(self):
        if self.pose_input.fresh_input:
            data = self.pose_input()
            if data is not None:
                self.body.update_quats(data)

        if self.joint_data_input.fresh_input:
            self.latest_joint_data = self.joint_data_input()

        if self.limb_lengths_input.fresh_input:
            limb_data = self.limb_lengths_input()
            if limb_data is not None:
                self._last_limb_data = limb_data
                self._apply_limb_lengths(limb_data)

        if self.mgl_input.fresh_input:
            msg = self.mgl_input()
            if msg == 'draw':
                self.ctx = MGLContext.get_instance()
                self.draw()
                self.ctx = None
            self.mgl_output.send(msg)

    def draw(self):
        if not self.ctx or not self.ctx.ctx: return
        
        # 5. Instanced Visualization (Optional) - MOVED TO END
        # if hasattr(self, 'latest_joint_data') and self.latest_joint_data is not None:
        #      self.draw_instanced(self.latest_joint_data)
        
        if not getattr(self, 'limb_top_fix_vao', None) or self._gl_dirty:
             if self._gl_dirty:
                 print("MGLBodyNode: re-initializing GL for updated limb lengths")
             else:
                 print("MGLBodyNode: initializing GL for top cap normal fix")
             self.initialize_gl()
             self._gl_dirty = False

        
        # 1. Compute Global Matrices for all joints
        self.global_matrices = {}
        self.joint_tip_matrices = {}  # Actual joint positions for sphere rendering
        
        if len(self.ctx.model_matrix_stack) > 0:
            world_mat = self.ctx.model_matrix_stack[-1]
        else:
            world_mat = np.identity(4, dtype=np.float32)

        s = self.scale_input()
        scale_mat = np.diag([s, s, s, 1.0]).astype(np.float32)
        
        root_mat = world_mat @ scale_mat
        
        self.traverse_matrices(t_PelvisAnchor, root_mat)
        

        # 2. Upload Bone Matrices
        bones_bytes = bytearray()
        MAX_BONES = 50
        
        for i in range(MAX_BONES):
            if i in self.global_matrices:
                m = self.global_matrices[i]
            else:
                m = np.identity(4, dtype='f4')
            bones_bytes.extend(m.T.astype('f4').flatten().tobytes())

        # Determine Render Mode
        mode = self.display_mode_input()
        use_unlit = (mode == 'lines')
        
        # Select Program
        target_prog = self.unlit_prog if use_unlit and hasattr(self, 'unlit_prog') else self.prog
            
        if 'bones' in target_prog:
            target_prog['bones'].write(bones_bytes)
            
        # 3. Uniforms
        vp = self.ctx.view_matrix @ self.ctx.projection_matrix
        if 'VP' in target_prog:
            target_prog['VP'].write(vp.flatten().tobytes())
        
        # Lit-only uniforms
        if not use_unlit:
            try:
                inv_view = np.linalg.inv(self.ctx.view_matrix)
                cam_pos = inv_view[3, :3]
                if 'view_pos' in target_prog:
                    target_prog['view_pos'].value = tuple(cam_pos)
            except:
                pass
            
            self.ctx.update_lights(target_prog)
            self.ctx.update_material(target_prog)

        if 'color' in target_prog:
            c = self.ctx.current_color
            target_prog['color'].value = tuple(c)
        
        # 4. Draw
        if use_unlit:
            if hasattr(self, 'skeleton_unlit_vao'):
                self.ctx.ctx.disable(moderngl.CULL_FACE)
                self.skeleton_unlit_vao.render(mode=moderngl.LINES)
                self.ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            if getattr(self, 'limb_top_fix_vao', None):
                self.ctx.ctx.disable(moderngl.CULL_FACE)
                self.limb_top_fix_vao.render(mode=moderngl.TRIANGLES)
                self.ctx.ctx.enable(moderngl.CULL_FACE)

        # 5. Joint Callbacks (Optional)
        if self.enable_callbacks_input():
            # Iterate all joints and trigger callback with their global matrix
            for i, joint in enumerate(self.body.joints):
                # Only exclude PelvisAnchor (hips/shoulder blades have distinct positions)
                if i in [t_PelvisAnchor]:
                    continue
                if joint is None:
                    continue
                
                # Use joint_tip_matrices (actual position) if available, else global_matrices
                tip_mat = self.joint_tip_matrices.get(i) if hasattr(self, 'joint_tip_matrices') else None
                mat = tip_mat if tip_mat is not None else self.global_matrices.get(i)
                if mat is not None:
                    if not hasattr(joint, 'do_draw') or joint.do_draw:
                        self.ctx.model_matrix_stack.append(mat)
                        self.joint_id_out.send(i)
                        
                        # Pass Joint Data Payload if available
                        if hasattr(self, 'latest_joint_data') and self.latest_joint_data is not None:
                             # Data arrives in body t_ index order, use directly
                             if i < len(self.latest_joint_data):
                                  self.joint_data_out.send(self.latest_joint_data[i])
                        
                        self.joint_callback.send('draw')
                        self.ctx.model_matrix_stack.pop()
        
        # 6. Instanced Visualization (Final Step)
        if hasattr(self, 'latest_joint_data') and self.latest_joint_data is not None:
             if self.draw_spheres_input():
                 self.draw_instanced(self.latest_joint_data)

    def traverse_matrices(self, joint_index, current_mat):
        if joint_index == t_PelvisAnchor:
             joint = self.body.joints[joint_index]
             trans = np.identity(4, dtype='f4')
             t_vec = joint.bone_translation * 0.01 
             trans[:3, 3] = t_vec
             
             anim_rot = np.identity(4, dtype='f4')
             if self.body.joint_matrices is not None and -1 < joint_index < self.body.joint_matrices.shape[1]: 
                  anim_rot = self.body.joint_matrices[self.body.current_body, joint_index].reshape((4,4)).T
                  if np.abs(anim_rot).sum() < 0.1: anim_rot = np.identity(4, dtype='f4')

             current_mat = current_mat @ trans @ anim_rot
        
        # self.global_matrices[joint_index] = current_mat # DO NOT OVERWRITE! This destroys limb alignment
        
        children = self.body.joints[joint_index].immed_children
        for child_idx in children:
             self.process_child_matrices(child_idx, current_mat)

    def process_child_matrices(self, joint_index, parent_mat):
        joint = self.body.joints[joint_index]
        if joint is None: return

        rest_rot = np.identity(4, dtype='f4')
        trans_bone = np.identity(4, dtype='f4')
        trans_bone[:3, 3] = joint.bone_translation 
        
        anim_rot = np.identity(4, dtype='f4')
        if self.body.joint_matrices is not None and -1 < joint_index < self.body.joint_matrices.shape[1]: 
             anim_matrix = self.body.joint_matrices[self.body.current_body, joint_index].reshape((4,4)).T
             if np.abs(anim_matrix).sum() > 0.1: 
                  anim_rot = anim_matrix
        
        # Calculate Limb Matrix
        b_vec = joint.bone_translation
        b_len = np.linalg.norm(b_vec)
        
        target_axis = b_vec / (b_len + 1e-6)
        up = np.array([0.0, 0.0, 1.0], dtype='f4')
        
        rot_align = np.identity(4, dtype='f4')
        dot_p = np.dot(up, target_axis)
        
        if np.abs(dot_p) < 0.99:
             axis = np.cross(up, target_axis)
             angle = math.acos(dot_p)
             rot_align = rotation_matrix(math.degrees(angle), axis).T
        elif dot_p < -0.99:
             rot_align = rotation_matrix(180, [1, 0, 0]).T
        
        limb_model = parent_mat @ rest_rot @ rot_align
        self.global_matrices[joint_index] = limb_model
        
        # Store actual joint TIP position (for sphere rendering)
        # limb_model position = parent joint position. Actual joint is at parent + bone_translation.
        joint_tip = parent_mat @ rest_rot @ trans_bone
        self.joint_tip_matrices[joint_index] = joint_tip
        
        child_global = parent_mat @ rest_rot @ trans_bone @ anim_rot
        self.traverse_matrices(joint_index, child_global)

