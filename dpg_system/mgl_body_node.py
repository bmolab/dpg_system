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
        self.color_input = self.add_input('color', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0])
        self.instanced_display_mode_input = self.add_input('instanced_mode', widget_type='combo', default_value='solid')
        self.instanced_display_mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        
        self.joint_callback = self.add_output('joint_callback')
        self.joint_data_out = self.add_output('joint_data')
        self.joint_id_out = self.add_output('joint_id')
        
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
        # 1. Sphere Geometry (Unit Sphere)
        # Reusing similar logic to MGLSphereNode but simplified
        sphere_res = 16
        # Parametric sphere generation... or just reuse cube for testing?
        # Let's generate a proper sphere
        
        verts = []
        norms = []
        indices = []
        
        import math
        
        for i in range(sphere_res + 1):
            lat = math.pi * i / sphere_res
            for j in range(sphere_res + 1):
                lon = 2 * math.pi * j / sphere_res
                x = math.sin(lat) * math.cos(lon)
                y = math.sin(lat) * math.sin(lon)
                z = math.cos(lat)
                verts.extend([x, y, z])
                norms.extend([x, y, z])
                
        for i in range(sphere_res):
            for j in range(sphere_res):
                p1 = i * (sphere_res + 1) + j
                p2 = p1 + 1
                p3 = (i + 1) * (sphere_res + 1) + j
                p4 = p3 + 1
                indices.extend([p1, p2, p3, p2, p4, p3])
        
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
        
        # DEBUG
        # print(f"MGLBodyNode: draw_instanced entered. Data len: {len(joint_data)}/{num_joints}. Global Mats: {len(self.global_matrices)}")
        
        # Build full_data array mapping bone indices to their correct data indices.
        # For bone indices 0-19: data index == bone index (active joints, t_ order)
        # For bone indices >= 20: remap to match MoCapGLBody.joint_callback mapping
        #   bone 20 (TopOfHead)       -> skip (no data)
        #   bone 21 (LeftBallOfFoot)   -> data index 20
        #   bone 22 (LeftToeTip)       -> data index 24
        #   bone 23 (RightBallOfFoot)  -> data index 21
        #   bone 24 (RightToeTip)      -> data index 25
        #   bone 25 (LeftKnuckle)      -> data index 22
        #   bone 26 (LeftFingerTip)    -> data index 26
        #   bone 27 (RightKnuckle)     -> data index 23
        #   bone 28 (RightFingerTip)   -> data index 27
        #   bone >= 29                 -> skip
        bone_to_data = {
            21: 20, 22: 24, 23: 21, 24: 25,
            25: 22, 26: 26, 27: 23, 28: 27
        }
        
        full_data = np.zeros(num_joints, dtype=np.float32)
        for i in range(min(num_joints, 20)):
            if i < len(joint_data):
                full_data[i] = joint_data[i]
        for bone_idx, data_idx in bone_to_data.items():
            if bone_idx < num_joints and data_idx < len(joint_data):
                full_data[bone_idx] = joint_data[data_idx]

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
        
        # Disable culling
        self.ctx.ctx.disable(moderngl.CULL_FACE) 
        # Enable Standard Alpha Blending
        self.ctx.ctx.enable(moderngl.BLEND)
        self.ctx.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        count = len(scaled_data)
        
        if mode == 'wireframe':
            self.ctx.ctx.wireframe = True
            self.instanced_vao.render(instances=count)
            self.ctx.ctx.wireframe = False
        elif mode == 'points':
            self.instanced_vao.render(instances=count, mode=moderngl.POINTS)
        else:
            self.instanced_vao.render(instances=count)
            
        self.ctx.ctx.disable(moderngl.BLEND)
        self.ctx.ctx.enable(moderngl.CULL_FACE)
        
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

    def execute(self):
        if self.pose_input.fresh_input:
            data = self.pose_input()
            if data is not None:
                self.body.update_quats(data)

        if self.joint_data_input.fresh_input:
            self.latest_joint_data = self.joint_data_input()
    
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
        
        if not getattr(self, 'limb_top_fix_vao', None): 
             print("MGLBodyNode: initializing GL for top cap normal fix")
             self.initialize_gl()

        
        # 1. Compute Global Matrices for all joints
        self.global_matrices = {}
        self.joint_tip_matrices = {}  # Actual joint positions for sphere rendering
        
        if len(self.ctx.model_matrix_stack) > 0:
            world_mat = self.ctx.model_matrix_stack[-1]
        else:
            world_mat = np.identity(4, dtype=np.float32)

        s = self.scale_input()
        scale_mat = np.diag([s, s, s, 1.0])
        rot_flip = rotation_matrix(180, [0, 0, 1]).T
        
        root_mat = world_mat @ rot_flip @ scale_mat
        
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
                             # Remap non-active joints (>= 20) to compacted data indices
                             bone_to_data = {
                                 21: 20, 22: 24, 23: 21, 24: 25,
                                 25: 22, 26: 26, 27: 23, 28: 27
                             }
                             data_idx = bone_to_data.get(i, i)  # Use remapped index or direct index
                             if data_idx < len(self.latest_joint_data) and (i < 20 or i in bone_to_data):
                                  self.joint_data_out.send(self.latest_joint_data[data_idx])
                        
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

