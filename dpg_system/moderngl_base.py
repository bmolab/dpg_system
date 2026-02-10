
import moderngl
import numpy as np
from dpg_system.matrix_utils import *

class MGLRenderTarget:
    def __init__(self, ctx, width, height, samples=4):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.samples = samples
        self.fbo = None
        self.texture = None
        self.depth_texture = None
        self.msaa_fbo = None
        self.msaa_texture = None
        self.msaa_depth_texture = None
        self.create()

    def release(self):
        if self.texture: self.texture.release()
        if self.depth_texture: self.depth_texture.release()
        if self.fbo: self.fbo.release()
        if self.msaa_texture: self.msaa_texture.release()
        if self.msaa_depth_texture: self.msaa_depth_texture.release()
        if self.msaa_fbo: self.msaa_fbo.release()
        
        self.texture = None
        self.depth_texture = None
        self.fbo = None
        self.msaa_texture = None
        self.msaa_depth_texture = None
        self.msaa_fbo = None

    def create(self):
        # 1. Standard Texture & FBO (Resolve Target)
        self.texture = self.ctx.texture((self.width, self.height), 4)
        self.depth_texture = self.ctx.depth_texture((self.width, self.height))
        self.fbo = self.ctx.framebuffer(self.texture, self.depth_texture)

        # 2. MSAA Target
        if self.samples > 0:
            valid_options = [16, 8, 6, 4, 2]
            candidates = [s for s in valid_options if s <= self.samples]
            if self.samples not in valid_options:
                candidates.insert(0, self.samples)
            
            success = False
            for s in candidates:
                try:
                    self.msaa_texture = self.ctx.texture((self.width, self.height), 4, samples=s)
                    self.msaa_depth_texture = self.ctx.depth_texture((self.width, self.height), samples=s)
                    self.msaa_fbo = self.ctx.framebuffer(self.msaa_texture, self.msaa_depth_texture)
                    self.samples = s
                    success = True
                    break
                except Exception as e:
                    # Release partials
                    if self.msaa_texture: self.msaa_texture.release()
                    if self.msaa_depth_texture: self.msaa_depth_texture.release()
                    if self.msaa_fbo: self.msaa_fbo.release()
            
            if not success:
                print("MGLRenderTarget: MSAA failed, falling back to 0 samples")
                self.samples = 0
                self.msaa_fbo = None
        else:
            self.msaa_fbo = None

    def use(self):
        target = self.msaa_fbo if (self.samples > 0 and self.msaa_fbo) else self.fbo
        if target:
            target.use()

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        target = self.msaa_fbo if (self.samples > 0 and self.msaa_fbo) else self.fbo
        if target:
            target.clear(r, g, b, a)

    def get_pixel_data(self):
        # Resolve MSAA if needed
        if self.samples > 0 and self.msaa_fbo:
            self.ctx.copy_framebuffer(self.fbo, self.msaa_fbo)
        if self.fbo:
            return self.fbo.read(components=4)
        return None


class MGLContext:
    _instance = None

    @staticmethod
    def get_instance():
        if MGLContext._instance is None:
            MGLContext._instance = MGLContext()
        return MGLContext._instance

    def __init__(self):
        if MGLContext._instance is not None:
            raise Exception("This class is a singleton!")
        
        try:
            self.ctx = moderngl.create_context(standalone=True)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        except Exception as e:
            print(f"Failed to create standalone context: {e}")
            return

        # Default Render State
        init_width = 1280
        init_height = 720
        init_samples = 4
        
        # Render Target Management
        self.default_target = MGLRenderTarget(self.ctx, init_width, init_height, init_samples)
        self.active_target = self.default_target
        
        self.current_color = (1.0, 1.0, 1.0, 1.0)
        self.lights = [] 
        self.current_material = {
            'ambient': [0.1, 0.1, 0.1],
            'diffuse': [1.0, 1.0, 1.0],
            'specular': [0.5, 0.5, 0.5],
            'shininess': 32.0
        }
        
        self.model_matrix_stack = [np.identity(4, dtype=np.float32)]
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix = np.identity(4, dtype=np.float32)
        
        # Default smooth shader
        self.default_shader = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 M;
                uniform mat4 V;
                uniform mat4 P;
                uniform float point_size;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                out vec3 v_normal;
                out vec3 v_pos;
                out vec2 v_texcoord;
                void main() {
                    vec4 world_pos = M * vec4(in_position, 1.0);
                    gl_Position = P * V * world_pos;
                    if (point_size > 1.5) {
                        gl_PointSize = point_size + 2.0;
                    } else {
                        gl_PointSize = max(1.0, point_size);
                    }
                    mat3 normal_matrix = transpose(inverse(mat3(M)));
                    v_normal = normal_matrix * in_normal;
                    v_pos = world_pos.xyz;
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                #define MAX_LIGHTS 8
                
                uniform vec4 color;
                uniform vec3 view_pos;
                uniform float point_size;
                uniform bool point_culling;
                uniform bool round_points;
                
                // Texturing
                uniform sampler2D diffuse_map;
                uniform bool has_texture;
                in vec2 v_texcoord;

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
                uniform vec3 material_ambient;
                uniform vec3 material_diffuse;
                uniform vec3 material_specular;
                uniform float material_shininess;
                
                in vec3 v_normal;
                in vec3 v_pos;
                out vec4 f_color;
                
                void main() {
                    // Manual Culling for Points
                    if (point_culling) {
                        vec3 n = normalize(v_normal);
                        vec3 v = normalize(view_pos - v_pos);
                        if (dot(n, v) < 0.0) discard;
                    }
                    
                    vec4 base_color = color;
                    float alpha = base_color.a;

                    // Apply Texture
                    if (has_texture) {
                        base_color *= texture(diffuse_map, v_texcoord);
                    }
                    
                    // Round Points
                    if (round_points && point_size > 1.5) {
                        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                        float dist = length(cxy);
                        float limit = point_size / (point_size + 2.0);
                        if (dist > limit) discard;
                    }
                    
                    vec3 norm = normalize(v_normal);
                    vec3 viewDir = normalize(view_pos - v_pos);
                    vec3 final_rgb = vec3(0.0);
                    
                    // Default Ambient if no lights
                    if (num_lights == 0) {
                        final_rgb = vec3(0.2) * base_color.rgb; // Simple fallback
                    } else {
                        vec3 total_ambient = vec3(0.0);
                        vec3 total_diffuse = vec3(0.0);
                        vec3 total_specular = vec3(0.0);
                        
                        for (int i = 0; i < MAX_LIGHTS; i++) {
                            if (i >= num_lights) break;
                            
                            // Ambient
                            total_ambient += lights[i].ambient * material_ambient;
                            
                            // Diffuse
                            vec3 lightDir = normalize(lights[i].pos - v_pos);
                            float diff = max(dot(norm, lightDir), 0.0); // One-sided
                            total_diffuse += diff * lights[i].diffuse * material_diffuse * lights[i].intensity;
                            
                            // Specular (Blinn-Phong)
                            vec3 halfwayDir = normalize(lightDir + viewDir);
                            float spec = pow(max(dot(norm, halfwayDir), 0.0), material_shininess);
                            total_specular += spec * lights[i].specular * material_specular * lights[i].intensity; 
                        }
                        
                        final_rgb = (total_ambient + total_diffuse + total_specular) * base_color.rgb;
                    }
                    
                    f_color = vec4(final_rgb, alpha);
                }
            '''
        )

    # Properties for backward compatibility (if needed) and convenience
    @property
    def fbo(self): return self.active_target.fbo
    @property
    def texture(self): return self.active_target.texture
    @property
    def width(self): return self.active_target.width
    @width.setter
    def width(self, value): pass # Read-only strictly speaking from external view
    @property
    def height(self): return self.active_target.height
    @height.setter
    def height(self, value): pass
    @property
    def samples(self): return self.active_target.samples
    @samples.setter
    def samples(self, value): pass

    def create_render_target(self, width, height, samples=4):
        return MGLRenderTarget(self.ctx, width, height, samples)

    def use_render_target(self, target):
        self.active_target = target
        target.use()

    # Legacy update_framebuffer for simple singleton usage
    def update_framebuffer(self, width, height, samples=None):
        if samples is None: samples = self.active_target.samples
        # Check if active target matches
        t = self.active_target
        if t.width != width or t.height != height or t.samples != samples:
            # Recreate active target (if it's the default one)
            if t == self.default_target:
                t.release()
                self.default_target = MGLRenderTarget(self.ctx, width, height, samples)
                self.active_target = self.default_target
            else:
                # If using a custom target, we don't automatically update it here usually?
                # Or we assume single-context mode.
                pass

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.active_target.clear(r, g, b, a)
        # Reset stacks
        self.model_matrix_stack = [np.identity(4, dtype=np.float32)]
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

    def get_pixel_data(self):
        return self.active_target.get_pixel_data()

    def update_lights(self, prog):
        # Set num_lights
        if 'num_lights' in prog:
            prog['num_lights'].value = len(self.lights)
        
        for i, light in enumerate(self.lights):
            if i >= 8: break # Shield against overflow
            
            # Position
            name = f'lights[{i}].pos'
            if name in prog:
                prog[name].value = tuple(light['pos'])
            
            # Ambient
            name = f'lights[{i}].ambient'
            if name in prog:
                prog[name].value = tuple(light['ambient'])
                
            # Diffuse
            name = f'lights[{i}].diffuse'
            if name in prog:
                prog[name].value = tuple(light['diffuse'])
                
            # Specular
            name = f'lights[{i}].specular'
            if name in prog:
                prog[name].value = tuple(light['specular'])
                
            # Intensity
            name = f'lights[{i}].intensity'
            if name in prog:
                prog[name].value = light['intensity']

    def update_material(self, prog):
        if 'material_ambient' in prog:
            prog['material_ambient'].value = tuple(self.current_material['ambient'])
        if 'material_diffuse' in prog:
            prog['material_diffuse'].value = tuple(self.current_material['diffuse'])
        if 'material_specular' in prog:
            prog['material_specular'].value = tuple(self.current_material['specular'])
        if 'material_shininess' in prog:
            prog['material_shininess'].value = self.current_material['shininess']

    # Matrix Stack Operations
    def push_matrix(self):
        if self.model_matrix_stack:
            self.model_matrix_stack.append(self.model_matrix_stack[-1].copy())
        else:
            self.model_matrix_stack.append(np.identity(4, dtype=np.float32))

    def pop_matrix(self):
        if len(self.model_matrix_stack) > 1:
            self.model_matrix_stack.pop()
    
    def multiply_matrix(self, matrix):
        if self.model_matrix_stack:
            # Note: matrix multiplication order usually: current * new
            # But due to row-major (numpy) vs column-major (GL) implicit transpose,
            # we need pre-multiplication in numpy to achieve post-multiplication in GL
            # M_gl_new = M_gl_old @ T_gl  => M_numpy_new = T_numpy @ M_numpy_old
            self.model_matrix_stack[-1] = np.dot(matrix, self.model_matrix_stack[-1])
            
    def get_model_matrix(self):
        if self.model_matrix_stack:
            return self.model_matrix_stack[-1]
        return np.identity(4, dtype=np.float32)

    def set_view_matrix(self, matrix):
        self.view_matrix = matrix

    def set_projection_matrix(self, matrix):
        self.projection_matrix = matrix

