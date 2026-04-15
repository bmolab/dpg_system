import numpy as np
import time
import os
import moderngl
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
from dpg_system.moderngl_nodes import MGLNode

try:
    import torch
except ImportError:
    torch = None


def register_mgl_shader_nodes():
    Node.app.register_node('mgl_shader', MGLShaderNode.factory)


class MGLShaderNode(MGLNode):
    """ShaderToy-compatible fragment shader node for the MGL scene graph.

    Supports two modes:
      - generator:    Renders a fullscreen quad with a custom fragment shader.
                      iChannel0-3 come from user inputs.
      - post_process: Captures the current framebuffer content before drawing,
                      binds it as iChannel0, then renders the fullscreen quad.
                      iChannel1-3 come from user inputs.

    Shader source can be loaded from a .glsl file (with auto-reload on change)
    or pasted as text into the shader_text input.

    ShaderToy compatibility:
      If the shader contains ``mainImage``, it is automatically wrapped with
      the standard ShaderToy uniforms (iTime, iResolution, iFrame, iMouse,
      iChannel0-3, iTimeDelta).  Otherwise the source is used as-is
      (but still gets the uniform declarations prepended if no #version is found).
    """

    # ------------------------------------------------------------------ #
    #  Fullscreen quad vertex shader (always the same)                    #
    # ------------------------------------------------------------------ #
    _VERT_SRC = '''\
#version 330
in vec2 in_position;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_position * 0.5 + 0.5;
}
'''

    # ------------------------------------------------------------------ #
    #  ShaderToy compatibility wrapper                                    #
    # ------------------------------------------------------------------ #
    _SHADERTOY_HEADER = '''\
#version 330
precision highp float;

uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

out vec4 _st_fragColor;
'''

    _SHADERTOY_FOOTER = '''\

void main() {
    mainImage(_st_fragColor, gl_FragCoord.xy);
}
'''

    # Uniform block for raw (non-ShaderToy) shaders that lack a #version line
    _RAW_UNIFORMS = '''\
#version 330
precision highp float;

uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
'''

    # ------------------------------------------------------------------ #
    #  Factory / init                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def factory(name, data, args=None):
        return MGLShaderNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # GL resources
        self._prog = None
        self._vao = None
        self._quad_vbo = None
        self._error = None
        self._needs_compile = False  # deferred compilation flag

        # Backbuffer copy for post-process mode
        self._backbuffer_tex = None
        self._backbuffer_fbo = None
        self._bb_w = 0
        self._bb_h = 0

        # Internal textures for numpy/torch channel inputs
        self._channel_textures = [None, None, None, None]
        self._channel_sizes = [(0, 0, 0)] * 4

        # Default 1×1 black texture (lazily created)
        self._default_tex = None

        # Timing
        self._start_time = time.time()
        self._last_time = self._start_time
        self._frame_count = 0

        # File watching
        self._loaded_path = None
        self._loaded_mtime = 0
        self._last_source = None

    def initialize(self, args):
        super().initialize(args)

        default_file = ''
        if len(args) > 0:
            default_file = args[0]

        # Shader source inputs
        self.file_input = self.add_input('file', widget_type='text_input',
                                         default_value=default_file,
                                         callback=self._on_file_changed)
        self.file_input.widget.width = 220
        self.shader_text_input = self.add_input('shader_text',
                                                callback=self._on_text_changed)

        # Texture / data channel inputs
        self.channel_inputs = []
        for i in range(4):
            inp = self.add_input(f'iChannel{i}')
            self.channel_inputs.append(inp)

        # Overridable uniforms
        self.time_input = self.add_input('iTime')
        self.mouse_input = self.add_input('iMouse')

        # Mode
        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value='generator')
        self.mode_input.widget.combo_items = ['generator', 'post_process']

        # Controls
        self.reload_button = self.add_property('reload', widget_type='button',
                                               callback=self._force_reload)
        self.auto_reload_option = self.add_option('auto_reload',
                                                   widget_type='checkbox',
                                                   default_value=True)
        self.status_output = self.add_output('status')

    # ------------------------------------------------------------------ #
    #  Input callbacks — never compile here, just mark dirty             #
    # ------------------------------------------------------------------ #
    def _on_file_changed(self):
        """Called when the file text input changes. Load source, defer compile."""
        path = self._resolve_path(self.file_input())
        if path:
            self._read_file_source(path)

    def _on_text_changed(self):
        """Called when shader text is received via input. Defer compile."""
        text = self.shader_text_input()
        if text and isinstance(text, str) and len(text.strip()) > 0:
            self._last_source = text
            self._needs_compile = True

    def _force_reload(self):
        """Reload button callback."""
        path = self._resolve_path(self.file_input())
        if path:
            self._read_file_source(path, force=True)
        elif self._last_source:
            self._needs_compile = True

    def _resolve_path(self, path):
        """Resolve a file path, trying absolute then relative to app_path."""
        if not path or not isinstance(path, str) or not path.strip():
            return None
        path = path.strip()
        if os.path.isfile(path):
            return path
        if hasattr(self, 'app') and hasattr(self.app, 'app_path'):
            full = os.path.join(self.app.app_path, path)
            if os.path.isfile(full):
                return full
        return None

    # ------------------------------------------------------------------ #
    #  File loading & watching                                            #
    # ------------------------------------------------------------------ #
    def _read_file_source(self, path, force=False):
        """Read shader source from file and mark for compilation."""
        try:
            mtime = os.path.getmtime(path)
            if not force and path == self._loaded_path and mtime == self._loaded_mtime:
                return  # no change
            with open(path, 'r') as f:
                source = f.read()
            self._loaded_path = path
            self._loaded_mtime = mtime
            self._last_source = source
            self._needs_compile = True
        except Exception as e:
            self._error = f'File error: {e}'
            print(f'[mgl_shader] {self._error}')

    def _check_file_reload(self):
        """Called each frame when auto_reload is on."""
        if not self._loaded_path or not os.path.isfile(self._loaded_path):
            return
        try:
            mtime = os.path.getmtime(self._loaded_path)
            if mtime != self._loaded_mtime:
                self._read_file_source(self._loaded_path, force=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Shader compilation                                                 #
    # ------------------------------------------------------------------ #
    def _compile_shader(self, frag_src):
        """Compile fragment shader, wrapping in ShaderToy compat if needed."""
        if not frag_src or not frag_src.strip():
            return

        self._last_source = frag_src

        # Determine wrapping mode
        has_main_image = 'mainImage' in frag_src
        has_version = '#version' in frag_src

        if has_main_image:
            # ShaderToy-style shader
            full_frag = self._SHADERTOY_HEADER + frag_src + self._SHADERTOY_FOOTER
        elif not has_version:
            # Raw GLSL without #version — prepend uniforms
            full_frag = self._RAW_UNIFORMS + frag_src
        else:
            # Full GLSL with #version — use as-is
            full_frag = frag_src

        # Need a GL context to compile
        ctx_instance = MGLContext.get_instance()
        if ctx_instance is None or ctx_instance.ctx is None:
            self._error = 'No GL context available'
            return

        ctx = ctx_instance.ctx

        # Release old program and invalidate VAO
        if self._prog is not None:
            try:
                self._prog.release()
            except Exception:
                pass
            self._prog = None
        if self._vao is not None:
            try:
                self._vao.release()
            except Exception:
                pass
            self._vao = None

        try:
            self._prog = ctx.program(
                vertex_shader=self._VERT_SRC,
                fragment_shader=full_frag,
            )
            self._error = None
            # Reset timing on successful compile
            self._start_time = time.time()
            self._last_time = self._start_time
            self._frame_count = 0
            print(f'[mgl_shader] Shader compiled successfully')
            self.status_output.send('ok')
        except Exception as e:
            self._error = str(e)
            print(f'[mgl_shader] Compilation error:\n{self._error}')
            self.status_output.send(f'error: {self._error}')

    # ------------------------------------------------------------------ #
    #  GL resource management                                             #
    # ------------------------------------------------------------------ #
    def _ensure_quad(self, ctx):
        """Lazily create the fullscreen quad VAO."""
        if self._vao is not None:
            return
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self._quad_vbo = ctx.buffer(verts.tobytes())
        # VAO will be recreated per-program since programs may differ
        self._rebuild_vao(ctx)

    def _rebuild_vao(self, ctx):
        """Rebuild VAO for current program."""
        if self._prog is None or self._quad_vbo is None:
            return
        if self._vao is not None:
            try:
                self._vao.release()
            except Exception:
                pass
        self._vao = ctx.simple_vertex_array(self._prog, self._quad_vbo, 'in_position')

    def _ensure_default_texture(self, ctx):
        """Create a 1×1 black texture for unconnected channels."""
        if self._default_tex is None:
            self._default_tex = ctx.texture((1, 1), 4,
                                            data=b'\x00\x00\x00\xff')
            self._default_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _update_channel_texture(self, ctx, index, data):
        """Convert numpy/torch data to a moderngl texture for the given channel."""
        if data is None:
            return None

        # Already a moderngl texture
        if isinstance(data, moderngl.Texture):
            return data

        # Convert torch tensor
        if torch is not None and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if isinstance(data, list):
            data = np.array(data, dtype=np.uint8)

        if not isinstance(data, np.ndarray):
            return None

        # Normalize float data to uint8
        if data.dtype in (np.float32, np.float64):
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

        # Determine dimensions
        if data.ndim == 2:
            h, w = data.shape
            c = 1
        elif data.ndim == 3:
            h, w, c = data.shape
        else:
            return None

        if c not in (1, 3, 4):
            return None

        # Create or resize texture
        old_w, old_h, old_c = self._channel_sizes[index]
        if (self._channel_textures[index] is None or
                w != old_w or h != old_h or c != old_c):
            if self._channel_textures[index] is not None:
                try:
                    self._channel_textures[index].release()
                except Exception:
                    pass
            self._channel_textures[index] = ctx.texture((w, h), c)
            self._channel_textures[index].filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._channel_sizes[index] = (w, h, c)

        self._channel_textures[index].write(data.tobytes())
        return self._channel_textures[index]

    def _ensure_backbuffer(self, ctx, w, h):
        """Create or resize the backbuffer copy texture+FBO for post-process mode."""
        if self._bb_w != w or self._bb_h != h or self._backbuffer_tex is None:
            if self._backbuffer_tex is not None:
                try:
                    self._backbuffer_tex.release()
                except Exception:
                    pass
            if self._backbuffer_fbo is not None:
                try:
                    self._backbuffer_fbo.release()
                except Exception:
                    pass
            self._backbuffer_tex = ctx.texture((w, h), 4)
            self._backbuffer_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._backbuffer_fbo = ctx.framebuffer(color_attachments=[self._backbuffer_tex])
            self._bb_w = w
            self._bb_h = h

    # ------------------------------------------------------------------ #
    #  Save / load custom state                                           #
    # ------------------------------------------------------------------ #
    def save_custom(self, container):
        if self._loaded_path:
            container['shader_file'] = self._loaded_path
        if self._last_source and not self._loaded_path:
            container['shader_text'] = self._last_source

    def load_custom(self, container):
        if 'shader_file' in container:
            path = container['shader_file']
            self.file_input.set(path)
            # Compilation deferred to first draw (need GL context)
        elif 'shader_text' in container:
            # Text will be compiled on first draw
            self._last_source = container['shader_text']

    # ------------------------------------------------------------------ #
    #  Custom create — trigger initial file load if set                   #
    # ------------------------------------------------------------------ #
    def custom_create(self, from_file):
        path = self.file_input()
        if path and isinstance(path, str) and len(path.strip()) > 0:
            self._loaded_path = None  # force load on first draw

    # ------------------------------------------------------------------ #
    #  Draw                                                               #
    # ------------------------------------------------------------------ #
    def draw(self):
        if self.ctx is None:
            return
        inner_ctx = self.ctx.ctx

        # Auto-reload file if enabled
        if self.auto_reload_option():
            self._check_file_reload()

        # Compile if dirty or first draw
        if self._needs_compile and self._last_source:
            self._compile_shader(self._last_source)
            self._needs_compile = False
        elif self._prog is None and self._error is None:
            # First draw: try to load file
            path = self._resolve_path(self.file_input())
            if path:
                self._read_file_source(path)
                if self._needs_compile and self._last_source:
                    self._compile_shader(self._last_source)
                    self._needs_compile = False
            elif self._last_source:
                self._compile_shader(self._last_source)

        # Nothing to render without a compiled program
        if self._prog is None:
            return

        # Ensure quad geometry
        self._ensure_quad(inner_ctx)
        if self._vao is None:
            self._rebuild_vao(inner_ctx)
        if self._vao is None:
            return

        # Ensure default texture
        self._ensure_default_texture(inner_ctx)

        # ---- Post-process: capture current framebuffer ---- #
        mode = self.mode_input()
        is_post = (mode == 'post_process')

        if is_post and self.ctx.active_target is not None:
            rt = self.ctx.active_target
            w, h = rt.width, rt.height
            self._ensure_backbuffer(inner_ctx, w, h)
            # Resolve MSAA if needed
            src_fbo = rt.fbo
            if rt.samples > 0 and rt.msaa_fbo:
                inner_ctx.copy_framebuffer(rt.fbo, rt.msaa_fbo)
            # Copy resolved FBO to our backbuffer
            inner_ctx.copy_framebuffer(self._backbuffer_fbo, rt.fbo)
            # Re-activate the original render target for our quad draw
            rt.use()

        # ---- Update timing ---- #
        now = time.time()
        time_override = self.time_input()
        if time_override is not None:
            try:
                current_time = float(any_to_float(time_override))
            except Exception:
                current_time = now - self._start_time
        else:
            current_time = now - self._start_time

        time_delta = now - self._last_time
        self._last_time = now
        self._frame_count += 1

        # ---- Get viewport resolution ---- #
        if self.ctx.active_target is not None:
            res_w = self.ctx.active_target.width
            res_h = self.ctx.active_target.height
        else:
            res_w, res_h = 640, 480

        # ---- Set uniforms ---- #
        prog = self._prog

        if 'iResolution' in prog:
            prog['iResolution'].value = (float(res_w), float(res_h), 1.0)
        if 'iTime' in prog:
            prog['iTime'].value = current_time
        if 'iTimeDelta' in prog:
            prog['iTimeDelta'].value = time_delta
        if 'iFrame' in prog:
            prog['iFrame'].value = self._frame_count

        # Mouse
        mouse_val = self.mouse_input()
        if mouse_val is not None:
            try:
                mv = any_to_array(mouse_val).flatten()
                if len(mv) >= 4:
                    mouse_vec = tuple(float(x) for x in mv[:4])
                elif len(mv) >= 2:
                    mouse_vec = (float(mv[0]), float(mv[1]), 0.0, 0.0)
                else:
                    mouse_vec = (0.0, 0.0, 0.0, 0.0)
            except Exception:
                mouse_vec = (0.0, 0.0, 0.0, 0.0)
        else:
            mouse_vec = (0.0, 0.0, 0.0, 0.0)
        if 'iMouse' in prog:
            prog['iMouse'].value = mouse_vec

        # ---- Bind texture channels ---- #
        for i in range(4):
            channel_name = f'iChannel{i}'
            if channel_name not in prog:
                continue

            tex = None

            if is_post and i == 0:
                # Post-process mode: iChannel0 = captured backbuffer
                tex = self._backbuffer_tex
            else:
                # Resolve input
                raw = self.channel_inputs[i]()
                if raw is not None:
                    if isinstance(raw, moderngl.Texture):
                        tex = raw
                    else:
                        tex = self._update_channel_texture(inner_ctx, i, raw)

            if tex is None:
                tex = self._default_tex

            tex.use(location=i)
            prog[channel_name].value = i

        # ---- Render fullscreen quad ---- #
        inner_ctx.disable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLE_STRIP)
        inner_ctx.enable(moderngl.DEPTH_TEST)

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #
    def custom_cleanup(self):
        for tex in self._channel_textures:
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
        if self._default_tex is not None:
            try:
                self._default_tex.release()
            except Exception:
                pass
        if self._backbuffer_tex is not None:
            try:
                self._backbuffer_tex.release()
            except Exception:
                pass
        if self._backbuffer_fbo is not None:
            try:
                self._backbuffer_fbo.release()
            except Exception:
                pass
        if self._prog is not None:
            try:
                self._prog.release()
            except Exception:
                pass
        if self._vao is not None:
            try:
                self._vao.release()
            except Exception:
                pass
        if self._quad_vbo is not None:
            try:
                self._quad_vbo.release()
            except Exception:
                pass
