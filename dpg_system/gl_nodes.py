import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import freetype
from dpg_system.open_gl_base import *
from dpg_system.glfw_base import *
from dpg_system.colormaps import _viridis_data, _magma_data, _plasma_data, _inferno_data, make_heatmap, make_coldmap
# can create command parsers for various types of GLObjects?


def register_gl_nodes():
    Node.app.register_node('gl_context', GLContextNode.factory)
    Node.app.register_node('gl_sphere', GLSphereNode.factory)
    Node.app.register_node('gl_cylinder', GLCylinderNode.factory)
    Node.app.register_node('gl_disk', GLDiskNode.factory)
    Node.app.register_node('gl_partial_disk', GLPartialDiskNode.factory)
    Node.app.register_node('gl_translate', GLTransformNode.factory)
    Node.app.register_node('gl_rotate', GLTransformNode.factory)
    Node.app.register_node('gl_scale', GLTransformNode.factory)
    Node.app.register_node('gl_material', GLMaterialNode.factory)
    Node.app.register_node('gl_align', GLAlignNode.factory)
    Node.app.register_node('gl_quaternion_rotate', GLQuaternionRotateNode.factory)
    Node.app.register_node('gl_text', GLTextNode.factory)
    Node.app.register_node('gl_korean_text', GLKoreanTextNode.factory)

    Node.app.register_node('gl_billboard', GLBillboard.factory)
    Node.app.register_node('gl_rotation_disk', GLXYZDiskNode.factory)
    Node.app.register_node('gl_button_grid', GLButtonGridNode.factory)

    Node.app.register_node('gl_line_array', GLNumpyLines.factory)
    Node.app.register_node('gl_color', GLColorNode.factory)
    Node.app.register_node('gl_enable', GLEnableNode.factory)
    Node.app.register_node('gl_axis_angle_rotate', GLAxisAngleRotateNode.factory)


class GLCommandParser:
    def __init__(self):
        self.dict = {}

    def perform(self, command, object, args):
        if command in self.dict:
            self.dict[command](object, args)


class GLContextCommandParser(GLCommandParser):
    def __init__(self):
        super().__init__()
        self.dict['ortho'] = self.set_ortho
        self.dict['frustum'] = self.set_frustum
        self.dict['perspective'] = self.set_perspective
        self.dict['size'] = self.set_size
        self.dict['position'] = self.set_position
        self.dict['clear_color'] = self.set_clear_color

    def set_clear_color(self, me, args):
        alpha = 1.0
        if args is not None and len(args) > 2:
            red = any_to_float(args[0])
            green = any_to_float(args[1])
            blue = any_to_float(args[2])
            if len(args) > 3:
                alpha = any_to_float(args[3])
            me.context.clear_color = [red, green, blue, alpha]

    def set_size(self, me, args):
        if args is not None and len(args) > 0:
            if len(args) > 1:
                width = any_to_int(args[0])
                height = any_to_int(args[1])
                glfw.set_window_size(me.context.window, width, height)

    def set_position(self, me, args):
        if args is not None and len(args) > 0:
            if len(args) > 1:
                x = any_to_int(args[0])
                y = any_to_int(args[1])
                glfw.set_window_pos(me.context.window, x, y)

    def set_frustum(self, me, args):  # context must be established
        hold_context = glfw.get_current_context()
        if args is not None and len(args) > 0:
            glfw.make_context_current(me.context.window)
            near = .1
            far = 1000
            focal_length = 2.0
            if len(args) > 0:
                focal_length = any_to_float(args[0])
            if len(args) > 1:
                near = any_to_float(args[1])
            if len(args) > 2:
                far = any_to_float(args[2])
            width = me.context.width
            height = me.context.height
            height_over_width = float(height) / float(width)
            f = near / focal_length
            h = height_over_width * f
            current_matrix_mode = gl.glGetInteger(gl.GL_MATRIX_MODE)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glFrustum(-f, f, -h, h, near, far)
            gl.glMatrixMode(current_matrix_mode)
        glfw.make_context_current(hold_context)

    def set_ortho(self, me, args):  # context must be established
        hold_context = glfw.get_current_context()
        glfw.make_context_current(me.context.window)
        if args is not None and len(args) > 3:
            near = .1
            far = 1000
            left = any_to_float(args[0])
            right = any_to_float(args[1])
            bottom = any_to_float(args[2])
            top = any_to_float(args[3])
            dest_rect = [left, right, bottom, top]
            if len(args) > 4:
                near = any_to_float(args[4])
            if len(args) > 5:
                far = any_to_float(args[5])
            width = dest_rect[2] - dest_rect[0]
            height = dest_rect[3] - dest_rect[1]
            current_matrix_mode = gl.glGetInteger(gl.GL_MATRIX_MODE)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(-width / 2.0, width / 2.0, height / 2.0, -height / 2.0, near, far)
            gl.glMatrixMode(current_matrix_mode)
        glfw.make_context_current(hold_context)

    def set_perspective(self, me, args):  # context must be established
        hold_context = glfw.get_current_context()
        glfw.make_context_current(me.context.window)
        aspect = me.context.width / me.context.height
        if args is not None and len(args) > 0:
            fov = 50.0
            near = 0.1
            far = 1000
            if len(args) > 0:
                fov = any_to_float(args[0])
            if len(args) > 1:
                near = any_to_float(args[1])
            if len(args) > 2:
                far = any_to_float(args[2])
            fov_radians = fov / 180. * math.pi
            cotan = 1.0 / math.tan(fov_radians / 2.0)
            current_matrix_mode = gl.glGetInteger(gl.GL_MATRIX_MODE)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            m = np.array([cotan / aspect, 0.0, 0.0, 0.0, 0.0, cotan, 0.0, 0.0, 0.0, 0.0, (far + near) / (near - far), -1.0, 0.0, 0.0, (2.0 * far * near) / (near - far), 0.0])
            m = m.reshape((4, 4))
            gl.glMultMatrixd(m)
            gl.glMatrixMode(current_matrix_mode)
        glfw.make_context_current(hold_context)


class GLContextNode(Node):
    context_list = []
    pending_contexts = []
    pending_deletes = []

    @staticmethod
    def factory(name, data, args=None):
        node = GLContextNode(name, data, args)
        return node

    @staticmethod
    def maintenance_loop():
        for p in GLContextNode.pending_contexts:
            p.create_context()
        GLContextNode.pending_contexts = []
        for c in GLContextNode.context_list:
            if c.ready:
                if not glfw.window_should_close(c.context.window):
                    c.draw()
        deleted = []
        for c in GLContextNode.pending_deletes:
            c.close()
            deleted.append(c)
        for c in deleted:
            GLContextNode.pending_deletes.remove(c)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.title = 'untitled'
        self.width = 0
        self.height = 0
        self.samples = 1
        self.ready = False
        self.fov = 60.0
        self.command_parser = GLContextCommandParser()
        self.pending_commands = []
        self.shifted_keys = {}
        self.shifted_keys['1'] = '!'
        self.shifted_keys['2'] = '@'
        self.shifted_keys['3'] = '#'
        self.shifted_keys['4'] = '$'
        self.shifted_keys['5'] = '%'
        self.shifted_keys['6'] = '^'
        self.shifted_keys['7'] = '&'
        self.shifted_keys['8'] = '*'
        self.shifted_keys['9'] = '('
        self.shifted_keys['0'] = ')'
        self.shifted_keys['`'] = '~'
        self.shifted_keys['-'] = '_'
        self.shifted_keys['='] = '+'
        self.shifted_keys['['] = '{'
        self.shifted_keys[']'] = '}'
        self.shifted_keys['\\'] = '|'
        self.shifted_keys[';'] = ':'
        self.shifted_keys["'"] = '"'
        self.shifted_keys[','] = '<'
        self.shifted_keys['.'] = '>'
        self.shifted_keys["/"] = '?'

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == str:
                self.title = val
            elif t in [int, float]:
                if self.width == 0:
                    self.width = val
                else:
                    if self.height == 0:
                        self.height = val
                    else:
                        self.samples = val
        if self.width == 0:
            self.width = 640
        if self.height == 0:
            self.height = 480

        self.pending_contexts.append(self)
        print('appended to pending contexts')
        self.context = None
        self.command_input = self.add_input('commands', triggers_execution=True)
        self.output = self.add_output('gl_chain')
        self.ui_output = self.add_output('ui')
        self.fov_option = self.add_option('fov', widget_type='drag_float', default_value=self.fov, callback=self.fov_changed)
        self.fov_option.widget.speed = 0.5

    def execute(self):
        if self.command_input.fresh_input:
            data = self.command_input()
            if type(data) == list:
                self.pending_commands.append(data)

    def fov_changed(self):
        self.fov = self.fov_option()
        if self.fov < 1:
            self.fov = 1
        elif self.fov > 180:
            self.fov = 180
        if self.context:
            self.context.set_fov(self.fov)

    def handle_key(self, key, mods):
        if mods == 1:
            if ord('A') <= key <= ord('Z'):
                char = chr(key)
                char = char.upper()
                key = ord(char)
            else:
                char = chr(key)
                if char in self.shifted_keys:
                    char = self.shifted_keys[char]
                    key = ord(char)
        else:
            if ord('A') <= key <= ord('Z'):
                char = chr(key)
                char = char.lower()
                key = ord(char)
        self.ui_output.send(['key', key])

    def create_context(self):
        self.context = MyGLContext(self.title, self.width, self.height, self.samples)
        self.context_list.append(self)
        self.context.node = self
        self.ready = True

    def custom_cleanup(self):
        self.ready = False
        if self in self.context_list:
            self.context_list.remove(self)
        if self.context:
            self.pending_deletes.append(self.context)

    def init(self):
        pass

    def draw(self):
        if self.context and self.ready:
            try:
                if len(self.pending_commands) > 0:
                    for command in self.pending_commands:
                        self.command_parser.perform(command[0], self, command[1:])
            except:
                self.pending_commands = []
            self.pending_commands = []
            self.context.prepare_draw()
            glPushMatrix()
            glTranslate(0.0, 0.0, -1.0)
            self.output.send('draw')
            glPopMatrix()
            self.context.end_draw()

    def predisplay_callback(self):
        pass


class GLNode(Node):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.gl_input = None
        self.gl_output = None
        self.initialize(args)

    def initialize(self, args):
        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.gl_output = self.add_output('gl chain out')

    def draw(self):
        pass

    def remember_state(self):
        pass

    def restore_state(self):
        pass

    def execute(self):
        if self.gl_input.fresh_input:
            input_list = self.gl_input()
            do_draw = False
            t = type(input_list)
            if t == list:
                if type(input_list[0]) == str:
                    if input_list[0] == 'draw':
                        do_draw = True
            elif t == str:
                if input_list == 'draw':
                    do_draw = True
            if do_draw:
                self.remember_state()
                self.draw()
                self.gl_output.send('draw')
                self.restore_state()
            else:
                self.handle_other_messages(input_list)

    def handle_other_messages(self, message):
        pass

class TexturedGLNode(GLNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.texture_shape = None
        self.numpy_texture = None
        self.texture_data_pending = None

    def initialize(self, args):
        super().initialize(args)

    def update_texture(self):
        if self.texture_data_pending is not None:
            if self.numpy_texture is None:
                self.numpy_texture = GLNumpyTexture(self.texture_data_pending)
            else:
                self.numpy_texture.update(self.texture_data_pending)
            self.texture_data_pending = None

    def prepare_texture_for_drawing(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.numpy_texture.texture)

    def finish_texture_drawing(self):
        glBindTexture(GL_TEXTURE_2D, 0)


    def receive_texture_data(self, data):
        if type(data) == np.ndarray:
            self.texture_data_pending = data  # * 255.0
        elif self.app.torch_available and type(data) == torch.Tensor:
            texture_data = data
            if len(texture_data.shape) > 2:
                if texture_data.shape[-3] <= 5:
                    if texture_data.shape[-1] > 5:
                        texture_data = texture_data.permute(-2, -1, -3)
            self.texture_data_pending = texture_data.cpu().numpy()
        elif type(data) == float:
            self.texture_data_pending = np.ones((16, 16, 1), dtype=np.float32) * data
        elif type(data) == int:
            self.texture_data_pending = np.ones((16, 16, 1), dtype=np.uint8) * data
        elif type(data) == list:
            component_count = len(data)
            t = type(data[0])
            if component_count == 3:
                if t == float:
                    self.texture_data_pending = np.ones((16, 16, 3), dtype=np.float32)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier
                elif t == int:
                    self.texture_data_pending = np.ones((16, 16, 3), dtype=np.uint8)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier
            elif component_count == 4:
                if t == float:
                    self.texture_data_pending = np.ones((16, 16, 4), dtype=np.float32)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier
                elif t == int:
                    self.texture_data_pending = np.ones((16, 16, 4), dtype=np.uint8)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier
            elif component_count == 1:
                if t == float:
                    self.texture_data_pending = np.ones((16, 16, 1), dtype=np.float32)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier
                elif t == int:
                    self.texture_data_pending = np.ones((16, 16, 1), dtype=np.uint8)
                    multiplier = np.array(data)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    multiplier = np.expand_dims(multiplier, axis=0)
                    self.texture_data_pending = self.texture_data_pending * multiplier


class GLQuadricCommandParser(GLCommandParser):
    def __init__(self, quadric_node):
        super().__init__()
        self.quadric_node = quadric_node
        self.dict['style'] = self.set_style
        self.dict['orient_scale'] = self.set_orient_scale
        # self.dict['frustum'] = self.set_frustum
        # self.dict['perspective'] = self.set_perspective

    def set_style(self, node, args):
        if args is not None and len(args) > 0:
            mode = any_to_string(args[0])
            if mode == 'fill':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            elif mode == 'line':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            elif mode == 'point':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_POINT)

    def set_orient_scale(self, node, args):
        orient = list_to_array(args)
        scale = np.linalg.norm(orient) + 1e-6
        axis = orient / scale

        if scale > 0.001:
            # note that up vector is not same for all joints... see refVector
            up_vector = np.array([0.0, 0.0, 1.0])
            v = np.cross(axis, up_vector)
            c = np.dot(axis, up_vector)
            k = 1.0 / (1.0 + c)

            node.alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                                         v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                                         v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                                         0.0, 0.0, 0.0, 1.0])

            node.alignment_matrix.reshape((4, 4))
            node.set_size(scale)
        else:
            node.set_size(0.0)
            # restore_matrix = glGetInteger(GL_MATRIX_MODE)
            # glMatrixMode(GL_MODELVIEW)
            # glPushMatrix()
            # glMultMatrixf(alignment_matrix)
            # gluDisk(self.joint_sphere, 0.0, scale, 32, 1)
            # glPopMatrix()
            # glMatrixMode(restore_matrix)


# class GLNumpyVertices:
#     def __init__(self, texture_array):
#         self.texture = -1
#         self.width = 256
#         self.height = 256
#         self.channels = 3
#         self.type = GL_UNSIGNED_BYTE
#
#         self.allocate(texture_array)

    # def allocate(self, texture_array):
    #     self.width = texture_array.shape[1]
    #     self.height = texture_array.shape[0]
    #     num_dims = len(texture_array.shape)
    #     if num_dims == 3:
    #         self.channels = texture_array.shape[2]
    #     else:
    #         self.channels = 1
    #     # self.type = GL_UNSIGNED_BYTE
    #     if texture_array.dtype == np.float32:
    #         self.type = GL_FLOAT
    #     elif texture_array.dtype == np.uint8:
    #         self.type = GL_UNSIGNED_BYTE
    #
    #     if self.texture == -1:
    #         self.texture = glGenTextures(1)
    #
    #     glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    #     glEnable(GL_TEXTURE_2D)
    #     glBindTexture(GL_TEXTURE_2D, self.texture)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    #     glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    #     contiguous_texture = np.ascontiguousarray(texture_array)
    #     if self.type == GL_FLOAT:
    #         if self.channels == 3:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, contiguous_texture)
    #         elif self.channels == 4:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_FLOAT, contiguous_texture)
    #         elif self.channels == 1:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.width, self.height, 0, GL_LUMINANCE, GL_FLOAT, contiguous_texture)
    #     elif self.type == GL_UNSIGNED_BYTE:
    #         if self.channels == 3:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, contiguous_texture)
    #         elif self.channels == 4:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, contiguous_texture)
    #         elif self.channels == 1:
    #             glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.width, self.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
    #                          contiguous_texture)
    #     glBindTexture(GL_TEXTURE_2D, 0)
    #
    # def update(self, texture_array):
    #     reshape = False
    #     if self.width != texture_array.shape[1]:
    #         reshape = True
    #     if self.height != texture_array.shape[0]:
    #         reshape = True
    #     num_dims = len(texture_array.shape)
    #     if num_dims == 3:
    #         channels = texture_array.shape[2]
    #     else:
    #         channels = 1
    #     if self.channels != channels:
    #         reshape = True
    #
    #     if reshape:
    #         self.allocate(texture_array)
    #     else:
    #         glEnable(GL_TEXTURE_2D)
    #         glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    #
    #         glBindTexture(GL_TEXTURE_2D, self.texture)
    #         contiguous_texture = np.ascontiguousarray(texture_array)
    #         if self.type == GL_FLOAT:
    #             if self.channels == 3:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_FLOAT, contiguous_texture)
    #             elif self.channels == 4:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, contiguous_texture)
    #             elif self.channels == 1:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_LUMINANCE, GL_FLOAT, contiguous_texture)
    #
    #         elif self.type == GL_UNSIGNED_BYTE:
    #             if self.channels == 3:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, contiguous_texture)
    #             elif self.channels == 4:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, contiguous_texture)
    #             elif self.channels == 1:
    #                 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_LUMINANCE, GL_UNSIGNED_BYTE, contiguous_texture)
    #         glBindTexture(GL_TEXTURE_2D, 0)


class GLNumpyTexture:
    def __init__(self, texture_array):
        self.texture = -1
        self.width = 256
        self.height = 256
        self.channels = 3
        self.type = GL_UNSIGNED_BYTE
        self.contiguous_texture = None
        self.allocate(texture_array)

    def allocate(self, texture_array):
        self.width = texture_array.shape[1]
        self.height = texture_array.shape[0]
        num_dims = len(texture_array.shape)
        if num_dims == 3:
            self.channels = texture_array.shape[2]
        else:
            self.channels = 1
        # self.type = GL_UNSIGNED_BYTE
        if texture_array.dtype == np.float32:
            self.type = GL_FLOAT
        elif texture_array.dtype == np.uint8:
            self.type = GL_UNSIGNED_BYTE

        if self.texture == -1:
            self.texture = glGenTextures(1)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        self.contiguous_texture = np.ascontiguousarray(texture_array)
        if self.type == GL_FLOAT:
            if self.channels == 3:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, self.contiguous_texture)
            elif self.channels == 4:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_FLOAT, self.contiguous_texture)
            elif self.channels == 1:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.width, self.height, 0, GL_LUMINANCE, GL_FLOAT, self.contiguous_texture)
        elif self.type == GL_UNSIGNED_BYTE:
            if self.channels == 3:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.contiguous_texture)
            elif self.channels == 4:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.contiguous_texture)
            elif self.channels == 1:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.width, self.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                             self.contiguous_texture)
        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, texture_array):
        reshape = False
        if self.width != texture_array.shape[1]:
            reshape = True
        if self.height != texture_array.shape[0]:
            reshape = True
        num_dims = len(texture_array.shape)
        if num_dims == 3:
            channels = texture_array.shape[2]
        else:
            channels = 1
        if self.channels != channels:
            reshape = True

        if reshape:
            self.allocate(texture_array)
        else:
            glEnable(GL_TEXTURE_2D)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            glBindTexture(GL_TEXTURE_2D, self.texture)
            self.contiguous_texture = np.ascontiguousarray(texture_array)
            if self.type == GL_FLOAT:
                if self.channels == 3:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_FLOAT, self.contiguous_texture)
                elif self.channels == 4:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, self.contiguous_texture)
                elif self.channels == 1:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_LUMINANCE, GL_FLOAT, self.contiguous_texture)

            elif self.type == GL_UNSIGNED_BYTE:
                if self.channels == 3:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.contiguous_texture)
                elif self.channels == 4:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, self.contiguous_texture)
                elif self.channels == 1:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_LUMINANCE, GL_UNSIGNED_BYTE, self.contiguous_texture)
            glBindTexture(GL_TEXTURE_2D, 0)



class GLBillboard(TexturedGLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLBillboard(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.width = self.add_input('width', widget_type='drag_float', default_value=1.6)
        self.height = self.add_input('height', widget_type='drag_float', default_value=0.9)
        self.texture = self.add_input('texture')
        # self.gl_output = self.add_output('gl chain out')

    def execute(self):
        if self.texture.fresh_input:
            data = self.texture()
            self.receive_texture_data(data)
        super().execute()

    def draw(self):
        self.update_texture()
        # if self.texture_data_pending is not None:
        #     if self.numpy_texture is None:
        #         self.numpy_texture = GLNumpyTexture(self.texture_data_pending)
        #     else:
        #         self.numpy_texture.update(self.texture_data_pending)
        #     self.texture_data_pending = None
        half_width = self.width() / 2
        half_height = self.height() / 2
        # color = self.color_input.get_widget_value()

        if self.numpy_texture is not None and self.numpy_texture.texture != -1:
            self.prepare_texture_for_drawing()
            # glEnable(GL_TEXTURE_2D)
            # glBindTexture(GL_TEXTURE_2D, self.numpy_texture.texture)
            gl.glBegin(gl.GL_TRIANGLE_STRIP)
            glTexCoord2f(0, 0)
            gl.glVertex2f(-half_width, half_height)
            glTexCoord2f(0, 1)
            gl.glVertex2f(-half_width, -half_height)
            glTexCoord2f(1, 0)
            gl.glVertex2f(half_width, half_height)
            glTexCoord2f(1, 1)
            gl.glVertex2f(half_width, -half_height)
            gl.glEnd()
            self.finish_texture_drawing()
            # glBindTexture(GL_TEXTURE_2D, 0)
        else:
            gl.glBegin(gl.GL_TRIANGLE_STRIP)
            gl.glVertex2f(-half_width, -half_height)
            gl.glVertex2f(-half_width, half_height)
            gl.glVertex2f(half_width, -half_height)
            gl.glVertex2f(half_width, half_height)
            gl.glEnd()


class GLQuadricNode(TexturedGLNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        try:
            self.quadric = gluNewQuadric()

        except Exception as e:
            print('self.quadric failed')
            print(e)
        self.shading_option = None
        self.pending_commands = []
        self.command_parser = GLQuadricCommandParser(self)
        self.alignment_matrix = None
        self.texture = self.add_input('texture')
        super().initialize(args)

    def process_pending_commands(self):
        if self.pending_commands is not None:
            if len(self.pending_commands) > 0:
                for command in self.pending_commands:
                    self.command_parser.perform(command[0], self, command[1:])
            self.pending_commands = []

    def shading_changed(self):
        shading = self.shading_option()
        if shading == 'flat':
            self.shading = glu.GLU_FLAT
        elif shading == 'smooth':
            self.shading = glu.GLU_SMOOTH
        else:
            self.shading = glu.GLU_NONE
        glu.gluQuadricNormals(self.quadric, self.shading)

    def set_size(self, scaler):
        pass

    def add_shading_option(self):
        self.shading_option = self.add_option('shading', widget_type='combo', default_value='smooth', callback=self.shading_changed)
        self.shading_option.widget.combo_items = ['none', 'flat', 'smooth']

    def handle_other_messages(self, message):
        if type(message) == list:
            self.pending_commands.append(message)

    def quadric_draw(self):
        pass

    def draw(self):
        self.process_pending_commands()
        if self.numpy_texture is not None:
            glu.gluQuadricTexture(self.quadric, True)
        self.update_texture()
        restore_matrix = glGetInteger(GL_MATRIX_MODE)
        if self.alignment_matrix is not None:
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glMultMatrixf(self.alignment_matrix)
        if self.numpy_texture is not None and self.numpy_texture.texture != -1:
            self.prepare_texture_for_drawing()
        self.quadric_draw()
        if self.numpy_texture is not None and self.numpy_texture.texture != -1:
            self.finish_texture_drawing()
        if self.alignment_matrix is not None:
            glPopMatrix()
            glMatrixMode(restore_matrix)

    def execute(self):
        if self.texture.fresh_input:
            data = self.texture()
            self.receive_texture_data(data)
        super().execute()


class GLSphereNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLSphereNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        size = self.arg_as_float(default_value=0.5)
        slices = self.arg_as_int(index=1, default_value=32)
        stacks = self.arg_as_int(index =2, default_value=32)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.size = self.add_input('size', widget_type='drag_float', default_value=size)
        # self.gl_output = self.add_output('gl chain out')

        self.slices = self.add_input('slices', widget_type='drag_int', default_value=slices)
        self.stacks = self.add_input('stacks', widget_type='drag_int', default_value=stacks)
        self.add_shading_option()

    def set_size(self, scaler):
        self.size.set(scaler)

    def quadric_draw(self):
        gluSphere(self.quadric, self.size(), self.slices(), self.stacks())


class GLDiskNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLDiskNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)

        inner_radius = self.arg_as_float(default_value=0.0)
        outer_radius = self.arg_as_float(index=1, default_value=0.5)
        slices = self.arg_as_int(index=2, default_value=32)
        rings = self.arg_as_int(index=3, default_value=1)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.outer_radius = self.add_input('outer radius', widget_type='drag_float', default_value=outer_radius)
        # self.gl_output = self.add_output('gl chain out')

        self.inner_radius = self.add_input('inner radius', widget_type='drag_float', default_value=inner_radius)
        self.slices = self.add_input('slices', widget_type='drag_int', default_value=slices)
        self.rings = self.add_input('rings', widget_type='drag_int', default_value=rings)
        self.add_shading_option()

    def set_size(self, scaler):
        self.outer_radius.set(scaler)

    def quadric_draw(self):
        gluDisk(self.quadric, self.inner_radius(), self.outer_radius(), self.slices(), self.rings())


class GLPartialDiskNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLPartialDiskNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)

        inner_radius = self.arg_as_float(default_value=0.0)
        outer_radius = self.arg_as_float(index=1, default_value=0.5)
        slices = self.arg_as_int(index=2, default_value=32)
        rings = self.arg_as_int(index=3, default_value=1)
        start_angle = self.arg_as_float(index=4, default_value=0.0)
        sweep_angle = self.arg_as_float(index=5, default_value=90)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.outer_radius = self.add_input('outer radius', widget_type='drag_float', default_value=outer_radius)
        # self.gl_output = self.add_output('gl chain out')

        self.inner_radius = self.add_input('inner radius', widget_type='drag_float', default_value=inner_radius)
        self.slices = self.add_input('slices', widget_type='drag_int', default_value=slices)
        self.rings = self.add_input('rings', widget_type='drag_int', default_value=rings)
        self.start_angle = self.add_input('start angle', widget_type='drag_float', default_value=start_angle)
        self.start_angle.widget.speed = 1
        self.sweep_angle = self.add_input('sweep angle', widget_type='drag_float', default_value=sweep_angle)
        self.sweep_angle.widget.speed = 1
        self.add_shading_option()

    def set_size(self, scaler):
        self.outer_radius.set(scaler)

    def quadric_draw(self):
        gluPartialDisk(self.quadric, self.inner_radius(), self.outer_radius(), self.slices(), self.rings(), self.start_angle(), self.sweep_angle())


class GLCylinderNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLCylinderNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)

        base_radius = self.arg_as_float(default_value=0.5)
        top_radius = self.arg_as_float(index=1, default_value=0.5)
        height = self.arg_as_float(index=2, default_value=0.5)
        slices = self.arg_as_int(index=3, default_value=40)
        stacks = self.arg_as_int(index=4, default_value=1)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        # self.gl_output = self.add_output('gl chain out')

        self.base_radius = self.add_input('base radius', widget_type='drag_float', default_value=base_radius)
        self.top_radius = self.add_input('top radius', widget_type='drag_float', default_value=top_radius)
        self.height = self.add_input('height', widget_type='drag_float', default_value=height)
        self.slices = self.add_input('slices', widget_type='drag_int', default_value=slices)
        self.stacks = self.add_input('stacks', widget_type='drag_int', default_value=stacks)
        self.add_shading_option()

    def set_size(self, scaler):
        self.base_radius.set(scaler)
        self.top_radius.set(scaler)
        self.height.set(scaler)

    def quadric_draw(self):
        gluCylinder(self.quadric, self.base_radius(), self.top_radius(), self.height(), self.slices(), self.stacks())


class GLQuaternionRotateNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLQuaternionRotateNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.previous_quat = None
        self.joint_sphere = gluNewQuadric()
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.show_axis = False
        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.quat_input = self.add_input('quaternion')
        # self.gl_output = self.add_output('gl chain out')
        self.show_axis_option = self.add_option('show axis', widget_type='checkbox', default_value=self.show_axis)

    def remember_state(self):
        gl.glPushMatrix()

    def restore_state(self):
        gl.glPopMatrix()

    def transform_to_opengl(self, transform):
        if transform is not None and len(transform) == 16:
            # Transpose matrix for OpenGL column-major order.
            for i in range(0, 4):
                for j in range((i + 1), 4):
                    temp = transform[4 * i + j]
                    transform[4 * i + j] = transform[4 * j + i]
                    transform[4 * j + i] = temp
        return transform

    def draw(self):
        input_ = self.quat_input()
        t = type(input_)
        if t == list:
            input_ = np.array(input_)
            t = np.ndarray
        if t == np.ndarray:
            rotation_q = Quaternion(input_)
            if self.show_axis_option():
                if self.previous_quat:
                    up_vector = np.array([0.0, 0.0, 1.0])
                    d = Quaternion.sym_distance(rotation_q, self.previous_quat) * 20
                    change = rotation_q - self.previous_quat
                    axis = change.unit.axis
                    gl.glBegin(gl.GL_LINES)
                    gl.glVertex3f(-axis[0] * d, -axis[1] * d, -axis[2] * d)
                    gl.glVertex3f(axis[0] * d, axis[1] * d, axis[2] * d)
                    gl.glEnd()

                    v = np.cross(axis, up_vector)
                    c = np.dot(axis, up_vector)
                    k = 1.0 / (1.0 + c)

                    alignment_matrix = np.array(
                        [v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                         v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                         v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                         0.0, 0.0, 0.0, 1.0])

                    alignment_matrix.reshape((4, 4))
                    restore_matrix = glGetInteger(GL_MATRIX_MODE)
                    glMatrixMode(GL_MODELVIEW)
                    glPushMatrix()
                    glMultMatrixf(alignment_matrix)
                    gluDisk(self.joint_sphere, 0.0, d / 2, 32, 1)
                    glPopMatrix()
                    glMatrixMode(restore_matrix)

                self.previous_quat = rotation_q
            transform = quaternion_to_R3_rotation(rotation_q)
            transform = self.transform_to_opengl(transform)
            glMultMatrixf(transform)


class GLAxisAngleRotateNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLAxisAngleRotateNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.show_axis = False
        self.restore_matrix = None
        self.rotvec_input = self.add_input('rotation vector')

    def remember_state(self):
        self.restore_matrix = glGetInteger(GL_MATRIX_MODE)
        gl.glPushMatrix()

    def restore_state(self):
        gl.glPopMatrix()
        glMatrixMode(self.restore_matrix)

    def transform_to_opengl(self, transform):
        if transform is not None and len(transform) == 16:
            # Transpose matrix for OpenGL column-major order.
            for i in range(0, 4):
                for j in range((i + 1), 4):
                    temp = transform[4 * i + j]
                    transform[4 * i + j] = transform[4 * j + i]
                    transform[4 * j + i] = temp
        return transform

    def draw(self):
        input_ = self.rotvec_input()
        t = type(input_)
        if t == list:
            input_ = np.array(input_)
            t = np.ndarray
        if t == np.ndarray:
            axis = input_[:3]
            up_vector = np.array([0.0, 0.0, 1.0])

            v = np.cross(axis, up_vector)
            c = np.dot(axis, up_vector)
            k = 1.0 / (1.0 + c)
            alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                                         v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                                         v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                                         0.0, 0.0, 0.0, 1.0])

            alignment_matrix.reshape((4, 4))

            glMatrixMode(GL_MODELVIEW)
            glMultMatrixf(alignment_matrix)



class GLTransformNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLTransformNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.values = [0.0, 0.0, 0.0]
        self.values[0] = self.arg_as_float(default_value=0.0)
        self.values[1] = self.arg_as_float(index=1, default_value=0.0)
        self.values[2] = self.arg_as_float(index=2, default_value=0.0)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.x = self.add_input('x', widget_type='drag_float', default_value=self.values[0])
        self.y = self.add_input('y', widget_type='drag_float', default_value=self.values[1])
        self.z = self.add_input('z', widget_type='drag_float', default_value=self.values[2])
        if self.label == 'gl_rotate':
            self.x.widget.speed = 1
            self.y.widget.speed = 1
            self.z.widget.speed = 1
        self.reset_button = self.add_property('reset', widget_type='button', callback=self.reset)
        # self.gl_output = self.add_output('gl chain out')

    def reset(self):
        if self.label in ['gl_translate', 'gl_rotate']:
            self.x.set(0.0)
            self.y.set(0.0)
            self.z.set(0.0)
        elif self.label == 'gl_scale':
            self.x.set(1.0)
            self.y.set(1.0)
            self.z.set(1.0)

    def remember_state(self):
        gl.glPushMatrix()

    def restore_state(self):
        gl.glPopMatrix()

    def draw(self):
        self.values[0] = self.x()
        self.values[1] = self.y()
        self.values[2] = self.z()

        gl.glMatrixMode(gl.GL_MODELVIEW)

        if self.label == 'gl_translate':
            gl.glTranslatef(self.values[0], self.values[1], self.values[2])
        elif self.label == 'gl_rotate':
            if self.values[0] != 0.0:
                gl.glRotatef(self.values[0], 1.0, 0.0, 0.0)
            if self.values[2] != 0.0:
                gl.glRotatef(self.values[2], 0.0, 0.0, 1.0)
            if self.values[1] != 0.0:
                gl.glRotatef(self.values[1], 0.0, 1.0, 0.0)
        elif self.label == 'gl_scale':
            gl.glScalef(self.values[0], self.values[1], self.values[2])


class GLColorNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLColorNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.hold_color = (GLfloat * 4)()
        self.color = (GLfloat * 4)()
        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.red_input = self.add_input('red', widget_type='slider_float', min=0.0, max=1.0, default_value=0.0)
        self.green_input = self.add_input('green', widget_type='slider_float', min=0.0, max=1.0, default_value=0.0)
        self.blue_input = self.add_input('blue', widget_type='slider_float', min=0.0, max=1.0, default_value=0.0)
        self.alpha_input = self.add_input('alpha', widget_type='slider_float', min=0.0, max=1.0, default_value=0.0)
        # self.gl_output = self.add_output('gl chain out')

    def remember_state(self):
        gl.glGetFloatv(GL_CURRENT_COLOR, self.hold_color)

    def restore_state(self):
        gl.glColor4fv(self.hold_color)


    def draw(self):
        self.color[0] = self.red_input()
        self.color[1] = self.green_input()
        self.color[2] = self.blue_input()
        self.color[3] = self.alpha_input()
        gl.glColor4fv(self.color)

class GLEnableNode(GLNode):
    state_dict = {}
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLEnableNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.state_code = -1
        self.hold_state = False
        if len(GLEnableNode.state_dict) == 0:
            self.create_state_dict()
        state_arg = ''

        if len(args) > 0:
            state_arg = args[0]
            if state_arg in self.state_dict:
                self.state_code = GLEnableNode.state_dict[state_arg]
        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.state_input = self.add_input(state_arg, widget_type='checkbox', default_value=False)
        # self.gl_output = self.add_output('gl chain out')

    def remember_state(self):
        if self.state_code != -1:
            self.hold_state = gl.glIsEnabled(self.state_code)

    def restore_state(self):
        if self.state_code != -1:
            if self.hold_state:
                gl.glEnable(self.state_code)
            else:
                gl.glDisable(self.state_code)

    def draw(self):
        if self.state_code != -1:
            if self.state_input():
                gl.glEnable(self.state_code)
            else:
                gl.glDisable(self.state_code)

    def create_state_dict(self):
        if len(GLEnableNode.state_dict) == 0:
            self.state_dict['GL_LIGHTING'] = gl.GL_LIGHTING
            self.state_dict['GL_TEXTURE_1D'] = gl.GL_TEXTURE_1D
            self.state_dict['GL_TEXTURE_2D'] = gl.GL_TEXTURE_2D
            self.state_dict['GL_TEXTURE_3D'] = gl.GL_TEXTURE_3D
            self.state_dict['GL_LINE_STIPPLE'] = gl.GL_LINE_STIPPLE
            self.state_dict['GL_POLYGON_STIPPLE'] = gl.GL_POLYGON_STIPPLE
            self.state_dict['GL_CULL_FACE'] = gl.GL_CULL_FACE
            self.state_dict['GL_ALPHA_TEST'] = gl.GL_ALPHA_TEST
            self.state_dict['GL_BLEND'] = gl.GL_BLEND
            self.state_dict['GL_INDEX_LOGIC_OP'] = gl.GL_INDEX_LOGIC_OP
            self.state_dict['GL_COLOR_LOGIC_OP'] = gl.GL_COLOR_LOGIC_OP
            self.state_dict['GL_DITHER'] = gl.GL_DITHER
            self.state_dict['GL_STENCIL_TEST'] = gl.GL_STENCIL_TEST
            self.state_dict['GL_DEPTH_TEST'] = gl.GL_DEPTH_TEST
            self.state_dict['GL_CLIP_PLANE0'] = gl.GL_CLIP_PLANE0
            self.state_dict['GL_CLIP_PLANE1'] = gl.GL_CLIP_PLANE1
            self.state_dict['GL_CLIP_PLANE2'] = gl.GL_CLIP_PLANE2
            self.state_dict['GL_CLIP_PLANE3'] = gl.GL_CLIP_PLANE3
            self.state_dict['GL_CLIP_PLANE4'] = gl.GL_CLIP_PLANE4
            self.state_dict['GL_CLIP_PLANE5'] = gl.GL_CLIP_PLANE5
            self.state_dict['GL_LIGHT0'] = gl.GL_LIGHT0
            self.state_dict['GL_LIGHT1'] = gl.GL_LIGHT1
            self.state_dict['GL_LIGHT2'] = gl.GL_LIGHT2
            self.state_dict['GL_LIGHT3'] = gl.GL_LIGHT3
            self.state_dict['GL_LIGHT4'] = gl.GL_LIGHT4
            self.state_dict['GL_LIGHT5'] = gl.GL_LIGHT5
            self.state_dict['GL_LIGHT6'] = gl.GL_LIGHT6
            self.state_dict['GL_LIGHT7'] = gl.GL_LIGHT7
            self.state_dict['GL_TEXTURE_GEN_S'] = gl.GL_TEXTURE_GEN_S
            self.state_dict['GL_TEXTURE_GEN_T'] = gl.GL_TEXTURE_GEN_T
            self.state_dict['GL_TEXTURE_GEN_Q'] = gl.GL_TEXTURE_GEN_Q
            self.state_dict['GL_TEXTURE_GEN_R'] = gl.GL_TEXTURE_GEN_R
            self.state_dict['GL_MAP1_VERTEX_3'] = gl.GL_MAP1_VERTEX_3
            self.state_dict['GL_MAP1_VERTEX_4'] = gl.GL_MAP1_VERTEX_4
            self.state_dict['GL_MAP1_COLOR_4'] = gl.GL_MAP1_COLOR_4
            self.state_dict['GL_MAP1_INDEX'] = gl.GL_MAP1_INDEX
            self.state_dict['GL_MAP1_NORMAL'] = gl.GL_MAP1_NORMAL
            self.state_dict['GL_MAP1_TEXTURE_COORD_1'] = gl.GL_MAP1_TEXTURE_COORD_1
            self.state_dict['GL_MAP1_TEXTURE_COORD_2'] = gl.GL_MAP1_TEXTURE_COORD_2
            self.state_dict['GL_MAP1_TEXTURE_COORD_3'] = gl.GL_MAP1_TEXTURE_COORD_3
            self.state_dict['GL_MAP1_TEXTURE_COORD_4'] = gl.GL_MAP1_TEXTURE_COORD_4
            self.state_dict['GL_MAP2_VERTEX_3'] = gl.GL_MAP2_VERTEX_3
            self.state_dict['GL_MAP2_VERTEX_4'] = gl.GL_MAP2_VERTEX_4
            self.state_dict['GL_MAP2_COLOR_4'] = gl.GL_MAP2_COLOR_4
            self.state_dict['GL_MAP2_INDEX'] = gl.GL_MAP2_INDEX
            self.state_dict['GL_MAP2_NORMAL'] = gl.GL_MAP2_NORMAL
            self.state_dict['GL_POINT_SMOOTH'] = gl.GL_POINT_SMOOTH
            self.state_dict['GL_LINE_SMOOTH'] = gl.GL_LINE_SMOOTH
            self.state_dict['GL_POLYGON_SMOOTH'] = gl.GL_POLYGON_SMOOTH
            self.state_dict['GL_SCISSOR_TEST'] = gl.GL_SCISSOR_TEST
            self.state_dict['GL_COLOR_MATERIAL'] = gl.GL_COLOR_MATERIAL
            self.state_dict['GL_NORMALIZE'] = gl.GL_NORMALIZE
            self.state_dict['GL_AUTO_NORMAL'] = gl.GL_AUTO_NORMAL
            self.state_dict['GL_VERTEX_ARRAY'] = gl.GL_VERTEX_ARRAY
            self.state_dict['GL_NORMAL_ARRAY'] = gl.GL_NORMAL_ARRAY
            self.state_dict['GL_COLOR_ARRAY'] = gl.GL_COLOR_ARRAY
            self.state_dict['GL_INDEX_ARRAY'] = gl.GL_INDEX_ARRAY
            self.state_dict['GL_TEXTURE_COORD_ARRAY'] = gl.GL_TEXTURE_COORD_ARRAY
            self.state_dict['GL_EDGE_FLAG_ARRAY'] = gl.GL_EDGE_FLAG_ARRAY
            self.state_dict['GL_POLYGON_OFFSET_POINT'] = gl.GL_POLYGON_OFFSET_POINT
            self.state_dict['GL_POLYGON_OFFSET_LINE'] = gl.GL_POLYGON_OFFSET_LINE
            self.state_dict['GL_POLYGON_OFFSET_FILL'] = gl.GL_POLYGON_OFFSET_FILL
            self.state_dict['GL_COLOR_TABLE'] = gl.GL_COLOR_TABLE
            self.state_dict['GL_POST_CONVOLUTION_COLOR_TABLE'] = gl.GL_POST_CONVOLUTION_COLOR_TABLE
            self.state_dict['GL_POST_COLOR_MATRIX_COLOR_TABLE'] = gl.GL_POST_COLOR_MATRIX_COLOR_TABLE
            self.state_dict['GL_CONVOLUTION_1D'] = gl.GL_CONVOLUTION_1D
            self.state_dict['GL_CONVOLUTION_2D'] = gl.GL_CONVOLUTION_2D
            self.state_dict['GL_SEPARABLE_2D'] = gl.GL_SEPARABLE_2D
            self.state_dict['GL_HISTOGRAM'] = gl.GL_HISTOGRAM
            self.state_dict['GL_MINMAX'] = gl.GL_MINMAX
            self.state_dict['GL_RESCALE_NORMAL'] = gl.GL_RESCALE_NORMAL


class GLMaterial:
    def __init__(self):
        self.ambient = [0.2, 0.2, 0.2, 1.0]
        self.diffuse = [0.8, 0.8, 0.8, 1.0]
        self.specular = [0.0, 0.0, 0.0, 1.0]
        self.emission = [0.0, 0.0, 0.0, 1.0]
        self.shininess = 0.0


class GLMaterialNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLMaterialNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.presets = {}
        super().__init__(label, data, args)

    def save_custom(self, container):
        container['ambient'] = list(self.material.ambient)
        container['diffuse'] = list(self.material.diffuse)
        container['specular'] = list(self.material.specular)
        container['emission'] = list(self.material.emission)

    def load_custom(self, container):
        if 'ambient' in container:
            self.material.ambient = container['ambient']
        if 'diffuse' in container:
            self.material.diffuse = container['diffuse']
        if 'specular' in container:
            self.material.specular = container['specular']
        if 'emission' in container:
            self.material.emission = container['emission']

    def initialize(self, args):
        super().initialize(args)
        self.material = GLMaterial()
        self.material.ambient = [0.2, 0.2, 0.2, 1.0]
        self.material.diffuse = [0.8, 0.8, 0.8, 1.0]
        self.material.specular = [0.0, 0.0, 0.0, 1.0]
        self.material.emission = [0.0, 0.0, 0.0, 1.0]
        self.material.shininess = 0.0

        self.hold_material = GLMaterial()

        self.create_material_presets()

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.ambient_input = self.add_input('ambient')
        self.diffuse_input = self.add_input('diffuse')
        self.specular_input = self.add_input('specular')
        self.emission_input = self.add_input('emission')
        self.shininess_input = self.add_input('shininess', widget_type='drag_float', default_value=self.material.shininess)
        self.alpha_input = self.add_input('alpha', widget_type='drag_float', default_value=1.0, callback=self.set_alpha)

        self.preset_menu = self.add_property('presets', widget_type='combo', callback=self.preset_selected)
        presets = list(self.presets.keys())
        self.preset_menu.widget.combo_items = presets
        # self.gl_output = self.add_output('gl chain out')

    def set_alpha(self):
        alpha = self.alpha_input()
        self.material.ambient[3] = alpha
        self.material.diffuse[3] = alpha
        self.material.specular[3] = alpha
        self.material.emission[3] = alpha

    def preset_selected(self):
        selected_preset = self.preset_menu()
        if selected_preset in self.presets:
            p = self.presets[selected_preset]
            self.material.ambient = p.ambient
            self.material.diffuse = p.diffuse
            self.material.specular = p.specular
            self.material.emission = p.emission
            self.material.shininess = p.shininess

    def remember_state(self):
        self.hold_material.ambient = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT)
        self.hold_material.diffuse = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE)
        self.hold_material.specular = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR)
        self.hold_material.emission = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_EMISSION)
        self.hold_material.shininess = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_SHININESS)

    def restore_state(self):
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, self.hold_material.ambient)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, self.hold_material.diffuse)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, self.hold_material.specular)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, self.hold_material.emission)
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, self.hold_material.shininess)

    def execute(self):
        if self.ambient_input.fresh_input:
            ambient = self.ambient_input()
            t = type(ambient)
            if t == np.ndarray:
                if ambient.shape[0] == 4:
                    self.material.ambient = ambient
            elif t == list:
                ambient = any_to_numerical_list(ambient)
                if len(ambient) == 4:
                    self.material.ambient = ambient
                elif len(ambient) == 3:
                    self.material.ambient = ambient + [1.0]
            elif t in [float, np.double]:
                self.material.ambient = [ambient, ambient, ambient, 1.0]

        if self.diffuse_input.fresh_input:
            diffuse = self.diffuse_input()
            t = type(diffuse)
            if t == np.ndarray:
                if diffuse.shape[0] == 4:
                    self.material.diffuse = diffuse
            if t == list:
                diffuse = any_to_numerical_list(diffuse)
                if len(diffuse) == 4:
                    self.material.diffuse = diffuse
                elif len(diffuse) == 3:
                    self.material.diffuse = diffuse + [1.0]
            elif t in [float, np.double]:
                self.material.diffuse = [diffuse, diffuse, diffuse, 1.0]

        if self.specular_input.fresh_input:
            specular = self.specular_input()
            t = type(specular)
            if t == np.ndarray:
                if specular.shape[0] == 4:
                    self.material.specular = specular
            if t == list:
                specular = any_to_numerical_list(specular)
                if len(specular) == 4:
                    self.material.specular = specular
                elif len(specular) == 3:
                    self.material.specular = specular + [1.0]
            elif t in [float, np.double]:
                self.material.specular = [specular, specular, specular, 1.0]

        if self.emission_input.fresh_input:
            emission = self.emission_input()
            t = type(emission)
            if t == np.ndarray:
                if emission.shape[0] == 4:
                    self.material.emission = emission
            if t == list:
                emission = any_to_numerical_list(emission)
                if len(emission) == 4:
                    self.material.emission = emission
                elif len(emission) == 3:
                    self.material.emission = emission + [1.0]
            elif t in [float, np.double]:
                self.material.emission = [emission, emission, emission, 1.0]

        if self.shininess_input.fresh_input:
            shininess = self.shininess_input()
            t = type(shininess)
            shininess = any_to_float(shininess)
            if shininess < 0:
                shininess = 0
            self.material.shininess = shininess
        super().execute()

    def draw(self):
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, self.material.ambient)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, self.material.diffuse)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, self.material.specular)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, self.material.emission)
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, self.material.shininess)

    def create_material_presets(self):
        red_clay = GLMaterial()
        red_clay.diffuse = [1.0, .5, .2, 1]
        self.presets['red_clay'] = red_clay

        green_plastic = GLMaterial()
        green_plastic.ambient = [.5, 1.0, .5, 1]
        green_plastic.diffuse = [0, 1, 0, 1]
        green_plastic.specular = [.5, 1.0, .5, 1]
        green_plastic.shininess = 40.0
        self.presets['green_plastic'] = green_plastic

        silver_metal = GLMaterial()
        silver_metal.ambient = [1.0, 1.0, 1.0, 1]
        silver_metal.diffuse = [.5, .5, .5, 1]
        silver_metal.specular = [1.0, 1.0, 1.0, 1]
        silver_metal.shininess = 5.0
        self.presets['silver_metal'] = silver_metal

        brass = GLMaterial()
        brass.ambient= [0.329412, 0.223529, 0.027451, 1.0]
        brass.diffuse = [0.780392, 0.568627, 0.113725, 1.0]
        brass.specular = [0.992157, 0.941176, 0.807843, 1.0]
        brass.shininess = 27.8974
        self.presets['brass'] = brass

        bronze = GLMaterial()
        bronze.ambient = [0.2125, 0.1275, 0.054, 1.0]
        bronze.diffuse = [0.714, 0.4284, 0.18144, 1.0]
        bronze.specular = [0.393548, 0.271906, 0.166721, 1.0]
        bronze.shininess = 25.6
        self.presets['bronze'] = bronze

        polished_bronze = GLMaterial()
        polished_bronze.ambient = [0.25, 0.148, 0.06475, 1.0]
        polished_bronze.diffuse = [0.4, 0.2368, 0.1036, 1.0]
        polished_bronze.specular = [0.774597, 0.458561, 0.200621, 1.0]
        polished_bronze.shininess = 76.8
        self.presets['polished_bronze'] = polished_bronze

        chrome = GLMaterial()
        chrome.ambient = [0.25, 0.25, 0.25, 1.0]
        chrome.diffuse = [0.4, 0.4, 0.4, 1.0]
        chrome.specular = [0.774597, 0.774597, 0.774597, 1.0]
        chrome.shininess = 76.8
        self.presets['chrome'] = chrome

        dull_copper = GLMaterial()
        dull_copper.ambient = [0.26, 0.26, 0.26, 1.0]
        dull_copper.diffuse = [0.3, 0.11, 0.0, 1.0]
        dull_copper.specular = [.75, .33, 0.0, 1.0]
        dull_copper.shininess = 10
        self.presets['dull_copper'] = dull_copper

        copper = GLMaterial()
        copper.ambient = [0.19125, 0.0735, 0.0225, 1.0]
        copper.diffuse = [0.7038, 0.27048, 0.0828, 1.0]
        copper.specular = [0.256777, 0.137622, 0.086014, 1.0]
        copper.shininess = 12.8
        self.presets['copper'] = copper

        polished_copper = GLMaterial()
        polished_copper.ambient = [0.2295, 0.08825, 0.0275, 1.0]
        polished_copper.diffuse = [0.5508, 0.2118, 0.066, 1.0]
        polished_copper.specular = [0.580594, 0.223257, 0.0695701, 1.0]
        polished_copper.shininess = 51.2
        self.presets['polished_copper'] = polished_copper

        gold = GLMaterial()
        gold.ambient = [0.24725, 0.1995, 0.0745, 1.0]
        gold.diffuse = [0.75164, 0.60648, 0.22648, 1.0]
        gold.specular = [0.628281, 0.555802, 0.366065, 1.0]
        gold.shininess = 51.2
        self.presets['gold'] = gold

        dull_gold = GLMaterial()
        dull_gold.ambient = [.4, .4, .4, 1.0]
        dull_gold.diffuse = [0.22, 0.15, 0., 1.0]
        dull_gold.specular = [0.71, .7, .56, 1.0]
        dull_gold.shininess = 20
        self.presets['dull_gold'] = dull_gold

        polished_gold = GLMaterial()
        polished_gold.ambient = [0.24725, 0.2245, 0.0645, 1.0]
        polished_gold.diffuse = [0.34615, 0.3143, 0.0903, 1.0]
        polished_gold.specular = [0.797357, 0.723991, 0.208006, 1.0]
        polished_gold.shininess = 83.2
        self.presets['polished_gold'] = polished_gold

        pewter = GLMaterial()
        pewter.ambient = [0.105882, 0.058824, 0.113725, 1.0]
        pewter.diffuse = [0.427451, 0.470588, 0.541176, 1.0]
        pewter.specular = [0.333333, 0.333333, 0.521569, 1.0]
        pewter.shininess = 9.84615
        self.presets['pewter'] = pewter

        steel = GLMaterial()
        steel.ambient = [0.231250, 0.231250, 0.231250, 1.000000]
        steel.diffuse = [0.277500, 0.277500, 0.277500, 1.000000]
        steel.specular = [0.773911, 0.773911, 0.773911, 1.000000]
        steel.shininess = 40.599998
        self.presets['steel'] = steel

        dull_aluminum = GLMaterial()
        dull_aluminum.ambient = [.3, .3, .3, 1.000000]
        dull_aluminum.diffuse = [.3, .3, .5, 1.000000]
        dull_aluminum.specular = [.7, .7, .8, 1.000000]
        dull_aluminum.shininess = 13
        self.presets['dull_aluminum'] = dull_aluminum

        aluminum = GLMaterial()
        aluminum.ambient = [0.250000, 0.250000, 0.250000, 1.000000]
        aluminum.diffuse = [0.400000, 0.400000, 0.400000, 1.000000]
        aluminum.specular = [0.5774597, 0.5774597, 0.5774597, 1.000000]
        aluminum.shininess = 25.800003
        self.presets['aluminum'] = aluminum

        silver = GLMaterial()
        silver.ambient = [0.19225, 0.19225, 0.19225, 1.0]
        silver.diffuse = [0.50754, 0.50754, 0.50754, 1.0]
        silver.specular = [0.508273, 0.508273, 0.508273, 1.0]
        silver.shininess = 51.2
        self.presets['silver'] = silver

        polished_silver = GLMaterial()
        polished_silver.ambient = [0.23125, 0.23125, 0.23125, 1.0]
        polished_silver.diffuse = [0.2775, 0.2775, 0.2775, 1.0]
        polished_silver.specular = [0.773911, 0.773911, 0.773911, 1.0]
        polished_silver.shininess = 89.6
        self.presets['polished_silver'] = polished_silver

        emerald = GLMaterial()
        emerald.ambient = [0.0215, 0.1745, 0.0215, 1.0]
        emerald.diffuse = [0.07568, 0.61424, 0.07568, 1.0]
        emerald.specular = [0.633, 0.727811, 0.633, 1.0]
        emerald.shininess = 76.8
        self.presets['emerald'] = emerald

        jade = GLMaterial()
        jade.ambient = [0.135, 0.2225, 0.1575, 0.95]
        jade.diffuse = [0.54, 0.89, 0.63, 0.95]
        jade.specular = [0.316228, 0.316228, 0.316228, 0.95]
        jade.shininess = 12.8
        self.presets['jade'] = jade

        obsidian = GLMaterial()
        obsidian.ambient = [0.05375, 0.05, 0.06625, 0.82]
        obsidian.diffuse = [0.18275, 0.17, 0.22525, 0.82]
        obsidian.specular = [0.332741, 0.328634, 0.346435, 0.82]
        obsidian.shininess = 38.4
        self.presets['obsidian'] = obsidian

        pearl = GLMaterial()
        pearl.ambient = [0.25, 0.20725, 0.20725, 0.922]
        pearl.diffuse = [1.0, 0.829, 0.829, 0.922]
        pearl.specular = [0.296648, 0.296648, 0.296648, 0.922]
        pearl.shininess = 11.264
        self.presets['pearl'] = pearl

        ruby = GLMaterial()
        ruby.ambient = [0.1745, 0.01175, 0.01175, 0.55]
        ruby.diffuse = [0.61424, 0.04136, 0.04136, 0.55]
        ruby.specular = [0.727811, 0.626959, 0.626959, 0.55]
        ruby.shininess = 76.8
        self.presets['ruby'] = ruby

        turquoise = GLMaterial()
        turquoise.ambient = [0.1, 0.18725, 0.1745, 0.8]
        turquoise.diffuse = [0.396, 0.74151, 0.69102, 0.8]
        turquoise.specular = [0.297254, 0.30829, 0.306678, 0.8]
        turquoise.shininess = 12.8
        self.presets['turquoise'] = turquoise

        black_plastic = GLMaterial()
        black_plastic.ambient = [0.0, 0.0, 0.0, 1.0]
        black_plastic.diffuse = [0.01, 0.01, 0.01, 1.0]
        black_plastic.specular = [0.50, 0.50, 0.50, 1.0]
        black_plastic.shininess = 32
        self.presets['black_plastic'] = black_plastic

        gray_plastic = GLMaterial()
        gray_plastic.ambient = [0.500000, 0.500000, 0.500000, 1.000000]
        gray_plastic.diffuse = [0.010000, 0.010000, 0.010000, 1.000000]
        gray_plastic.specular = [0.500000, 0.500000, 0.500000, 1.000000]
        gray_plastic.shininess = 32.000000
        self.presets['gray_plastic'] = gray_plastic

        white_plastic = GLMaterial()
        white_plastic.ambient = [0.00000, 0.00000, 0.00000, 1.000000]
        white_plastic.diffuse = [0.550000, 0.550000, 0.550000, 1.000000]
        white_plastic.specular = [0.700000, 0.700000, 0.700000, 1.000000]
        white_plastic.shininess = 32.000000
        self.presets['white_plastic'] = white_plastic

        blue_blastic = GLMaterial()
        blue_blastic.ambient = [0.100000, 0.100000, 0.100000, 1.000000]
        blue_blastic.diffuse = [0.020000, 0.020000, 0.710000, 1.000000]
        blue_blastic.specular = [0.830000, 0.830000, 0.830000, 1.000000]
        blue_blastic.shininess = 15.000000
        self.presets['blue_blastic'] = blue_blastic

        metallic_red = GLMaterial()
        metallic_red.ambient = [0.15, 0.15, 0.15, 1.000000]
        metallic_red.diffuse = [0.27, 0, 0, 1.000000]
        metallic_red. specular = [0.610000, 0.130000, 0.180000, 1.000000]
        metallic_red.shininess = 26.
        self.presets['metallic_red'] = metallic_red

        metallic_purple = GLMaterial()
        metallic_purple.ambient = [0.17, 0.17, 0.17, 1.000000]
        metallic_purple.diffuse = [0.1, 0.03, 0.22, 1.000000]
        metallic_purple.specular = [0.640000, 0.000000, 0.980000, 1.000000]
        metallic_purple.shininess = 26.0
        self.presets['metallic_purple'] = metallic_purple

        gray_rubber = GLMaterial()
        gray_rubber.ambient = [0.520000, 0.520000, 0.520000, 1.0]
        gray_rubber.diffuse = [0.01, 0.01, 0.01, 1.0]
        gray_rubber.specular = [0.4, 0.4, 0.4, 1.0]
        gray_rubber.shininess = 10
        self.presets['gray_rubber'] = gray_rubber

        black_rubber = GLMaterial()
        black_rubber.ambient = [0.02, 0.02, 0.02, 1.0]
        black_rubber.diffuse = [0.01, 0.01, 0.01, 1.0]
        black_rubber.specular = [0.4, 0.4, 0.4, 1.0]
        black_rubber.shininess = 10
        self.presets['black_rubber'] = black_rubber

        orange_metal = GLMaterial()
        orange_metal.ambient = [0.19125, 0.0735, 0.0225, 0.5]
        orange_metal.diffuse = [0.7038, 0.27048, 0.0828, 0.5]
        orange_metal.specular = [0.256777, 0.137622, 0.086014, 0.5]
        orange_metal.shininess = 12.8
        self.presets['orange_metal'] = orange_metal

class GLAlignNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLAlignNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.ready = False
        super().__init__(label, data, args)
        self.x = None
        self.y = None
        self.z = None
        self.axis = np.array([0.0, 1.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])

    def initialize(self, args):
        super().initialize(args)
        if args is not None:
            float_count = 0
            for i in range(len(args)):
                val, t = decode_arg(args, i)
                if float_count < 3:
                    self.axis[float_count] = val
                float_count += 1

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.x = self.add_input('x', widget_type='drag_float', default_value=self.axis[0], callback=self.axis_changed)
        self.y = self.add_input('y', widget_type='drag_float', default_value=self.axis[1], callback=self.axis_changed)
        self.z = self.add_input('z', widget_type='drag_float', default_value=self.axis[2], callback=self.axis_changed)
        # self.gl_output = self.add_output('gl chain out')
        self.align()
        self.ready = True

    def remember_state(self):
        if self.ready:
            self.restore_matrix = gl.glGetInteger(gl.GL_MATRIX_MODE)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()

    def restore_state(self):
        if self.ready:
            gl.glPopMatrix()
            gl.glMatrixMode(self.restore_matrix)

    def draw(self):
        if self.ready:
            gl.glLineWidth(2)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f(self.axis[0], self.axis[1], self.axis[2])
            gl.glEnd()
            gl.glMultMatrixf(self.alignment_matrix)

    def axis_changed(self):
        self.axis[0] = self.x()
        self.axis[1] = self.y()
        self.axis[2] = self.z()
        self.axis /= (math.sqrt(self.axis[0] * self.axis[0] + self.axis[1] * self.axis[1] + self.axis[2] * self.axis[2]))
        self.align()

    def align(self):
        v = np.cross(self.axis, self.up)
        c = np.dot(self.axis, self.up)
        k = 1.0 / (1.0 + c)

        self.alignment_matrix = np.array([v[0] * v[0] * k + c, v[1] * v[1] * k - v[2], v[2] * v[0] * k + v[1], 0.0,
                  v[0] * v[1] * k + v[2], v[1] * v[1] * k + c, v[2] * v[1] * k - v[0], 0.0,
                  v[0] * v[2] * k - v[1], v[1] * v[2] * k + v[0], v[2] * v[2] * k + c, 0.0,
                  0.0, 0.0, 0.0, 1.0])
        self.alignment_matrix.reshape((4, 4))

class CharacterSlot:
    def __init__(self, cell, glyph_box, glyph, texture_size):
        # self.texture = texture
        # self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)
        # self.origin = [0, 0]
        left_in_texture = cell[0] * glyph_box[0]
        top_in_texture = cell[1] * glyph_box[1]
        right_in_texture = left_in_texture + glyph_box[0]
        bottom_in_texture = top_in_texture + glyph_box[1]
        left_in_texture /= texture_size[0]
        right_in_texture /= texture_size[0]
        top_in_texture /= texture_size[1]
        bottom_in_texture /= texture_size[1]

        self.texture_coords = [left_in_texture, top_in_texture, right_in_texture, bottom_in_texture]
        # print(self.texture_coords)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
            # self.origin = [glyph.bitmap_left, glyph.bitmap.rows - glyph.bitmap_top]
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
            # self.origin = [glyph.bitmap_left, glyph.bitmap.rows - glyph.bitmap_top]
        else:
            raise RuntimeError('unknown glyph type')


class GLTextNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLTextNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ready = False
        self.new_text = False
        self.display_list = -1
        self.characters = {}
        self.initialized = False
        self.font_size = 24
        self.coords = None
        self.text_buffer = -1
        self.text_vertex_buffer = -1
        self.color = [1.0, 1.0, 1.0, 1.0]
        self.font_path = "Inconsolata-g.otf"
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [float, int]:
                self.font_size = v
            elif t == str:
                self.font_path = v
        self.face = None

        self.text_input = self.add_input('text', callback=self.text_changed)
        self.position_x_input = self.add_input('position_x', widget_type='drag_float', default_value=0.0, callback=self.text_changed)
        self.position_y_input = self.add_input('position_y', widget_type='drag_float', default_value=0.0, callback=self.text_changed)
        self.text_alpha_input = self.add_input('alpha', widget_type='drag_float', default_value=1.0, callback=self.text_changed)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.text_changed)
        self.text_font = self.add_string_input('font', widget_type='text_input', default_value=self.font_path, callback=self.font_changed)
        self.text_color = self.add_option('alpha', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0], callback=self.color_changed)
        self.text_size = self.add_option('size', widget_type='drag_int', default_value=self.font_size, callback=self.size_changed)
        self.alpha_power = self.add_option('alpha power', widget_type='drag_float', default_value=1.0)
        self.separator = self.add_option('separator', widget_type='text_input', default_value=' ')

        self.ready = True
        self.context = None
        self.texture = -1
        self.was_lit = False
        self.was_depth = False

    def custom_create(self, from_file):
        dpg.configure_item(self.text_color.widget.uuid, no_alpha=True)
        dpg.configure_item(self.text_color.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)

    def text_changed(self):
        self.new_text = True

    def color_changed(self):
        self.color = self.text_color()
        self.color[0] /= 255.0
        self.color[1] /= 255.0
        self.color[2] /= 255.0

    def size_changed(self):
        size = self.text_size()
        if size != self.font_size:
            self.font_size = size
            self.initialized = False
            self.new_text = True

    def font_changed(self):
        path = self.text_font()
        if self.font_path != path:
            self.font_path = path
            self.initialized = False
            self.new_text = True

    def update_font(self):
        print('update font')
        hold_context = glfw.get_current_context()
        glfw.make_context_current(self.context)

        self.ready = False
        if self.texture != -1:
            glDeleteTextures(1, self.texture)
        self.characters = {}
        if self.face is not None:
            del self.face
        self.face = freetype.Face(self.font_path)
        size = self.font_size * 256.0
        self.face.set_char_size(int(size))
        self.create_glyph_textures()
        self.ready = True
        glfw.make_context_current(hold_context)

    def create_glyph_textures(self):
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        bottom = 1000
        top = 0
        left = 1000
        right = 0
        printable_count = 0

        for i in range(32, 255):
            a = chr(i)
            if a.isprintable():
                printable_count += 1
                self.face.load_char(chr(i))
                glyph = self.face.glyph
                if glyph.bitmap_top > top:
                    top = glyph.bitmap_top
                if glyph.bitmap_left < left:
                    left = glyph.bitmap_left
                if glyph.bitmap.width + glyph.bitmap_left > right:
                    right = glyph.bitmap.rows + glyph.bitmap_left
                if glyph.bitmap_top - glyph.bitmap.rows < bottom:
                    bottom = glyph.bitmap_top - glyph.bitmap.rows

        self.glyph_shape = [0, 0]
        self.glyph_shape[0] = right - left
        self.glyph_shape[1] = top - bottom

        font_atlas_shape = [self.glyph_shape[0] * 16, self.glyph_shape[1] * 16]
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, font_atlas_shape[0], font_atlas_shape[1], 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        for i in range(32, 255):
            a = chr(i)
            if a.isprintable():
                self.face.load_char(chr(i))
                glyph = self.face.glyph
                bm = glyph.bitmap.buffer
                rgb_bm = [0.0] * (glyph.bitmap.rows * glyph.bitmap.width * 4)
                for k in range(glyph.bitmap.rows):
                    for j in range(glyph.bitmap.width):
                        rgb_bm[(k * glyph.bitmap.width + j) * 4] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 1] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 2] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 3] = float(bm[k * glyph.bitmap.width + j]) / 255.0

                cell = [i % 16, i // 16]
                glyph_offset = [cell[0] * self.glyph_shape[0], cell[1] * self.glyph_shape[1]]
                glTexSubImage2D(GL_TEXTURE_2D, 0, glyph_offset[0] + glyph.bitmap_left, glyph_offset[1] + (self.glyph_shape[1] - glyph.bitmap_top) + bottom, glyph.bitmap.width, glyph.bitmap.rows, GL_RGBA, GL_FLOAT, rgb_bm)
                self.characters[chr(i)] = CharacterSlot(cell, self.glyph_shape, glyph, font_atlas_shape)

        glBindTexture(GL_TEXTURE_2D, 0)

    def remember_state(self):
        self.was_lit = glGetBoolean(GL_LIGHTING)
        if self.was_lit:
            glDisable(GL_LIGHTING)
        self.was_depth = glGetBoolean(GL_DEPTH_TEST)
        if self.was_depth:
            glDisable(GL_DEPTH_TEST)
        glPushMatrix()

    def restore_state(self):
        glPopMatrix()
        if self.was_lit:
            glEnable(GL_LIGHTING)
        if self.was_depth:
            glEnable(GL_DEPTH_TEST)

    def render_text(self):
        if self.initialized:
            self.new_text = False
            if self.display_list != -1:
                glDeleteLists(self.display_list, 1)
            self.display_list = glGenLists(1)

            pos = [self.position_x_input(), self.position_y_input()]
            scale = self.scale_input() / 100
            text = self.text_input()

            if type(text) == str:
                glNewList(self.display_list, GL_COMPILE)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_TEXTURE_2D)

                # self.coords = np.ndarray((len(text), 24))
                for index, c in enumerate(text):
                    if c in self.characters:
                        ch = self.characters[c]
                        width = self.glyph_shape[0] * scale
                        height = self.glyph_shape[1] * scale
                        vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
                        glBegin(GL_TRIANGLES)
                        for i in range(6):
                            glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
                            glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
                        glEnd()
                        # self.coords[index] = vertices.copy()
                        pos[0] += ((ch.advance >> 6) * scale)

                glEndList()
            elif type(text) == list:
                alpha_power = self.alpha_power()
                separator = self.separator()

                glNewList(self.display_list, GL_COMPILE)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_TEXTURE_2D)
                for fragment in text:
                    this_text = ''
                    if type(fragment) == list:
                        this_alpha = fragment[1]

                        if this_alpha > 0.0:
                            this_alpha = pow(this_alpha, alpha_power)
                            this_text = fragment[0] + separator
                            glColor4f(self.color[0], self.color[1], self.color[2], self.text_alpha_input() * this_alpha)

                    elif type(fragment) == str:
                        this_text = fragment + separator
                    if len(this_text) > 0:
                        for index, c in enumerate(this_text):
                            if c in self.characters:
                                ch = self.characters[c]
                                width = self.glyph_shape[0] * scale
                                height = self.glyph_shape[1] * scale
                                vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
                                glBegin(GL_TRIANGLES)
                                for i in range(6):
                                    glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
                                    glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
                                glEnd()
                                # self.coords[index] = vertices.copy()
                                pos[0] += ((ch.advance >> 6) * scale)

                glEndList()

    # figure out drawing the text using drawlists or vertex_arrays
    def draw(self):
        if not self.ready:
            return
        if not self.initialized:
            self.context = glfw.get_current_context()
            self.update_font()
            self.initialized = True
        if self.new_text:
            self.render_text()
        # if self.coords is None:
        #     self.render_text()
        # if self.text_buffer == -1:
        #     self.text_buffer = glGenBuffers(1)
        #     self.text_vertex_buffer = glGenVertexArrays(1)
        #     glBindVertexArray(self.text_vertex_buffer)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTranslatef(0, 0, -2)

        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_TEXTURE_2D)
        glColor4f(self.color[0], self.color[1], self.color[2], self.text_alpha_input())
        if self.display_list != -1:
            glCallList(self.display_list)
        # pos = [self.position_x_input(), self.position_y_input()]
        # scale = self.scale_input() / 100
        # text = self.text_input()
        # glBindTexture(GL_TEXTURE_2D, self.texture)
        #
        # for c in text:
        #     ch = self.characters[c]
        #     width = self.glyph_shape[0] * scale
        #     height = self.glyph_shape[1] * scale
        #     vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
        #
        #     glBegin(GL_TRIANGLES)
        #     for i in range(6):
        #         glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
        #         glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
        #     glEnd()
        #
        #     pos[0] += ((ch.advance >> 6) * scale)
        # glBindBuffer(GL_ARRAY_BUFFER, self.text_buffer)
        # glBufferData(GL_ARRAY_BUFFER, self.coords.size * self.coords.itemsize, self.coords.data, GL_DYNAMIC_DRAW)
        # glBindBuffer(GL_ARRAY_BUFFER, 0)
        #
        # glDrawArrays(GL_TRIANGLES, 0, self.coords.shape[0] * 6)

        glBindTexture(GL_TEXTURE_2D, 0)

        glColor4f(1.0, 1.0, 1.0, 1.0)


    def get_rendering_buffer(self, xpos, ypos, width, height, texture_coords, zfix=0.):
        return np.asarray([
            xpos, ypos - height, texture_coords[0], texture_coords[3],
            xpos, ypos, texture_coords[0], texture_coords[1],
            xpos + width, ypos, texture_coords[2], texture_coords[1],
            xpos, ypos - height, texture_coords[0], texture_coords[3],
            xpos + width, ypos, texture_coords[2], texture_coords[1],
            xpos + width, ypos - height, texture_coords[2], texture_coords[3]
        ], np.float32)

    # def get_rendering_buffer(self, xpos, ypos, width, height, texture_coords, zfix=0.):
    #     return np.asarray([
    #         xpos, ypos - height, texture_coords[0], texture_coords[3],
    #         xpos, ypos, texture_coords[0], texture_coords[1],
    #         # xpos, ypos - height, texture_coords[0], texture_coords[3],
    #         # xpos + width, ypos, texture_coords[2], texture_coords[1],
    #         xpos + width, ypos - height, texture_coords[2], texture_coords[3],
    #         xpos + width, ypos, texture_coords[2], texture_coords[1]
    #     ], np.float32)

class GLKoreanTextNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLKoreanTextNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ready = False
        self.new_text = False
        self.display_list = -1
        self.characters = {}
        self.initialized = False
        self.font_size = 6
        self.coords = None
        self.text_buffer = -1
        self.text_vertex_buffer = -1
        self.color = [1.0, 1.0, 1.0, 1.0]
        self.font_path = "Inconsolata-g.otf"
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [float, int]:
                self.font_size = v
            elif t == str:
                self.font_path = v
        self.face = None

        self.text_input = self.add_input('text', callback=self.text_changed)
        self.position_x_input = self.add_input('position_x', widget_type='drag_float', default_value=0.0, callback=self.text_changed)
        self.position_y_input = self.add_input('position_y', widget_type='drag_float', default_value=0.0, callback=self.text_changed)
        self.text_alpha_input = self.add_input('alpha', widget_type='drag_float', default_value=1.0, callback=self.text_changed)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.text_changed)
        self.text_font = self.add_string_input('font', widget_type='text_input', default_value=self.font_path, callback=self.font_changed)
        self.text_color = self.add_option('alpha', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0], callback=self.color_changed)
        self.text_size = self.add_option('size', widget_type='drag_int', default_value=self.font_size, callback=self.size_changed)
        self.alpha_power = self.add_option('alpha power', widget_type='drag_float', default_value=1.0)
        self.separator = self.add_option('separator', widget_type='text_input', default_value=' ')

        self.ready = True
        self.context = None
        self.texture = -1
        self.was_lit = False
        self.was_depth = False

    def custom_create(self, from_file):
        dpg.configure_item(self.text_color.widget.uuid, no_alpha=True)
        dpg.configure_item(self.text_color.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)

    def text_changed(self):
        self.new_text = True

    def color_changed(self):
        self.color = self.text_color()
        self.color[0] /= 255.0
        self.color[1] /= 255.0
        self.color[2] /= 255.0

    def size_changed(self):
        size = self.text_size()
        if size != self.font_size:
            self.font_size = size
            self.initialized = False
            self.new_text = True

    def font_changed(self):
        path = self.text_font()
        if self.font_path != path:
            self.font_path = path
            self.initialized = False
            self.new_text = True

    def update_font(self):
        print('update font')
        hold_context = glfw.get_current_context()
        glfw.make_context_current(self.context)

        self.ready = False
        if self.texture != -1:
            glDeleteTextures(1, self.texture)
        self.characters = {}
        if self.face is not None:
            del self.face
        self.face = freetype.Face(self.font_path)
        size = self.font_size * 256.0
        self.face.set_char_size(int(size))
        self.create_glyph_textures()
        self.ready = True
        glfw.make_context_current(hold_context)

    def create_glyph_textures(self):
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        bottom = 1000
        top = 0
        left = 1000
        right = 0
        printable_count = 0

        for i in range(0xAC00, 0xD7A3):
            a = chr(i)
            if a.isprintable():
                printable_count += 1
                self.face.load_char(chr(i))
                glyph = self.face.glyph
                if glyph.bitmap_top > top:
                    top = glyph.bitmap_top
                if glyph.bitmap_left < left:
                    left = glyph.bitmap_left
                if glyph.bitmap.width + glyph.bitmap_left > right:
                    right = glyph.bitmap.rows + glyph.bitmap_left
                if glyph.bitmap_top - glyph.bitmap.rows < bottom:
                    bottom = glyph.bitmap_top - glyph.bitmap.rows

        self.glyph_shape = [0, 0]
        self.glyph_shape[0] = right - left
        self.glyph_shape[1] = top - bottom

        font_atlas_shape = [self.glyph_shape[0] * 16, self.glyph_shape[1] * 16]
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, font_atlas_shape[0], font_atlas_shape[1], 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        for i in range(0xAC00, 0xD7A3):
            a = chr(i)
            if a.isprintable():
                self.face.load_char(chr(i))
                glyph = self.face.glyph
                bm = glyph.bitmap.buffer
                rgb_bm = [0.0] * (glyph.bitmap.rows * glyph.bitmap.width * 4)
                for k in range(glyph.bitmap.rows):
                    for j in range(glyph.bitmap.width):
                        rgb_bm[(k * glyph.bitmap.width + j) * 4] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 1] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 2] = 1.0
                        rgb_bm[(k * glyph.bitmap.width + j) * 4 + 3] = float(bm[k * glyph.bitmap.width + j]) / 255.0

                cell = [i % 128, i // 128]
                a = cell[0] * self.glyph_shape[0]
                b = cell[1] * self.glyph_shape[1]
                glyph_offset = [cell[0] * self.glyph_shape[0], cell[1] * self.glyph_shape[1]]
                glTexSubImage2D(GL_TEXTURE_2D, 0, glyph_offset[0] + glyph.bitmap_left, glyph_offset[1] + (self.glyph_shape[1] - glyph.bitmap_top) + bottom, glyph.bitmap.width, glyph.bitmap.rows, GL_RGBA, GL_FLOAT, rgb_bm)
                self.characters[chr(i)] = CharacterSlot(cell, self.glyph_shape, glyph, font_atlas_shape)

        glBindTexture(GL_TEXTURE_2D, 0)

    def remember_state(self):
        self.was_lit = glGetBoolean(GL_LIGHTING)
        if self.was_lit:
            glDisable(GL_LIGHTING)
        self.was_depth = glGetBoolean(GL_DEPTH_TEST)
        if self.was_depth:
            glDisable(GL_DEPTH_TEST)
        glPushMatrix()

    def restore_state(self):
        glPopMatrix()
        if self.was_lit:
            glEnable(GL_LIGHTING)
        if self.was_depth:
            glEnable(GL_DEPTH_TEST)

    def render_text(self):
        if self.initialized:
            self.new_text = False
            if self.display_list != -1:
                glDeleteLists(self.display_list, 1)
            self.display_list = glGenLists(1)

            pos = [self.position_x_input(), self.position_y_input()]
            scale = self.scale_input() / 100
            text = self.text_input()

            if type(text) == str:
                glNewList(self.display_list, GL_COMPILE)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_TEXTURE_2D)

                # self.coords = np.ndarray((len(text), 24))
                for index, c in enumerate(text):
                    if c in self.characters:
                        ch = self.characters[c]
                        width = self.glyph_shape[0] * scale
                        height = self.glyph_shape[1] * scale
                        vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
                        glBegin(GL_TRIANGLES)
                        for i in range(6):
                            glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
                            glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
                        glEnd()
                        # self.coords[index] = vertices.copy()
                        pos[0] += ((ch.advance >> 6) * scale)

                glEndList()
            elif type(text) == list:
                alpha_power = self.alpha_power()
                separator = self.separator()

                glNewList(self.display_list, GL_COMPILE)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_TEXTURE_2D)
                for fragment in text:
                    this_text = ''
                    if type(fragment) == list:
                        this_alpha = fragment[1]

                        if this_alpha > 0.0:
                            this_alpha = pow(this_alpha, alpha_power)
                            this_text = fragment[0] + separator
                            glColor4f(self.color[0], self.color[1], self.color[2], self.text_alpha_input() * this_alpha)

                    elif type(fragment) == str:
                        this_text = fragment + separator
                    if len(this_text) > 0:
                        for index, c in enumerate(this_text):
                            if c in self.characters:
                                ch = self.characters[c]
                                width = self.glyph_shape[0] * scale
                                height = self.glyph_shape[1] * scale
                                vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
                                glBegin(GL_TRIANGLES)
                                for i in range(6):
                                    glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
                                    glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
                                glEnd()
                                # self.coords[index] = vertices.copy()
                                pos[0] += ((ch.advance >> 6) * scale)

                glEndList()

    # figure out drawing the text using drawlists or vertex_arrays
    def draw(self):
        if not self.ready:
            return
        if not self.initialized:
            self.context = glfw.get_current_context()
            self.update_font()
            self.initialized = True
        if self.new_text:
            self.render_text()
        # if self.coords is None:
        #     self.render_text()
        # if self.text_buffer == -1:
        #     self.text_buffer = glGenBuffers(1)
        #     self.text_vertex_buffer = glGenVertexArrays(1)
        #     glBindVertexArray(self.text_vertex_buffer)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTranslatef(0, 0, -2)

        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_TEXTURE_2D)
        glColor4f(self.color[0], self.color[1], self.color[2], self.text_alpha_input())
        if self.display_list != -1:
            glCallList(self.display_list)
        # pos = [self.position_x_input(), self.position_y_input()]
        # scale = self.scale_input() / 100
        # text = self.text_input()
        # glBindTexture(GL_TEXTURE_2D, self.texture)
        #
        # for c in text:
        #     ch = self.characters[c]
        #     width = self.glyph_shape[0] * scale
        #     height = self.glyph_shape[1] * scale
        #     vertices = self.get_rendering_buffer(pos[0], pos[1], width, height, ch.texture_coords)
        #
        #     glBegin(GL_TRIANGLES)
        #     for i in range(6):
        #         glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
        #         glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
        #     glEnd()
        #
        #     pos[0] += ((ch.advance >> 6) * scale)
        # glBindBuffer(GL_ARRAY_BUFFER, self.text_buffer)
        # glBufferData(GL_ARRAY_BUFFER, self.coords.size * self.coords.itemsize, self.coords.data, GL_DYNAMIC_DRAW)
        # glBindBuffer(GL_ARRAY_BUFFER, 0)
        #
        # glDrawArrays(GL_TRIANGLES, 0, self.coords.shape[0] * 6)

        glBindTexture(GL_TEXTURE_2D, 0)

        glColor4f(1.0, 1.0, 1.0, 1.0)


    def get_rendering_buffer(self, xpos, ypos, width, height, texture_coords, zfix=0.):
        return np.asarray([
            xpos, ypos - height, texture_coords[0], texture_coords[3],
            xpos, ypos, texture_coords[0], texture_coords[1],
            xpos + width, ypos, texture_coords[2], texture_coords[1],
            xpos, ypos - height, texture_coords[0], texture_coords[3],
            xpos + width, ypos, texture_coords[2], texture_coords[1],
            xpos + width, ypos - height, texture_coords[2], texture_coords[3]
        ], np.float32)

    # def get_rendering_buffer(self, xpos, ypos, width, height, texture_coords, zfix=0.):
    #     return np.asarray([
    #         xpos, ypos - height, texture_coords[0], texture_coords[3],
    #         xpos, ypos, texture_coords[0], texture_coords[1],
    #         # xpos, ypos - height, texture_coords[0], texture_coords[3],
    #         # xpos + width, ypos, texture_coords[2], texture_coords[1],
    #         xpos + width, ypos - height, texture_coords[2], texture_coords[3],
    #         xpos + width, ypos, texture_coords[2], texture_coords[1]
    #     ], np.float32)


class GLXYZDiskNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLXYZDiskNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        # self.scale = self.arg_as_float(default_value=1.0)

        # self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.scale = self.add_input('gl chain in', widget_type='drag_float', default_value=1.0)
        self.quat = self.add_input('quaternion in', callback=self.set_quaternion)
        # self.gl_output = self.add_output('gl chain out')

        self.inner_radius = 0.0
        self.slices = 32
        self.rings = 1
        self.degree_factor = 180.0 / math.pi
        self.add_shading_option()
        self.size_x = 0.0
        self.size_y = 0.0
        self.size_z = 0.0

    def set_quaternion(self):
        data = any_to_array(self.quat())
        if data.shape[-1] % 4 == 0:
            # q = quaternion.as_quat_array(data)
            # euler = quaternion.as_euler_angles(q) * self.degree_factor
            scale = self.scale()
            self.size_x = data[1] * scale
            self.size_y = data[2] * scale
            self.size_z = data[3] * scale

    def quadric_draw(self):
        # hold_ambient_material = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT)
        # hold_diffuse_material = gl.glGetMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE)
        glPushMatrix()
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.0, 0.0, 0.5])
        gluDisk(self.quadric, self.inner_radius, self.size_x, self.slices, self.rings)
        glRotatef(90.0, 0.0, 1.0, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0, 0.0, 1.0, 0.5])
        gluDisk(self.quadric, self.inner_radius, self.size_y, self.slices, self.rings)
        glRotatef(90.0, 1.0, 0.0, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0, 1.0, 0.0, 0.5])
        gluDisk(self.quadric, self.inner_radius, self.size_z, self.slices, self.rings)
        glPopMatrix()
        # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, hold_ambient_material)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, hold_diffuse_material)


class GLButtonGridNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLButtonGridNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.display_list = -1
        self.initialized = False
        self.new_selection = True
        self.selection = 0
        self.selection_input = self.add_input('selection', widget_type='drag_int', default_value=self.selection, callback=self.display_changed)
        self.position_x_input = self.add_input('position_x', widget_type='drag_float', default_value=0.0,
                                               callback=self.display_changed)
        self.position_y_input = self.add_input('position_y', widget_type='drag_float', default_value=0.0,
                                               callback=self.display_changed)
        self.spacing_input = self.add_input('spacing', widget_type='drag_float', default_value=0.20, callback=self.display_changed)
        self.line_thickness_input = self.add_input('thickness', widget_type='drag_int', default_value=4,
                                            callback=self.display_changed)

        self.text_alpha_input = self.add_input('alpha', widget_type='drag_float', default_value=1.0,
                                               callback=self.display_changed)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=.1,
                                          callback=self.display_changed)
        # self.gl_output = self.add_output('gl chain out')


    def display_changed(self):
        self.selection = self.selection_input()
        self.new_selection = True

    def create_call_list(self, which_highlight):
        self.space = self.spacing_input()
        if self.display_list != -1:
            glDeleteLists(self.display_list, 1)
        self.display_list = glGenLists(1)

        pos = [self.position_x_input(), self.position_y_input()]
        scale = self.scale_input()

        glNewList(self.display_list, GL_COMPILE)

        glEnable(GL_BLEND)
        glLineWidth(self.line_thickness_input())
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor4f(1.0, 1.0, 1.0, self.text_alpha_input())
        glBegin(GL_QUADS)

        for row in range(4):
            for column in range(4):
                button_id = row * 4 + column
                fill = False
                if which_highlight == button_id:
                    fill = True
                if fill:
                    glColor4f(1.0, 0.0, 0.0, self.text_alpha_input())
                else:
                    glColor4f(0.5, 0.5, 0.5, self.text_alpha_input())
                glVertex2f(column * scale + pos[0], row * scale + pos[1])
                glVertex2f(column * scale + pos[0], (row + 1) * scale - scale * self.space + pos[1])
                glVertex2f((column + 1) * scale - scale * self.space + pos[0], (row + 1) * scale - scale * self.space + pos[1])
                glVertex2f((column + 1) * scale - scale * self.space + pos[0], row * scale + pos[1])

        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glEndList()

    def draw(self):
        if self.new_selection:
            self.create_call_list(self.selection)
            self.new_selection = False

        glTranslatef(0, 0, -2)
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, self.text_alpha_input())
        if self.display_list != -1:
            glCallList(self.display_list)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


class GLNumpyLines(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLNumpyLines(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.array_input = self.add_input('array', triggers_execution=True)
        self.alpha_fade = self.add_bool_input('alpha_fade', widget_type='checkbox', default_value=True)
        self.motion_accent = self.add_input('accent_motion', widget_type='checkbox', default_value=False)
        self.accent_color = self.add_input('accent_colour', widget_type='checkbox', default_value=False)
        self.accent_scale = self.add_input('accent_scale', widget_type='drag_int', default_value=50)
        self.line_width = self.add_input('line_width', widget_type='drag_int', default_value=1)
        self.selected_joints = self.add_input('selected_joints', widget_type='text_input', default_value='')
        # self.gl_output = self.add_output('gl chain out')

        self.colors = []
        self.previous_array = None
        color = [1.0, 1.0, 1.0, 1.0]
        for i in range(20):
            self.colors.append(color.copy())
        # self.colors_input = self.add_input('colors', triggers_execution=True)
        self.color_index = self.add_input('color index', widget_type='input_int', default_value=0)
        self.color_control = self.add_input('color_control', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0], callback=self.color_changed)
        self.line_array = None
        self.new_array = False
        self.motion_array = None



    def custom_create(self, from_file):
        dpg.configure_item(self.color_control.widget.uuid, no_alpha=True)
        dpg.configure_item(self.color_control.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)

    def color_changed(self):
        index = self.color_index()
        color = self.color_control()
        if index == -1:
            for i in range(20):
                self.colors[i][0] = color[0] / 255.0
                self.colors[i][1] = color[1] / 255.0
                self.colors[i][2] = color[2] / 255.0
                self.colors[i][3] = 1.0
            return
        if 0 >= index >= len(self.colors):
            index = len(self.colors) - 1
        color = self.color_control()
        self.colors[index][0] = color[0] / 255.0
        self.colors[index][1] = color[1] / 255.0
        self.colors[index][2] = color[2] / 255.0
        self.colors[index][3] = 1.0

    def remember_state(self):
        self.was_lit = glGetBoolean(GL_LIGHTING)
        if self.was_lit:
            glDisable(GL_LIGHTING)
        self.was_depth = glGetBoolean(GL_DEPTH_TEST)
        if self.was_depth:
            glDisable(GL_DEPTH_TEST)
        glPushMatrix()

    def restore_state(self):
        glPopMatrix()
        if self.was_lit:
            glEnable(GL_LIGHTING)
        if self.was_depth:
            glEnable(GL_DEPTH_TEST)

    def execute(self):
        if self.active_input == self.array_input:
            incoming = any_to_array(self.array_input())
            self.line_array = incoming.copy()

            accent_scale = self.accent_scale()
            if self.motion_accent():
                if self.previous_array is not None:
                    self.motion_array = np.linalg.norm(self.line_array - self.previous_array, axis=2) * accent_scale
                self.previous_array = self.line_array.copy()
            else:
                self.previous_array = None
                self.motion_array = None
        elif self.active_input == self.gl_input:
            super().execute()

    def draw(self):
        if self.line_array is not None:
            number_of_lines = self.line_array.shape[1]
            number_of_points = self.line_array.shape[0]
            select_list = any_to_list(self.selected_joints())
            if len(select_list) == 0:
                selected = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            else:
                selected = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
                for select in select_list:
                    select_int = any_to_int(select)
                    if 0 <= select_int < number_of_points:
                        selected[select_int] = True
            accent_scale = self.accent_scale()

            gl.glLineWidth(self.line_width())
            for i in range(number_of_lines):
                if selected[i]:
                    color = self.colors[i]
                    gl.glBegin(GL_LINE_STRIP)
                    for j in range(number_of_points):
                        alpha = 1.0
                        if self.motion_accent() and self.motion_array is not None:
                            alpha = self.motion_array[j, i]
                            # if self.previous_array is not None:
                            #     motion = np.linalg.norm(self.line_array - self.previous_array) * accent_scale
                            # if j > 0:
                            #     alpha = np.linalg.norm(self.line_array[j, i] - self.line_array[j - 1, i]) * accent_scale
                            # else:
                            #     alpha = 0.0
                            if alpha > 1.0:
                                alpha = 1.0
                            if self.accent_color():
                                color = _viridis_data[int(alpha * 255.0)]
                                alpha = 1.0
                        if self.alpha_fade():
                            alpha = (number_of_points - j) / number_of_points * alpha
                        gl.glColor4f(color[0], color[1], color[2], alpha)
                        gl.glVertex(self.line_array[j, i])
                    gl.glEnd()


