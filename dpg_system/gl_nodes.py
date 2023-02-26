import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import freetype
from dpg_system.open_gl_base import *
from dpg_system.glfw_base import *

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

    def set_frustum(self, context, args):  # context must be established
        if args is not None and len(args) > 0:
            near = .1
            far = 1000
            focal_length = 2.0
            if len(args) > 0:
                focal_length = any_to_float(args[0])
            if len(args) > 1:
                near = any_to_float(args[1])
            if len(args) > 2:
                far = any_to_float(args[2])
            width = context.width
            height = context.height
            height_over_width = float(height) / float(width)
            f = near / focal_length
            h = height_over_width * f
            current_matrix_mode = gl.glGetInteger(gl.GL_MATRIX_MODE)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glFrustum(-f, f, -h, h, near, far)
            gl.glMatrixMode(current_matrix_mode)

    def set_ortho(self, context, args):  # context must be established
        if args is not None and len(args) > 3:
            near = .1
            far = 1000
            dest_rect = [any_to_float(args[0]), any_to_float(args[1]), any_to_float(args[2]), any_to_float(args[3])]
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

    def set_perspective(self, context, args):  # context must be established
        aspect = context.width / context.height
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
            p.create()
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
        self.ready = False
        self.fov = 60.0
        self.command_parser = GLContextCommandParser()
        self.pending_commands = []

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == str:
                self.title = val
            elif t in [int, float]:
                if self.width == 0:
                    self.width = val
                else:
                    self.height = val
        if self.width == 0:
            self.width = 640
        if self.height == 0:
            self.height = 480

        self.pending_contexts.append(self)
        self.context = None
        self.command_input = self.add_input('commands', triggers_execution=True)
        self.output = self.add_output('gl_chain')
        self.fov_option = self.add_option('fov', widget_type='drag_float', default_value=self.fov, callback=self.fov_changed)
        self.fov_option.widget.speed = 0.5

    def execute(self):
        if self.command_input.fresh_input:
            data = self.command_input.get_received_data()
            if type(data) == list:
                self.pending_commands.append(data)

    def fov_changed(self):
        self.fov = self.fov_option.get_widget_value()
        if self.fov < 1:
            self.fov = 1
        elif self.fov > 180:
            self.fov = 180
        if self.context:
            self.context.set_fov(self.fov)

    def create(self):
        self.context = MyGLContext(self.title, self.width, self.height)
        self.context_list.append(self)
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
            if len(self.pending_commands) > 0:
                for command in self.pending_commands:
                    self.command_parser.perform(command[0], self.context, command[1:])
            self.pending_commands = []
            self.context.prepare_draw()
            self.output.send('draw')
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
            input_list = self.gl_input.get_received_data()
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


class GLQuadricCommandParser(GLCommandParser):
    def __init__(self):
        super().__init__()
        self.dict['style'] = self.set_style
        # self.dict['frustum'] = self.set_frustum
        # self.dict['perspective'] = self.set_perspective

    def set_style(self, quadric, args):
        if args is not None and len(args) > 0:
            mode = any_to_string(args[0])
            if mode == 'fill':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            elif mode == 'line':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            elif mode == 'point':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_POINT)


class GLQuadricNode(GLNode):
    def __init__(self, label: str, data, args):
        self.quadric = gluNewQuadric()
        self.shading_option = None
        self.command_parser = GLQuadricCommandParser()
        self.pending_commands = []
        super().__init__(label, data, args)

    def process_pending_commands(self):
        if len(self.pending_commands) > 0:
            for command in self.pending_commands:
                self.command_parser.perform(command[0], self.quadric, command[1:])
        self.pending_commands = []

    def shading_changed(self):
        shading = self.shading_option.get_widget_value()
        if shading == 'flat':
            self.shading = glu.GLU_FLAT
        elif shading == 'smooth':
            self.shading = glu.GLU_SMOOTH
        else:
            self.shading = glu.GLU_NONE
        glu.gluQuadricNormals(self.quadric, self.shading)

    def add_shading_option(self):
        self.shading_option = self.add_option('shading', widget_type='combo', default_value='smooth', callback=self.shading_changed)
        self.shading_option.widget.combo_items = ['none', 'flat', 'smooth']

    def handle_other_messages(self, message):
        if type(message) == list:
            self.pending_commands.append(message)


class GLSphereNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLSphereNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        self.size = self.arg_as_float(default_value=0.5)
        self.slices = self.arg_as_int(index=1, default_value=32)
        self.stacks = self.arg_as_int(index =2, default_value=32)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.size_property = self.add_property('size', widget_type='drag_float', default_value=self.size)
        self.gl_output = self.add_output('gl chain out')

        self.slices_option = self.add_option('slices', widget_type='drag_int', default_value=self.slices, callback=self.options_changed)
        self.stacks_option = self.add_option('stacks', widget_type='drag_int', default_value=self.stacks, callback=self.options_changed)
        self.add_shading_option()

    def options_changed(self):
        self.shading_changed()
        self.slices = self.slices_option.get_widget_value()
        self.stacks = self.stacks_option.get_widget_value()

    def draw(self):
        self.process_pending_commands()
        self.size = self.size_property.get_widget_value()
        gluSphere(self.quadric, self.size, self.slices, self.stacks)


class GLDiskNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLDiskNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        self.inner_radius = self.arg_as_float(default_value=0.0)
        self.outer_radius = self.arg_as_float(index=1, default_value=0.5)
        self.slices = self.arg_as_int(index=2, default_value=32)
        self.rings = self.arg_as_int(index=3, default_value=1)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.outer_radius_property = self.add_property('outer radius', widget_type='drag_float', default_value=self.outer_radius, callback=self.outer_radius_changed)
        self.gl_output = self.add_output('gl chain out')

        self.inner_radius_option = self.add_option('inner radius', widget_type='drag_float', default_value=self.inner_radius, callback=self.options_changed)
        self.slices_option = self.add_option('slices', widget_type='drag_int', default_value=self.slices, callback=self.options_changed)
        self.rings_option = self.add_option('rings', widget_type='drag_int', default_value=self.rings, callback=self.options_changed)
        self.add_shading_option()

    def outer_radius_changed(self):
        self.outer_radius = self.outer_radius_property.get_widget_value()

    def options_changed(self):
        self.shading_changed()
        self.inner_radius = self.inner_radius_option.get_widget_value()
        self.slices = self.slices_option.get_widget_value()
        self.rings = self.rings_option.get_widget_value()

    def draw(self):
        self.process_pending_commands()
        gluDisk(self.quadric, self.inner_radius, self.outer_radius, self.slices, self.rings)


class GLPartialDiskNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLPartialDiskNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        self.inner_radius = self.arg_as_float(default_value=0.0)
        self.outer_radius = self.arg_as_float(index=1, default_value=0.5)
        self.slices = self.arg_as_int(index=2, default_value=32)
        self.rings = self.arg_as_int(index=3, default_value=1)
        self.start_angle = self.arg_as_float(index=4, default_value=0.0)
        self.sweep_angle = self.arg_as_float(index=5, default_value=90)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.outer_radius_property = self.add_property('outer radius', widget_type='drag_float', default_value=self.outer_radius, callback=self.outer_radius_changed)
        self.gl_output = self.add_output('gl chain out')

        self.inner_radius_option = self.add_option('inner radius', widget_type='drag_float', default_value=self.inner_radius, callback=self.options_changed)
        self.slices_option = self.add_option('slices', widget_type='drag_int', default_value=self.slices, callback=self.options_changed)
        self.rings_option = self.add_option('rings', widget_type='drag_int', default_value=self.rings, callback=self.options_changed)
        self.start_angle_option = self.add_option('start angle', widget_type='drag_float', default_value=self.start_angle, callback=self.options_changed)
        self.start_angle_option.widget.speed = 1
        self.sweep_angle_option = self.add_option('sweep angle', widget_type='drag_float', default_value=self.sweep_angle, callback=self.options_changed)
        self.sweep_angle_option.widget.speed = 1
        self.add_shading_option()

    def outer_radius_changed(self):
        self.outer_radius = self.outer_radius_property.get_widget_value()

    def options_changed(self):
        self.shading_changed()
        self.inner_radius = self.inner_radius_option.get_widget_value()
        self.slices = self.slices_option.get_widget_value()
        self.rings = self.rings_option.get_widget_value()
        self.start_angle = self.start_angle_option.get_widget_value()
        self.sweep_angle = self.sweep_angle_option.get_widget_value()

    def draw(self):
        self.process_pending_commands()
        gluPartialDisk(self.quadric, self.inner_radius, self.outer_radius, self.slices, self.rings, self.start_angle, self.sweep_angle)


class GLCylinderNode(GLQuadricNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLCylinderNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        self.base_radius = self.arg_as_float(default_value=0.5)
        self.top_radius = self.arg_as_float(index=1, default_value=0.5)
        self.height = self.arg_as_float(index=2, default_value=0.5)
        self.slices = self.arg_as_int(index=3, default_value=40)
        self.stacks = self.arg_as_int(index=4, default_value=1)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.gl_output = self.add_output('gl chain out')

        self.base_radius_option = self.add_option('base radius', widget_type='drag_float', default_value=self.base_radius, callback=self.options_changed)
        self.top_radius_option = self.add_option('top radius', widget_type='drag_float', default_value=self.top_radius, callback=self.options_changed)
        self.height_option = self.add_option('height', widget_type='drag_float', default_value=self.height, callback=self.options_changed)
        self.slices_option = self.add_option('slices', widget_type='drag_int', default_value=self.slices, callback=self.options_changed)
        self.stacks_option = self.add_option('stacks', widget_type='drag_int', default_value=self.stacks, callback=self.options_changed)
        self.add_shading_option()

    def options_changed(self):
        self.shading_changed()
        self.base_radius = self.base_radius_option.get_widget_value()
        self.top_radius = self.top_radius_option.get_widget_value()
        self.height = self.height_option.get_widget_value()
        self.slices = self.slices_option.get_widget_value()
        self.stacks = self.stacks_option.get_widget_value()

    def draw(self):
        self.process_pending_commands()
        gluCylinder(self.quadric, self.base_radius, self.top_radius, self.height, self.slices, self.stacks)


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
        self.show_axis = False
        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.quat_input = self.add_input('quaternion')
        self.gl_output = self.add_output('gl chain out')
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
        input_ = self.quat_input.get_received_data()
        t = type(input_)
        if t == list:
            input_ = np.array(input_)
            t = np.ndarray
        if t == np.ndarray:
            rotation_q = Quaternion(input_)
            if self.show_axis_option.get_widget_value():
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


class GLTransformNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLTransformNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        self.values = [0.0, 0.0, 0.0]
        self.values[0] = self.arg_as_float(default_value=0.0)
        self.values[1] = self.arg_as_float(index=1, default_value=0.0)
        self.values[2] = self.arg_as_float(index=2, default_value=0.0)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.x_input = self.add_input('x', widget_type='drag_float', default_value=self.values[0])
        self.y_input = self.add_input('y', widget_type='drag_float', default_value=self.values[1])
        self.z_input = self.add_input('z', widget_type='drag_float', default_value=self.values[2])
        if self.label == 'gl_rotate':
            self.x_input.widget.speed = 1
            self.y_input.widget.speed = 1
            self.z_input.widget.speed = 1
        self.reset_button = self.add_property('reset', widget_type='button', callback=self.reset)
        self.gl_output = self.add_output('gl chain out')

    def reset(self):
        if self.label in ['gl_translate', 'gl_rotate']:
            self.x_input.set(0.0)
            self.y_input.set(0.0)
            self.z_input.set(0.0)
        elif self.label == 'gl_scale':
            self.x_input.set(1.0)
            self.y_input.set(1.0)
            self.z_input.set(1.0)

    def remember_state(self):
        gl.glPushMatrix()

    def restore_state(self):
        gl.glPopMatrix()

    def draw(self):
        self.values[0] = self.x_input.get_widget_value()
        self.values[1] = self.y_input.get_widget_value()
        self.values[2] = self.z_input.get_widget_value()

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

    def save_custom_setup(self, container):
        container['ambient'] = list(self.material.ambient)
        container['diffuse'] = list(self.material.diffuse)
        container['specular'] = list(self.material.specular)
        container['emission'] = list(self.material.emission)

    def load_custom_setup(self, container):
        if 'ambient' in container:
            self.material.ambient = container['ambient']
        if 'diffuse' in container:
            self.material.diffuse = container['diffuse']
        if 'specular' in container:
            self.material.specular = container['specular']
        if 'emission' in container:
            self.material.emission = container['emission']

    def initialize(self, args):
        self.material = GLMaterial()
        self.material.ambient = [0.2, 0.2, 0.2, 1.0]
        self.material.diffuse = [0.8, 0.8, 0.8, 1.0]
        self.material.specular = [0.0, 0.0, 0.0, 1.0]
        self.material.emission = [0.0, 0.0, 0.0, 1.0]
        self.material.shininess = 0.0

        self.hold_material = GLMaterial()

        self.create_material_presets()

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.ambient_input = self.add_input('ambient')
        self.diffuse_input = self.add_input('diffuse')
        self.specular_input = self.add_input('specular')
        self.emission_input = self.add_input('emission')
        self.shininess_input = self.add_input('shininess', widget_type='drag_float', default_value=self.material.shininess, callback=self.shininess_changed)
        self.preset_menu = self.add_property('presets', widget_type='combo', callback=self.preset_selected)
        presets = list(self.presets.keys())
        self.preset_menu.widget.combo_items = presets
        self.gl_output = self.add_output('gl chain out')

    def preset_selected(self):
        selected_preset = self.preset_menu.get_widget_value()
        if selected_preset in self.presets:
            p = self.presets[selected_preset]
            self.material.ambient = p.ambient
            self.material.diffuse = p.diffuse
            self.material.specular = p.specular
            self.material.emission = p.emission
            self.material.shininess = p.shininess

    def shininess_changed(self, input=None):
        self.material.shininess = self.shininess_input.get_widget_value()

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
            ambient = self.ambient_input.get_received_data()
            t = type(ambient)
            if t == np.ndarray:
                if ambient.shape[0] == 4:
                    self.material.ambient = ambient
            elif t == list:
                if len(ambient) == 4:
                    self.material.ambient = ambient
                elif len(ambient) == 3:
                    self.material.ambient = ambient + [1.0]
            elif t in [float, np.float, np.double]:
                self.material.ambient = [ambient, ambient, ambient, 1.0]

        if self.diffuse_input.fresh_input:
            diffuse = self.diffuse_input.get_received_data()
            t = type(diffuse)
            if t == np.ndarray:
                if diffuse.shape[0] == 4:
                    self.material.diffuse = diffuse
            if t == list:
                if len(diffuse) == 4:
                    self.material.diffuse = diffuse
                elif len(diffuse) == 3:
                    self.material.diffuse = diffuse + [1.0]
            elif t in [float, np.float, np.double]:
                self.material.diffuse = [diffuse, diffuse, diffuse, 1.0]

        if self.specular_input.fresh_input:
            specular = self.specular_input.get_received_data()
            t = type(specular)
            if t == np.ndarray:
                if specular.shape[0] == 4:
                    self.material.specular = specular
            if t == list:
                if len(specular) == 4:
                    self.material.specular = specular
                elif len(specular) == 3:
                    self.material.specular = specular + [1.0]
            elif t in [float, np.float, np.double]:
                self.material.specular = [specular, specular, specular, 1.0]

        if self.emission_input.fresh_input:
            emission = self.emission_input.get_received_data()
            t = type(emission)
            if t == np.ndarray:
                if emission.shape[0] == 4:
                    self.material.emission = emission
            if t == list:
                if len(emission) == 4:
                    self.material.emission = emission
                elif len(emission) == 3:
                    self.material.emission = emission + [1.0]
            elif t in [float, np.float, np.double]:
                self.material.emission = [emission, emission, emission, 1.0]

        if self.shininess_input.fresh_input:
            shininess = self.shininess_input.get_received_data()
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


class GLAlignNode(GLNode):
    @staticmethod
    def factory(node_name, data, args=None):
        node = GLAlignNode(node_name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.ready = False
        super().__init__(label, data, args)

    def initialize(self, args):
        self.axis = np.array([0.0, 1.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        if args is not None:
            float_count = 0
            for i in range(len(args)):
                val, t = decode_arg(args, i)
                if float_count < 3:
                    self.axis[float_count] = val
                float_count += 1

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.x_input = self.add_input('x', widget_type='drag_float', default_value=self.axis[0], callback=self.axis_changed)
        self.y_input = self.add_input('y', widget_type='drag_float', default_value=self.axis[1], callback=self.axis_changed)
        self.z_input = self.add_input('z', widget_type='drag_float', default_value=self.axis[2], callback=self.axis_changed)
        self.gl_output = self.add_output('gl chain out')
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

    def axis_changed(self, input=None):
        self.axis[0] = self.x_input.get_widget_value()
        self.axis[1] = self.y_input.get_widget_value()
        self.axis[2] = self.z_input.get_widget_value()
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
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)
        self.origin = [0, 0]

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
            self.origin = [glyph.bitmap_left, glyph.bitmap.rows - glyph.bitmap_top]
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
            self.origin = [glyph.bitmap_left, glyph.bitmap.rows - glyph.bitmap_top]
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
        self.characters = {}
        self.initialized = False
        self.font_size = 24
        self.color = [1.0, 1.0, 1.0, 1.0]
        self.font_path = "Inconsolata-g.otf"
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [float, int]:
                self.font_size = v
            elif t == str:
                self.font_path = v
        self.face = None
        # self.face = freetype.Face(self.font_path)
        # size = self.font_size * 256.0
        # self.face.set_char_size(int(size))

        self.text_input = self.add_input('text', widget_type='text_input', default_value='text')
        self.position_x_input = self.add_input('position_x', widget_type='drag_float', default_value=0.0)
        self.position_y_input = self.add_input('position_y', widget_type='drag_float', default_value=0.0)
        self.text_alpha_input = self.add_input('alpha', widget_type='drag_float', default_value=1.0)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0)
        self.text_color = self.add_option('alpha', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0], callback=self.color_changed)
        self.text_font = self.add_option('font', widget_type='text_input', default_value=self.font_path, callback=self.font_changed)
        self.text_size = self.add_option('size', widget_type='drag_int', default_value=self.font_size, callback=self.size_changed)
        self.ready = True
        self.context = None

    def custom_setup(self, from_file):
        dpg.configure_item(self.text_color.widget.uuid, no_alpha=True)
        dpg.configure_item(self.text_color.widget.uuid, alpha_preview=dpg.mvColorEdit_AlphaPreviewNone)

    def color_changed(self):
        self.color = self.text_color.get_widget_value()
        self.color[0] /= 255.0
        self.color[1] /= 255.0
        self.color[2] /= 255.0

    def size_changed(self):
        size = self.text_size.get_widget_value()
        if size != self.font_size:
            self.font_size = size
            self.initialized = False

    def font_changed(self):
        path = self.text_font.get_widget_value()
        if self.font_path != path:
            self.font_path = path
            self.initialized = False

    def update_font(self):
        hold_context = glfw.get_current_context()
        glfw.make_context_current(self.context)

        self.ready = False
        t = []
        for ch in self.characters:
            character = self.characters[ch]
            # delete textures
            t.append(character.texture)
        if len(t) > 0:
            glDeleteTextures(len(t), t)
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

        for i in range(0, 128):
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

                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, glyph.bitmap.width, glyph.bitmap.rows, 0,
                             GL_RGBA, GL_FLOAT, rgb_bm)

                # texture options
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

                # now store character for later use
                self.characters[chr(i)] = CharacterSlot(texture, glyph)
        glBindTexture(GL_TEXTURE_2D, 0)

    def get_rendering_buffer(self, xpos, ypos, width, height, zfix=0.):
        return np.asarray([
            xpos, ypos - height, 0, 1,
            xpos, ypos, 0, 0,
                  xpos + width, ypos, 1, 0,
            xpos, ypos - height, 0, 1,
                  xpos + width, ypos, 1, 0,
                  xpos + width, ypos - height, 1, 1
        ], np.float32)

    def draw(self):
        if not self.ready:
            return
        if not self.initialized:
            self.context = glfw.get_current_context()
            self.update_font()
            self.initialized = True
        glActiveTexture(GL_TEXTURE0)
        glPushMatrix()
        glTranslatef(0, 0, -2)
        glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glColor4f(self.color[0], self.color[1], self.color[2], self.text_alpha_input.get_widget_value())

        pos = [self.position_x_input.get_widget_value(), self.position_y_input.get_widget_value()]
        scale = self.scale_input.get_widget_value() / 100
        text = self.text_input.get_widget_value()

        for c in text:
            ch = self.characters[c]
            width, height = ch.textureSize
            width = width * scale
            height = height * scale
            origin = ch.origin[1] * scale
            vertices = self.get_rendering_buffer(pos[0] + ch.bearing[0] * scale, pos[1] + ch.bearing[1] * scale, width, height)
            glBindTexture(GL_TEXTURE_2D, ch.texture)

            glBegin(GL_TRIANGLES)
            for i in range(6):
                glTexCoord2f(vertices[i * 4 + 2], vertices[i * 4 + 3])
                glVertex2f(vertices[i * 4], vertices[i * 4 + 1])
            glEnd()

            pos[0] += (ch.advance >> 6) * scale
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_COLOR_MATERIAL)

        glPopMatrix()

