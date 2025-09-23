from OpenGL.GLUT import *
from dpg_system.body_base import *

# openGL Context is a node
# it outputs a draw message per render cycle
# children draw when they receive this message (and register as child, save gl_context)
# body received quaternions in input and draw pose


class OpenGLThread(threading.Thread):
    def __init__(self, window_width=1024, window_height=1024):
        threading.Thread.__init__(self, target=self.run_loop)
        self.__mutex = None
        self.__mutex = threading.Lock()
        self.children = []
        self.window_width = window_width
        self.window_height = window_height

    def __del__(self):
        if self.__mutex is not None:
            lock = ScopedLock(self.__mutex)
            self.__mutex = None

    def run_loop(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)

        glutInitWindowPosition(0, 0)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow("Example")

        self.init_additional()

        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutVisibilityFunc(self.visibility)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keys)

        glutMouseFunc(self.mouse)
        glutMotionFunc(self.mouse_drag)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (.5, .5, .5))
        glLightfv(GL_LIGHT0, GL_POSITION, (-3.0, 3.0, 0.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (-3.0, 3.0, 3.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.25))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (.7, .7, .7))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.25, 0.25, 0.25))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (.7, .7, .7))
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glShadeModel(GL_FLAT)
        glEnable(GL_AUTO_NORMAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glutMainLoop()

    def add_child(self, child):
        self.children.append(child)
        child.set_display_context(self)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def keys(self, key, x, y):
        for child in self.children:
            if child.handle_key(key, x, y):
                break

    def mouse(self, button, state, x, y):
        for child in self.children:
            if child.handle_mouse(button, state, x, y):
                break

    def reshape(self, width, height):
        if height <= 0:
            height = 1

        height_real = float(width) / float(height)

        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, height_real, 1, 1000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(-2, 2, 2, 0, 0, 0, 0, 1, 0)

    def idle(self):
        glutPostRedisplay()

    def visibility(self, visible):
        if visible == GLUT_VISIBLE:
            glutIdleFunc(self.idle)
        else:
            glutIdleFunc(None)

    def mouse_drag(self, x, y):
        for child in self.children:
            if child.handle_mouse_drag(x, y):
                break

    def display(self):
        self.predisplay_callbacks()
        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.display_items()
        glutSwapBuffers()

    def display_internal(self):
        pass

    def display_items(self):
        self.display_internal()
        for child in self.children:
            child.draw()

    def init_additional(self):
        for child in self.children:
            child.init()

    def predisplay_callbacks(self):
        for child in self.children:
            child.predisplay_callback()


# render node creates a gl child
class OpenGLRenderer:
    gl_thread = None

    def __init__(self, data, args=None):
        self.gl_context = None

    def __del__(self):
        if self.gl_context is not None:
            self.gl_context.remove_child(self)

    def set_gl_thread(self, threader):
        self.gl_thread = threader




class OpenGLObject:
    def __init__(self):
        self.gl_context = None

    def __del__(self):
        if self.gl_context is not None:
            self.gl_context.remove_child(self)

    def init(self):
        pass

    def predisplay_callback(self):
        pass

    def draw(self):
        pass

    def handle_key(self, key, x, y):
        return False

    def handle_mouse(self, button, state, x, y):
        return False

    def handle_mouse_drag(self, x, y):
        return False

    def set_display_context(self, context):
        self.gl_context = context


# this should not subclass OpenGLObject, but should receive render related messages from OpenGLObject

class OpenGLBodyBase(OpenGLObject):
    def __init__(self):
        super().__init__()
        self.body = BodyData()

        self.joint_order = []
        self.new_data = False
        self.show_rotation_spheres = False
        self.pose_quats = None

    def draw(self):
        if self.pose_quats is not None:
            self.body.draw(self.show_rotation_spheres)

    def update_pose(self, quats, positions=None):
        if quats is not None:
            for jointID, joint_name in enumerate(self.joint_order):
                self.body.update(jointID, quats[jointID])
            return True


class OpenGLBody(OpenGLObject):
    def __init__(self):
        super().__init__()
        self.body = BodyData()
        self.take_quats = None
        self.take_positions = None

        self.frame_count = 0
        self.currentFrame = 0
        self.joint_order = []
        self.speed = 1
        self.new_data = False
        self.show_rotation_spheres = False

    def init(self):
        self.load_take_from_numpy()

    def predisplay_callbacks(self):
        self.body.adjust_clear_colour()

    def draw(self):
        if self.take_quats is not None:
            display = False
            while not display:
                display = self.update_from_take()
                if not display:
                    self.currentFrame += 1
                    if self.currentFrame >= self.frame_count:
                        self.currentFrame = 0

            self.body.draw(self.show_rotation_spheres)

    def update_from_take(self):
        if self.take_quats is not None:
            thisFrame = int(self.currentFrame)
            glutSetWindowTitle(str(thisFrame))

            this_take_frame_quats = self.take_quats[thisFrame]
            this_take_frame_positions = self.take_positions[thisFrame]

            if self.new_data:
                self.new_data = False
            for jointID, joint_name in enumerate(self.joint_order):
                self.body.update(jointID, this_take_frame_quats[jointID], this_take_frame_positions[jointID])

            self.currentFrame += self.speed

            if self.currentFrame >= self.frame_count:
                self.currentFrame = 0
            return True

    def load_take_from_numpy(self):
        self.take_quats, self.take_positions, _, _ = load_take_from_npz('take.npz')
        self.frame_count = self.take_quats.shape[0]
        self.joint_order = []
        for j, name in enumerate(JointTranslator.shadow_joint_index_to_name):
            actual_name = JointTranslator.shadow_joint_index_to_name[name]
            self.joint_order.append(actual_name)

    def handle_key(self, key, x, y):
        return False

    def handle_mouse(self, button, state, x, y):
         if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            if self.take_quats is not None:
                self.speed = 0
                pos = x / self.gl_context.window_width
                self.currentFrame = int(pos * self.take_quats.shape[0])
                return True
         return False

    def handle_mouse_drag(self, x, y):
        if self.take_quats is not None:
            pos = x / self.gl_context.window_width
            self.currentFrame = int(pos * self.take_quats.shape[0])
            return True
        return False

#
# def main():
#     example = OpenGLThread(window_width=1920, window_height=1080)
#     body = OpenGLBody()
#     example.add_child(body)
#     example.run_loop()
#
#
# if __name__ == "__main__":
#     sys.exit(main())
