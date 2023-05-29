import glfw
import dearpygui.dearpygui as dpg
import OpenGL.GL as gl
import OpenGL.GLU as glu
import math
import numpy as np


class MyGLContext:
    inited = False

    @staticmethod
    def poll_glfw_events():
        glfw.poll_events()

    def __init__(self, name='untitled', width=640, height=480):
        if not self.inited:
            # print('about to init glfw')
            if not glfw.init():
                print("library is not initialized")
                return
            self.inited = True
#        Create a windowed mode window and its OpenGL context
#         print(glfw.get_version())
        # print('glfw inited')
        self.rotation_angle = 0
        self.d_x = 0
        self.height = height
        self.width = width
        self.pending_fov = 60
        self.fov = 30
        self.node = None
        # print('about to create window')

#        gl.glutInitDisplayMode(glfw.GLUT_RGB | glfw.GLUT_DOUBLE | glfw.GLUT_DEPTH)
        self.window = glfw.create_window(width, height, name, None, None)
        if self.window:
            # print('window created')
            glfw.make_context_current(self.window)
            glfw.set_key_callback(self.window, self.on_key)

            gl.glEnable(gl.GL_DEPTH_TEST)
            self.update_fov()
#            glu.gluPerspective(60, width / height, .1, 1000)
            gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
            gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (.5, .5, .5))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (-3.0, 3.0, 0.0))
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, (-3.0, 3.0, 3.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.25, 0.25, 0.25))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (.7, .7, .7))
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_AMBIENT, (0.25, 0.25, 0.25))
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, (.7, .7, .7))
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            gl.glShadeModel(gl.GL_FLAT)
            gl.glEnable(gl.GL_AUTO_NORMAL)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            gl.glShadeModel(gl.GL_SMOOTH)

        # else:
        #     glfw.terminate()


    def prepare_draw(self):
        if self.window:
            glfw.make_context_current(self.window)
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glColor4f(1.0, 1.0, 1.0, 1.0)

    #            self.update_fov()

    def end_draw(self):
        if self.window:
            glfw.swap_buffers(self.window)

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
 #       glfw.terminate()

    def set_fov(self, fov):
        self.pending_fov = fov

    def update_fov(self):
        if self.pending_fov != self.fov:
            if self.window:
                aspect = self.width / self.height
                # print('fov', self.pending_fov, aspect)
                current_matrix_mode = gl.glGetInteger(gl.GL_MATRIX_MODE)
                gl.glMatrixMode(gl.GL_PROJECTION)
                projectionD = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
                gl.glLoadIdentity()
                fov_radians = self.pending_fov / 180. * math.pi
                cotan = 1.0 / math.tan(fov_radians / 2.0)
                far = 1000
                near = 0.1
                m = np.array([cotan / aspect, 0.0, 0.0, 0.0, 0.0, cotan, 0.0, 0.0, 0.0, 0.0, (far + near) / (near - far), -1.0, 0.0, 0.0, (2.0 * far * near) / (near - far), 0.0])
                m = m.reshape((4, 4))
                gl.glMultMatrixd(m)
                gl.glMatrixMode(current_matrix_mode)
#                glu.gluPerspective(self.pending_fov, aspect, .1, 1000.0)
                self.fov = self.pending_fov

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if self.node is not None:
                self.node.handle_key(key, mods)

    # def on_key(self, window, key, scancode, action, mods):
    #     if action == glfw.PRESS:
    #         # ESC to quit
    #         if key == glfw.KEY_ESCAPE:
    #             glfw.set_window_should_close(window, 1)
    #         elif key == glfw.KEY_UP:
    #             self.d_x += 0.1
    #         elif key == glfw.KEY_DOWN:
    #             self.d_x += -0.1
    #         elif key == glfw.KEY_SPACE:
    #             color.red += color.blue
    #             color.blue = color.red - color.blue
    #             color.red = color.red - color.blue


