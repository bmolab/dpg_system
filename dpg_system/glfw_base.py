import glfw
import dearpygui.dearpygui as dpg
import OpenGL.GL as gl
import OpenGL.GLU as glu
import math
import numpy as np
import threading


class MyGLContext:
    gl_thread = None
    inited = False

    @staticmethod
    def poll_glfw_events():
        glfw.poll_events()

    def __init__(self, name='untitled', width=640, height=480, samples=1):

        if not MyGLContext.inited:
            if not glfw.init():
                print("MyGLContext: glfw.init() failed; GLFW unavailable")
                self.window = None
                return
            # MyGLContext.gl_thread = threading.Thread(target=run_gl_thread)
            MyGLContext.inited = True
#        Create a windowed mode window and its OpenGL context
        self.rotation_angle = 0
        self.d_x = 0
        self.height = height
        self.width = width
        self.pending_fov = 60
        self.fov = 30
        self.node = None
        self.last_key = -1
        self.last_mods = 0
        self.clear_color = [0.0, 0.0, 0.0, 1.0]

#        gl.glutInitDisplayMode(glfw.GLUT_RGB | glfw.GLUT_DOUBLE | glfw.GLUT_DEPTH)
        if samples != 1 and samples < 8:
            glfw.window_hint(glfw.SAMPLES, samples)
        # glfw.window_hint(glfw.SCALE_TO_MONITOR, False)
        self.window = glfw.create_window(width, height, name, None, None)
        if not self.window:
            print(f'MyGLContext: glfw.create_window({width}x{height}, {name!r}) failed; degrading to no-op')
            return
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
            gl.glClearColor(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3])
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            # gl.glColor4fv(self.clear_color)

    #            self.update_fov()

    def end_draw(self):
        if self.window:
            glfw.swap_buffers(self.window)

    def close(self):
        if self.window:
            try:
                glfw.destroy_window(self.window)
            except Exception as e:
                print(f'MyGLContext.close: glfw.destroy_window failed ({e})')
            self.window = None
 #       glfw.terminate()

    def set_fov(self, fov):
        self.pending_fov = fov

    def update_fov(self):
        if self.pending_fov == self.fov:
            return
        if not self.window:
            return
        # Guard degenerate inputs: height==0 (divide-by-zero on aspect) and
        # pending_fov at 0 or 180 (tan goes to 0 or inf, producing a bad
        # projection matrix that breaks every subsequent draw).
        if self.height == 0:
            return
        if self.pending_fov <= 0 or self.pending_fov >= 180:
            print(f'MyGLContext.update_fov: invalid fov {self.pending_fov}, skipping')
            return
        aspect = self.width / self.height
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
        self.fov = self.pending_fov

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if self.node is not None:
                self.node.handle_key(key, mods)
                self.last_key = key
                self.last_mods = mods
        elif action == glfw.REPEAT:
            if self.node is not None:
                self.node.handle_key(self.last_key, self.last_mods)

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



