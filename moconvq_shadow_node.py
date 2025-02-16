
import numpy
import torch
import functools
import numpy as np
import threading
import glfw
import os
import json
import time
import OpenGL.GLU as glu

from dpg_system.dpg_app import App, Node, Conduit
from glumpy import app, gloo, gl

from dpg_system.conversion_utils import *

import torch._dynamo as dynamo

app.use('glfw')
back = app.__backend__

shadow_app = None

screens_wide = 1
image_aspect_ratio = screens_wide * 16 / 9
window_width = 1920

window_height = int(window_width / image_aspect_ratio)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ShadowApp(App):
    def __init__(self):
        super().__init__()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = app.Window(window_width, window_height, fullscreen=False)
        self.clear_color = [0.0, 0.0, 0.0, 1.0]
        self.window.set_title('shadow')
        self.window_height = self.window.height
        self.window_width = self.window.width
        self.image_aspect_ratio = self.window.width / self.window_height
        self.scene_graph_conduit = self.add_conduit('gl_scene')
        # glfw.make_context_current(self.window.native_window.contents)
        gl.glEnable(gl.GL_DEPTH_TEST)
        glu.gluPerspective(60, self.window_width / self.window_height, .1, 1000)
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
        self.clear_color_red_var = self.add_variable(variable_name='clear_color_red', default_value=0, setter=self.set_clear)
        self.clear_color_green_var = self.add_variable(variable_name='clear_color_green', default_value=0, setter=self.set_clear)
        self.clear_color_blue_var = self.add_variable(variable_name='clear_color_blue', default_value=0, setter=self.set_clear)

    def set_clear(self, val):
        self.clear_color[0] = self.clear_color_red_var.get()
        self.clear_color[1] = self.clear_color_green_var.get()
        self.clear_color[2] = self.clear_color_blue_var.get()

    def draw(self):
        gl.glClearColor(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        self.scene_graph_conduit.transmit('draw')
        gl.glFlush()


shadow_app = ShadowApp()

window1 = shadow_app.window


def run_shadow():
    global shadow_app
    shadow_app.start()
    shadow_app.run_loop()


shadow_thread = threading.Thread(target=run_shadow)
shadow_thread.start()


@window1.event
def on_draw(dt):
    shadow_app.draw()


backend = app.__backend__
clock = app.__init__(backend=backend)
app.run()
