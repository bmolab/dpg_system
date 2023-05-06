import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import quaternion
import freetype
from dpg_system.open_gl_base import *
from dpg_system.glfw_base import *
import cv2

# can create command parsers for various types of GLObjects?


def register_opencv_nodes():
    Node.app.register_node('cv_image', CVImageNode.factory)
    Node.app.register_node('cv_capture', CVVideoCaptureNode.factory)
    Node.app.register_node('cv_camera', CVVideoCaptureNode.factory)
    # Node.app.register_node('gl_sphere', GLSphereNode.factory)
    # Node.app.register_node('gl_cylinder', GLCylinderNode.factory)
    # Node.app.register_node('gl_disk', GLDiskNode.factory)
    # Node.app.register_node('gl_partial_disk', GLPartialDiskNode.factory)
    # Node.app.register_node('gl_translate', GLTransformNode.factory)
    # Node.app.register_node('gl_rotate', GLTransformNode.factory)
    # Node.app.register_node('gl_scale', GLTransformNode.factory)
    # Node.app.register_node('gl_material', GLMaterialNode.factory)
    # Node.app.register_node('gl_align', GLAlignNode.factory)
    # Node.app.register_node('gl_quaternion_rotate', GLQuaternionRotateNode.factory)
    # Node.app.register_node('gl_text', GLTextNode.factory)
    # Node.app.register_node('gl_billboard', GLBillboard.factory)

class CVImageNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CVImageNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if len(args) > 0:
            self.image_path = any_to_string(args[0])

        self.path = ''
        self.image = None
        self.input = self.add_input('show image', triggers_execution=True)
        self.path_input = self.add_input('path in', triggers_execution=True)
        self.output = self.add_output("")

    def execute(self):
        if self.path_input.fresh_input:
            path = self.path_input.get_received_data()
            if type(path) == list:
                self.path = list_to_string(path)
            elif type(path) == str:
                self.path = path
            if type(self.path) == str and self.path != '':
                self.image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if self.image is not None:
            self.output.send(self.image)


class CVVideoCaptureNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CVVideoCaptureNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.streaming = False
        self.input = self.add_input('on/off', widget_type='checkbox', callback=self.on_off)
        self.output = self.add_output("")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")

    def on_off(self):
        on = self.input.get_widget_value()
        if on != self.streaming:
            if on:
                self.add_frame_task()
            else:
                self.remove_frame_tasks()
            self.streaming = on

    def cleanup(self):
        self.remove_frame_tasks()

    def frame_task(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?)")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.output.send(rgb)

    def execute(self):
        self.on_off()


