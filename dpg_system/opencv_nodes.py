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

class CVImageNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CVImageNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        image_path = ''
        if len(args) > 0:
            image_path = any_to_string(args[0])

        self.image = None
        self.input = self.add_input('show image', triggers_execution=True)
        self.path_input = self.add_input('path in', widget_type='text_input', default_value=image_path, callback=self.path_changed)
        self.output = self.add_output('')

    def custom_create(self, from_file):
        self.path_changed()

    def path_changed(self):
        path = self.path_input()
        if type(path) == list:
            path = list_to_string(path)
        elif type(path) == str:
            path = path
        if type(path) == str and path != '':
            self.image = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.image is not None:
                self.execute()

    def execute(self):
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
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.turn_on_off)
        self.output = self.add_output('')
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")

    def turn_on_off(self):
        on = self.on_off()
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
        self.turn_on_off()


