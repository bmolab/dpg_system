import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import cv2
import dearpygui.dearpygui as dpg

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
        if type(path) == str and path != '':
            self.image = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.image is None:
                print(f"CVImageNode: failed to read image: {path}")
                return
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
        self.cap = None
        self.camera_indices = []

        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.turn_on_off)
        self.source_selector = self.add_input('source', widget_type='combo', default_value='0', callback=self.change_source)
        self.refresh_button = self.add_input('refresh', widget_type='button', callback=self.refresh_sources)
        self.output = self.add_output('')

    def post_creation_callback(self):
        self.refresh_sources()

    def enumerate_cameras(self, max_cameras=10):
        """Probe camera indices to find available cameras."""
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def refresh_sources(self):
        self.camera_indices = self.enumerate_cameras()
        camera_names = [str(i) for i in self.camera_indices]
        if not camera_names:
            camera_names = ['0']
        self.source_selector.widget.combo_items = camera_names
        dpg.configure_item(self.source_selector.widget.uuid, items=camera_names)

    def change_source(self):
        idx_str = self.source_selector()
        if idx_str is None or idx_str == '':
            return
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            return

        # Release existing capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            print(f"Cannot open camera {idx}")
            self.cap = None

    def turn_on_off(self):
        on = self.on_off()
        if on != self.streaming:
            if on:
                # If no camera is open yet, try to open the selected source
                if self.cap is None or not self.cap.isOpened():
                    self.change_source()
                if self.cap is not None and self.cap.isOpened():
                    self.add_frame_task()
                else:
                    print("No camera available to stream")
                    return
            else:
                self.remove_frame_tasks()
            self.streaming = on

    def cleanup(self):
        self.remove_frame_tasks()
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:
                pass
            self.cap = None

    def frame_task(self):
        if self.cap is None:
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?)")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"CVVideoCaptureNode: capture error ({e})")
            return
        self.output.send(rgb)

    def execute(self):
        self.turn_on_off()
