"""Real-time dense optical-flow nodes for dpg_system.

Camera images are expected as uint8 RGB arrays (H, W, 3).
Flow is float32 (H, W, 2) in pixels per processed frame.
"""
import cv2
import numpy as np
from dpg_system.node import Node


def register_cv_motion_nodes():
    Node.app.register_node('cv_optical_flow', CVOpticalFlowNode.factory)
    Node.app.register_node('cv_flow_features', CVFlowFeaturesNode.factory)
    Node.app.register_node('cv_flow_visualize', CVFlowVisualizeNode.factory)


def _valid_flow(value):
    return (isinstance(value, np.ndarray) and value.ndim == 3
            and value.shape[2] == 2 and value.size > 0)


class CVOpticalFlowNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return CVOpticalFlowNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.previous_gray = None
        self.previous_size = None
        self.image_input = self.add_input('RGB image', triggers_execution=True)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset)
        self.width_option = self.add_option('process width (0=original)', widget_type='drag_int', default_value=320, min=0, max=4096)
        self.pyr_scale_option = self.add_option('pyramid scale', widget_type='drag_float', default_value=0.5, min=0.1, max=0.9)
        self.levels_option = self.add_option('pyramid levels', widget_type='drag_int', default_value=3, min=1, max=8)
        self.window_option = self.add_option('window size', widget_type='drag_int', default_value=15, min=3, max=101)
        self.iterations_option = self.add_option('iterations', widget_type='drag_int', default_value=3, min=1, max=20)
        self.flow_output = self.add_output('flow (dx,dy)')

    def reset(self):
        self.previous_gray = None
        self.previous_size = None

    def execute(self):
        image = self.image_input()
        if not isinstance(image, np.ndarray) or image.size == 0:
            return
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            # cv_camera sends RGB, not BGR.
            conversion = cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
            gray = cv2.cvtColor(image, conversion)
        else:
            return

        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.floating):
                gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

        desired_width = max(0, int(self.width_option()))
        if desired_width and gray.shape[1] > desired_width:
            target_height = max(1, round(gray.shape[0] * desired_width / gray.shape[1]))
            gray = cv2.resize(gray, (desired_width, target_height), interpolation=cv2.INTER_AREA)
        gray = np.ascontiguousarray(gray)

        if self.previous_gray is None or self.previous_size != gray.shape:
            self.previous_gray = gray.copy()
            self.previous_size = gray.shape
            # First frame (or size change) has no previous frame for comparison.
            return

        window = max(3, int(self.window_option()))
        if window % 2 == 0:
            window += 1
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray, gray, None,
                pyr_scale=float(np.clip(self.pyr_scale_option(), 0.1, 0.9)),
                levels=max(1, int(self.levels_option())),
                winsize=window,
                iterations=max(1, int(self.iterations_option())),
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        except cv2.error as exc:
            print(f'cv_optical_flow: {exc}')
            self.reset()
            return

        self.previous_gray = gray.copy()
        self.flow_output.send(flow.astype(np.float32, copy=False))


class CVFlowFeaturesNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return CVFlowFeaturesNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.flow_input = self.add_input('flow', triggers_execution=True)
        self.threshold_option = self.add_option('motion threshold (px/frame)', widget_type='drag_float', default_value=1.0, min=0.0, max=100.0)
        self.energy_output = self.add_output('motion energy')
        self.horizontal_output = self.add_output('horizontal motion')
        self.vertical_output = self.add_output('vertical motion')
        self.activity_output = self.add_output('active fraction (0..1)')
        self.max_output = self.add_output('max motion')

    def execute(self):
        flow = self.flow_input()
        if not _valid_flow(flow):
            return
        flow = np.nan_to_num(flow.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        magnitude = np.linalg.norm(flow, axis=2)
        threshold = max(0.0, float(self.threshold_option()))
        moving = magnitude > threshold
        # Mean magnitude is measured in pixels per processed frame, not pixels/sec.
        self.energy_output.send(float(np.mean(magnitude)))
        # Signed averages can cancel when objects move in opposite directions.
        self.horizontal_output.send(float(np.mean(flow[..., 0])))
        self.vertical_output.send(float(np.mean(flow[..., 1])))
        self.activity_output.send(float(np.mean(moving)))
        self.max_output.send(float(np.max(magnitude)))


class CVFlowVisualizeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return CVFlowVisualizeNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.flow_input = self.add_input('flow', triggers_execution=True)
        self.scale_option = self.add_option('full brightness at (px/frame)', widget_type='drag_float', default_value=8.0, min=0.1, max=100.0)
        self.image_output = self.add_output('RGB image')

    def execute(self):
        flow = self.flow_input()
        if not _valid_flow(flow):
            return
        flow = np.nan_to_num(flow.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue: 0..179
        hsv[..., 1] = 255
        scale = max(0.1, float(self.scale_option()))
        hsv[..., 2] = np.clip(magnitude * (255.0 / scale), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.image_output.send(rgb)
