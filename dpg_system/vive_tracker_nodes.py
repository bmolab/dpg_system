from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
from dpg_system.triad_openvr.triad_openvr import *
import numpy as np

def register_vive_tracker_nodes():
    Node.app.register_node('vive_tracker', ViveTrackerNode.factory)
    # Node.app.register_node('continuous_rotation', ContinuousRotationNode.factory)




class ViveTrackerNode(Node):
    open_vr = None

    @staticmethod
    def factory(name, data, args=None):
        node = ViveTrackerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if ViveTrackerNode.open_vr is None:
            ViveTrackerNode.open_vr = triad_openvr()
            ViveTrackerNode.open_vr.print_discovered_objects()

        self.interval = 1/250
        self.enable_in = self.add_input('enable_in', widget_type='checkbox', triggers_execution=True)
        self.output_format_in = self.add_input('output_format', widget_type='combo', default_value='quaternion')
        self.which_tracker_in = self.add_input('which_tracker', widget_type='combo', default_value='tracker_1')
        self.which_tracker_in.widget.combo_items = ['tracker_1', 'tracker_2', 'tracker_3', 'tracker_4']
        self.output_format_in.widget.combo_items = ['quaternion', 'euler', 'matrix']
        self.orientation_out = self.add_output('orientation')
        self.position_out = self.add_output('position')
        self.orientation = None
        self.previous_orientation = None
        self.position = None
        self.__mutex = threading.Lock()
        self.thread = threading.Thread(target=self.vive_service_loop)
        self.thread_started = False
        if not self.thread_started:
            self.thread.start()
            self.thread_started = True

    def vive_service_loop(self):
        while True:
            if self.enable_in():
                self.get_data()
            time.sleep(self.interval)

    def frame_task(self):
        self.get_data()

    def custom_cleanup(self) -> None:
        if self.thread.is_alive():
            self.thread.join(1)

    def get_data(self):
        if self.which_tracker_in() in ViveTrackerNode.open_vr.devices:
            if self.output_format_in() == 'quaternion':
                orientation = ViveTrackerNode.open_vr.devices[self.which_tracker_in()].get_pose_quaternion()
                if orientation is not None:
                    self.orientation = any_to_array(orientation[3:])
                    self.position = any_to_array(orientation[:3])
                    self.orientation_out.send(self.orientation)
                    self.position_out.send(self.position)
            elif self.output_format_in() == 'euler':
                orientation = ViveTrackerNode.open_vr.devices[self.which_tracker_in()].get_pose_euler()
                if orientation is not None:
                    self.orientation = any_to_array(orientation[3:])
                    self.position = any_to_array(orientation[:3])
                    if self.previous_orientation is not None:
                        if self.previous_orientation[0] - self.orientation[0] > 180:
                            self.orientation[0] += 360
                        elif self.previous_orientation[0] - self.orientation[0] < -180:
                            self.orientation[0] -= 360
                        if self.previous_orientation[1] - self.orientation[1] > 180:
                            self.orientation[1] += 360
                        elif self.previous_orientation[1] - self.orientation[1] < -180:
                            self.orientation[1] -= 360
                        if self.previous_orientation[2] - self.orientation[2] > 180:
                            self.orientation[2] += 360
                        elif self.previous_orientation[2] - self.orientation[2] < -180:
                            self.orientation[2] -= 360
                    self.previous_orientation = self.orientation
                    self.orientation_out.send(self.orientation)
                    self.position_out.send(self.position)


    # def execute(self):
    #     if self.enable_in():
    #         if not self.has_frame_task:
    #             self.add_frame_task()
    #     else:
    #         if self.has_frame_task:
    #             self.remove_frame_tasks()








