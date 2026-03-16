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
        self.connected_out = self.add_output('connected')
        self.orientation = None
        self.previous_orientation = None
        self.position = None
        self.connected = False
        self.tracker_serial = None       # serial of the tracker we're bound to
        self.tracker_device_name = None   # current name in open_vr.devices
        self.__mutex = threading.Lock()
        self.thread = threading.Thread(target=self.vive_service_loop, daemon=True)
        self.thread_started = False
        if not self.thread_started:
            self.thread.start()
            self.thread_started = True

    def _cache_tracker_serial(self):
        """Cache the serial number of the currently selected tracker so we can find it after reconnection."""
        target_name = self.which_tracker_in()
        if target_name in ViveTrackerNode.open_vr.devices:
            device = ViveTrackerNode.open_vr.devices[target_name]
            self.tracker_serial = device.get_serial()
            self.tracker_device_name = target_name
            self.connected = True
            self.connected_out.send(1)
            print(f'Tracker bound to "{target_name}" (serial: {self.tracker_serial})')
            return True
        return False

    def _find_tracker_by_serial(self):
        """Find the tracker in open_vr.devices by serial number, regardless of its current name."""
        if self.tracker_serial is None:
            return None
        for name, device in ViveTrackerNode.open_vr.devices.items():
            if device.device_class == "Tracker" and device.get_serial() == self.tracker_serial:
                return name
        return None

    def vive_service_loop(self):
        while True:
            try:
                ViveTrackerNode.open_vr.poll_vr_events()
            except Exception as e:
                print(f'poll_vr_events error (non-fatal): {e}')
            if self.enable_in():
                self.get_data()
            time.sleep(self.interval)

    def frame_task(self):
        self.get_data()

    def custom_cleanup(self) -> None:
        if self.thread.is_alive():
            self.thread.join(1)

    def get_data(self):
        target_name = self.which_tracker_in()

        # First time: cache the serial of the selected tracker
        if self.tracker_serial is None:
            if not self._cache_tracker_serial():
                # Tracker not yet available at all
                if self.connected:
                    self.connected = False
                    self.connected_out.send(0)
                return

        # Check if the tracker is still under its known name
        if self.tracker_device_name not in ViveTrackerNode.open_vr.devices:
            # Tracker disappeared — try to find it by serial (it may have reconnected under a new name)
            new_name = self._find_tracker_by_serial()
            if new_name is not None:
                self.tracker_device_name = new_name
                if not self.connected:
                    self.connected = True
                    self.connected_out.send(1)
                    print(f'Tracker reconnected as "{new_name}" (serial: {self.tracker_serial})')
            else:
                # Tracker is genuinely offline
                if self.connected:
                    self.connected = False
                    self.connected_out.send(0)
                    print(f'Tracker disconnected (serial: {self.tracker_serial})')
                return

        # If the user changed the tracker selection, re-cache
        if target_name != self.tracker_device_name and target_name in ViveTrackerNode.open_vr.devices:
            self._cache_tracker_serial()

        try:
            device = ViveTrackerNode.open_vr.devices[self.tracker_device_name]
            if device is not None:
                if self.output_format_in() == 'quaternion':
                    orientation = device.get_pose_quaternion()
                    if orientation is not None:
                        self.orientation = any_to_array(orientation[3:])
                        self.position = any_to_array(orientation[:3])
                        self.orientation_out.send(self.orientation)
                        self.position_out.send(self.position)
                        if not self.connected:
                            self.connected = True
                            self.connected_out.send(1)
                    else:
                        if self.connected:
                            self.connected = False
                            self.connected_out.send(0)
                elif self.output_format_in() == 'euler':
                    orientation = device.get_pose_euler()
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

                        if not self.connected:
                            self.connected = True
                            self.connected_out.send(1)
                    else:
                        if self.connected:
                            self.connected = False
                            self.connected_out.send(0)
            else:
                print('tracker not found')
        except (ZeroDivisionError, KeyError, Exception) as e:
            # ZeroDivisionError: degenerate pose matrix (r_w == 0 in quaternion conversion)
            # KeyError: device removed from dict between our check and access
            # Skip this frame silently — the tracker may be in a transient bad state
            pass







