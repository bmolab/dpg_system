from dpg_system.node import Node
import threading
import math
from collections import deque
from dpg_system.conversion_utils import *
from dpg_system.triad_openvr.triad_openvr import *
import numpy as np

def register_vive_tracker_nodes():
    Node.app.register_node('vive_tracker', ViveTrackerNode.factory)
    Node.app.register_node('vive_base_stations', ViveBaseStationNode.factory)
    # Node.app.register_node('continuous_rotation', ContinuousRotationNode.factory)




class ViveTrackerNode(Node):
    open_vr = None
    chaperone_setup = None

    @staticmethod
    def factory(name, data, args=None):
        node = ViveTrackerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if ViveTrackerNode.open_vr is None:
            try:
                ViveTrackerNode.open_vr = triad_openvr()
                ViveTrackerNode.open_vr.print_discovered_objects()
            except Exception as e:
                print(f'ViveTrackerNode: OpenVR initialization failed ({e}); node will report disconnected')
                ViveTrackerNode.open_vr = None

        if ViveTrackerNode.open_vr is not None and ViveTrackerNode.chaperone_setup is None:
            try:
                ViveTrackerNode.chaperone_setup = openvr.VRChaperoneSetup()
            except Exception as e:
                print(f'ViveTrackerNode: IVRChaperoneSetup unavailable ({e})')
                ViveTrackerNode.chaperone_setup = None

        self.interval = 1/250
        self.enable_in = self.add_input('enable_in', widget_type='checkbox', triggers_execution=True)
        self.output_format_in = self.add_input('output_format', widget_type='combo', default_value='quaternion')
        self.which_tracker_in = self.add_input('which_tracker', widget_type='combo', default_value='tracker_1')
        self.which_tracker_in.widget.combo_items = ['tracker_1', 'tracker_2', 'tracker_3', 'tracker_4']
        self.output_format_in.widget.combo_items = ['quaternion', 'euler', 'matrix']
        self.play_area_x_in = self.add_input('play_area_x_m', widget_type='drag_float', default_value=3.0)
        self.play_area_z_in = self.add_input('play_area_z_m', widget_type='drag_float', default_value=3.0)
        self.play_area_yaw_in = self.add_input('play_area_yaw_deg', widget_type='drag_float', default_value=0.0)
        self.play_area_center_x_in = self.add_input('play_area_center_x_m', widget_type='drag_float', default_value=0.0)
        self.play_area_center_z_in = self.add_input('play_area_center_z_m', widget_type='drag_float', default_value=0.0)
        self.floor_height_in = self.add_input('floor_height_m', widget_type='drag_float', default_value=0.0)
        self.capture_fl_in = self.add_input('capture_FL_corner', widget_type='button', callback=self.capture_fl)
        self.capture_fr_in = self.add_input('capture_FR_corner', widget_type='button', callback=self.capture_fr)
        self.capture_br_in = self.add_input('capture_BR_corner', widget_type='button', callback=self.capture_br)
        self.capture_bl_in = self.add_input('capture_BL_corner', widget_type='button', callback=self.capture_bl)
        self.compute_from_corners_in = self.add_input('compute_from_corners', widget_type='button', callback=self.compute_from_corners)
        self.clear_corners_in = self.add_input('clear_corners', widget_type='button', callback=self.clear_corners)
        self.apply_chaperone_in = self.add_input('apply_chaperone', widget_type='button', callback=self.apply_chaperone)
        self.orientation_out = self.add_output('orientation')
        self.position_out = self.add_output('position')
        self.connected_out = self.add_output('connected')
        self.orientation = None
        self.previous_orientation = None
        self.position = None
        self.connected = False
        self.tracker_serial = None       # serial of the tracker we're bound to
        self.tracker_device_name = None   # current name in open_vr.devices
        self.captured_corners = [None, None, None, None]  # FL, FR, BR, BL in raw tracking coords
        self.__mutex = threading.Lock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.vive_service_loop, daemon=True)
        self.thread_started = False
        if ViveTrackerNode.open_vr is not None and not self.thread_started:
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
        while not self._stop_event.is_set():
            try:
                if ViveTrackerNode.open_vr is not None:
                    try:
                        ViveTrackerNode.open_vr.poll_vr_events()
                    except Exception as e:
                        print(f'poll_vr_events error (non-fatal): {e}')
                    if self.enable_in():
                        self.get_data()
            except Exception as e:
                # Never let the polling thread die on an unexpected error.
                print(f'ViveTrackerNode service loop error (non-fatal): {e}')
            time.sleep(self.interval)

    def frame_task(self):
        self.get_data()

    def apply_chaperone(self):
        if ViveTrackerNode.chaperone_setup is None:
            print('ViveTrackerNode: chaperone setup unavailable')
            return
        try:
            size_x = float(self.play_area_x_in())
            size_z = float(self.play_area_z_in())
            yaw_rad = math.radians(float(self.play_area_yaw_in()))
            center_x = float(self.play_area_center_x_in())
            center_z = float(self.play_area_center_z_in())
            floor_y = float(self.floor_height_in())

            pose = openvr.HmdMatrix34_t()
            c = math.cos(yaw_rad)
            s = math.sin(yaw_rad)
            pose.m[0][0] = c;   pose.m[0][1] = 0.0; pose.m[0][2] = s;   pose.m[0][3] = center_x
            pose.m[1][0] = 0.0; pose.m[1][1] = 1.0; pose.m[1][2] = 0.0; pose.m[1][3] = floor_y
            pose.m[2][0] = -s;  pose.m[2][1] = 0.0; pose.m[2][2] = c;   pose.m[2][3] = center_z

            ViveTrackerNode.chaperone_setup.setWorkingPlayAreaSize(size_x, size_z)
            ViveTrackerNode.chaperone_setup.setWorkingStandingZeroPoseToRawTrackingPose(pose)
            ViveTrackerNode.chaperone_setup.commitWorkingCopy(openvr.EChaperoneConfigFile_Live)
            print(f'Chaperone applied: size=({size_x:.3f}, {size_z:.3f}) m, yaw={math.degrees(yaw_rad):.2f}°, '
                  f'center=({center_x:.3f}, {center_z:.3f}) m, floor={floor_y:.3f} m')
        except Exception as e:
            print(f'ViveTrackerNode: apply_chaperone failed ({e})')

    def _get_current_tracker_position(self):
        if ViveTrackerNode.open_vr is None:
            return None
        target_name = self.tracker_device_name or self.which_tracker_in()
        if target_name not in ViveTrackerNode.open_vr.devices:
            return None
        try:
            pose = ViveTrackerNode.open_vr.devices[target_name].get_pose_quaternion()
        except Exception as e:
            print(f'ViveTrackerNode: pose read failed ({e})')
            return None
        if pose is None:
            return None
        return np.array(pose[:3], dtype=float)

    def _capture_corner(self, idx, label):
        pos = self._get_current_tracker_position()
        if pos is None:
            print(f'capture {label}: no valid tracker pose (check tracker selection and connection)')
            return
        self.captured_corners[idx] = pos
        ys = [c[1] for c in self.captured_corners if c is not None]
        floor_y = float(sum(ys) / len(ys))
        self.floor_height_in.set(floor_y)
        print(f'captured {label} at raw ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}); '
              f'floor_height updated to {floor_y:.3f} m (mean of {len(ys)} corner(s))')

    def capture_fl(self):
        self._capture_corner(0, 'FL')

    def capture_fr(self):
        self._capture_corner(1, 'FR')

    def capture_br(self):
        self._capture_corner(2, 'BR')

    def capture_bl(self):
        self._capture_corner(3, 'BL')

    def clear_corners(self):
        self.captured_corners = [None, None, None, None]
        print('captured corners cleared')

    def compute_from_corners(self):
        if any(c is None for c in self.captured_corners):
            missing = [n for n, c in zip(['FL', 'FR', 'BR', 'BL'], self.captured_corners) if c is None]
            print(f'compute_from_corners: missing corners {missing}')
            return
        fl, fr, br, bl = self.captured_corners
        center = 0.25 * (fl + fr + br + bl)
        floor_y = float(center[1])

        # standing +X (in raw XZ) averaged from the two "front-to-back" parallel edges
        edge_x = 0.5 * ((fr - fl) + (br - bl))
        size_x = 0.5 * (float(np.linalg.norm(fr - fl)) + float(np.linalg.norm(br - bl)))
        size_z = 0.5 * (float(np.linalg.norm(bl - fl)) + float(np.linalg.norm(br - fr)))

        # R_y(yaw) @ (1,0,0) = (cos yaw, 0, -sin yaw)
        yaw_rad = math.atan2(-float(edge_x[2]), float(edge_x[0]))
        yaw_deg = math.degrees(yaw_rad)

        self.play_area_x_in.set(size_x)
        self.play_area_z_in.set(size_z)
        self.play_area_yaw_in.set(yaw_deg)
        self.play_area_center_x_in.set(float(center[0]))
        self.play_area_center_z_in.set(float(center[2]))
        self.floor_height_in.set(floor_y)
        print(f'computed from corners: size=({size_x:.3f}, {size_z:.3f}) m, yaw={yaw_deg:.2f}°, '
              f'center=({center[0]:.3f}, {center[2]:.3f}) m, floor={floor_y:.3f} m')

    def custom_cleanup(self) -> None:
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print('ViveTrackerNode: service thread did not exit within 1s')

    def get_data(self):
        if ViveTrackerNode.open_vr is None:
            return

        try:
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
        except Exception:
            # ZeroDivisionError: degenerate pose matrix (r_w == 0 in quaternion conversion)
            # KeyError: device removed from dict between our check and access
            # Skip this frame silently — the tracker may be in a transient bad state
            pass


def _quat_angle_deg(q_a, q_b):
    """Angular difference (degrees) between two [w, x, y, z]-ish unit quaternions.

    Accepts the triad_openvr convention where get_pose_quaternion returns
    [x, y, z, r_w, r_x, r_y, r_z]; callers pass in just the [r_w, r_x, r_y, r_z] part.
    """
    a = np.asarray(q_a, dtype=float)
    b = np.asarray(q_b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = abs(float(np.dot(a / na, b / nb)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class _StationStats:
    """Rolling pose history + stability metrics for a single base station (lighthouse)."""

    def __init__(self, serial, window):
        self.serial = serial
        self.mode = None
        self.positions = deque(maxlen=window)   # each: np.array([x, y, z]) in metres
        self.quaternions = deque(maxlen=window)  # each: np.array([r_w, r_x, r_y, r_z])
        self.baseline = None                     # np.array([x, y, z]) captured reference
        self.valid = False
        self.last_seen_valid = False

    def resize(self, window):
        if self.positions.maxlen == window:
            return
        self.positions = deque(self.positions, maxlen=window)
        self.quaternions = deque(self.quaternions, maxlen=window)

    def add_sample(self, pos, quat):
        self.positions.append(pos)
        self.quaternions.append(quat)

    def set_baseline(self):
        if self.positions:
            self.baseline = np.mean(np.array(self.positions), axis=0)
            return True
        return False

    def position_jitter_mm(self):
        # RMS Euclidean distance of samples from their window mean, in mm.
        if len(self.positions) < 2:
            return 0.0
        arr = np.array(self.positions)
        mean = arr.mean(axis=0)
        d2 = np.sum((arr - mean) ** 2, axis=1)
        return float(np.sqrt(d2.mean()) * 1000.0)

    def orientation_jitter_deg(self):
        # RMS angular deviation of samples from the most recent orientation.
        if len(self.quaternions) < 2:
            return 0.0
        ref = self.quaternions[-1]
        angles = np.array([_quat_angle_deg(q, ref) for q in self.quaternions])
        return float(np.sqrt(np.mean(angles ** 2)))

    def drift_mm(self):
        # Distance of the current window mean from the captured baseline, in mm.
        if self.baseline is None or not self.positions:
            return 0.0
        mean = np.mean(np.array(self.positions), axis=0)
        return float(np.linalg.norm(mean - self.baseline) * 1000.0)

    def current_position(self):
        # Most recent sampled position [x, y, z] in metres, or None.
        if not self.positions:
            return None
        return self.positions[-1]

    def current_orientation(self):
        # Most recent sampled orientation quaternion [r_w, r_x, r_y, r_z], or None.
        if not self.quaternions:
            return None
        return self.quaternions[-1]


class ViveBaseStationNode(Node):
    """Monitors the stability of the Vive/SteamVR base stations (lighthouses).

    Base stations appear to OpenVR as "Tracking Reference" devices, and SteamVR
    continuously reports their estimated pose. If a base station is solidly
    mounted and well-synced, that pose is essentially motionless — so any jitter
    or drift in it, or a station dropping out entirely, points straight at the
    usual causes of erratic tracker readings (loose mounts, vibration, sync loss,
    reflections). This node surfaces those metrics so they can be watched or wired
    into scopes/alarms.
    """

    @staticmethod
    def factory(name, data, args=None):
        node = ViveBaseStationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # Share the single process-wide OpenVR session with ViveTrackerNode
        # (openvr.init() must only be called once per process).
        if ViveTrackerNode.open_vr is None:
            try:
                ViveTrackerNode.open_vr = triad_openvr()
                ViveTrackerNode.open_vr.print_discovered_objects()
            except Exception as e:
                print(f'ViveBaseStationNode: OpenVR initialization failed ({e}); node will report no stations')
                ViveTrackerNode.open_vr = None

        self.interval = 1 / 50
        self.window = 250  # samples held for the rolling jitter/drift statistics

        self.enable_in = self.add_input('enable_in', widget_type='checkbox', default_value=True, triggers_execution=True)
        self.window_in = self.add_input('window_size', widget_type='input_int', default_value=self.window, min=2)
        self.jitter_threshold_in = self.add_input('jitter_threshold_mm', widget_type='drag_float', default_value=1.0)
        self.drift_threshold_in = self.add_input('drift_threshold_mm', widget_type='drag_float', default_value=5.0)
        self.set_baseline_in = self.add_input('set_baseline', widget_type='button', callback=self.set_baseline)
        self.print_report_in = self.add_input('print_report', widget_type='button', callback=self.print_report)
        self.reset_in = self.add_input('reset', widget_type='button', callback=self.reset_stats)

        self.count_out = self.add_output('num_stations')
        self.serials_out = self.add_output('serials')          # list of serial strings (row order of the arrays below)
        self.positions_out = self.add_output('positions_m')    # np.array (N, 3) current x,y,z per station, metres
        self.orientations_out = self.add_output('orientations')  # np.array (N, 4) current quaternion [w,x,y,z] per station
        self.jitter_out = self.add_output('jitter_mm')          # np.array, one per station (sorted by serial)
        self.orient_jitter_out = self.add_output('orient_jitter_deg')
        self.drift_out = self.add_output('drift_mm')            # np.array, one per station
        self.stable_out = self.add_output('all_stable')        # 1 = every station within thresholds, else 0
        self.report_out = self.add_output('report')            # human-readable multi-line string

        self.stations = {}          # serial -> _StationStats
        self.last_station_count = None
        self.__mutex = threading.Lock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread_started = False
        if ViveTrackerNode.open_vr is not None:
            self.thread.start()
            self.thread_started = True

    @staticmethod
    def _device_string(vr, index, prop):
        try:
            s = vr.getStringTrackedDeviceProperty(index, prop)
        except Exception:
            return None
        if isinstance(s, bytes):
            s = s.decode('utf-8', errors='replace')
        return s

    def _enumerate_tracking_references(self):
        """Enumerate base stations straight from OpenVR on every sample.

        We deliberately do NOT use ViveTrackerNode.open_vr.devices here: that dict
        is only populated at triad_openvr() init and thereafter kept up to date by
        poll_vr_events(), which runs solely in ViveTrackerNode's thread. Base
        stations that wake up after init (or any activation event that dict missed)
        would otherwise never appear, so this node would report fewer stations than
        SteamVR shows. Querying the tracked-device poses directly reflects the true
        current state each frame.

        Returns a list of (serial, mode, valid, pose_mat), one per *connected*
        tracking reference — matching SteamVR's active-base-station count.
        """
        ovr = ViveTrackerNode.open_vr
        if ovr is None:
            return []
        vr = ovr.vr
        try:
            poses = ovr.get_pose()
        except Exception as e:
            print(f'ViveBaseStationNode: pose read failed ({e})')
            return []
        result = []
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            pose_i = poses[i]
            if not pose_i.bDeviceIsConnected:
                continue
            try:
                if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_TrackingReference:
                    continue
            except Exception:
                continue
            serial = self._device_string(vr, i, openvr.Prop_SerialNumber_String) or f'index_{i}'
            mode = self._device_string(vr, i, openvr.Prop_ModeLabel_String)
            if mode:
                mode = mode.strip().upper() or None
            valid = bool(pose_i.bPoseIsValid)
            pose_mat = pose_i.mDeviceToAbsoluteTracking if valid else None
            result.append((serial, mode, valid, pose_mat))
        return result

    def monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                if ViveTrackerNode.open_vr is not None and self.enable_in():
                    self.sample_stations()
            except Exception as e:
                print(f'ViveBaseStationNode monitor loop error (non-fatal): {e}')
            time.sleep(self.interval)

    def sample_stations(self):
        window = int(self.window_in())
        if window < 2:
            window = 2
        self.window = window

        refs = self._enumerate_tracking_references()
        seen_serials = set()

        with self.__mutex:
            for serial, mode, valid, pose_mat in refs:
                seen_serials.add(serial)
                stats = self.stations.get(serial)
                if stats is None:
                    stats = _StationStats(serial, window)
                    stats.mode = mode
                    self.stations[serial] = stats
                else:
                    stats.resize(window)
                    if mode and not stats.mode:
                        stats.mode = mode

                if not valid or pose_mat is None:
                    # Connected but not yet localized by SteamVR — counted, flagged NO POSE.
                    stats.valid = False
                    continue
                try:
                    quat = convert_to_quaternion(pose_mat)  # [x, y, z, r_w, r_x, r_y, r_z]
                except Exception:
                    stats.valid = False
                    continue
                pos = np.array(quat[:3], dtype=float)
                rot = np.array(quat[3:], dtype=float)
                stats.add_sample(pos, rot)
                stats.valid = True
                stats.last_seen_valid = True

            # Drop stations that are no longer present at all (physical dropout).
            for serial in list(self.stations.keys()):
                if serial not in seen_serials:
                    del self.stations[serial]

        count = len(seen_serials)
        if count != self.last_station_count:
            if self.last_station_count is not None:
                print(f'ViveBaseStationNode: base station count changed {self.last_station_count} -> {count} '
                      f'(a dropout/reappearance is a common cause of erratic tracking)')
            self.last_station_count = count

        self._emit()

    def _sorted_stats(self):
        return [self.stations[s] for s in sorted(self.stations.keys())]

    def _emit(self):
        jitter_thresh = float(self.jitter_threshold_in())
        drift_thresh = float(self.drift_threshold_in())

        with self.__mutex:
            stats_list = self._sorted_stats()
            serials = [s.serial for s in stats_list]
            positions = [s.current_position() for s in stats_list]
            orientations = [s.current_orientation() for s in stats_list]
            jitters = [s.position_jitter_mm() for s in stats_list]
            orient_jitters = [s.orientation_jitter_deg() for s in stats_list]
            drifts = [s.drift_mm() for s in stats_list]
            valids = [s.valid for s in stats_list]

        # Stations that have never yielded a valid pose get a zero-filled row so
        # the position/orientation arrays stay aligned with `serials` and the metrics.
        pos_rows = [p if p is not None else np.zeros(3) for p in positions]
        rot_rows = [q if q is not None else np.array([1.0, 0.0, 0.0, 0.0]) for q in orientations]

        all_stable = 1
        for j, d, v in zip(jitters, drifts, valids):
            if (not v) or j > jitter_thresh or d > drift_thresh:
                all_stable = 0
                break
        if len(stats_list) == 0:
            all_stable = 0

        self.count_out.send(len(stats_list))
        self.serials_out.send(serials)
        self.positions_out.send(np.array(pos_rows, dtype=float).reshape(-1, 3))
        self.orientations_out.send(np.array(rot_rows, dtype=float).reshape(-1, 4))
        self.jitter_out.send(np.array(jitters, dtype=float))
        self.orient_jitter_out.send(np.array(orient_jitters, dtype=float))
        self.drift_out.send(np.array(drifts, dtype=float))
        self.stable_out.send(all_stable)
        self.report_out.send(self._format_report(stats_list, jitters, orient_jitters, drifts, valids,
                                                  jitter_thresh, drift_thresh))

    def _format_report(self, stats_list, jitters, orient_jitters, drifts, valids, jitter_thresh, drift_thresh):
        if not stats_list:
            return 'no base stations detected'
        lines = [f'{len(stats_list)} base station(s):']
        for s, j, oj, d, v in zip(stats_list, jitters, orient_jitters, drifts, valids):
            flags = []
            if not v:
                flags.append('NO POSE')
            if j > jitter_thresh:
                flags.append('JITTER')
            if d > drift_thresh:
                flags.append('DRIFT')
            status = 'OK' if not flags else ' '.join(flags)
            mode = f' mode={s.mode}' if s.mode else ''
            base = f' baseline@{d:.2f}mm' if s.baseline is not None else ' (no baseline)'
            p = s.current_position()
            pos = f'pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})m' if p is not None else 'pos=(none)'
            lines.append(f'  {s.serial}{mode}: {pos} pos_jitter={j:.3f}mm orient_jitter={oj:.4f}deg'
                         f'{base} n={len(s.positions)} [{status}]')
        return '\n'.join(lines)

    def set_baseline(self):
        with self.__mutex:
            done = [s.serial for s in self.stations.values() if s.set_baseline()]
        if done:
            print(f'ViveBaseStationNode: baseline captured for {len(done)} station(s): {", ".join(done)}')
        else:
            print('ViveBaseStationNode: no station samples yet to baseline (enable the node and wait a moment)')

    def reset_stats(self):
        with self.__mutex:
            self.stations.clear()
        self.last_station_count = None
        print('ViveBaseStationNode: statistics and baselines cleared')

    def print_report(self):
        with self.__mutex:
            stats_list = self._sorted_stats()
            jitters = [s.position_jitter_mm() for s in stats_list]
            orient_jitters = [s.orientation_jitter_deg() for s in stats_list]
            drifts = [s.drift_mm() for s in stats_list]
            valids = [s.valid for s in stats_list]
        print(self._format_report(stats_list, jitters, orient_jitters, drifts, valids,
                                   float(self.jitter_threshold_in()), float(self.drift_threshold_in())))

    def custom_cleanup(self) -> None:
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print('ViveBaseStationNode: monitor thread did not exit within 1s')







