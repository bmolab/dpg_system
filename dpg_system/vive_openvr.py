# Vendored from triad_openvr (https://github.com/TriadSemi/triad_openvr, at d389aac),
# with dpg_system changes: device names stable per serial, a serial -> name
# names file (load_names / save_names), and a lock around the device list.
#
# Needs the 'openvr' package and a running SteamVR, so in practice Linux only.
# Importing this without openvr raises ModuleNotFoundError, which is how
# dpg_app skips vive_tracker_nodes on platforms that don't have it.

import time
import sys
import openvr
import math
import json
import os
import threading

from functools import lru_cache

# Function to print out text but instead of starting a new line it will overwrite the existing line
def update_text(txt):
    sys.stdout.write('\r'+txt)
    sys.stdout.flush()

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Euler angles (in degrees)
def convert_to_euler(pose_mat):
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,yaw,pitch,roll]

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Quaternion
def convert_to_quaternion(pose_mat):
    # Per issue #2, adding a abs() so that sqrt only results in real numbers
    r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
    r_x = (pose_mat[2][1]-pose_mat[1][2])/(4*r_w)
    r_y = (pose_mat[0][2]-pose_mat[2][0])/(4*r_w)
    r_z = (pose_mat[1][0]-pose_mat[0][1])/(4*r_w)

    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,r_w,r_x,r_y,r_z]

#Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer():
    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self,pose_mat,t):
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] /pose_mat[0][0]))
        self.pitch.append(180 / math.pi * math.atan(-1 * pose_mat[2][0] / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2))))
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] /pose_mat[2][2]))
        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1]-pose_mat[1][2])/(4*r_w))
        self.r_y.append((pose_mat[0][2]-pose_mat[2][0])/(4*r_w))
        self.r_z.append((pose_mat[1][0]-pose_mat[0][1])/(4*r_w))

def get_pose(vr_obj):
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


class vr_tracked_device():
    def __init__(self,vr_obj,index,device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj

    @lru_cache(maxsize=None)
    def get_serial(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)

    def is_charging(self):
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)


    def sample(self,num_samples,sample_rate):
        interval = 1/sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(pose[self.index].mDeviceToAbsoluteTracking,time.time()-sample_start)
            sleep_time = interval- (time.time()-start)
            if sleep_time>0:
                time.sleep(sleep_time)
        return rtn

    def get_pose_euler(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_pose_matrix(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None

    def get_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vVelocity
        else:
            return None

    def get_angular_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vAngularVelocity
        else:
            return None

    def get_pose_quaternion(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def controller_state_to_dict(self, pControllerState):
        # This function is graciously borrowed from https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
        # docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
        d = {}
        d['unPacketNum'] = pControllerState.unPacketNum
        # on trigger .y is always 0.0 says the docs
        d['trigger'] = pControllerState.rAxis[1].x
        # 0.0 on trigger is fully released
        # -1.0 to 1.0 on joystick and trackpads
        d['trackpad_x'] = pControllerState.rAxis[0].x
        d['trackpad_y'] = pControllerState.rAxis[0].y
        # These are published and always 0.0
        # for i in range(2, 5):
        #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
        #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
        d['ulButtonPressed'] = pControllerState.ulButtonPressed
        d['ulButtonTouched'] = pControllerState.ulButtonTouched
        # To make easier to understand what is going on
        # Second bit marks menu button
        d['menu_button'] = bool(pControllerState.ulButtonPressed >> 1 & 1)
        # 32 bit marks trackpad
        d['trackpad_pressed'] = bool(pControllerState.ulButtonPressed >> 32 & 1)
        d['trackpad_touched'] = bool(pControllerState.ulButtonTouched >> 32 & 1)
        # third bit marks grip button
        d['grip_button'] = bool(pControllerState.ulButtonPressed >> 2 & 1)
        # System button can't be read, if you press it
        # the controllers stop reporting
        return d

    def get_controller_inputs(self):
        result, state = self.vr.getControllerState(self.index)
        return self.controller_state_to_dict(state)

    def trigger_haptic_pulse(self, duration_micros=1000, axis_id=0):
        """
        Causes devices with haptic feedback to vibrate for a short time.
        """
        self.vr.triggerHapticPulse(self.index ,axis_id, duration_micros)

class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        return self.vr.getStringTrackedDeviceProperty(self.index,openvr.Prop_ModeLabel_String).decode('utf-8').upper()
    def sample(self,num_samples,sample_rate):
        print("Warning: Tracking References do not move, sample isn't much use...")

class triad_openvr():
    def __init__(self, configfile_path=None):
        # Initialize OpenVR in the
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

        # Initializing object to hold indexes for various tracked objects
        self.object_names = {"Tracking Reference":[],"HMD":[],"Controller":[],"Tracker":[]}
        self.devices = {}
        self.device_index_map = {}
        # serial -> name, so a device that drops out and comes back keeps its
        # original name instead of being renumbered (and colliding with others)
        self.serial_names = {}
        self.config_entries = []   # entries read from the names file
        # devices / object_names are mutated by poll_vr_events and read from
        # several node threads, so all access goes through this lock
        self.lock = threading.RLock()
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                               openvr.k_unMaxTrackedDeviceCount)

        # A config file pins names to serials ({"devices": [{"name", "type", "serial"}]}).
        # Devices not listed in it are still picked up and numbered as usual.
        if configfile_path:
            try:
                self.load_names(configfile_path)
            except Exception as e:
                print(f'triad_openvr: could not read names file {configfile_path} ({e})')

        # Iterate through the pose list to find the active devices and determine their type
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                self.add_tracked_device(i)

    def __del__(self):
        openvr.shutdown()

    def get_pose(self):
        return get_pose(self.vr)

    def poll_vr_events(self):
        """
        Used to poll VR events and find any new tracked devices or ones that are no longer tracked.
        """
        event = openvr.VREvent_t()
        with self.lock:
            while self.vrsystem.pollNextEvent(event):
                if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                    self.add_tracked_device(event.trackedDeviceIndex)
                elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                    #If we were already tracking this device, quit tracking it.
                    if event.trackedDeviceIndex in self.device_index_map:
                        self.remove_tracked_device(event.trackedDeviceIndex)

    def _name_for(self, i, prefix):
        # Reuse the name this serial had before; otherwise take the lowest free
        # number. (Numbering by list length reissued names still in use, so a
        # reconnecting tracker could overwrite a different tracker's entry.)
        serial = self.vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
        name = self.serial_names.get(serial)
        if name is None or name in self.devices:
            taken = set(self.devices) | set(self.serial_names.values())
            n = 1
            while prefix + str(n) in taken:
                n += 1
            name = prefix + str(n)
            self.serial_names[serial] = name
        return name

    def add_tracked_device(self, tracked_device_index):
        i = tracked_device_index
        kinds = {
            openvr.TrackedDeviceClass_Controller: ("controller_", "Controller", vr_tracked_device),
            openvr.TrackedDeviceClass_HMD: ("hmd_", "HMD", vr_tracked_device),
            openvr.TrackedDeviceClass_GenericTracker: ("tracker_", "Tracker", vr_tracked_device),
            openvr.TrackedDeviceClass_TrackingReference: ("tracking_reference_", "Tracking Reference", vr_tracking_reference),
        }
        with self.lock:
            if i in self.device_index_map:
                # duplicate activation for a device we already hold
                return
            kind = kinds.get(self.vr.getTrackedDeviceClass(i))
            if kind is None:
                return
            prefix, device_type, cls = kind
            device_name = self._name_for(i, prefix)
            self.object_names[device_type].append(device_name)
            self.devices[device_name] = cls(self.vr, i, device_type)
            self.device_index_map[i] = device_name

    def load_names(self, path):
        """Read serial -> name assignments from a config file and rename any
        connected devices to match. Returns False if the file doesn't exist."""
        if not os.path.exists(path):
            return False
        with open(path, 'r') as f:
            entries = json.load(f).get('devices', [])
        with self.lock:
            self.config_entries = entries
            configured = {e['serial']: e['name'] for e in entries}
            claimed = set(configured.values())
            # keep automatic names for unlisted devices unless the file claims them
            kept = {s: n for s, n in self.serial_names.items() if s not in configured and n not in claimed}
            self.serial_names = {**kept, **configured}
            live = list(self.device_index_map)
            for i in live:
                self.remove_tracked_device(i)
            for i in live:
                self.add_tracked_device(i)
        return True

    def save_names(self, path):
        """Write the current name of every connected device, plus any entries
        from the loaded file for devices that aren't connected right now."""
        with self.lock:
            entries = [{'name': name, 'type': self.devices[name].device_class,
                        'serial': self.devices[name].get_serial()}
                       for name in sorted(self.devices)]
            live_serials = {e['serial'] for e in entries}
            entries += [e for e in self.config_entries if e['serial'] not in live_serials]
            self.config_entries = entries
        with open(path, 'w') as f:
            json.dump({'devices': entries}, f, indent=4)

    def remove_tracked_device(self, tracked_device_index):
        with self.lock:
            if tracked_device_index in self.device_index_map:
                device_name = self.device_index_map[tracked_device_index]
                self.object_names[self.devices[device_name].device_class].remove(device_name)
                del self.device_index_map[tracked_device_index]
                del self.devices[device_name]
                return
        raise Exception("Tracked device index {} not valid. Not removing.".format(tracked_device_index))

    def rename_device(self,old_device_name,new_device_name):
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        for i in range(len(self.object_names[self.devices[new_device_name].device_class])):
            if self.object_names[self.devices[new_device_name].device_class][i] == old_device_name:
                self.object_names[self.devices[new_device_name].device_class][i] = new_device_name

    def print_discovered_objects(self):
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type])!=1:
                plural+="s"
            print("Found "+str(len(self.object_names[device_type]))+" "+plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", Mode "+self.devices[device].get_model()+
                          ", "+self.devices[device].get_model()+
                          ")")
                else:
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", "+self.devices[device].get_model()+")")
