import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import socket
import struct
import struct
import threading
import time

def register_visca_nodes():
    Node.app.register_node('visca_camera', ViscaNode.factory)
    Node.app.register_node('ptz_camera', ViscaNode.factory)

class ViscaNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ViscaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.default_ip = '10.1.1.160'
        self.port = 52381  # Default Sony VISCA over IP port
        self.sequence_number = 1

        if len(args) > 0:
            self.default_ip = any_to_string(args[0])
        if len(args) > 1:
            self.port = any_to_int(args[1])
            
        print(f"VISCA Node Initialized: IP={self.default_ip}, Port={self.port}")

        self.socket = None
        self.connected = False
        self.pan_active = False
        self.tilt_active = False
        self.zoom_active = False

        # --- Inputs ---
        # Control Inputs
        self.pan_input = self.add_input('pan', widget_type='slider_int', widget_width=120, default_value=0.0, min=-20.0, max=20.0, callback=self.drive_pan)
        self.tilt_input = self.add_input('tilt', widget_type='slider_int', widget_width=120, default_value=0.0, min=-20.0, max=20.0, callback=self.drive_tilt)
        
        self.zoom_input = self.add_input('zoom', widget_type='slider_int', widget_width=120, default_value=0.0, min=-7.0, max=7.0, callback=self.drive_zoom_init)
        self.focus_input = self.add_input('focus', widget_type='slider_int', widget_width=120, default_value=0, min=-1, max=1, callback=self.drive_focus)
        self.auto_focus_input = self.add_input('auto_focus', widget_type='checkbox', callback=self.set_auto_focus)
        
        # Triggers
        self.stop_input = self.add_input('stop', widget_type='button', callback=self.stop_all)
        self.home_input = self.add_input('home', widget_type='button', callback=self.go_home)
        self.reset_seq_input = self.add_input('reset_sequence', widget_type='button', callback=self.reset_sequence)
        self.reconnect_input = self.add_input('reconnect', widget_type='button', callback=self.create_socket)
        self.power_input = self.add_input('power', widget_type='checkbox', callback=self.set_power)
        
        # Presets
        self.preset_recall_input = self.add_input('preset_recall', widget_type='input_int', callback=self.recall_preset)
        self.preset_store_input = self.add_input('preset to store', widget_type='input_int')
        self.preset_store_button = self.add_input('store preset', widget_type='button', callback=self.store_preset)

        # Debug / Custom
        self.custom_cmd_input = self.add_input('custom_cmd_hex', widget_type='text_input', default_value='81 01 06 04 FF', callback=self.send_custom)
        self.send_custom_btn = self.add_input('send_custom', widget_type='button', callback=self.send_custom)
        
        self.pan_abs_input = self.add_input('pan_abs', widget_type='input_int', default_value=0)
        self.tilt_abs_input = self.add_input('tilt_abs', widget_type='input_int', default_value=0)
        self.abs_speed_input = self.add_input('abs_speed', widget_type='drag_int', default_value=5, min=1, max=18)
        self.drive_abs_btn = self.add_input('drive_absolute', widget_type='button', callback=self.drive_absolute)

        self.show_options_property = self.add_property('show_connection_options', widget_type='checkbox', callback=self.show_options_local)
        self.ip_property = self.add_option('ip', widget_type='text_input', default_value=self.default_ip)
        self.port_property = self.add_option('port', widget_type='input_int', default_value=self.port)

        self.print_debug_opt = self.add_option('print debug', widget_type='checkbox')
        # Setup socket
        self.create_socket()

        self.rx_thread = None
        self.rx_running = False
        self.start_rx_thread()
        self.pan_initiated = False
        self.tilt_initiated = False
        self.zoom_initiated = False

    def show_options_local(self):
        self.show_options('show', self.show_options_property())

    def custom_create(self, from_file):
        self.add_frame_task()

    def start_rx_thread(self):
        if self.rx_thread and self.rx_thread.is_alive():
            return
        self.rx_running = True
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.rx_thread.start()

    def frame_task(self):
        try:
            if dpg.is_item_deactivated(self.pan_input.widget.uuid) and self.pan_active:
                if self.pan_active:
                    self.pan_release()
                self.pan_initiated = False

            if dpg.is_item_deactivated(self.tilt_input.widget.uuid) and self.tilt_active:
                if self.tilt_active:
                    self.tilt_release()
                self.tilt_initiated = False

            if dpg.is_item_deactivated(self.zoom_input.widget.uuid) and self.zoom_active:
                if self.zoom_active:
                    self.zoom_release()
                self.zoom_initiated = False

            # print(self.pan_initiated, self.tilt_initiated, self.zoom_initiated)
            if self.pan_initiated or self.tilt_initiated:
                self.drive_pan_tilt()
            if self.zoom_initiated:
                self.drive_zoom()

        except Exception:
            # Ignore DPG thread errors (SystemError, Item not found, etc.)
            pass

    def stop_all(self):
        # Pan/Tilt Stop: 81 01 06 01 VV WW 03 03 FF (VV=Pan speed, WW=Tilt Speed) - Wait, Stop is 03 03? 
        # Actually standard PT Drive Stop is: 8x 01 06 01 VV WW 03 03 FF
        ps = 5
        ts = 5
        cmd = bytearray([0x06, 0x01, ps, ts, 0x03, 0x03])
        self.send_packet(self.build_visca_command(cmd))
        
        self.pan_input.set(0.0)
        self.tilt_input.set(0.0)
        
        # Zoom Stop: 8x 01 04 07 00 FF
        cmd_zoom = bytearray([0x04, 0x07, 0x00])
        self.send_packet(self.build_visca_command(cmd_zoom))
        
        # Focus Stop: 8x 01 04 08 00 FF
        cmd_focus = bytearray([0x04, 0x08, 0x00])
        self.send_packet(self.build_visca_command(cmd_focus))

    def go_home(self):
        # Home: 8x 01 06 04 FF
        cmd = bytearray([0x06, 0x04])
        self.send_packet(self.build_visca_command(cmd))

    def set_power(self):
        # Power On: 8x 01 04 00 02 FF
        # Power Off: 8x 01 04 00 03 FF
        state = self.power_input()
        if state:
            cmd = bytearray([0x04, 0x00, 0x02])
        else:
            cmd = bytearray([0x04, 0x00, 0x03])
        self.send_packet(self.build_visca_command(cmd))

    def pan_release(self):
        # Reset to 0 and Stop
        self.pan_initiated = False
        self.pan_input.set(0.0)
        # We need to send a stop command OR just re-trigger drive_pan which will see 0
        self.drive_pan_tilt()

    def tilt_release(self):
        self.tilt_active = False

        self.tilt_input.set(0.0)
        self.drive_pan_tilt()

    def zoom_release(self):
        self.zoom_active = False
        self.zoom_input.set(0.0)
        self.drive_zoom()
        # Force stop
        # cmd = bytearray([0x04, 0x07, 0x00])
        # self.send_packet(self.build_visca_command(cmd))

    def drive_pan_tilt(self):
        val = self.pan_input()
        # Calculate Speed from value magnitude (1-24)
        # Clamp value to -20, 20
        # Map 0-20 to 1-24? Or just use raw value capped at 24?
        # User asked for range += 20.
        # Let's map 0-20 to roughly 0-24? Or just use it as is?
        # Let's assume 1:1 for simplicity, maybe cap at max speed (0x18 = 24).
        
        speed = int(abs(val))
        if speed > 24: speed = 24
        if speed < 1: speed = 1 # Minimum speed if moving
        
        # Pan Left: 0x01, Right: 0x02, Stop: 0x03
        pan_dir = 0x03
        
        if abs(val) < 0.1: # Deadzone/Stop
            pan_dir = 0x03
            speed = 0 # Dummy
        elif val > 0:
            pan_dir = 0x02 # Right
        elif val < 0:
            pan_dir = 0x01 # Left
            
        # We need a tilt speed. Since we are driving independent axes, 
        # normally we should respect the other axis state.
        # But for momentary sliders, presumably we only control one at a time OR 
        # we need to query the other slider's current state.
        # Since 'drive_pan' is called by pan slider, let's look at tilt slider value too.
        
        tilt_val = self.tilt_input()
        t_speed = int(abs(tilt_val))
        if t_speed > 20: t_speed = 20 # Tilt max is usually lower (0x14 = 20)
        if t_speed < 1: t_speed = 1
        
        tilt_dir = 0x03
        if abs(tilt_val) < 0.1:
            tilt_dir = 0x03
            t_speed = 0 #dummy
        elif tilt_val > 0:
            tilt_dir = 0x01 # Up
        elif tilt_val < 0:
            tilt_dir = 0x02 # Down

        if pan_dir != 0x03:
            self.pan_active = True
        else:
            self.pan_active = False
        if tilt_dir != 0x03:
            self.tilt_active = True
        else:
            self.tilt_active = False

        # 8x 01 06 01 VV WW 03 03 FF (Stop)
        # Note: If we are stopping Pan (dir 03), speed matters less but usually 0 or current.
        cmd = bytearray([0x06, 0x01, speed, t_speed, pan_dir, tilt_dir])
        self.send_packet(self.build_visca_command(cmd))

    def drive_pan(self):
        if self.pan_active or self.pan_input() != 0.00:
            self.pan_initiated = True

    def drive_tilt(self):
        if self.tilt_active or self.tilt_input() != 0.00:
            self.tilt_initiated = True

    def drive_zoom_init(self):
        if self.zoom_active or self.zoom_input() != 0.00:
            self.zoom_initiated = True

    def drive_zoom(self):
        val = self.zoom_input()
        # Zoom Tele (Standard): 8x 01 04 07 02 FF
        # Zoom Wide (Standard): 8x 01 04 07 03 FF
        # Zoom Stop: 8x 01 04 07 00 FF
        # Variable speed is: 8x 01 04 07 2p FF (Tele), 3p (Wide) where p=0-7 speed

        # squared_val = val * val
        # if val < 0:
        #     squared_val = -squared_val
        # squared_val *= 7
        # val = int(squared_val)
        #
        speed = int(abs(val))
        if speed > 7: speed = 7

        if val > 0:
            # Tele
            byte = 0x20 | speed
            cmd = bytearray([0x04, 0x07, byte])
            self.zoom_active = True
        elif val < 0:
            # Wide
            byte = 0x30 | speed
            cmd = bytearray([0x04, 0x07, byte])
            self.zoom_active = True
        else:
            # Stop
            cmd = bytearray([0x04, 0x07, 0x00])
            self.zoom_active = False
            
        self.send_packet(self.build_visca_command(cmd))

    def drive_focus(self):
        val = self.focus_input()
        # Focus Far (Standard): 8x 01 04 08 02 FF
        # Focus Near (Standard): 8x 01 04 08 03 FF
        # Focus Stop: 8x 01 04 08 00 FF

        if val > 0:
            # Far
            cmd = bytearray([0x04, 0x08, 0x02])
        elif val < 0:
            # Near
            cmd = bytearray([0x04, 0x08, 0x03])
        else:
            # Stop
            cmd = bytearray([0x04, 0x08, 0x00])
            
        self.send_packet(self.build_visca_command(cmd))

    def set_auto_focus(self):
        # Auto Focus On: 8x 01 04 38 02 FF
        # Auto Focus Off (Manual): 8x 01 04 38 03 FF
        state = self.auto_focus_input()
        if state:
            # Auto Focus
            cmd = bytearray([0x04, 0x38, 0x02])
        else:
            # Manual Focus
            cmd = bytearray([0x04, 0x38, 0x03])
        self.send_packet(self.build_visca_command(cmd))

    def recall_preset(self):
        preset_id = self.preset_recall_input()
        if preset_id < 0 or preset_id > 255:
            return 
        # Recall: 8x 01 04 3F 02 pp FF
        cmd = bytearray([0x04, 0x3F, 0x02, preset_id])
        self.send_packet(self.build_visca_command(cmd))

    def store_preset(self):
        preset_id = self.preset_store_input()
        if preset_id < 0 or preset_id > 255:
            return
        # Store: 8x 01 04 3F 01 pp FF
        cmd = bytearray([0x04, 0x3F, 0x01, preset_id])
        self.send_packet(self.build_visca_command(cmd))

    def rx_loop(self):
        if self.connected and self.socket:
            try:
                data, addr = self.socket.recvfrom(1024)
                if self.print_debug_opt():
                    print(f"VISCA RX ({addr[0]}:{addr[1]}) | {' '.join(f'{b:02X}' for b in data)}")
            except socket.timeout:
                pass
            except OSError:
                # Socket closed or error
                pass
            except Exception as e:
                if self.print_debug_opt():
                    print(f"VISCA RX Error: {e}")
        else:
            time.sleep(0.1)

    def create_socket(self):
        if self.socket:
            self.socket.close()
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', 0))  # Explicitly bind to ephemeral port
            self.socket.settimeout(0.5)  # Non-blocking-ish for thread
            self.connected = True
            if self.print_debug_opt():
                local_port = self.socket.getsockname()[1]
                print(
                    f"VISCA Socket Created & Bound on Local Port: {local_port}. Destination: {self.ip_property()}:{self.port_property()}")
        except Exception as e:
            print(f"VISCA: Error creating socket: {e}")
            self.connected = False

    def send_packet(self, payload):
        if not self.connected or not self.socket:
            self.create_socket()

        # VISCA over IP Header
        # Byte 0: Payload type (0x01 = VISCA Command, 0x02 = VISCA Inquiry)
        # Byte 1: Payload length (high byte)
        # Byte 2: Payload length (low byte)
        # Byte 3: Sequence number (0x00 to 0xFF, uint32 really but usually byte sequence)

        # Byte 3: Sequence number (0x00 to 0xFF, uint32 really but usually byte sequence)

        # Note: Sony VISCA over IP header is 8 bytes.
        # 0x01 0x00 (Payload Type: Command)
        # 0x00 0xXX (Length of payload)
        # 0xXX 0xXX 0xXX 0xXX (Sequence Number)

        payload_len = len(payload)

        # Always use header (Canon/Sony over IP standard)
        header = bytearray([0x01, 0x00, 0x00, payload_len])

        # Sequence number handling (4 bytes, big endian)
        seq_bytes = self.sequence_number.to_bytes(4, 'big')
        header.extend(seq_bytes)

        message = header + payload

        try:
            target_ip = self.ip_property()
            target_port = self.port_property()
            self.socket.sendto(message, (target_ip, target_port))
            if self.print_debug_opt():
                local_port = self.socket.getsockname()[1]
                print(
                    f"VISCA TX ({target_ip}:{target_port}) [Src:{local_port}] Seq:{self.sequence_number} | {' '.join(f'{b:02X}' for b in message)}")

            self.sequence_number += 1
            if self.sequence_number > 0xFFFFFFFF:
                self.sequence_number = 1
        except Exception as e:
            print(f"VISCA: Send Error: {e}")

    def reset_sequence(self):
        self.sequence_number = 1

        # IF_Clear Broadcast: 88 01 00 01 FF
        cmd = bytearray([0x88, 0x01, 0x00, 0x01, 0xFF])

        if self.print_debug_opt():
            print("VISCA: Resetting Sequence Number...")

        self.send_packet(cmd)

    def build_visca_command(self, cmd_bytes):
        # 8x 01 ... FF
        # x is camera address, usually 1 for VISCA over IP
        prefix = bytearray([0x81, 0x01]) 
        terminator = bytearray([0xFF])
        return prefix + cmd_bytes + terminator

    def send_custom(self):
        # Expects hex string like "81 01 06 04 FF"
        hex_str = self.custom_cmd_input()
        try:
            # Remove spaces and convert to bytes
            clean_hex = hex_str.replace(' ', '')
            cmd = bytearray.fromhex(clean_hex)
            # Send directly (send_packet handles header)
            # Wait, send_packet expects payload. 
            # If user provides full 81...FF command, we should pass it as payload?
            # Yes, build_visca_command adds 81 01 and FF. 
            # But custom command might be fully formed or not.
            # Let's assume user provides full command including 81 and FF.
            # So we SHOULD NOT use build_visca_command if it already has 81.
            
            # Check if it starts with 81
            if cmd[0] == 0x81:
                # Raw payload
                self.send_packet(cmd)
            else:
                # Assume it's the inner part
                self.send_packet(self.build_visca_command(cmd))
        except Exception as e:
            print(f"Error parsing hex: {e}")

    def drive_absolute(self):
        pan = self.pan_abs_input()
        tilt = self.tilt_abs_input()
        speed = self.abs_speed_input()
        
        # 8x 01 06 02 VV WW 0Y 0Y 0Y 0Y 0Z 0Z 0Z 0Z FF
        # VV = Pan Speed, WW = Tilt Speed (use same for both for now)
        vv = speed
        ww = speed
        
        # Pan Bytes (4 bytes big endian generally, but VISCA might use nibbles?)
        # Spec says: 0Y 0Y 0Y 0Y. 
        # This usually means Y is a nibble. e.g. Position 0x1234 -> 00 00 01 02 03 04? 
        # Wait, YYYY is the hex value. 
        # Standard VISCA Absolute Position:
        # YYYY: Pan Position (0xYYYY)
        # ZZZZ: Tilt Position (0xZZZZ)
        # 
        # BUT the packet has 0Y 0Y 0Y 0Y.
        # This implies standard VISCA usage where bytes are spread out?
        # NO, usually it's just 4 bytes: Y1 Y2 Y3 Y4?
        # Let's check spec details carefully.
        # "0Y 0Y 0Y 0Y"
        # If position is P = 0xABCD.
        # Bytes are: 0A 0B 0C 0D.
        # Yes, nibblized!
        
        # Helper to nibblize
        def to_nibbles(val):
            # Val is int.
            # Handle negative (two's complement 16 bit? or just 4 nibbles?)
            # Usually 16-bit signed integer for Pan/Tilt.
            val = int(val)
            if val < 0:
                val = (1 << 16) + val # 2's complement for 16 bit
            
            # Now we have 0 to 65535.
            # 0xABCD
            n1 = (val >> 12) & 0xF
            n2 = (val >> 8) & 0xF
            n3 = (val >> 4) & 0xF
            n4 = val & 0xF
            return [n1, n2, n3, n4] # Actually need 0Y .. so [n1, n2, n3, n4] is correct if we prefix 0.
            
        p_nibbles = to_nibbles(pan)
        t_nibbles = to_nibbles(tilt)
        
        # 81 01 06 02 VV WW 0Y 0Y 0Y 0Y 0Z 0Z 0Z 0Z FF
        # We need to construct the inner core (06 02 ...) if using build_visca
        # Or just build the whole thing.
        # Let's use build_visca_command which takes the inner part (Command Data).
        # Inner: 06 02 VV WW ...
        
        payload = bytearray([0x06, 0x02, vv, ww])
        payload.extend(p_nibbles) # 0Y 0Y 0Y 0Y? No wait, bytearray takes bytes.
        # We need [0x0Y, 0x0Y, ...]
        
        for n in p_nibbles:
            payload.append(n) # n is 0-15, which is 0x0N. Correct.
            
        for n in t_nibbles:
            payload.append(n)
            
        self.send_packet(self.build_visca_command(payload))

    def cleanup(self):
        self.rx_running = False
        if self.socket:
            self.socket.close()
