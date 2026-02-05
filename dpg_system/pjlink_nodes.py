import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import socket
import hashlib
import time
import threading

def register_pjlink_nodes():
    Node.app.register_node('pjlink_projector', PJLinkNode.factory)

class PJLinkNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PJLinkNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.default_ip = '10.1.1.141'
        self.port = 4352 # Standard PJLink Port
        self.password = '@Panasonic'

        if len(args) > 0:
            self.default_ip = any_to_string(args[0])
        if len(args) > 1:
            self.password = any_to_string(args[1])
            
        self.socket = None
        self.connected = False
        self.auth_token = ''

        # Properties
        # Controls
        self.connect_input = self.add_input('connect', widget_type='checkbox', callback=self.connect)
        # self.disconnect_input = self.add_input('disconnect', widget_type='button', callback=self.disconnect)
        
        self.power_input = self.add_input('power_on', widget_type='checkbox', callback=self.set_power)
        self.shutter_input = self.add_input('shutter_mute', widget_type='checkbox', callback=self.set_shutter)
        self.freeze_input = self.add_input('freeze', widget_type='checkbox', callback=self.set_freeze)
        self.volume_input = self.add_input('volume', widget_type='slider_float', default_value=0.0, min=0.0, max=100.0, callback=self.set_volume)
        

        # Inputs: 1=RGB, 2=Video, 3=Digital, 4=Storage, 5=Network
        # Usually followed by check digit, e.g. 11 = RGB1, 31 = Digital1 (HDMI)
        self.input_code_property = self.add_input('input_code', widget_type='combo', widget_width=100, default_value='HDMI 1', callback=self.set_input)
        self.input_dict = {'RGB 1': 11, 'RGB 2': 12, 'Video 1': 21, 'Video 2': 22, 'HDMI 1': 31, 'HDMI 2': 32,
                           'HDMI 3': 33, 'Storage': 41, 'Network': 51}

        self.input_code_property.widget.combo_items = list(self.input_dict.keys())

        self.status_property = self.add_property('status', widget_type='text_input', default_value='')
        self.response_output = self.add_output('response')

        self.show_options_property = self.add_property('show_connection_options', widget_type='checkbox', callback=self.show_options_local)

        self.ip_property = self.add_option('ip', widget_type='text_input', default_value=self.default_ip)
        self.port_property = self.add_option('port', widget_type='input_int', default_value=self.port)
        self.password_property = self.add_option('password', widget_type='text_input', default_value=self.password)
        self.password_property.widget.password = True # Hide text

        # Status
        self.print_debug_opt = self.add_option('print_debug', widget_type='checkbox', default_value=False)
        self.custom_cmd_opt = self.add_option('custom_cmd', widget_type='text_input', default_value='', callback=self.send_custom)
        # self.send_custom_opt = self.add_input('send_custom', widget_type='button', callback=self.send_custom)

    def show_options_local(self):
        self.show_options('show', self.show_options_property())

    def connect(self):
        pending_connect = self.connect_input()

        if not pending_connect:
            if self.connected:
                self.disconnect()
            return

        ip = self.ip_property()
        port = self.port_property()
        password = self.password_property()
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((ip, port))
            
            # Reset Auth
            self.auth_token = ''
            
            # Initial Handshake (Read PJLINK 0 or 1 <seed>)
            data = self.socket.recv(1024).decode('utf-8').strip()
            if self.print_debug_opt():
                print(f"PJLink Handshake: {data}")
                
            if 'PJLINK 0' in data:
                # No Auth
                self.auth_token = ''
            elif 'PJLINK 1' in data:
                # Auth Required
                # PJLINK 1 <seed>
                    parts = data.split()
                    if len(parts) > 2:
                        seed = parts[2]
                        # MD5(seed + password) - Correct order usually
                        src = f"{seed}{password}"
                        self.auth_token = hashlib.md5(src.encode('utf-8')).hexdigest()
                        if self.print_debug_opt():
                            print(f"PJLink Auth Token Generated for Seed {seed}")
            
            self.connected = True
            self.status_property.set('Connected')
            
            # Query Power Status
            self.query_power()
            
        except Exception as e:
            if self.print_debug_opt():
                print(f"PJLink Connection Error: {e}")
            self.connected = False
            self.status_property.set(f'Error: {e}')

    def disconnect(self):
        if self.socket:
            self.socket.close()
        self.socket = None
        self.connected = False
        self.status_property.set('Disconnected')

    def send_command(self, cmd):
        # cmd should be e.g. "%1POWR 1"
        # We prepend auth token if it exists
        
        if not self.connected:
            self.connect()
            if not self.connected:
                return

        full_cmd = f"{self.auth_token}{cmd}\r"
        
        try:
            self.socket.sendall(full_cmd.encode('utf-8'))
            if self.print_debug_opt():
                print(f"PJLink TX: {full_cmd.strip()}")
                
            # Read response
            response = self.socket.recv(1024).decode('utf-8').strip()
            if self.print_debug_opt():
                print(f"PJLink RX: {response}")
                
            self.response_output.send(response)
            return response
        except Exception as e:
            if self.print_debug_opt():
                print(f"PJLink Send Error: {e}")
            self.disconnect()
            return None

    def set_power(self):
        state = self.power_input()
        # %1POWR 1 (On) or 0 (Off)
        val = 1 if state else 0
        cmd = f"%1POWR {val}"
        self.send_command(cmd)

    def query_power(self):
        resp = self.send_command("%1POWR ?")
        # Resp: %1POWR=1
        if resp and 'POWR=' in resp:
            try:
                state_str = resp.split('=')[1]
                state = int(state_str)
                # 0=Off, 1=On, 2=Cooling, 3=Warmup
                if state == 1:
                    self.power_input.set(True)
                else:
                    self.power_input.set(False)
            except:
                pass

    def set_shutter(self):
        state = self.shutter_input()
        # %1AVMT 30 (Mute Video+Audio open/off)
        # %1AVMT 31 (Mute Video+Audio closed/on)
        # 10/11 = Video Only
        # 20/21 = Audio Only
        # 30/31 = Both
        
        # Let's assume AV Mute (31)
        val = 31 if state else 30
        cmd = f"%1AVMT {val}"
        self.send_command(cmd)

    def set_input(self):
        code_str = self.input_code_property()
        if code_str in self.input_dict:
            code = self.input_dict[code_str]
            # Extract code "31" from "31 (Digital 1)"
            # code = code_str.split()[0]
            cmd = f"%1INPT {code}"
            self.send_command(cmd)

    def set_freeze(self):
        # %2FREZ 1 (On), 0 (Off)
        state = self.freeze_input()
        val = 1 if state else 0
        cmd = f"%2FREZ {val}"
        self.send_command(cmd)

    def set_volume(self):
        # %2VOLM <val>
        # Note: Volume range might vary or need scaling.
        # Assuming standard integer value if accepted.
        val = int(self.volume_input())
        # PJLink spec usually says volume is 0-something, e.g. 0-100 or device dependent.
        cmd = f"%2VOLM {val}"
        self.send_command(cmd)

    def send_custom(self):
        cmd = self.custom_cmd_opt()
        if cmd:
            self.send_command(cmd)

    def cleanup(self):
        self.disconnect()
