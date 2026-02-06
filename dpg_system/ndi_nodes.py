import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from dpg_system.ndi_receiver import NDIReceiver

def register_ndi_nodes():
    Node.app.register_node('ndi_receiver', NDIReceiverNode.factory)

class NDIReceiverNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NDIReceiverNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.receiver = None
        self.streaming = False
        self.source_list = []

        # Inputs
        self.on_off = self.add_input('on/off', widget_type='checkbox', callback=self.toggle_streaming)
        self.source_selector = self.add_input('source', widget_type='combo', callback=self.change_source)
        self.output_type_selector = self.add_input('output_type', widget_type='combo', default_value='numpy', callback=self.change_output_type)
        self.output_type_selector.widget.combo_items = ['numpy', 'torch']
        self.bandwidth_selector = self.add_input('bandwidth', widget_type='combo', default_value='highest', callback=self.change_bandwidth)
        self.bandwidth_selector.widget.combo_items = ['active', 'highest', 'lowest', 'audio only']
        self.refresh_button = self.add_input('refresh sources', widget_type='button', callback=self.refresh_sources)

        # Output
        self.output = self.add_output('image')

        # Initialize Receiver
        try:
            self.receiver = NDIReceiver()
        except Exception as e:
            print(f"NDI Init Error: {e}")

        # Initial scan
        # self.refresh_sources() # deferred to avoid startup hang if network scan is slow

    def refresh_sources(self):
        if self.receiver:
            sources = self.receiver.find_sources(timeout_secs=2.0)
            # Store source objects or just names? NDIReceiver.connect takes name or object.
            # But the combo box needs strings.
            self.source_list = sources # Keep full objects if needed, but current NDIReceiver.find_sources returns objects I think?
            # actually my NDIReceiver.find_sources returns a list of NDIlib_source objects usually?
            # Let's check my implementation of ndi_receiver.py.
            # It returns a list of sources. And connect can handle strings (via search) or objects.
            
            # Let's verify what find_sources returns. In my implementation it returned 'sources' from find_get_current_sources
            # which are NDIlib objects.
            
            source_names = [s.ndi_name for s in self.source_list]
            self.source_selector.widget.combo_items = source_names
            dpg.configure_item(self.source_selector.widget.uuid, items=source_names)
            
            if source_names:
                # Optional: Auto-select first if none selected?
                pass

    def change_source(self):
        name = self.source_selector()
        if self.receiver and name:
            # Pass string to connect, my NDIReceiver handles re-finding or we can pass object if we mapped it.
            # My NDIReceiver.connect handles string scan.
            self.receiver.connect(name)

    def change_output_type(self):
        pass

    def change_bandwidth(self):
        if self.receiver:
            import NDIlib as ndi
            bw_map = {
                'metadata': ndi.RECV_BANDWIDTH_METADATA_ONLY,
                'audio only': ndi.RECV_BANDWIDTH_AUDIO_ONLY,
                'lowest': ndi.RECV_BANDWIDTH_LOWEST,
                'highest': ndi.RECV_BANDWIDTH_HIGHEST
            }
            selection = self.bandwidth_selector()
            if selection in bw_map:
                self.receiver.set_bandwidth(bw_map[selection])

    def toggle_streaming(self):
        on = self.on_off()
        if on != self.streaming:
            if on:
                if not self.receiver.connected_source:
                   # Try to connect to current combo value if not connected
                   name = self.source_selector()
                   if name:
                       self.receiver.connect(name)
                
                # Start threaded capture
                self.receiver.start_capture()
                self.add_frame_task()
            else:
                self.remove_frame_tasks()
                self.receiver.stop_capture()
            self.streaming = on

    def frame_task(self):
        if self.receiver:
            # Get latest frame from thread
            frame, _ = self.receiver.read()
            
            if frame is not None:
                out_type = self.output_type_selector()
                
                if out_type == 'torch' and torch is not None:
                    # Convert to tensor on demand
                    tensor = torch.from_numpy(frame)
                    # Permute to (C, H, W)
                    tensor = tensor.permute(2, 0, 1)
                    self.output.send(tensor)
                else:
                    self.output.send(frame)

    def cleanup(self):
        self.remove_frame_tasks()
        if self.receiver:
            self.receiver.destroy()
