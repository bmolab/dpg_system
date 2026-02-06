import NDIlib as ndi
import numpy as np
import time
import sys

# Optional: Torch for tensor output
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import threading

class NDIReceiver:
    def __init__(self, source_name=None, color_format=ndi.RECV_COLOR_FORMAT_RGBX_RGBA, bandwidth=ndi.RECV_BANDWIDTH_HIGHEST):
        """
        Initialize the NDI Receiver.
        
        Args:
            source_name (str, optional): The name of the NDI source to connect to immediately.
            color_format: NDI color format preference. Default is BGRX/BGRA (suitable for OpenCV).
            bandwidth: NDI bandwidth preference (HIGHEST or LOWEST).
        """
        if not ndi.initialize():
            raise RuntimeError("Cannot initialize NDI runtime.")
        
        self.ndi_recv = None
        self.ndi_find = None
        self.color_format = color_format
        self.bandwidth = bandwidth
        self.connected_source = None
        
        # Threading state
        self.capture_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0
        
        # Create the receiver instance
        self._create_receiver()
        # Create the finder instance
        self._create_finder()
        
        if source_name:
            self.connect(source_name)

    def _create_receiver(self):
        recv_create_desc = ndi.RecvCreateV3()
        recv_create_desc.color_format = self.color_format
        recv_create_desc.bandwidth = self.bandwidth
        self.ndi_recv = ndi.recv_create_v3(recv_create_desc)
        if self.ndi_recv is None:
            raise RuntimeError("Failed to create NDI receiver.")
            
    def set_bandwidth(self, bandwidth):
        """
        Change the bandwidth setting. Recreates the receiver.
        """
        if self.bandwidth != bandwidth:
            self.bandwidth = bandwidth
            
            was_running = self.running
            if was_running:
                self.stop_capture()
            
            # Destroy old receiver
            if self.ndi_recv:
                ndi.recv_destroy(self.ndi_recv)
                self.ndi_recv = None
                
            # Recreate
            self._create_receiver()
            
            # Reconnect if we were connected
            if self.connected_source:
                ndi.recv_connect(self.ndi_recv, self.connected_source)
                
            if was_running:
                self.start_capture()

    def _create_finder(self):
        find_desc = ndi.FindCreate()
        self.ndi_find = ndi.find_create_v2(find_desc)
        if self.ndi_find is None:
            print("Failed to create NDI find instance.")

    def find_sources(self, timeout_secs=3.0):
        """
        Scan for available NDI sources on the network.
        
        Args:
            timeout_secs (float): How long to wait for sources to announce themselves.
            
        Returns:
            list: A list of source names (str).
        """
        if self.ndi_find is None:
            return []
        
        sources_list = []
        start_time = time.time()
        
        # Wait for sources
        print(f"Scanning for NDI sources for {timeout_secs} seconds...")
        while time.time() - start_time < timeout_secs:
            ndi.find_wait_for_sources(self.ndi_find, 1000)
            sources = ndi.find_get_current_sources(self.ndi_find)
            sources_list = sources
            # Keep scanning for the full timeout to find all sources
            
        return sources_list

    def connect(self, source):
        """
        Connect to a specific NDI source.
        
        Args:
            source: Can be a source name string or an NDIlib_source object from find_sources.
        """
        target_source = None
        
        if isinstance(source, str):
            # If string, we need to find the source object or create a manual one
            print(f"Searching for source matching: {source}")
            # We assume find_sources has been called or populated, or we wait a bit
            # Use a quick scan if we don't have it
            sources = self.find_sources(timeout_secs=1.0)
            for s in sources:
                try:
                    name = s.ndi_name
                except:
                    continue
                    
                if source == name or source in name:
                    target_source = s
                    break
            
            if target_source is None:
                print(f"Could not find source: {source}")
                return False
        else:
            target_source = source
            
        try:
            print(f"Connecting to {target_source.ndi_name}...")
        except:
            print("Connecting to <Unknown Source Name>...")
            
        ndi.recv_connect(self.ndi_recv, target_source)
        self.connected_source = target_source
        return True
        
    def start_capture(self):
        """Start the background capture thread."""
        if self.running:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop_capture(self):
        """Stop the background capture thread."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
            self.capture_thread = None

    def _capture_loop(self):
        """Background loop to capture frames."""
        print("NDI Capture thread started.")
        while self.running:
            # Use a short timeout so we can check 'running' frequently
            # But not too short to cause busy waiting if NDI is slow. 
            # 50ms is decent (20fps min polling), but usually NDI returns faster if frame exists.
            frame, ts = self.receive_array(timeout_ms=50)
            
            if frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_ts = ts
                    
        print("NDI Capture thread user stopped.")

    def read(self):
        """
        Get the latest captured frame (thread-safe). 
        Returns (frame, timestamp) or (None, None).
        """
        # If not threaded, user should use receive_array directly or we could support fallback?
        # For this implementation, read() implies using the threaded buffer.
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame, self.latest_ts
        return None, None

    def _get_frame_shape(self, video_frame):
        """
        Calculate numpy shape from video frame metadata.
        """
        height = video_frame.yres
        width = video_frame.xres
        channels = 4 # Assumes BGRX or BGRA which are 4 channel
        return (height, width, channels)

    def receive_array(self, timeout_ms=1000):
        """
        Capture a frame and return it as a numpy array.
        
        Returns:
            tuple: (frame_array, timestamp) or (None, None) if no frame.
                   frame_array is shape (H, W, 4) (BGRA/BGRX).
                   timestamp is the NDI timestamp.
        """
        t, v, a, m = ndi.recv_capture_v2(self.ndi_recv, timeout_ms)
        
        if t == ndi.FRAME_TYPE_VIDEO:
            # We have a video frame
            height = v.yres
            width = v.xres
            
            try:
                # The buffer Protocol in newer numpy/python might accept v.data
                # We make a copy because we need to free the NDI frame immediately after.
                frame_data = np.copy(np.frombuffer(v.data, dtype=np.uint8))
                
                # Reshape
                # BGRX/BGRA is 4 bytes per pixel.
                frame_data = frame_data.reshape((height, width, 4))
                
                timestamp = v.timestamp
                return frame_data, timestamp
                
            except Exception as e:
                print(f"Error converting frame: {e}")
                return None, None
            finally:
                # Crucial: Release the frame back to the SDK
                ndi.recv_free_video_v2(self.ndi_recv, v)
                
        elif t == ndi.FRAME_TYPE_AUDIO:
            # We ignore audio for this task, but we must free it
            ndi.recv_free_audio_v2(self.ndi_recv, a)
        elif t == ndi.FRAME_TYPE_METADATA:
            ndi.recv_free_metadata(self.ndi_recv, m)
            
        return None, None

    def receive_tensor(self, timeout_ms=1000, device='cpu'):
        """
        Capture a frame and return it as a PyTorch tensor.
        
        Args:
            device: 'cpu' or 'cuda'.
            
        Returns:
            tuple: (tensor, timestamp) or (None, None).
                   Tensor format is (C, H, W) normalized 0-1 if preferred, or kept as uint8 (H,W,C).
                   Video/Vision usually prefers (C, H, W).
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed.")
            
        frame, ts = self.receive_array(timeout_ms)
        if frame is None:
            return None, None
        
        # Convert to tensor
        # numpy frame is (H, W, C), uint8
        tensor = torch.from_numpy(frame)
        
        if device != 'cpu':
            tensor = tensor.to(device)
            
        # Permute to (C, H, W) standard for Torch Vision
        tensor = tensor.permute(2, 0, 1)
        
        return tensor, ts

    def destroy(self):
        """Clean up resources."""
        self.stop_capture()
        if self.ndi_recv:
            ndi.recv_destroy(self.ndi_recv)
        if self.ndi_find:
            ndi.find_destroy(self.ndi_find)
        ndi.destroy()


def main():
    print("Initializing NDI Receiver...")
    receiver = NDIReceiver()
    
    print("Looking for sources...")
    sources = receiver.find_sources()
    
    if not sources:
        print("No NDI sources found. Make sure an NDI sender is running on the network.")
        receiver.destroy()
        return

    print(f"Found {len(sources)} sources.")
    for i, s in enumerate(sources):
        try:
            name = s.ndi_name
            print(f"{i}: {name}")
        except Exception as e:
            print(f"{i}: <Name Decode Error: {e}>")
            # We can still connect to 's' even if the name is garbage, 
            # because 's' is the struct passed to C++.
        
    # Connect to the first one for demonstration
    if len(sources) > 0:
        target = sources[0]
        # Try to print target name safely
        try:
             t_name = target.ndi_name
        except:
             t_name = "Unknown"
             
        print(f"Connecting to source 0 ({t_name})...")
        receiver.connect(target)
    
    print("Receiving frames (Ctrl+C to stop)...")
    try:
        frame_count = 0
        start_t = time.time()
        
        while True:
            frame, ts = receiver.receive_array()
            if frame is not None:
                frame_count += 1
                if frame_count % 60 == 0:
                    fps = frame_count / (time.time() - start_t)
                    print(f"Received frame: {frame.shape}, Timestamp: {ts}, FPS: {fps:.2f}")
            else:
                # No frame received (timeout or other type)
                pass
                
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        receiver.destroy()

if __name__ == "__main__":
    main()
