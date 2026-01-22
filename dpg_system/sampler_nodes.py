from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.torch_base_nodes import *

try:
    import sounddevice as sd
    from dpg_system.node import LoadDialog
except Exception as e:
    print(e)
import platform
import numpy as np
import threading
import os
import torchaudio
import time
# Import our new engine
from dpg_system.sampler import SamplerEngine, Sample
import dearpygui.dearpygui as dpg


def register_sampler_nodes():
    Node.app.register_node('sampler_voice', SamplerVoiceNode.factory)
    Node.app.register_node('sampler_engine', SamplerEngineNode.factory)
    Node.app.register_node('multi_voice_sampler', SamplerMultiVoiceNode.factory)
    Node.app.register_node('polyphonic_sampler', PolyphonicSamplerNode.factory)
    Node.app.register_node('granular_sampler', GranularSamplerNode.factory)
    Node.app.register_node('scratch_sampler', ScratchSamplerNode.factory)


class SamplerEngineNode(Node):
    """
    Manages the global SamplerEngine instance.
    """
    engine = None

    @staticmethod
    def factory(name, data, args=None):
        node = SamplerEngineNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # Initialize engine if not exists
        if SamplerEngineNode.engine is None:
            SamplerEngineNode.engine = SamplerEngine()
            SamplerEngineNode.engine.start()
        self.stop_input = self.add_input('stop engine', widget_type='button', callback=self.stop_engine)
        self.restart_input = self.add_input('restart engine', widget_type='button', callback=self.restart_engine)
        self.master_vol_input = self.add_input('master volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0, callback=self.set_engine_volume)
        self.level_out = self.add_output('output level')
        self.add_frame_task()

    def frame_task(self):
        if SamplerEngineNode.engine:
            self.level_out.send(SamplerEngineNode.engine.output_level)
            
    def set_engine_volume(self):
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.set_master_volume(self.master_vol_input())

    def stop_engine(self):
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.stop()

    def restart_engine(self):
        if SamplerEngineNode.engine:
            if not SamplerEngineNode.engine.stream or not SamplerEngineNode.engine.stream.active:
                SamplerEngineNode.engine.start()
                # Restore volume
                SamplerEngineNode.engine.set_master_volume(self.master_vol_input())

    def custom_cleanup(self):
        # We generally don't want to kill the static engine just because one node is deleted,
        # unless it is the last node?
        # For now, let's leave it running or rely on app shutdown?
        # User snippet had strict cleanup.
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.stop()
            SamplerEngineNode.engine = None
        print("SamplerEngineNode cleaned up.")


class SamplerVoiceNode(Node):
    """
    A node to control a single voice (or arbitrary voice) of the SamplerEngine.
    """

    @staticmethod
    def factory(name, data, args=None):
        node = SamplerVoiceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.sample = None
        self.last_voice_idx = 0
        # Ensure engine is initialized
        if SamplerEngineNode.engine is None:
            SamplerEngineNode.engine = SamplerEngine()
            SamplerEngineNode.engine.start()

        # Inputs
        self.play_toggle = self.add_input('play', widget_type='checkbox', default_value=False,
                                          callback=self.toggle_play)
        
        # Internal properties
        self.path_input = self.add_property('path', widget_type='text_input', default_value="")
        self.voice_idx_property = self.add_property('voice_index', widget_type='input_int', default_value=0)
        
        # UI Layout
        self.load_input = self.add_input('load', widget_type='button', callback=self.on_load)
        self.file_label = self.add_label('')
        self.length_label = self.add_label('')
        
        self.sample_start_input = self.add_input('sample start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240,
                                               callback=self.update_loop_params)
        self.sample_end_input = self.add_input('sample end', widget_type='slider_int', default_value=-1, min=0, max=100000, widget_width=240,
                                             callback=self.update_loop_params)
        
        self.volume_input = self.add_input('volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0,
                                           callback=self.update_params)
        self.pitch_input = self.add_input('pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0,
                                          callback=self.update_params)
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=False,
                                         callback=self.update_loop_params)

        self.loop_start_input = self.add_input('loop start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240,
                                               callback=self.update_loop_params)
        self.loop_end_input = self.add_input('loop end', widget_type='slider_int', default_value=-1, min=0, max=100000, widget_width=240,
                                             callback=self.update_loop_params)


        self.crossfade_input = self.add_input('crossfade ratio', widget_type='drag_float', default_value=0.0, min=0.0, max=0.5, widget_width=240,
                                              callback=self.update_loop_params)

    def set_widget_max(self, input_obj, max_val):
        if hasattr(input_obj, 'widget'):
            try:
                dpg.configure_item(input_obj.widget.uuid, max_value=int(max_val))
                if hasattr(input_obj.widget, 'max'):
                    input_obj.widget.max = int(max_val)
            except:
                pass

    def request_load_file(self):
        # Assuming LoadDialog exists in the environment
        try:
            from dpg_system.element_loader import LoadDialog
            LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])
        except ImportError:
            pass

    def load_file_callback(self, path):
        # self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        # New: Use Sample factory/init to load
        self.sample = Sample(filepath)
        self.path_input.set(filepath) # Update property for persistence

        # Update label
        self.file_label.set(os.path.basename(filepath))
        slen = int(len(self.sample.data))
        self.length_label.set(f"{slen} samples")
        
        # Update sliders
        self.set_widget_max(self.loop_start_input, slen)
        self.set_widget_max(self.loop_end_input, slen)
        self.set_widget_max(self.sample_start_input, slen)
        self.set_widget_max(self.sample_end_input, slen)
        # self.set_widget_max(self.crossfade_input, slen // 2)
        
        # Always reset loop points to full sample on load
        self.loop_start_input.set(0)
        self.loop_end_input.set(slen)
        self.sample_start_input.set(0)
        self.sample_end_input.set(slen)

    def on_load(self):
        data = self.load_input()
        if data == 'bang' or (isinstance(data, list) and not data) or data is None:
             self.request_load_file()
             return
        
        path = ""
        if isinstance(data, str):
            path = data
        elif isinstance(data, list) and len(data) >= 2:
            try:
                # Optional: Check ID? 
                target_id = int(data[0])
                current_id = int(self.voice_idx_property())
                if target_id != current_id:
                     return
                path = data[1]
            except:
                return

        if path:
             self.load_file_with_path(path)
             
    def save_custom(self, container):
        if hasattr(self, 'sample') and self.sample and self.sample.path:
            container['path'] = self.sample.path

    def load_custom(self, container):
        if 'path' in container:
            self.load_file_with_path(container['path'])

        if SamplerEngineNode.engine:
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            idx = int(self.voice_idx_property())
            SamplerEngineNode.engine.set_voice_volume(idx, vol)
            SamplerEngineNode.engine.set_voice_pitch(idx, pitch)

    def update_loop_params(self):
        if self.sample and SamplerEngineNode.engine:
            l_start = int(self.loop_start_input())
            l_end = int(self.loop_end_input())
            xf_ratio = float(self.crossfade_input())
            idx = int(self.voice_idx_property())
            if l_end < 0:
                l_end = len(self.sample.data)
            
            loop_len = l_end - l_start
            if loop_len < 0: loop_len = 0
            xf = int(loop_len * xf_ratio)
            
            loop = bool(self.loop_input())
            SamplerEngineNode.engine.set_voice_loop_window(idx, loop, l_start, l_end, xf)
            
            s_start = int(self.sample_start_input())
            s_end = int(self.sample_end_input())
            if s_end < 0: s_end = len(self.sample.data)
            SamplerEngineNode.engine.set_voice_playback_range(idx, s_start, s_end)

    def toggle_play(self):
        is_playing = bool(self.play_toggle())
        if is_playing:
            self.play()
        else:
            self.stop_voice()

    def update_params(self):
        if hasattr(self, 'ignore_updates') and self.ignore_updates: return
        
        if SamplerEngineNode.engine:
            idx = int(self.voice_idx_property())
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            
            SamplerEngineNode.engine.set_voice_volume(idx, vol)
            SamplerEngineNode.engine.set_voice_pitch(idx, pitch)

    def play(self):
        if self.sample and SamplerEngineNode.engine:
            idx = int(self.voice_idx_property())
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            loop = bool(self.loop_input())
            l_start = int(self.loop_start_input())
            l_end = int(self.loop_end_input())
            xf_ratio = float(self.crossfade_input())
            if l_end < 0:
                l_end = len(self.sample.data)
            
            loop_len = l_end - l_start
            if loop_len < 0: loop_len = 0
            xf = int(loop_len * xf_ratio)

            self.sample.default_volume = vol
            self.sample.default_pitch = pitch
            self.sample.loop = loop
            self.sample.loop_start = l_start
            self.sample.loop_end = l_end
            self.sample.loop_end = l_end
            self.sample.crossfade_frames = xf
            
            s_start = int(self.sample_start_input())
            s_end = int(self.sample_end_input())
            if s_end < 0: s_end = len(self.sample.data)
            self.sample.sample_start = s_start
            self.sample.sample_end = s_end
            
            SamplerEngineNode.engine.play_voice(idx, self.sample, volume=vol, pitch=pitch)
            self.last_voice_idx = idx

    def stop_voice(self):
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.stop_voice(self.last_voice_idx)

    def execute(self):
        pass


class SamplerMultiVoiceNode(Node):
    """
    A node to control ALL voices of the SamplerEngine.
    """

    @staticmethod
    def factory(name, data, args=None):
        node = SamplerMultiVoiceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.voices_state = []
        for i in range(128):
            self.voices_state.append({
                "sample": None,
                "path": "",
                "sample_volume": 1.0,  # Base volume
                "fade": 1.0,           # Interactive volume
                "sample_pitch": 1.0,   # Base pitch
                "int_pitch": 1.0,      # Interactive pitch
                "loop": False,
                "loop_start": 0,
                "loop_end": -1,
                "sample_start": 0,
                "sample_end": -1,
                "crossfade": 0.0,
                "attack": 0.0,
                "decay": 0.0,
                "decay": 0.0,
                "decay_curve": 1.0,
                "playing": False,
                "fade_zero_since": None
            })

        self.current_idx = 0
        self.ignore_updates = False

        if SamplerEngineNode.engine is None:
            SamplerEngineNode.engine = SamplerEngine()
            SamplerEngineNode.engine.start()

        # Inputs
        self.voice_idx_input = self.add_input('voice index', widget_type='drag_int', default_value=0, min=0, max=127,
                                              callback=self.on_voice_idx_change)

        self.play_toggle = self.add_input('play', widget_type='checkbox', default_value=False,
                                          callback=self.toggle_play)

        # self.path_input = self.add_input('path', callback=self.load_file) # Removed
        self.load_btn = self.add_input('load', widget_type='button', callback=self.on_load)

        self.file_label = self.add_label('')
        self.length_label = self.add_label('')

        self.sample_start_input = self.add_input('sample start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240,
                                               callback=self.update_loop_params)
        self.sample_end_input = self.add_input('sample end', widget_type='slider_int', default_value=-1, min=0, max=100000, widget_width=240,
                                             callback=self.update_loop_params)

        self.voice_params_input = self.add_input('voice params', callback=self.on_voice_params_change)

        self.volume_input = self.add_input('sample volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0,
                                           callback=self.update_params)
        self.fade_input = self.add_input('fade', widget_type='drag_float', default_value=1.0, min=0.0, max=1.0,
                                           callback=self.update_params)
                                           
        self.pitch_input = self.add_input('sample pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0,
                                          callback=self.update_params)
        self.int_pitch_input = self.add_input('interactive pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0,
                                          callback=self.update_params)

        self.attack_input = self.add_input('attack (s)', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0,
                                           callback=self.update_params)
        self.decay_input = self.add_input('decay (s)', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0,
                                          callback=self.update_params)
        self.decay_curve_input = self.add_input('decay curve', widget_type='drag_float', default_value=1.0, min=0.1,
                                                max=20.0, callback=self.update_params)
        self.path_input = self.add_property('path', widget_type='text_input', default_value="")
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=False,
                                         callback=self.update_loop_params)
        self.loop_start_input = self.add_input('loop start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240,
                                               callback=self.update_loop_params)
        self.loop_end_input = self.add_input('loop end', widget_type='slider_int', default_value=-1, min=0, max=100000, widget_width=240,
                                             callback=self.update_loop_params)

        self.crossfade_input = self.add_input('crossfade ratio', widget_type='drag_float', default_value=0.0, min=0.0, max=0.5, widget_width=240,
                                              callback=self.update_loop_params)
        self.pos_output = self.add_output('position')
        self.active_out = self.add_output('active_voices')
        if not hasattr(self, 'message_handlers'):
            self.message_handlers = {}
        self.message_handlers['trigger'] = self.trigger_from_message
        self.message_handlers['stop'] = self.stop_from_message

    def set_widget_max(self, input_obj, max_val):
        if hasattr(input_obj, 'widget'):
            try:
                dpg.configure_item(input_obj.widget.uuid, max_value=int(max_val))
                if hasattr(input_obj.widget, 'max'):
                    input_obj.widget.max = int(max_val)
            except:
                pass

    def stop_from_message(self, message='', message_data=[]):
        data = message_data
        if not data or (isinstance(data, (list, tuple)) and len(data) == 0):
            if SamplerEngineNode.engine:
                SamplerEngineNode.engine.stop_all()
            for state in self.voices_state:
                state["playing"] = False
        else:
            indices = data if isinstance(data, (list, tuple)) else [data]
            for item in indices:
                try:
                    idx = int(item)
                    if 0 <= idx < 128:
                        if SamplerEngineNode.engine:
                            SamplerEngineNode.engine.stop_voice(idx)
                        self.voices_state[idx]["playing"] = False
                except:
                    pass
        if not self.voices_state[self.current_idx]["playing"]:
            # Check syncing
            self.sync_ui_to_state()

    def trigger_from_message(self, message='', message_data=[]):
        data = message_data
        if not isinstance(data, (list, tuple)) or len(data) == 0: return
        try:
            idx = int(data[0])
            if not (0 <= idx < 128): return
            state = self.voices_state[idx]
            if len(data) > 1 and data[1] is not None:
                fade = float(data[1])
                state["fade"] = fade
                if SamplerEngineNode.engine:
                    # Update engine with combined volume
                    eff_vol = state["sample_volume"] * fade
                    SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
            else:
                state["fade"] = 1.0
                if SamplerEngineNode.engine:
                     eff_vol = state["sample_volume"]
                     SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
                    
            if len(data) > 2 and data[2] is not None:
                i_pitch = float(data[2])
                state["int_pitch"] = i_pitch
                if SamplerEngineNode.engine:
                    eff_pitch = state["sample_pitch"] * i_pitch
                    SamplerEngineNode.engine.set_voice_pitch(idx, eff_pitch)

            self.trigger_voice(idx)
            state["playing"] = True
            state["fade_zero_since"] = None
            if idx == self.current_idx:
                self.sync_ui_to_state()
                self.add_frame_task()
        except Exception as e:
            print(f"Trigger message error: {e}")

    def custom_create(self, from_file):
        self.sync_ui_to_state()

    def request_load_file(self, *args):
        try:
            from dpg_system.element_loader import LoadDialog
            LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])
        except ImportError:
            pass

    def load_file_callback(self, path):
        # self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath, voice_idx=None):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
        target_idx = voice_idx if voice_idx is not None else self.current_idx
        try:
            sample = Sample(filepath)
            state = self.voices_state[target_idx]
            state["sample"] = sample
            state["path"] = filepath
            state["loop_end"] = -1
            state["loop_start"] = 0
            if target_idx == self.current_idx:
                self.file_label.set(os.path.basename(filepath))
                slen = int(len(sample.data))
                self.length_label.set(f"Length: {slen} samples")
                
                # Update sliders for current voice
                self.set_widget_max(self.loop_start_input, slen)
                self.set_widget_max(self.loop_end_input, slen)
                self.set_widget_max(self.loop_start_input, slen)
                self.set_widget_max(self.loop_end_input, slen)
                self.set_widget_max(self.sample_start_input, slen)
                self.set_widget_max(self.sample_end_input, slen)
                self.set_widget_max(self.crossfade_input, slen // 2)
                
                self.ignore_updates = True
                # self.path_input.set(filepath)
                self.loop_start_input.set(0)
                
                # Handle loop end -1 -> slen
                self.loop_end_input.set(slen)
                
                self.ignore_updates = False
            if state["playing"]:
                self.trigger_voice(target_idx)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")

    def on_voice_idx_change(self, *args):
        try:
            val = self.voice_idx_input()
            idx = int(val)
            if 0 <= idx < 128:
                self.current_idx = idx
                self.sync_ui_to_state()
        except Exception as e:
            print(f"Error changing voice index: {e}")

    def on_load(self):
        data = self.load_btn()
        if data == 'bang' or (isinstance(data, list) and not data) or data is None:
             self.request_load_file()
             return
        
        target_idx = self.current_idx
        path = ""
        
        if isinstance(data, str):
            path = data
        elif isinstance(data, list) and len(data) >= 2:
            try:
                target_idx = int(data[0])
                path = data[1]
            except:
                return

        if path:
             self.load_file_with_path(path, target_idx)

    def sync_ui_to_state(self):
        self.ignore_updates = True
        state = self.voices_state[self.current_idx]
        if SamplerEngineNode.engine:
            start_active = SamplerEngineNode.engine.voices[self.current_idx].active
            state["playing"] = start_active
            if start_active:
                self.add_frame_task()
            else:
                self.remove_frame_tasks()
        # self.path_input.set(state["path"])
        if state["path"]:
            self.file_label.set(os.path.basename(state["path"]))
            if state["sample"]:
                slen = int(len(state['sample'].data))
                self.length_label.set(f"{slen} samples")
                
                # Update sliders
                self.set_widget_max(self.loop_start_input, slen)
                self.set_widget_max(self.loop_end_input, slen)
                self.set_widget_max(self.sample_start_input, slen)
                self.set_widget_max(self.sample_end_input, slen)
                # self.set_widget_max(self.crossfade_input, slen // 2)
            else:
                self.length_label.set("")

        else:
            self.file_label.set("No File")
            self.length_label.set("")
        self.play_toggle.set(state["playing"])
        self.volume_input.set(state.get("sample_volume", 1.0))
        self.fade_input.set(state.get("fade", 1.0))
        self.pitch_input.set(state.get("sample_pitch", 1.0))
        self.int_pitch_input.set(state.get("int_pitch", 1.0))
        self.attack_input.set(state.get("attack", 0.0))
        self.decay_input.set(state.get("decay", 0.0))
        self.decay_curve_input.set(state.get("decay_curve", 1.0))
        self.loop_input.set(state["loop"])
        self.loop_start_input.set(state["loop_start"])
        
        le = state["loop_end"]
        if le < 0 and state["sample"]:
             le = int(len(state["sample"].data))
        self.loop_end_input.set(le)
        
        self.sample_start_input.set(state.get("sample_start", 0))
        se = state.get("sample_end", -1)
        if se < 0 and state["sample"]:
             se = int(len(state["sample"].data))
        self.sample_end_input.set(se)

        self.crossfade_input.set(state["crossfade"])
        self.ignore_updates = False

    def frame_task(self):
        if SamplerEngineNode.engine:
            voice = SamplerEngineNode.engine.voices[self.current_idx]
            
            # Update Position check for current UI voice
            self.pos_output.send(voice.position)

            # Global Auto-Stop & Cleanup Check
            any_playing = False
            now = time.time()
            
            for i, state in enumerate(self.voices_state):
                if state["playing"]:
                    # check actual engine active status
                    engine_voice = SamplerEngineNode.engine.voices[i]
                    is_active = engine_voice.active

                    # 1. Natural Stop (e.g. sample finished)
                    if not is_active:
                         state["playing"] = False
                         if i == self.current_idx:
                             self.ignore_updates = True
                             self.play_toggle.set(False)
                             self.ignore_updates = False
                    
                    # 2. Fade Timeout Stop
                    elif state["fade_zero_since"] is not None:
                        if now - state["fade_zero_since"] > 1.0:
                            # Timeout reached -> Stop Voice
                            SamplerEngineNode.engine.stop_voice(i)
                            state["playing"] = False
                            state["fade_zero_since"] = None
                            
                            if i == self.current_idx:
                                self.ignore_updates = True
                                self.play_toggle.set(False)
                                self.ignore_updates = False
                        else:
                            # Still timing out, but playing
                            any_playing = True
                    else:
                        # Playing normally
                        any_playing = True

            # Only remove frame task if NO voices are playing
            if not any_playing:
                self.remove_frame_tasks()
                
            # Build and send Active Grid
            # Fixed 128 voices for MultiVoiceNode
            # Shape (4, 32)
            grid_flat = np.zeros(128, dtype=int)
            
            # Optimization: We already looped voices_state, but we need actual engine state
            # voices_state maps 1:1 to engine voices 0..127
            # But checking engine.voices[i].active is the source of truth
            for i in range(128):
                if SamplerEngineNode.engine.voices[i].active:
                    grid_flat[i] = 1
            
            grid = grid_flat.reshape((4, 32))
            self.active_out.send(grid)

    def update_params(self, *args):
        if self.ignore_updates: return

        s_vol = float(self.volume_input())
        fade = float(self.fade_input())
        
        s_pitch = float(self.pitch_input())
        i_pitch = float(self.int_pitch_input())
        
        attack = float(self.attack_input())
        decay = float(self.decay_input())
        curve = float(self.decay_curve_input())

        state = self.voices_state[self.current_idx]
        state["sample_volume"] = s_vol
        state["fade"] = fade
        state["sample_pitch"] = s_pitch
        state["int_pitch"] = i_pitch
        
        state["attack"] = attack
        state["decay"] = decay
        state["decay_curve"] = curve

        if SamplerEngineNode.engine:
            eff_vol = s_vol * fade
            eff_pitch = s_pitch * i_pitch
            
            SamplerEngineNode.engine.set_voice_volume(self.current_idx, eff_vol)
            SamplerEngineNode.engine.set_voice_pitch(self.current_idx, eff_pitch)
            SamplerEngineNode.engine.set_voice_envelope(self.current_idx, attack, decay, curve)

    def update_loop_params(self, *args):
        if self.ignore_updates: return
        state = self.voices_state[self.current_idx]
        state["loop"] = bool(self.loop_input())
        state["loop_start"] = int(self.loop_start_input())
        state["loop_end"] = int(self.loop_end_input())
        state["sample_start"] = int(self.sample_start_input())
        state["sample_end"] = int(self.sample_end_input())
        state["crossfade"] = float(self.crossfade_input())
        effective_end = state["loop_end"]
        if effective_end < 0 and state["sample"]:
            effective_end = len(state["sample"].data)
        
        loop_len = effective_end - state["loop_start"]
        if loop_len < 0: loop_len = 0
        xf_frames = int(loop_len * state["crossfade"])

        if SamplerEngineNode.engine and state["sample"]:
            SamplerEngineNode.engine.set_voice_loop_window(self.current_idx, state["loop"], state["loop_start"], effective_end,
                                                           xf_frames)
            
            s_end = state["sample_end"]
            if s_end < 0: s_end = len(state["sample"].data)
            SamplerEngineNode.engine.set_voice_playback_range(self.current_idx, state["sample_start"], s_end)

    def toggle_play(self, *args):
        val = self.play_toggle()

        # Handle List Input (Advanced Triggering)
        if isinstance(val, (list, tuple)):
            if not SamplerEngineNode.engine: return

            def update_voice(idx, fade_val, int_pitch_val=None):
                if not (0 <= idx < 128): return

                state = self.voices_state[idx]
                state["fade"] = fade_val
                if int_pitch_val is not None:
                    state["int_pitch"] = int_pitch_val
                
                # Calculate effective product
                s_vol = state.get("sample_volume", 1.0)
                eff_vol = s_vol * fade_val
                
                SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
                
                if int_pitch_val is not None:
                     s_pitch = state.get("sample_pitch", 1.0)
                     eff_pitch = s_pitch * int_pitch_val
                     SamplerEngineNode.engine.set_voice_pitch(idx, eff_pitch)
                
                # Check actual engine voice state
                voice = SamplerEngineNode.engine.voices[idx]

                # Logic: If fade > 0 and not playing, trigger it. 
                if not voice.active and fade_val > 0.0:
                    # Trigger using current params (including new fade/pitch)
                    self.trigger_voice(idx)
                    self.voices_state[idx]["playing"] = True
                    self.voices_state[idx]["fade_zero_since"] = None
                    
                    # Update UI if this is the current voice
                    if idx == self.current_idx:
                         self.ignore_updates = True
                         self.play_toggle.set(True)
                         self.fade_input.set(fade_val)
                         if int_pitch_val is not None:
                             self.int_pitch_input.set(int_pitch_val)
                         self.ignore_updates = False
                    
                    # Ensure frame task is running
                    self.add_frame_task()
                
                elif idx == self.current_idx:
                    # Voice is active, just update UI knob
                    self.ignore_updates = True
                    self.fade_input.set(fade_val)
                    if int_pitch_val is not None:
                        self.int_pitch_input.set(int_pitch_val)
                    self.ignore_updates = False

                # Auto-stop Logic tracking
                if fade_val <= 0.001:
                    if self.voices_state[idx]["fade_zero_since"] is None:
                        self.voices_state[idx]["fade_zero_since"] = time.time()
                else:
                    self.voices_state[idx]["fade_zero_since"] = None

            if len(val) > 0:
                # Check format
                first = val[0]
                if isinstance(first, (list, tuple)):
                    # Format: [[idx, vol], [idx, vol]] OR [[idx, vol, pitch]]
                    for item in val:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            i_p = float(item[2]) if len(item) > 2 else None
                            try:
                                update_voice(int(item[0]), float(item[1]), i_p)
                            except:
                                pass
                else:
                    # Format: [.5, .2, 0, ...] (Implied indices 0..N)
                    for i, v_vol in enumerate(val):
                        try:
                            update_voice(i, float(v_vol))
                        except:
                            pass
            
            # Enforce UI Sync: The input pin auto-updates to True when receiving list data.
            # We must force it back to the real state of the current voice.
            current_state_playing = self.voices_state[self.current_idx]["playing"]
            if self.play_toggle() != current_state_playing:
                self.ignore_updates = True
                self.play_toggle.set(current_state_playing)
                self.ignore_updates = False
            return
            
        if self.ignore_updates: return
        is_playing = bool(val)
        state = self.voices_state[self.current_idx]
        state["playing"] = is_playing

        if is_playing:
            # Force fade to 1.0
            state["fade"] = 1.0
            state["fade_zero_since"] = None
            self.fade_input.set(1.0)
            
            self.trigger_voice(self.current_idx)
            self.add_frame_task()
        else:
            if SamplerEngineNode.engine:
                SamplerEngineNode.engine.stop_voice(self.current_idx)
            # Do NOT remove frame task yet, let it fade out and auto-remove when active=False in frame_task

    def trigger_voice(self, idx, attack=None, decay=None, decay_curve=None):
        if SamplerEngineNode.engine:
            state = self.voices_state[idx]
            if state["sample"]:
                s = state["sample"]
                
                # Check for keys, fallback for legacy data if needed (though init handles it)
                s_vol = state.get("sample_volume", 1.0)
                fade = state.get("fade", 1.0)
                s_pitch = state.get("sample_pitch", 1.0)
                i_pitch = state.get("int_pitch", 1.0)
                
                eff_vol = s_vol * fade
                eff_pitch = s_pitch * i_pitch
                
                s.default_volume = eff_vol
                s.default_pitch = eff_pitch
                s.loop = state["loop"]
                s.loop_start = state["loop_start"]
                le = state["loop_end"]
                if le < 0: le = len(s.data)
                s.loop_end = le
                
                s.sample_start = state.get("sample_start", 0)
                se = state.get("sample_end", -1)
                if se < 0: se = len(s.data)
                s.sample_end = se

                loop_len = le - s.loop_start
                if loop_len < 0: loop_len = 0
                s.crossfade_frames = int(loop_len * state["crossfade"])
                # Use provided overrides or state values
                a = attack if attack is not None else state.get("attack", 0.0)
                d = decay if decay is not None else state.get("decay", 0.0)
                c = decay_curve if decay_curve is not None else state.get("decay_curve", 1.0)
                SamplerEngineNode.engine.set_voice_envelope(idx, a, d, c)
                SamplerEngineNode.engine.play_voice(idx, s, volume=eff_vol, pitch=eff_pitch)

    def on_voice_params_change(self, *args):
        # Placeholder for bulk update
        pass

    def execute(self):
        pass

    def save_custom(self, container):
        voices_data = {}
        for i, state in enumerate(self.voices_state):
            has_file = bool(state["path"])
            # Save if modified from defaults
            if has_file or state.get("sample_volume", 1.0) != 1.0 or state.get("sample_pitch", 1.0) != 1.0:
                voices_data[str(i)] = state.copy()
                if "sample" in voices_data[str(i)]: del voices_data[str(i)]["sample"]
        container['voices'] = voices_data
        container['current_idx'] = self.current_idx

    def load_custom(self, container):
        if 'current_idx' in container:
            self.current_idx = int(container['current_idx'])
        voices_data = container.get('voices', {})
        for idx_str, data in voices_data.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < 128:
                    state = self.voices_state[idx]
                    for k, v in data.items():
                        state[k] = v

                    if state["path"]:
                        self.load_file_with_path(state["path"], idx)
            except:
                pass
        self.sync_ui_to_state()


class Sound:
    def __init__(self, filepath, base_params=None):
        self.sample = Sample(filepath)
        self.path = filepath
        self.params = {
            "volume": 1.0,
            "pitch": 1.0,
            "attack": 0.0,
            "decay": 0.0,
            "decay_curve": 1.0,
            "loop": False,
            "loop_start": 0,
            "loop_end": -1,
            "sample_start": 0,
            "sample_end": -1,
            "crossfade": 0,
            "density": 20.0,
            "grain_dur": 0.1,
            "jitter_pos": 0.05,
            "jitter_pitch": 0.0,
            "jitter_dur": 0.0
        }
        if base_params:
            self.params.update(base_params)
            
        # Apply to sample
        self.sample.loop = self.params.get("loop", False)
        self.sample.loop_start = int(self.params.get("loop_start", 0))
        
        le = self.params.get("loop_end", -1)
        if le is None: le = -1
        le = int(le)
        if le < 0: le = len(self.sample.data)
        self.sample.loop_end = le
        
        self.sample.sample_start = int(self.params.get("sample_start", 0))
        se = self.params.get("sample_end", -1)
        if se is None: se = -1
        se = int(se)
        if se < 0: se = len(self.sample.data)
        self.sample.sample_end = se
        
        loop_len = le - self.sample.loop_start
        if loop_len < 0: loop_len = 0
        self.sample.crossfade_frames = int(float(self.params.get("crossfade", 0.0)) * loop_len)


class PolyphonicSamplerNode(Node):
    """
    Dynamically allocates voices to play sounds.
    """
    @staticmethod
    def factory(name, data, args=None):
        node = PolyphonicSamplerNode(name, data, args)
        return node
        
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
        self.sounds = {} # id (str/int) -> Sound
        self.active_allocations = [] # list of dicts: {voice_idx, sound_id, note_id}
        
        # Start at 64 to avoid collision with default MultiVoiceNode usage (0..?)
        # though ideally user manages range
        self.start_voice_idx = 64
        self.voice_count = 16
        
        if SamplerEngineNode.engine is None:
            SamplerEngineNode.engine = SamplerEngine()
            SamplerEngineNode.engine.start()

        # Inputs
        self.trigger_input = self.add_input('trigger', callback=self.on_trigger)
        self.stop_input = self.add_input('stop', callback=self.on_stop)
        self.load_input = self.add_input('load', widget_type='button', callback=self.on_load)
        
        # Helper props
        self.voice_range_start = self.add_property('start_voice', widget_type='input_int', default_value=64, callback=self.on_config_change)
        self.voice_range_count = self.add_property('voice_count', widget_type='input_int', default_value=16, callback=self.on_config_change)
        
        # Inspection UI
        self.inspect_label = self.add_label('Inspection')
        self.inspect_id_input = self.add_input('sound_id', widget_type='input_int', default_value=0, callback=self.on_inspect_change)
        self.file_label = self.add_label('')
        self.length_label = self.add_label('')
        
        self.edit_sample_start = self.add_input('sample_start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240, callback=self.on_param_edit)
        self.edit_sample_end = self.add_input('sample_end', widget_type='slider_int', default_value=100000, min=0, max=100000, widget_width=240, callback=self.on_param_edit)
        
        self.edit_vol = self.add_input('volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0, callback=self.on_param_edit)
        self.edit_pitch = self.add_input('pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0, callback=self.on_param_edit)
        self.edit_att = self.add_input('attack', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_dec = self.add_input('decay', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_curve = self.add_input('decay curve', widget_type='drag_float', default_value=1.0, min=0.1, max=20.0, callback=self.on_param_edit)
        
        self.create_loop_widgets()
        
        # Default to inspecting ID 0 (int) so immediate edits work
        self.current_inspect_id = 0
        self.ignore_updates = False
        
        self.add_additional_parameters()
        
        # MIDI Support
        self.midi_input = self.add_input('midi', callback=self.on_midi)
        self.base_note_input = self.add_input('base_note', widget_type='drag_int', default_value=60, min=0, max=127)

        # Output
        self.active_out = self.add_output('active_voices')

    def add_additional_parameters(self):
        pass

    def set_widget_max(self, input_obj, max_val):
        """Helper to safely set max on a DPG widget if it exists."""
        if hasattr(input_obj, 'widget'):
            try:
                dpg.configure_item(input_obj.widget.uuid, max_value=int(max_val))
                if hasattr(input_obj.widget, 'max'):
                    input_obj.widget.max = int(max_val)
            except:
                pass

    def create_loop_widgets(self):
        self.edit_loop = self.add_input('loop', widget_type='checkbox', default_value=False, callback=self.on_param_edit)
        # Using slider_int for visual range. Max will be updated dynamically on inspection.
        self.edit_loop_start = self.add_input('loop_start', widget_type='slider_int', default_value=0, min=0, max=100000, widget_width=240, callback=self.on_param_edit)
        # Loop end can be -1? Slider usually enforces range. 
        # If we use slider, -1 might be awkward if min=0.
        # User asked for slider max=frames.
        # Let's assume loop_end should be positive index. If default is -1 (end), we might need to handle slider 
        # showing 'max' as default? Or create a special max value?
        # Let's stick to positive indices for slider. If existing param is -1, map it to length.
        self.edit_loop_end = self.add_input('loop_end', widget_type='slider_int', default_value=100000, min=0, max=100000, widget_width=240, callback=self.on_param_edit)

        self.edit_cross = self.add_input('crossfade ratio', widget_type='slider_float', default_value=0.0, min=0.0, max=0.5, widget_width=240, callback=self.on_param_edit)

    def on_config_change(self):
        self.start_voice_idx = self.voice_range_start()
        self.voice_count = self.voice_range_count()

    def on_load(self):
        data = self.load_input()
        if data == 'bang' or (isinstance(data, list) and not data) or data is None:
             self.request_load_file()
             return

        sid = self.current_inspect_id
        path = ""
        params = {}
        
        if isinstance(data, str):
            path = data
        elif isinstance(data, list) and len(data) >= 2:
            try:
                sid = int(data[0])
                path = data[1]
                # Legacy param support
                if len(data) > 2: params["volume"] = float(data[2])
                if len(data) > 3: params["pitch"] = float(data[3])
                if len(data) > 4: params["attack"] = float(data[4])
                if len(data) > 5: params["decay"] = float(data[5])
                if len(data) > 6: params["decay_curve"] = float(data[6])
            except:
                return

        if path and os.path.exists(path):
            try:
                snd = Sound(path, params)
                self.sounds[sid] = snd
                
                # Check granule params if Granular
                if hasattr(self, 'density_input'):
                     # Preserve granular params from existing if they exist in params?
                     # Ideally Sound() captures defaults.
                     pass
                     
                if sid == self.current_inspect_id:
                     self.sync_ui_to_sound(sid)
            except Exception as e:
                print(f"Error loading sound {sid}: {e}")

    def request_load_file(self):
        try:
            from dpg_system.element_loader import LoadDialog
            LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])
        except ImportError:
            pass

    def load_file_callback(self, path):
        if self.current_inspect_id is not None:
             self.load_sample_file(self.current_inspect_id, path)

    def load_sample_file(self, sid, path):
        if not os.path.exists(path): return
        try:
             snd = Sound(path)
             self.sounds[sid] = snd
             if sid == self.current_inspect_id:
                  self.sync_ui_to_sound(sid)
        except Exception as e:
             print(f"Error loading sample file: {e}")

    def on_inspect_change(self):
        val = self.inspect_id_input()
        try:
            self.current_inspect_id = int(val)
        except:
            pass # Keep previous if invalid
            
        self.sync_ui_to_sound(self.current_inspect_id)
        
    def sync_ui_to_sound(self, sid):
        self.ignore_updates = True
        if sid in self.sounds:
            snd = self.sounds[sid]
            # Update labels
            self.file_label.set(os.path.basename(snd.path))
            self.length_label.set(f"{len(snd.sample.data)} samples")
            
            # Update Slider Ranges
            slen = int(len(snd.sample.data))
            self.set_widget_max(self.edit_loop_start, slen)
            self.set_widget_max(self.edit_loop_end, slen)
            self.set_widget_max(self.edit_sample_start, slen)
            self.set_widget_max(self.edit_sample_end, slen)
            # self.set_widget_max(self.edit_cross, slen // 2)
            
            # Update Params
            self.edit_vol.set(snd.params.get("volume", 1.0))
            self.edit_pitch.set(snd.params.get("pitch", 1.0))
            self.edit_att.set(snd.params.get("attack", 0.0))
            self.edit_dec.set(snd.params.get("decay", 0.0))
            self.edit_curve.set(snd.params.get("decay_curve", 1.0))
            
            self.edit_loop.set(snd.params.get("loop", False))
            self.edit_loop_start.set(snd.params.get("loop_start", 0))
            l_end = snd.params.get("loop_end", -1)
            if l_end < 0: l_end = slen
            self.edit_loop_end.set(l_end)
            
            self.edit_sample_start.set(snd.params.get("sample_start", 0))
            s_end = snd.params.get("sample_end", -1)
            if s_end < 0: s_end = slen
            self.edit_sample_end.set(s_end)
            
            self.edit_cross.set(snd.params.get("crossfade", 0))

            # Granular Params
            if hasattr(self, 'density_input'):
                 self.density_input.set(snd.params.get("density", 20.0))
                 self.grain_dur_input.set(snd.params.get("grain_dur", 0.1))
                 self.jitter_pos_input.set(snd.params.get("jitter_pos", 0.05))
                 self.jitter_pitch_input.set(snd.params.get("jitter_pitch", 0.0))
                 self.jitter_dur_input.set(snd.params.get("jitter_dur", 0.0))
        else:
            self.file_label.set("Empty")
            self.length_label.set("")
            # Reset params?
            self.edit_vol.set(1.0)
            self.edit_pitch.set(1.0)
            self.edit_att.set(0.0)
            self.edit_dec.set(0.0)
            self.edit_curve.set(1.0)
            self.edit_loop.set(False)
            self.edit_loop_start.set(0)
            self.edit_loop_end.set(100000) # Default max for slider
            self.edit_sample_start.set(0)
            self.edit_sample_end.set(100000)
            self.edit_cross.set(0)

        self.ignore_updates = False

    def on_param_edit(self):
        if self.ignore_updates: return
        
        sid = self.current_inspect_id
        if sid is None or sid not in self.sounds: return
        
        snd = self.sounds[sid]
        snd.params["volume"] = float(self.edit_vol())
        snd.params["pitch"] = float(self.edit_pitch())
        snd.params["attack"] = float(self.edit_att())
        snd.params["decay"] = float(self.edit_dec())
        snd.params["decay_curve"] = float(self.edit_curve())
        
        loop = bool(self.edit_loop())
        l_start = int(self.edit_loop_start())
        l_end = int(self.edit_loop_end())
        s_start = int(self.edit_sample_start())
        s_end = int(self.edit_sample_end())
        cross = float(self.edit_cross())
        
        snd.params["loop"] = loop
        snd.params["loop_start"] = l_start
        snd.params["loop_end"] = l_end
        snd.params["sample_start"] = s_start
        snd.params["sample_end"] = s_end
        snd.params["crossfade"] = cross
        
        # Update underlying sample object (so new triggers pick it up)
        snd.sample.loop = loop
        snd.sample.loop_start = l_start
        if l_end < 0:
             l_end_actual = len(snd.sample.data)
             snd.sample.loop_end = l_end_actual
        else:
             l_end_actual = l_end
             snd.sample.loop_end = l_end
             
        snd.sample.sample_start = s_start
        if s_end < 0:
             s_end_actual = len(snd.sample.data)
             snd.sample.sample_end = s_end_actual
        else:
             s_end_actual = s_end
             snd.sample.sample_end = s_end
             
        loop_len = l_end_actual - l_start
        if loop_len < 0: loop_len = 0
        xf_frames = int(loop_len * cross)
        snd.sample.crossfade_frames = xf_frames
        
        # Update active voices
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                if alloc["sid"] == sid:
                    idx = alloc["idx"]
                    # Send update commands
                    SamplerEngineNode.engine.set_voice_loop_window(idx, loop, l_start, snd.sample.loop_end, xf_frames)
                    SamplerEngineNode.engine.set_voice_playback_range(idx, s_start, snd.sample.sample_end)
                    SamplerEngineNode.engine.set_voice_envelope(idx, snd.params["attack"], snd.params["decay"], snd.params["decay_curve"])
                    # Should we update volume/pitch?
                    # Volume/Pitch in params are BASE.
                    # Trigger might have had multipliers.
                    # We can't know the multipliers easily unless we stored them (we did: alloc["pitch"], but not vol mult?)
                    # Let's assume params edit only affects base.
                    # We can update pitch if we stored pitch mult.
                    if "pitch" in alloc:
                        pmult = alloc["pitch"]
                        eff_pitch = snd.params["pitch"] * pmult
                        SamplerEngineNode.engine.set_voice_pitch(idx, eff_pitch)
                
    def find_free_voice(self):
        if not SamplerEngineNode.engine: return None
        
        # Check pool
        # Preference: 
        # 1. Totally inactive engine voice
        # 2. Steal oldest release? (Not impl yet)
        
        start = self.start_voice_idx
        end = start + self.voice_count
        
        # We need to respect global engine limits (128)
        start = max(0, min(127, start))
        end = max(0, min(128, end))
        
        # Identify voices we have already claimed (even if engine hasn't started them yet)
        busy_indices = {alloc["idx"] for alloc in self.active_allocations}
        
        for i in range(start, end):
             v = SamplerEngineNode.engine.voices[i]
             if not v.active and i not in busy_indices:
                 return i
        return None

    def on_trigger(self):
        data = self.trigger_input()
        # Format: sound_id, [pitch_mult], [velocity/vol]
        
        # Support bare int (trigger specific sound with default params)
        if isinstance(data, (int, float, str)):
             data = [data]
             
        # Check for Fader Control Mode: List of lists [[id, val], ...]
        # OR simple list of 2 elements [id, val] if val is float?
        # Actually, standard trigger is [id, pitch_mult, vol_mult].
        # Ambiguity between [id, pitch] and [id, val].
        # User requested: "list of lists ... [[id, fader], [id, fader]]"
        # If user sends just [[id, val]], that works.
        # But if user sends [id, val], is it trigger or fader?
        # Standard trigger: [id, pitch, vol]
        # Fader mode logic is distinctive enough we should stick to List-of-Lists requirement 
        # to disambiguate.
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                 self.handle_fader_list(data)
                 return
            # Handle empty list case?
            if len(data) == 0: return

        if not data: return
        try:
             sid = int(data[0])
        except:
             return
        
        if sid not in self.sounds:
            return

        snd = self.sounds[sid]
        
        pitch_mult = 1.0
        if len(data) > 1: pitch_mult = float(data[1])
        
        vol_mult = 1.0
        if len(data) > 2: vol_mult = float(data[2])
        
        # Allocate
        idx = self.find_free_voice()
        if idx is not None:
             # Play
             voice_params = snd.params
             
             eff_vol = voice_params["volume"] * vol_mult
             eff_pitch = voice_params["pitch"] * pitch_mult
             
             e = SamplerEngineNode.engine
             if e:
                 e.set_voice_envelope(idx, voice_params["attack"], voice_params["decay"], voice_params["decay_curve"])
                 e.play_voice(idx, snd.sample, eff_vol, eff_pitch)
                 
                 # Record allocation
                 self.active_allocations.append({
                     "idx": idx,
                     "sid": sid,
                     "pitch": pitch_mult,
                     "midi_note": None,
                     "time": time.time()
                 })

                 # Add frame task for cleanup
                 self.add_frame_task()

    def on_midi(self):
        data = self.midi_input()
        if not isinstance(data, list) or len(data) < 2:
            return
            
        sid = None
        note = 0
        vel = 0
        
        try:
            if len(data) == 3:
                sid = int(data[0])
                note = int(data[1])
                vel = int(data[2])
            else:
                note = int(data[0])
                vel = int(data[1])
        except:
            return
            
        base_note = self.base_note_input()
        
        if vel > 0:
            # Note On
            if sid is None:
                # determine from inspect
                sid = self.current_inspect_id
                if sid is None:
                    if len(self.sounds) > 0:
                        sid = next(iter(self.sounds))
                    else:
                        return
                        
            if sid not in self.sounds:
                return
                
            snd = self.sounds[sid]
            
            # Calculate pitch
            # 1.0 at base_note
            # pitch = 2 ** ((note - base) / 12)
            pitch_ratio = 2 ** ((note - base_note) / 12.0)
            
            voice_params = snd.params
            eff_pitch = voice_params["pitch"] * pitch_ratio
            eff_vol = voice_params["volume"] * (vel / 127.0)
            
            idx = self.find_free_voice()
            if idx is not None:
                e = SamplerEngineNode.engine
                if e:
                    e.set_voice_envelope(idx, voice_params["attack"], voice_params["decay"], voice_params["decay_curve"])
                    e.play_voice(idx, snd.sample, eff_vol, eff_pitch)
                    
                    self.active_allocations.append({
                        "idx": idx,
                        "sid": sid,
                        "pitch": pitch_ratio,
                        "midi_note": note,
                        "time": time.time()
                    })
                    self.add_frame_task()
        else:
            # Note Off (vel == 0)
            # Find voices with this midi note
            to_release = []
            for alloc in self.active_allocations:
                if alloc.get("midi_note") == note:
                    if sid is not None:
                        # If a specific SID was targeted, only release that sound
                        # Note: if data was 2-element, sid is None here, so we release all matching note
                        if alloc["sid"] == sid:
                            to_release.append(alloc)
                    else:
                        # Legacy behavior: release any voice playing this note
                        to_release.append(alloc)
            
            for alloc in to_release:
                if SamplerEngineNode.engine:
                    SamplerEngineNode.engine.stop_voice(alloc["idx"])
                    # We rely on frame task to remove from active_allocations when it actually finishes

    def handle_fader_list(self, data):
        # Data: [[sid, fade], [sid, fade], ...]
        # 1. Parse desired state
        desired_state = {}
        for item in data:
            if len(item) >= 2:
                try:
                    sid = int(item[0]) # Enforce int
                    fade = float(item[1])
                    desired_state[sid] = fade
                except:
                    pass
        
        # 2. Identify active SIDs
        active_sids = set()
        for alloc in self.active_allocations:
            active_sids.add(alloc["sid"])
            
        # 3. Update or Start
        for sid, fade in desired_state.items():
            if sid not in self.sounds: continue
            
            if sid in active_sids:
                # Update existing
                for alloc in self.active_allocations:
                    if alloc["sid"] == sid:

                        idx = alloc["idx"]
                        # Update fade tracking
                        alloc["fade"] = fade
                        if fade <= 0.001:
                            if alloc.get("fade_zero_since") is None:
                                alloc["fade_zero_since"] = time.time()
                        else:
                            alloc["fade_zero_since"] = None
                        
                        # Apply volume
                        snd = self.sounds[sid]
                        eff_vol = snd.params["volume"] * fade
                        if SamplerEngineNode.engine:
                            SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
            
            else:
                # Start new voice (if fade > 0)
                if fade > 0.0:

                    idx = self.find_free_voice()
                    if idx is not None and SamplerEngineNode.engine:
                        snd = self.sounds[sid]
                        voice_params = snd.params
                        
                        eff_vol = voice_params["volume"] * fade
                        eff_pitch = voice_params["pitch"]
                        
                        # Start with 0 attack for fader mode?
                        # User said: "ignoring the attack parameter"
                        # We force attack to 0.0 for this trigger.
                        SamplerEngineNode.engine.set_voice_envelope(idx, 0.0, voice_params["decay"], voice_params["decay_curve"])
                        SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch)
                        
                        alloc_record = {
                            "idx": idx,
                            "sid": sid,
                            "pitch": 1.0, 
                            "time": time.time(),
                            "fade": fade,
                            "fade_zero_since": None,
                            "midi_note": None
                        }
                        self.active_allocations.append(alloc_record)
                        self.add_frame_task()
                
                # Check for zero fade timeout on START?
                # If we start with 0.0, we should track timeout immediately.
                if fade <= 0.001:
                    # Search for the alloc we just added or updated
                    for alloc in self.active_allocations:
                        if alloc["sid"] == sid and alloc["idx"] == idx:
                             if alloc.get("fade_zero_since") is None:
                                 alloc["fade_zero_since"] = time.time()

        # 4. Stop missing
        # "If there are currently playing sounds whose id is not in the list, they should stop playing."
        # Note: This logic assumes we ONLY control voices via this list mode if we are using it.
        # But normal triggers might also be active. 
        # However, the user request implies this input list defines the *set* of playing sounds.
        # We will stop ANY sound ID not in the desired_state keys?
        # Or only those that were started via fader mode?
        # User said "If there are currently playing sounds whose id is not in the list, they should stop playing."
        # This is quite aggressive. It turns the trigger input into a state description.
        # We will iterate and release those not in desired_state.
        
        for alloc in self.active_allocations:
            # Only stop voices that were involved in fader logic (have 'fade' key?)
            # Or just blindly stop? User said "currently playing sounds whose id is not in the list".
            # Safest is to only stop those that match our active sounds logic.
            # But let's assume this node is exclusive for these sounds if using fader mode.
            if alloc["sid"] not in desired_state:
                # Stop it
                if SamplerEngineNode.engine:
                    SamplerEngineNode.engine.stop_voice(alloc["idx"])
                # We can mark it as effectively stopped so we don't re-stop it?
                # Process in frame_task will clean it up.

    def on_stop(self):
        data = self.stop_input()
        # Format: sound_id, [pitch]
        
        if isinstance(data, (int, float, str)):
             data = [data]
             
        # logic: release all voices matching sound_id (and pitch if provided)
        if not data: return
        try:
             sid = int(data[0])
        except:
             return
        
        pitch_match = None
        if len(data) > 1: pitch_match = float(data[1])
        
        to_remove = []
        for alloc in self.active_allocations:
            try:
                alloc_sid = int(alloc["sid"])
            except:
                alloc_sid = alloc["sid"]
                
            if alloc_sid == sid:
                if pitch_match is None or abs(alloc["pitch"] - pitch_match) < 0.001:
                    # Release
                    if SamplerEngineNode.engine:
                        SamplerEngineNode.engine.stop_voice(alloc["idx"])
                    # Don't remove immediately, let frame task handle cleanup when inactive?
                    # Or just mark as 'released'?
                    # Engine releases are async.
                    # We can leave it in active_allocations until it actually stops (is_active=False).
                    # But we should stop matching it for new 'stop' commands maybe?
                    # Actually, if we spam stop, it's fine.
        
        # Only cleanup happens in frame_task

    def frame_task(self):
        # Cleanup inactive allocations
        if not SamplerEngineNode.engine: return
        
        alive = []
        now = time.time()
        
        for alloc in self.active_allocations:
            idx = alloc["idx"]
            v = SamplerEngineNode.engine.voices[idx]
            
            # Auto-stop check
            if alloc.get("fade_zero_since") is not None:
                if now - alloc["fade_zero_since"] > 1.0:
                    SamplerEngineNode.engine.stop_voice(idx)
                    # It will be caught as inactive below or next frame
            
            # Race Condition Fix:
            # Voice.active is updated by Audio Thread (async).
            # If we just triggered it, it might still be active=False on this thread.
            # We must check the trigger time.
            grace_period = 0.2
            is_new = (now - alloc.get("time", 0)) < grace_period
            
            if v.active or is_new:
                alive.append(alloc)
        
        last_count = getattr(self, 'last_output_count', -1)
        count_changed = len(alive) != last_count
        # Or always update to animate? 
        # For grid, we want to update if state changes? 
        # Active indices set changes.
        # Let's track set of active indices?
        
        current_indices = {alloc["idx"] for alloc in alive}
        last_indices = getattr(self, 'last_active_indices', set())
        
        if current_indices != last_indices:
             self.active_allocations = alive
             self.last_active_indices = current_indices
             self.last_output_count = len(alive)
             
             # Grid Output
             # Map active indices to local range
             start = self.start_voice_idx
             count = self.voice_count
             
             grid_flat = np.zeros(count, dtype=int)
             for idx in current_indices:
                 if start <= idx < start + count:
                     grid_flat[idx - start] = 1
            
             # Reshape logic: Max 32 cols
             cols = 32
             if count > cols:
                 rows = (count + cols - 1) // cols
                 pad = (rows * cols) - count
                 if pad > 0:
                     grid_flat = np.pad(grid_flat, (0, pad))
                 grid = grid_flat.reshape((rows, cols))
                 self.active_out.send(grid)
             else:
                 # If count <= 32, user might still want 2D (1, count)? 
                 # Or (count,)?
                 # "maximum of 32 voices per row"
                 # (1, count) is safe for grid display?
                 # Let's verify user intent. "32 x 4 array" for 128 voices.
                 # So simple reshaping.
                 
                 # Fix: Ensure at least 2D shape (1, count) to avoid scalar confusion in dpg_system
                 if count > 0:
                     grid = grid_flat.reshape((1, count))
                     self.active_out.send(grid)
                 else:
                     self.active_out.send(grid_flat)
        else:
             self.active_allocations = alive
             
        if not self.active_allocations:
            self.remove_frame_tasks()
        
    def custom_cleanup(self):
        pass

    def save_custom(self, container):
        sounds_data = {}
        for sid, snd in self.sounds.items():
            # Save path and params
            sounds_data[str(sid)] = {
                "path": snd.path,
                "params": snd.params
            }
        container['sounds'] = sounds_data
        if self.current_inspect_id is not None:
             container['inspect_id'] = self.current_inspect_id

    def load_custom(self, container):
        if 'inspect_id' in container:
            try:
                self.current_inspect_id = int(container['inspect_id'])
                # Update widget
                self.inspect_id_input.set(self.current_inspect_id)
            except:
                pass
                
        sounds_data = container.get('sounds', {})
        for sid_str, data in sounds_data.items():
            try:
                sid = int(sid_str)
                path = data.get("path", "")
                params = data.get("params", {})
                
                if path and os.path.exists(path):
                    try:
                        snd = Sound(path, params)
                        self.sounds[sid] = snd
                    except Exception as e:
                        print(f"Error restoring sound {sid}: {e}")
            except Exception as e:
                print(f"Error loading sound data: {e}")
        
        # Sync UI if needed
        if self.current_inspect_id is not None:
            self.sync_ui_to_sound(self.current_inspect_id)


class GranularSamplerNode(PolyphonicSamplerNode):
    @staticmethod
    def factory(name, data, args=None):
        node = GranularSamplerNode(name, data, args)
        return node
        
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
    def add_additional_parameters(self):
        # Additional Inputs
        # self.pos_mod_input = self.add_input('position_mod', callback=self.on_pos_change) # Removed from top
        
        # Granular Parameters Widgets
        self.add_label("Granular Params")
        self.g_density = self.add_input('grain density', widget_type='drag_float', default_value=20.0, min=0.1, max=2000.0, callback=self.on_granular_edit)
        self.g_dur = self.add_input('grain_size (s)', widget_type='drag_float', default_value=0.1, min=0.001, max=1.0, callback=self.on_granular_edit)
        
        # New: Grain Position Slider (Mod Input)
        self.pos_mod_input = self.add_input('grain position', widget_type='slider_float', default_value=0.0, min=0.0, max=1.0, callback=self.on_pos_change, widget_width=240)

        self.g_jit_pos = self.add_input('jitter pos', widget_type='drag_float', default_value=0.05, min=0.0, max=1.0, callback=self.on_granular_edit)
        self.g_jit_pitch = self.add_input('jitter pitch', widget_type='drag_float', default_value=0.0, min=0.0, max=1.0, callback=self.on_granular_edit)
        self.g_jit_dur = self.add_input('jitter size', widget_type='drag_float', default_value=0.0, min=0.0, max=1.0, callback=self.on_granular_edit)
        
    def create_loop_widgets(self):
        # Override to prevent creation of loop widgets in Granular Node.
        # Use dummy objects to prevent crashes in sync_ui_to_sound or on_param_edit.
        class DummyWidget:
            def __call__(self): return 0
            def set(self, val): pass
            
        self.edit_loop = DummyWidget()
        self.edit_loop_start = DummyWidget()
        self.edit_loop_end = DummyWidget()
        self.edit_cross = DummyWidget()
        

        
    # Override sync_ui_to_sound to handle granular params
    def sync_ui_to_sound(self, sid):
        super().sync_ui_to_sound(sid)
        if sid not in self.sounds:
            # Reset defaults
            # (Super already handles ignore_updates flag? No, super sets it for its own block)
            # We need to wrap or rely on super's state. Super sets ignore_updates=True then False.
            # So we should call super, then do our own block with ignore_updates=True
            self.ignore_updates = True
            self.g_density.set(20.0)
            self.g_dur.set(0.1)
            self.g_jit_pos.set(0.05)
            self.g_jit_pitch.set(0.0)
            self.g_jit_dur.set(0.0)
            self.ignore_updates = False
            return

        snd = self.sounds[sid]
        self.ignore_updates = True
        self.g_density.set(snd.params.get("density", 20.0))
        self.g_dur.set(snd.params.get("grain_dur", 0.1))
        self.g_jit_pos.set(snd.params.get("jitter_pos", 0.05))
        self.g_jit_pitch.set(snd.params.get("jitter_pitch", 0.0))
        self.g_jit_dur.set(snd.params.get("jitter_dur", 0.0))
        self.ignore_updates = False

    def on_granular_edit(self):
        if self.ignore_updates: return
        
        sid = self.current_inspect_id
        if sid is None or sid not in self.sounds: return
        
        snd = self.sounds[sid]
        
        # Read widgets
        d = float(self.g_density())
        dur = float(self.g_dur())
        jp = float(self.g_jit_pos())
        jpi = float(self.g_jit_pitch())
        jd = float(self.g_jit_dur())
        
        # Update Sound Params
        snd.params["density"] = d
        snd.params["grain_dur"] = dur
        snd.params["jitter_pos"] = jp
        snd.params["jitter_pitch"] = jpi
        snd.params["jitter_dur"] = jd
        
        # Update active voices for THIS sound
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                if alloc["sid"] == sid:
                    idx = alloc["idx"]
                    SamplerEngineNode.engine.set_voice_granular_params(idx, d, dur, jp, jpi, jd)

    def on_pos_change(self):
        # Modulation from input pin
        val = float(self.pos_mod_input())
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                SamplerEngineNode.engine.set_voice_grain_position(alloc["idx"], val)

    def on_trigger(self):
        data = self.trigger_input()
        
        if isinstance(data, (int, float, str)):
             data = [data]

        # Fader Mode Override: [[id, vol, pos], ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
             self.handle_fader_list(data)
             return
             
        # Standard Trigger: [id, pitch, vol]
        # We need to reimplement super().on_trigger LOGIC here to inject granular params.
        # Or let super spawn, then we update params immediately.
        # But super spawns with standard logic.
        # Copy-paste super logic is safest to ensure atomic param setup.
        
        if not data: return
        # Re-implement trigger logic for granular
        try:
             sid = int(data[0])
        except:
             return
        
        if sid not in self.sounds: return

        snd = self.sounds[sid]
        
        pitch_mult = 1.0
        if len(data) > 1: pitch_mult = float(data[1])
        
        vol_mult = 1.0
        if len(data) > 2: vol_mult = float(data[2])
        
        idx = self.find_free_voice()
        if idx is not None:
             vp = snd.params
             eff_vol = vp["volume"] * vol_mult
             eff_pitch = vp["pitch"] * pitch_mult
             
             e = SamplerEngineNode.engine
             if e:
                 # Standard Params
                 e.set_voice_envelope(idx, vp["attack"], vp["decay"], vp["decay_curve"])
                 
                 # Granular Params from Sound
                 d = vp.get("density", 20.0)
                 dur = vp.get("grain_dur", 0.1)
                 jp = vp.get("jitter_pos", 0.05)
                 jpi = vp.get("jitter_pitch", 0.0)
                 jd = vp.get("jitter_dur", 0.0)
                 
                 e.set_voice_mode(idx, "granular")
                 e.set_voice_granular_params(idx, d, dur, jp, jpi, jd)
                 
                 # Default pos? 0.0 usually unless fader.
                 # Actually, standard trigger doesn't set grain pos (keeps explicit set or 0?)
                 # Voice defaults grain_pos to 0.5 or 0? 0.0.
                 # Let's set to 0.0 on new trigger? Or randomize?
                 # Standard behavior: Start at beginning.
                 e.set_voice_grain_position(idx, 0.0)
                 
                 e.play_voice(idx, snd.sample, eff_vol, eff_pitch)
                 
                 self.active_allocations.append({
                     "idx": idx,
                     "sid": sid,
                     "pitch": pitch_mult,
                     "time": time.time()
                 })

                 self.add_frame_task()

    def handle_fader_list(self, data):
        # Override to handle [id, vol, pos]
        desired_state = {}
        for item in data:
            if len(item) >= 2:
                try:
                    sid = int(item[0])
                    vol = float(item[1])
                    pos = float(item[2]) if len(item) > 2 else 0.0
                    desired_state[sid] = {"vol": vol, "pos": pos}
                except:
                    pass

        active_sids = set()
        for alloc in self.active_allocations:
            active_sids.add(alloc["sid"])
        
        busy_indices = {alloc["idx"] for alloc in self.active_allocations}
        
        for sid, state in desired_state.items():
            vol = state["vol"]
            pos = state["pos"]
            
            if sid not in self.sounds: continue
            
            if sid in active_sids:
                # Update existing
                for alloc in self.active_allocations:
                    if alloc["sid"] == sid:
                        idx = alloc["idx"]
                        alloc["fade"] = vol
                        
                        if vol <= 0.001:
                            if alloc.get("fade_zero_since") is None:
                                alloc["fade_zero_since"] = time.time()
                        else:
                            alloc["fade_zero_since"] = None
                        
                        snd = self.sounds[sid]
                        eff_vol = snd.params["volume"] * vol
                        
                        if SamplerEngineNode.engine:
                            SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
                            SamplerEngineNode.engine.set_voice_grain_position(idx, pos)
            else:
                # Start new
                if vol > 0.0:
                    idx = self.find_free_voice()
                    if idx is not None and idx not in busy_indices:
                        if SamplerEngineNode.engine:
                            snd = self.sounds[sid]
                            vp = snd.params
                            
                            eff_vol = vp["volume"] * vol
                            eff_pitch = vp["pitch"]
                            
                            # Granular Params from Sound (Per-Sound!)
                            d = vp.get("density", 20.0)
                            dur_s = vp.get("grain_dur", 0.1)
                            jp = vp.get("jitter_pos", 0.05)
                            jpi = vp.get("jitter_pitch", 0.0)
                            jd = vp.get("jitter_dur", 0.0)
                            
                            SamplerEngineNode.engine.set_voice_granular_params(idx, d, dur_s, jp, jpi, jd)
                            SamplerEngineNode.engine.set_voice_grain_position(idx, pos)
                            
                            SamplerEngineNode.engine.set_voice_envelope(idx, 0.0, vp["decay"], vp["decay_curve"])
                            SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch, mode='granular')
                            
                            alloc_record = {
                                "idx": idx,
                                "sid": sid,
                                "pitch": 1.0, 
                                "time": time.time(),
                                "fade": vol,
                                "fade_zero_since": None
                            }
                            self.active_allocations.append(alloc_record)
                            busy_indices.add(idx)
                            self.add_frame_task()
 
    def on_pos_change(self):
        val = any_to_float(self.pos_mod_input())
        # Update all active granular voices for this node?
        # Or just the last one?
        # Usually, a modulation input affects ALL voices spawned by this node, 
        # or we need a way to map.
        # Simple approach: Update all active voices belonging to this node.
        
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                 idx = alloc["idx"]
                 # Check if voice is actually active?
                 if SamplerEngineNode.engine.voices[idx].active:
                     SamplerEngineNode.engine.set_voice_grain_position(idx, val)

    def handle_fader_list(self, data):
        # Override to handle [id, vol, pos]
        desired_state = {}
        for item in data:
            if len(item) >= 2:
                try:
                    sid = int(item[0])
                    vol = float(item[1])
                    pos = float(item[2]) if len(item) > 2 else 0.0
                    desired_state[sid] = {"vol": vol, "pos": pos}
                except:
                    pass

        active_sids = set()
        for alloc in self.active_allocations:
            active_sids.add(alloc["sid"])
        
        busy_indices = {alloc["idx"] for alloc in self.active_allocations}
        
        for sid, state in desired_state.items():
            vol = state["vol"]
            pos = state["pos"]
            
            if sid not in self.sounds: continue
            
            if sid in active_sids:
                # Update existing
                for alloc in self.active_allocations:
                    if alloc["sid"] == sid:
                        idx = alloc["idx"]
                        alloc["fade"] = vol
                        
                        if vol <= 0.001:
                            if alloc.get("fade_zero_since") is None:
                                alloc["fade_zero_since"] = time.time()
                        else:
                            alloc["fade_zero_since"] = None
                        
                        snd = self.sounds[sid]
                        eff_vol = snd.params["volume"] * vol
                        
                        if SamplerEngineNode.engine:
                            SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
                            SamplerEngineNode.engine.set_voice_grain_position(idx, pos)
            else:
                # Start new
                if vol > 0.0:
                    idx = self.find_free_voice()
                    if idx is not None and idx not in busy_indices:
                        if SamplerEngineNode.engine:
                            snd = self.sounds[sid]
                            vp = snd.params
                            
                            eff_vol = vp["volume"] * vol
                            eff_pitch = vp["pitch"]
                            
                            # Granular Params from Sound (Per-Sound!)
                            d = vp.get("density", 20.0)
                            dur_s = vp.get("grain_dur", 0.1)
                            jp = vp.get("jitter_pos", 0.05)
                            jpi = vp.get("jitter_pitch", 0.0)
                            jd = vp.get("jitter_dur", 0.0)
                            
                            SamplerEngineNode.engine.set_voice_granular_params(idx, d, dur_s, jp, jpi, jd)
                            SamplerEngineNode.engine.set_voice_grain_position(idx, pos)
                            
                            SamplerEngineNode.engine.set_voice_envelope(idx, 0.0, vp["decay"], vp["decay_curve"])
                            SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch, mode='granular')
                            
                            alloc_record = {
                                "idx": idx,
                                "sid": sid,
                                "pitch": 1.0, 
                                "time": time.time(),
                                "fade": vol,
                                "fade_zero_since": None
                            }
                            self.active_allocations.append(alloc_record)
                            busy_indices.add(idx)

                            self.add_frame_task()
 
        # Stop missing
        for alloc in self.active_allocations:
            if alloc["sid"] not in desired_state:
                if SamplerEngineNode.engine:
                    SamplerEngineNode.engine.stop_voice(alloc["idx"])

    def on_midi(self):
        data = self.midi_input()
        if not isinstance(data, list) or len(data) < 2:
            return
            
        sid = None
        note = 0
        vel = 0
        
        try:
            if len(data) == 3:
                sid = int(data[0])
                note = int(data[1])
                vel = int(data[2])
            else:
                note = int(data[0])
                vel = int(data[1])
        except:
            return
            
        if vel > 0:
            # Note On
            if sid is None:
                sid = self.current_inspect_id
                if sid is None:
                    if len(self.sounds) > 0:
                        sid = next(iter(self.sounds))
                    else:
                        return
            
            if sid not in self.sounds:
                return
                
            snd = self.sounds[sid]
            vol_mult = vel / 127.0
            
            idx = self.find_free_voice()
            if idx is not None:
                vp = snd.params
                
                eff_vol = vp["volume"] * vol_mult
                eff_pitch = vp["pitch"] # Use sound pitch, ignore MIDI note pitch
                
                # Granular Params
                d = vp.get("density", 20.0)
                dur_s = vp.get("grain_dur", 0.1)
                jp = vp.get("jitter_pos", 0.05)
                jpi = vp.get("jitter_pitch", 0.0)
                jd = vp.get("jitter_dur", 0.0)
                
                if SamplerEngineNode.engine:
                    SamplerEngineNode.engine.set_voice_granular_params(idx, d, dur_s, jp, jpi, jd)
                    
                    # Ensure position is set
                    pos_val = any_to_float(self.pos_mod_input())
                    SamplerEngineNode.engine.set_voice_grain_position(idx, pos_val)

                    SamplerEngineNode.engine.set_voice_envelope(idx, vp["attack"], vp["decay"], vp["decay_curve"])
                    SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch, mode='granular')
                    
                    self.active_allocations.append({
                        "idx": idx,
                        "sid": sid,
                        "pitch": 1.0,
                        "midi_note": note,
                        "time": time.time()
                    })
                    self.add_frame_task()
        else:
             super().on_midi()

    def on_trigger(self):
        data = self.trigger_input()
        
        if isinstance(data, (int, float, str)):
             data = [data]

        # Fader Mode Override: [[id, vol, pos], ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
             self.handle_fader_list(data)
             return
             
        # Standard Trigger: [id, pitch, vol]
        if not data: return
        try:
             sid = int(data[0])
        except:
             return
        
        if sid not in self.sounds:
            return

        snd = self.sounds[sid]
        
        pitch_mult = 1.0
        if len(data) > 1: pitch_mult = float(data[1])
        
        vol_mult = 1.0
        if len(data) > 2: vol_mult = float(data[2])
        
        # Allocate
        idx = self.find_free_voice()
        if idx is not None:
             vp = snd.params
             
             eff_vol = vp["volume"] * vol_mult
             eff_pitch = vp["pitch"] * pitch_mult
             
             # Granular Params from Sound
             d = vp.get("density", 20.0)
             dur_s = vp.get("grain_dur", 0.1)
             jp = vp.get("jitter_pos", 0.05)
             jpi = vp.get("jitter_pitch", 0.0)
             jd = vp.get("jitter_dur", 0.0)
             
             if SamplerEngineNode.engine:
                 SamplerEngineNode.engine.set_voice_granular_params(idx, d, dur_s, jp, jpi, jd)
                 # Wait, position modulation?
                 # If using trigger, position mod input pin is used in real-time.
                 # But we might want start pos? No, Granular engine reads pos mod.
                 # We just need to ensure start pos is what? 
                 # Usually grain pos is driven by set_voice_grain_position command.
                 # on_pos_change handles the pin.
                 # Do we need to force it?
                 # If we don't, it stays at last value.
                 # Let's read the pin now to be sync.
                 pos_val = any_to_float(self.pos_mod_input())
                 SamplerEngineNode.engine.set_voice_grain_position(idx, pos_val)

                 SamplerEngineNode.engine.set_voice_envelope(idx, vp["attack"], vp["decay"], vp["decay_curve"])
                 SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch, mode='granular')
                 
                 # Looping?
                 # Granular voices usually run until stopped or if envelope ends.
                 # But play_voice sets `voice.looping = sample.loop`.
                 # Granular engine ignores loop flag usually? 
                 # It just spawns grains based on active state.
                 # If sample.loop is true, does it matter?
                 # Voice.looping is used in `_process_granular`? No.
                 # `_process_granular` runs while `active` is true.
                 # `active` is managed by envelope/release.
                 
                 self.last_voice_idx = idx
                 
                 self.last_voice_idx = idx
                 
                 # Track allocation
                 self.active_allocations.append({
                     "idx": idx,
                     "sid": sid,
                     "pitch": pitch_mult,
                     "time": time.time(),
                     "fade": 1.0,
                     "fade_zero_since": None
                 })
                 self.add_frame_task()


class ScratchSamplerNode(PolyphonicSamplerNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ScratchSamplerNode(name, data, args)
        return node
        
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        
    def add_additional_parameters(self):
        self.add_label("Scratch Params")
        self.pos_input = self.add_input('position', widget_type='slider_float', default_value=0.0, min=0.0, max=1.0, callback=self.on_pos_change, widget_width=240)
        self.max_vel_input = self.add_input('max velocity', widget_type='drag_float', default_value=2.0, min=0.1, max=32.0, callback=self.on_param_change)
        self.accel_input = self.add_input('acceleration', widget_type='drag_float', default_value=1.0, min=0.1, max=50.0, callback=self.on_param_change)
        self.thresh_input = self.add_input('damp zone', widget_type='drag_float', default_value=0.15, min=0.001, max=1.0, callback=self.on_param_change)
        self.curve_input = self.add_input('response curve', widget_type='drag_float', default_value=2.0, min=1.0, max=5.0, callback=self.on_param_change)
        
    def create_loop_widgets(self):
        # Disable loop widgets
        class DummyWidget:
            def __call__(self): return 0
            def set(self, val): pass
        self.edit_loop = DummyWidget()
        self.edit_loop_start = DummyWidget()
        self.edit_loop_end = DummyWidget()
        self.edit_cross = DummyWidget()

    def on_pos_change(self):
        val = float(self.pos_input())
        # Update active voices
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                sid = alloc["sid"]
                if sid in self.sounds:
                    snd = self.sounds[sid]
                    target = val * len(snd.sample.data)
                    SamplerEngineNode.engine.set_voice_scratch_target(alloc["idx"], target)
                    
    def on_param_change(self):
        mv = float(self.max_vel_input())
        acc = float(self.accel_input())
        th = float(self.thresh_input())
        ex = float(self.curve_input())
        
        # Update active voices
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                 SamplerEngineNode.engine.set_voice_scratch_params(alloc["idx"], mv, acc)
                 SamplerEngineNode.engine.set_voice_scratch_tuning(alloc["idx"], th, ex)
                 
        # Update current sound params (if inspecting)
        sid = self.current_inspect_id
        if sid is not None and sid in self.sounds:
            snd = self.sounds[sid]
            snd.params["scratch_max_vel"] = mv
            snd.params["scratch_accel"] = acc
            snd.params["scratch_thresh"] = th
            snd.params["scratch_curve"] = ex

    def sync_ui_to_sound(self, sid):
        super().sync_ui_to_sound(sid)
        if sid in self.sounds:
            snd = self.sounds[sid]
            vp = snd.params
            self.max_vel_input.set(vp.get("scratch_max_vel", 2.0))
            self.accel_input.set(vp.get("scratch_accel", 1.0))
            self.thresh_input.set(vp.get("scratch_thresh", 0.15))
            self.curve_input.set(vp.get("scratch_curve", 2.0))

    def on_midi(self):
        data = self.midi_input()
        if not isinstance(data, list) or len(data) < 2:
            return
            
        sid = None
        note = 0
        vel = 0
        
        try:
            if len(data) == 3:
                sid = int(data[0])
                note = int(data[1])
                vel = int(data[2])
            else:
                note = int(data[0])
                vel = int(data[1])
        except:
            return
            
        if vel > 0:
            # Note On
            if sid is None:
                sid = self.current_inspect_id
                if sid is None:
                    if len(self.sounds) > 0:
                        sid = next(iter(self.sounds))
                    else:
                        return
            
            if sid not in self.sounds:
                return
                
            snd = self.sounds[sid]
            
            # Use velocity for volume, ignore pitch (Scratch uses position)
            # Or should we allow pitch? User said "behave the same as [id, volume]".
            # Trigger input ignores pitch arg in ScratchSampler (it parses checks > 2, but scratch mode usually 
            # ignores it? Wait, scratch mode uses `play_voice` with 0.0 velocity/pitch?
            # on_trigger: `e.play_voice(..., mode='scratch')`. Pitch is 0.0 passed to play_voice.
            # So scratch ignores pitch.
            
            vol_mult = vel / 127.0
            
            idx = self.find_free_voice()
            if idx is not None:
                vp = snd.params
                # Capture current UI params (or from sound if stored?)
                # trigger logic uses current UI input if sound is inspected? or always?
                # on_trigger uses input widgets: `mv = float(self.max_vel_input())`
                # We should do the same to match "trigger input" behavior
                
                mv = float(self.max_vel_input())
                acc = float(self.accel_input())
                th = float(self.thresh_input())
                ex = float(self.curve_input())
                
                eff_vol = vp["volume"] * vol_mult
                
                e = SamplerEngineNode.engine
                if e:
                    e.set_voice_envelope(idx, vp["attack"], vp["decay"], vp["decay_curve"])
                    e.set_voice_mode(idx, 'scratch')
                    e.set_voice_scratch_params(idx, mv, acc)
                    e.set_voice_scratch_tuning(idx, th, ex)
                    
                    pos_val = float(self.pos_input())
                    target = pos_val * len(snd.sample.data)
                    e.set_voice_scratch_target(idx, target)
                    
                    # Play
                    e.play_voice(idx, snd.sample, eff_vol, 0.0, mode='scratch')
                    
                    self.active_allocations.append({
                        "idx": idx,
                        "sid": sid,
                        "pitch": 1.0,
                        "midi_note": note,
                        "time": time.time()
                    })
                    self.add_frame_task()
        else:
             # Note Off (release)
             super().on_midi() # Use base implementation for Note Off logic
            
    def on_trigger(self):
        data = self.trigger_input()
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
             self.handle_fader_list(data)
             return
             
        if isinstance(data, (int, float, str)): data = [data]
        if not data: return
        
        try:
            sid = int(data[0])
        except:
            return
            
        if sid not in self.sounds: return
        snd = self.sounds[sid]
        
        # Parse multipliers
        vol_mult = 1.0
        if len(data) > 2: vol_mult = float(data[2])
        
        idx = self.find_free_voice()
        if idx is not None:
             vp = snd.params
             # Capture current UI params to sound
             mv = float(self.max_vel_input())
             acc = float(self.accel_input())
             th = float(self.thresh_input())
             ex = float(self.curve_input())
             
             vp["scratch_max_vel"] = mv
             vp["scratch_accel"] = acc
             vp["scratch_thresh"] = th
             vp["scratch_curve"] = ex
             
             eff_vol = vp["volume"] * vol_mult
             
             e = SamplerEngineNode.engine
             if e:
                 e.set_voice_envelope(idx, vp["attack"], vp["decay"], vp["decay_curve"])
                 e.set_voice_mode(idx, 'scratch')
                 e.set_voice_scratch_params(idx, mv, acc)
                 e.set_voice_scratch_tuning(idx, th, ex)
                 
                 # Set Initial Target from Slider
                 pos_val = float(self.pos_input())
                 target = pos_val * len(snd.sample.data)
                 e.set_voice_scratch_target(idx, target)
                 
                 # Play (Start at 0 velocity)
                 e.play_voice(idx, snd.sample, eff_vol, 0.0, mode='scratch')
                 
                 alloc_record = {
                     "idx": idx,
                     "sid": sid,
                     "pitch": 1.0,
                     "time": time.time(),
                     "fade": 1.0,
                     "fade_zero_since": None
                 }
                 self.active_allocations.append(alloc_record)
                 self.add_frame_task()

    def handle_fader_list(self, data):
        # Data: [[id, vol, pos], ...]
        desired_state = {}
        for item in data:
            if len(item) >= 2:
                try:
                    sid = int(item[0])
                    vol = float(item[1])
                    pos = float(item[2]) if len(item) > 2 else 0.0
                    desired_state[sid] = {"vol": vol, "pos": pos}
                except:
                    pass

        active_sids = set()
        for alloc in self.active_allocations:
            active_sids.add(alloc["sid"])
        
        busy_indices = {alloc["idx"] for alloc in self.active_allocations}
        
        for sid, state in desired_state.items():
            vol = state["vol"]
            pos = state["pos"]
            
            if sid not in self.sounds: continue
            snd = self.sounds[sid]
            target_samples = pos * len(snd.sample.data)
            
            if sid in active_sids:
                # Update existing
                for alloc in self.active_allocations:
                    if alloc["sid"] == sid:
                        idx = alloc["idx"]
                        alloc["fade"] = vol
                        
                        if vol <= 0.001:
                            if alloc.get("fade_zero_since") is None:
                                alloc["fade_zero_since"] = time.time()
                        else:
                            alloc["fade_zero_since"] = None
                        
                        eff_vol = snd.params["volume"] * vol
                        
                        if SamplerEngineNode.engine:
                            SamplerEngineNode.engine.set_voice_volume(idx, eff_vol)
                            SamplerEngineNode.engine.set_voice_scratch_target(idx, target_samples)
            else:
                # Start new
                if vol > 0.0:
                    idx = self.find_free_voice()
                    if idx is not None and idx not in busy_indices:
                        if SamplerEngineNode.engine:
                            vp = snd.params
                            
                            eff_vol = vp["volume"] * vol
                            # Params
                            mv = float(self.max_vel_input())
                            acc = float(self.accel_input())
                            th = float(self.thresh_input())
                            ex = float(self.curve_input())
                            
                            SamplerEngineNode.engine.set_voice_scratch_params(idx, mv, acc)
                            SamplerEngineNode.engine.set_voice_scratch_tuning(idx, th, ex)
                            SamplerEngineNode.engine.set_voice_scratch_target(idx, target_samples)
                            
                            SamplerEngineNode.engine.set_voice_envelope(idx, 0.0, vp["decay"], vp["decay_curve"])
                            SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, 0.0, mode='scratch')
                            
                            alloc_record = {
                                "idx": idx,
                                "sid": sid,
                                "pitch": 1.0, 
                                "time": time.time(),
                                "fade": vol,
                                "fade_zero_since": None
                            }
                            self.active_allocations.append(alloc_record)
                            self.add_frame_task()

