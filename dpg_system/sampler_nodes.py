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


def register_sampler_nodes():
    Node.app.register_node('sampler_voice', SamplerVoiceNode.factory)
    Node.app.register_node('sampler_engine', SamplerEngineNode.factory)
    Node.app.register_node('multi_voice_sampler', SamplerMultiVoiceNode.factory)
    Node.app.register_node('polyphonic_sampler', PolyphonicSamplerNode.factory)


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

    def stop_engine(self):
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.stop()

    def restart_engine(self):
        if SamplerEngineNode.engine:
            if not SamplerEngineNode.engine.stream or not SamplerEngineNode.engine.stream.active:
                SamplerEngineNode.engine.start()

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
        self.voice_idx_input = self.add_input('voice index', widget_type='drag_int', default_value=0, min=0, max=127)
        self.path_input = self.add_input('path', callback=self.load_file)
        self.load_btn = self.add_input('load', widget_type='button', callback=self.request_load_file)
        self.file_label = self.add_label('')
        self.length_label = self.add_label('')
        self.volume_input = self.add_input('volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0,
                                           callback=self.update_params)
        self.pitch_input = self.add_input('pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0,
                                          callback=self.update_params)
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=False,
                                         callback=self.update_loop_params)
        self.loop_start_input = self.add_input('loop start', widget_type='drag_int', default_value=0, min=0,
                                               callback=self.update_loop_params)
        self.loop_end_input = self.add_input('loop end', widget_type='drag_int', default_value=-1,
                                             callback=self.update_loop_params)
        self.crossfade_input = self.add_input('crossfade frames', widget_type='drag_int', default_value=0, min=0,
                                              callback=self.update_loop_params)

    def request_load_file(self):
        # Assuming LoadDialog exists in the environment
        try:
            from dpg_system.element_loader import LoadDialog
            LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])
        except ImportError:
            pass

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        # New: Use Sample factory/init to load
        self.sample = Sample(filepath)

        # Update label
        self.file_label.set(os.path.basename(filepath))
        self.length_label.set(f"Length: {len(self.sample.data)} samples")

    def load_file(self):
        val = self.path_input()
        if val == "bang":
            self.request_load_file()
        elif val:
            self.load_file_with_path(val)

    def update_params(self):
        if SamplerEngineNode.engine:
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            idx = int(self.voice_idx_input())
            SamplerEngineNode.engine.set_voice_volume(idx, vol)
            SamplerEngineNode.engine.set_voice_pitch(idx, pitch)

    def update_loop_params(self):
        if self.sample and SamplerEngineNode.engine:
            l_start = int(self.loop_start_input())
            l_end = int(self.loop_end_input())
            xf = int(self.crossfade_input())
            idx = int(self.voice_idx_input())
            if l_end < 0:
                l_end = len(self.sample.data)
            SamplerEngineNode.engine.set_voice_loop_window(idx, l_start, l_end, xf)

    def toggle_play(self):
        is_playing = bool(self.play_toggle())
        if is_playing:
            self.play()
        else:
            self.stop_voice()

    def play(self):
        if self.sample and SamplerEngineNode.engine:
            idx = int(self.voice_idx_input())
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            loop = bool(self.loop_input())
            l_start = int(self.loop_start_input())
            l_end = int(self.loop_end_input())
            xf = int(self.crossfade_input())
            if l_end < 0:
                l_end = len(self.sample.data)
            self.sample.default_volume = vol
            self.sample.default_pitch = pitch
            self.sample.loop = loop
            self.sample.loop_start = l_start
            self.sample.loop_end = l_end
            self.sample.crossfade_frames = xf
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
                "crossfade": 0,
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

        self.path_input = self.add_input('path', callback=self.load_file)
        self.load_btn = self.add_input('load', widget_type='button', callback=self.request_load_file)

        self.file_label = self.add_label('')
        self.length_label = self.add_label('')

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
        self.loop_input = self.add_input('loop', widget_type='checkbox', default_value=False,
                                         callback=self.update_loop_params)
        self.loop_start_input = self.add_input('loop start', widget_type='drag_int', default_value=0, min=0,
                                               callback=self.update_loop_params)
        self.loop_end_input = self.add_input('loop end', widget_type='drag_int', default_value=-1,
                                             callback=self.update_loop_params)
        self.crossfade_input = self.add_input('crossfade frames', widget_type='drag_int', default_value=0, min=0,
                                              callback=self.update_loop_params)
        self.pos_output = self.add_output('position')
        if not hasattr(self, 'message_handlers'):
            self.message_handlers = {}
        self.message_handlers['trigger'] = self.trigger_from_message
        self.message_handlers['stop'] = self.stop_from_message

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
        self.path_input.set(path)
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
                self.length_label.set(f"Length: {len(sample.data)} samples")
                self.ignore_updates = True
                self.path_input.set(filepath)
                self.loop_start_input.set(0)
                self.loop_end_input.set(-1)
                self.ignore_updates = False
            if state["playing"]:
                self.trigger_voice(target_idx)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")

    def load_file(self, *args):
        val = self.path_input()
        if val == "bang":
            self.request_load_file()
        elif val:
            self.load_file_with_path(val)

    def on_voice_idx_change(self, *args):
        try:
            val = self.voice_idx_input()
            idx = int(val)
            if 0 <= idx < 128:
                self.current_idx = idx
                self.sync_ui_to_state()
        except Exception as e:
            print(f"Error changing voice index: {e}")

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
        self.path_input.set(state["path"])
        if state["path"]:
            self.file_label.set(os.path.basename(state["path"]))
            if state["sample"]:
                self.length_label.set(f"Length: {len(state['sample'].data)} samples")
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
        self.loop_end_input.set(state["loop_end"])
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
        state["crossfade"] = int(self.crossfade_input())
        effective_end = state["loop_end"]
        if effective_end < 0 and state["sample"]:
            effective_end = len(state["sample"].data)
        if SamplerEngineNode.engine and state["sample"]:
            SamplerEngineNode.engine.set_voice_loop_window(self.current_idx, state["loop_start"], effective_end,
                                                           state["crossfade"])

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
                s.crossfade_frames = state["crossfade"]
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
            "crossfade": 0
        }
        if base_params:
            self.params.update(base_params)
            
        # Apply to sample
        self.sample.loop = self.params.get("loop", False)
        self.sample.loop_start = int(self.params.get("loop_start", 0))
        le = int(self.params.get("loop_end", -1))
        if le < 0: le = len(self.sample.data)
        self.sample.loop_end = le
        self.sample.crossfade_frames = int(self.params.get("crossfade", 0))


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
        self.load_input = self.add_input('load_sound', callback=self.on_load)
        
        # Helper props
        self.voice_range_start = self.add_property('start_voice', widget_type='input_int', default_value=64, callback=self.on_config_change)
        self.voice_range_count = self.add_property('voice_count', widget_type='input_int', default_value=16, callback=self.on_config_change)
        
        # Inspection UI
        self.inspect_label = self.add_label('Inspection')
        self.length_label = self.add_label('')
        self.inspect_id_input = self.add_input('sound_id', widget_type='input_int', default_value=0, callback=self.on_inspect_change)
        
        self.edit_vol = self.add_input('volume', widget_type='drag_float', default_value=1.0, min=0.0, max=2.0, callback=self.on_param_edit)
        self.edit_pitch = self.add_input('pitch', widget_type='drag_float', default_value=1.0, min=0.01, max=4.0, callback=self.on_param_edit)
        self.edit_att = self.add_input('attack', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_dec = self.add_input('decay', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_att = self.add_input('attack', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_dec = self.add_input('decay', widget_type='drag_float', default_value=0.0, min=0.0, max=10.0, callback=self.on_param_edit)
        self.edit_curve = self.add_input('curve', widget_type='drag_float', default_value=1.0, min=0.1, max=20.0, callback=self.on_param_edit)
        
        self.edit_loop = self.add_input('loop', widget_type='checkbox', default_value=False, callback=self.on_param_edit)
        self.edit_loop_start = self.add_input('loop_start', widget_type='drag_int', default_value=0, min=0, callback=self.on_param_edit)
        self.edit_loop_end = self.add_input('loop_end', widget_type='drag_int', default_value=-1, callback=self.on_param_edit)
        self.edit_cross = self.add_input('crossfade', widget_type='drag_int', default_value=0, min=0, callback=self.on_param_edit)
        
        # Default to inspecting ID 0 (int) so immediate edits work
        self.current_inspect_id = 0
        self.ignore_updates = False
        
        # Output
        self.active_out = self.add_output('active_voices')

    def on_config_change(self):
        self.start_voice_idx = self.voice_range_start()
        self.voice_count = self.voice_range_count()

    def on_load(self):
        data = self.load_input()
        # Format: id, path, [vol, pitch, att, dec, curve]
        if not data or len(data) < 2: return
        
        try:
            sid = int(data[0])
        except:
            return
            
        path = data[1]
        
        params = {}
        if len(data) > 2: params["volume"] = float(data[2])
        if len(data) > 3: params["pitch"] = float(data[3])
        if len(data) > 4: params["attack"] = float(data[4])
        if len(data) > 5: params["decay"] = float(data[5])
        if len(data) > 6: params["decay_curve"] = float(data[6])

        if os.path.exists(path):
            try:
                snd = Sound(path, params)
                self.sounds[sid] = snd
                # If we are inspecting this ID, update UI
                if self.current_inspect_id == sid:
                    self.sync_ui_to_sound(sid)
                # print(f"Loaded sound {sid} from {path}")
            except Exception as e:
                print(f"Error loading sound {path}: {e}")

    def on_inspect_change(self):
        val = self.inspect_id_input()
        try:
            self.current_inspect_id = int(val)
        except:
            pass # Keep previous if invalid
            
        self.sync_ui_to_sound(self.current_inspect_id)
        
    def sync_ui_to_sound(self, sid):
        if sid not in self.sounds:
            # Reset UI to defaults if sound not loaded
            self.ignore_updates = True
            self.edit_vol.set(1.0)
            self.edit_pitch.set(1.0)
            self.edit_att.set(0.0)
            self.edit_dec.set(0.0)
            self.edit_curve.set(1.0)
            self.edit_loop.set(False)
            self.edit_loop_start.set(0)
            self.edit_loop_end.set(-1)
            self.edit_cross.set(0)
            self.length_label.set("")
            self.ignore_updates = False
            return
        
        self.ignore_updates = True
        snd = self.sounds[sid]
        self.length_label.set(f"Length: {len(snd.sample.data)} samples")
        self.edit_vol.set(snd.params.get("volume", 1.0))
        self.edit_pitch.set(snd.params.get("pitch", 1.0))
        self.edit_att.set(snd.params.get("attack", 0.0))
        self.edit_dec.set(snd.params.get("decay", 0.0))
        self.edit_curve.set(snd.params.get("decay_curve", 1.0))
        self.edit_loop.set(snd.params.get("loop", False))
        self.edit_loop_start.set(snd.params.get("loop_start", 0))
        self.edit_loop_end.set(snd.params.get("loop_end", -1))
        self.edit_cross.set(snd.params.get("crossfade", 0))
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
        cross = int(self.edit_cross())
        
        snd.params["loop"] = loop
        snd.params["loop_start"] = l_start
        snd.params["loop_end"] = l_end
        snd.params["crossfade"] = cross
        
        # Update underlying sample object (so new triggers pick it up)
        snd.sample.loop = loop
        snd.sample.loop_start = l_start
        if l_end < 0:
             snd.sample.loop_end = len(snd.sample.data)
        else:
             snd.sample.loop_end = l_end
        snd.sample.crossfade_frames = cross
        
        # Update active voices
        if SamplerEngineNode.engine:
            for alloc in self.active_allocations:
                if alloc["sid"] == sid:
                    idx = alloc["idx"]
                    # Send update commands
                    SamplerEngineNode.engine.set_voice_loop_window(idx, loop, l_start, snd.sample.loop_end, cross)
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
                 
                 # Record allocation (for note off)
                 # We track by (sid, pitch_mult) to allow releasing specific note?
                 # Or just generic handle?
                 self.active_allocations.append({
                     "idx": idx,
                     "sid": sid,
                     "pitch": pitch_mult,
                     "time": time.time()
                 })
                 self.active_out.send(len(self.active_allocations))
                 
                 # Add frame task for cleanup
                 self.add_frame_task()

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
                        print('continue', sid, fade)
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
                    print('trigger', sid, fade)
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
                        
                        SamplerEngineNode.engine.set_voice_envelope(idx, 0.0, voice_params["decay"], voice_params["decay_curve"])
                        SamplerEngineNode.engine.play_voice(idx, snd.sample, eff_vol, eff_pitch)
                        
                        alloc_record = {
                            "idx": idx,
                            "sid": sid,
                            "pitch": 1.0, 
                            "time": time.time(),
                            "fade": fade,
                            "fade_zero_since": None
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
            if alloc["sid"] == sid:
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
            
            if v.active:
                alive.append(alloc)
        
        count_changed = len(alive) != len(self.active_allocations)
        self.active_allocations = alive
        
        if count_changed:
             self.active_out.send(len(self.active_allocations))
             
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
