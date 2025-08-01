from dpg_system.torch_base_nodes import *
try:
    import sounddevice as sd
except Exception as e:
    print(e)
import torch
import torchaudio
from dpg_system.node import LoadDialog

import threading
import time
import os
import platform

default_audio_string_seg = ''
if platform.system() == 'Darwin':
    default_audio_string_seg = 'speakers'

pyaudio_available = True

PYAUDIO_FORMAT_MAP = {
    # pyaudio.paFloat32: 'float32', # Uncomment if you have pyaudio installed to use the constant
    8: 'float32',
    # pyaudio.paInt32: 'int32',
    4: 'int32',
    # pyaudio.paInt24: 'int24', # sounddevice does not directly support 24-bit in this way
    # pyaudio.paInt16: 'int16',
    2: 'int16',
    # pyaudio.paInt8: 'int8',
    1: 'int8',
    # pyaudio.paUInt8: 'uint8'
    17: 'uint8'
}

def register_torchaudio_nodes():
    global pyaudio_available
    if pyaudio_available:
        Node.app.register_node('t.audio_source', TorchAudioSourceNode.factory)
        Node.app.register_node('t.audio.play', TorchAudioPlaySoundNode.factory)
        Node.app.register_node('t.audio.file', TorchAudioFileNode.factory)
    Node.app.register_node('t.audio.kaldi_pitch', TorchAudioKaldiPitchNode.factory)
    Node.app.register_node('t.audio.gain', TorchAudioGainNode.factory)
    Node.app.register_node('t.audio.contrast', TorchAudioContrastNode.factory)
    Node.app.register_node('t.audio.loudness', TorchAudioLoudnessNode.factory)
    Node.app.register_node('t.audio.overdrive', TorchAudioOverdriveNode.factory)
    Node.app.register_node('audio_mixer', AudioMixerNode.factory)
    Node.app.register_node('t.audio.multiplayer', TorchAudioMultiPlayerNode.factory)

    # Node.app.register_node('ta.vad', TorchAudioVADNode.factory) - does not seem to do anything

class AudioSource:
    audio = None

    def __init__(self, channels=1, rate=16000, chunk=1024, data_format=8):
        self.samplerate = int(rate)
        self.channels = channels
        self.blocksize = chunk
        # Convert pyaudio format to a NumPy dtype string
        self.dtype = PYAUDIO_FORMAT_MAP.get(data_format, 'float32')
        self.stream = None
        self.callback_routine = None

        self.sources = {}
        all_devices = sd.query_devices()
        for i, device_info in enumerate(all_devices):
            if device_info['max_input_channels'] > 0:
                self.sources[i] = device_info['name']

        try:
            self.device_index = sd.default.device['input']
            if self.device_index == -1: # No default device found
                # Fallback to the first available input device
                self.device_index = next(iter(self.sources))
        except (ValueError, StopIteration):
            raise IOError("No input audio device found.")

        print(f"Initialized with default device: '{self.sources[self.device_index]}' (index {self.device_index})")

    def get_device_list(self):
        """Returns a list of available input device names."""
        return list(self.sources.values())

    def change_source(self, source_name):
        """Changes the input device by its name."""
        found = False
        for index, name in self.sources.items():
            if name == source_name:
                self.device_index = index
                print(f"Changed audio source to: '{name}' (index {self.device_index})")
                found = True
                break
        if not found:
            print(f"Warning: Source '{source_name}' not found. No change made.")

    def set_callback(self, routine):
        """
        Sets the user-defined callback routine.

        IMPORTANT: The routine will receive a NumPy array, not raw bytes.
        Signature: your_callback(numpy_array, frame_count, time_info, status_flags)
        """
        self.callback_routine = routine

    def _internal_callback(self, indata, frames, time, status):
        """
        The internal callback passed to sounddevice. It calls the user's routine.
        `indata` is already a NumPy array.
        """
        if self.callback_routine:
            # We call the user's routine with a compatible signature.
            # NOTE: The 'flag' from pyaudio is analogous to 'status' in sounddevice.
            self.callback_routine(indata, frames, time, status)

    def start(self):
        """Starts the audio stream."""
        if self.stream and self.stream.active:
            print("Stream is already active.")
            return True

        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                device=self.device_index,
                dtype=self.dtype,
                blocksize=self.blocksize,
                callback=self._internal_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.stream = None
            return False

    def stop(self):
        """Stops the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped.")
        return self.stream is None  # Return True if successfully stopped (is None)

    def get_device_info(self):
        """Gets the device info dictionary for the current device."""
        return sd.query_devices(self.device_index)

    def get_max_input_channels(self):
        """Gets max input channels for the current device."""
        return self.get_device_info()['max_input_channels']

    def get_default_sample_rate(self):
        """Gets the default sample rate for the current device."""
        return int(self.get_device_info()['default_samplerate'])

    def check_format(self, rate, channels, data_format):
        """Checks if a given format is supported by the current device."""
        dtype = PYAUDIO_FORMAT_MAP.get(data_format, 'float32')
        try:
            sd.check_input_settings(device=self.device_index, channels=channels, samplerate=rate, dtype=dtype)
            return True
        except sd.PortAudioError:
            return False

    # Add context manager support for safer stream handling
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()






class TorchAudioSourceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.streaming = False

        self.format_dict = {
            'float': 'float32',
            'int32': 'int32',
            'int16': 'int16',
            # 'int24' is not directly supported by sounddevice as a standard dtype
            # It would require manual byte processing, which we want to avoid.
            # So it's best to remove it from the UI list.
        }

        self.dtype = torch.float32
        self.source = AudioSource()  # This now creates the sounddevice version

        # Get the initial device name from the new source's properties
        self.source_name = self.source.sources[self.source.device_index]
        self.source.set_callback(self.audio_callback)

        self.stream_input = self.add_input('stream', widget_type='checkbox', default_value=self.streaming,
                                           callback=self.stream_on_off)
        self.source_choice = self.add_property('source', widget_type='combo', width=180, default_value=self.source_name, callback=self.source_params_changed)
        self.source_choice.widget.combo_items = self.source.get_device_list()
        self.channels = self.add_input('channels', widget_type='input_int', default_value=1, callback=self.source_params_changed)
        self.sample_rate = self.add_input('sample_rate', widget_type='drag_int', default_value=16000, callback=self.source_params_changed)
        self.format = self.add_property('sample format', widget_type='combo', default_value='float', callback=self.source_params_changed)
        self.format.widget.combo_items = ['float', 'int32', 'int16']
        self.chunk_size = self.add_input('chunk_size', widget_type='drag_int', default_value=1024, callback=self.source_params_changed)
        self.output = self.add_output('audio tensors')

    def source_params_changed(self):
        changed = False
        source_changed = False
        source_name_from_ui = self.source_choice()
        if source_name_from_ui != self.source_name:
            source_changed = True
            self.source_name = source_name_from_ui

        channels = self.channels()
        if channels != self.source.channels:
            changed = True
        sample_rate = self.sample_rate()
        if sample_rate != self.source.sample_rate:
            changed = True
        dtype_str = self.format_dict.get(self.format(), 'float32')
        if dtype_str != self.source.dtype:
            changed = True

        chunk = self.chunk_size()
        if chunk != self.source.chunk:
            changed = True

        streaming = self.streaming
        if changed or source_changed:
            self.source.change_source(self.source_name)
            maxChannels = self.source.get_max_input_channels()
            if channels > maxChannels:
                channels = maxChannels
                self.channels.set(channels)

            if self.source.check_format(sample_rate, channels, dtype_str):
                self.source.stop()
                self.source.channels = channels
                self.source.sample_rate = sample_rate
                self.source.format = dtype_str
                self.source.chunk = chunk
                if streaming:
                    self.streaming = self.source.start()
            else:
                sample_rate = self.source.get_default_sample_rate()
                if self.source.check_format(sample_rate, channels, dtype_str):
                    self.source.stop()
                    self.source.channels = channels
                    self.source.sample_rate = sample_rate
                    self.sample_rate.set(self.source.rate)
                    self.source.format = dtype_str
                    self.source.chunk = chunk
                    if streaming:
                        self.streaming = self.source.start()
                else:
                    print('Audio Source format invalid: channels =', channels, 'rate =', sample_rate, 'format =', dtype_str)

    def audio_callback(self, indata, frame_count, time_info, flag):
        torch_ready_numpy = indata.T
        torch_audio_data = torch.from_numpy(torch_ready_numpy)
        # numpy_audio_data = np.frombuffer(buffer=indata, dtype=np.float32).copy()
        # chans = self.source.channels
        # if chans > 1:
        #     numpy_audio_data = numpy_audio_data.reshape(-1, chans).swapaxes(0, 1)
        # torch_audio_data = torch.from_numpy(numpy_audio_data)
        self.output.send(torch_audio_data)

    def stream_on_off(self):
        if self.stream_input():
            if not self.streaming:
                self.streaming = self.source.start()
        else:
            if self.streaming:
                is_stopped = self.source.stop()
                if is_stopped:
                    self.streaming = False

    def custom_cleanup(self):
        if self.streaming:
            self.source.stop()
            # self.source.audio.terminate()
            self.source = None


class TorchAudioKaldiPitchNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioKaldiPitchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.rate = self.add_property('sample_rate', widget_type='drag_int', default_value=16000)
        self.pitch_output = self.add_output('pitch out')
        self.nccf_output = self.add_output('nccf out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            pitch_feature = torchaudio.functional.compute_kaldi_pitch(data, self.rate())
            nccf, pitch = pitch_feature[..., 0], pitch_feature[..., 1]
            self.nccf_output.send(nccf)
            self.pitch_output.send(pitch)


class TorchAudioVADNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioVADNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.rate = self.add_property('sample_rate', widget_type='drag_int', default_value=16000)
        self.trigger_level = self.add_input('trigger_level', widget_type='drag_float', default_value=7.0)
        self.noise_reduction = self.add_input('noise_reduction', widget_type='drag_float', default_value=1.35)
        self.vad_output = self.add_output('active voice out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            print('trigger', self.trigger_level())
            active_audio = torchaudio.functional.vad(data, self.rate(), trigger_level=self.trigger_level(), noise_reduction_amount=self.noise_reduction())
            self.vad_output.send(active_audio)


class TorchAudioGainNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioGainNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.gain = self.add_input('gain in dB', widget_type='drag_float', default_value=1.0)
        self.output = self.add_output('audio out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            active_audio = torchaudio.functional.gain(data, self.gain())
            self.output.send(active_audio)


class TorchAudioContrastNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioContrastNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.contrast = self.add_input('contrast', widget_type='drag_float', default_value=75.0)
        self.output = self.add_output('audio out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            active_audio = torchaudio.functional.contrast(data, self.contrast())
            self.output.send(active_audio)


# loudness needs minimum of 6400 chunk size????
class TorchAudioLoudnessNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioLoudnessNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.rate = self.add_property('sample_rate', widget_type='drag_int', default_value=16000)
        self.loudness_output = self.add_output('loudness out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            if len(data.shape) < 2:
                data.unsqueeze(dim=0)
            if data.shape[-1] < 6400:
                print(self.label, 'too few samples to calculate loudness (min 6400)')
            else:
                active_audio = torchaudio.functional.loudness(data, self.rate())
                self.loudness_output.send(active_audio)


class TorchAudioOverdriveNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioOverdriveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.gain = self.add_input('gain', widget_type='drag_float', default_value=20.0)
        self.colour = self.add_input('colour', widget_type='drag_float', default_value=20.0)
        self.output = self.add_output('audio out')

    def execute(self):
        data = self.input_to_tensor()
        if data is not None:
            overdriven_audio = torchaudio.functional.overdrive(data, self.gain(), self.colour())
            self.output.send(overdriven_audio)


class TorchAudioFileNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioFileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.waveform = None
        self.sample_rate = None
        self.trigger_input = self.add_input('trigger', triggers_execution=True, trigger_button=True)
        self.path_input = self.add_input('path in', callback=self.load_file)
        self.load_file = self.add_input('load file', widget_type='button', callback=self.request_load_file)
        self.file_name = self.add_label('')

        self.output = self.add_output('audio data out')
        self.sample_rate_out = self.add_output('sample_rate')

    def request_load_file(self):
        loader = LoadDialog(self, self.load_file_callback, extensions=['.aif', '.wav', '.mp3'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found at: {filepath}")
            return None, None

            # torchaudio.load returns a tuple of (waveform, sample_rate)
        self.waveform, self.sample_rate = torchaudio.load(filepath)
        self.sample_rate_out.send(int(self.sample_rate))
        file_name = filepath.split('/')[-1]
        self.file_name.set(file_name)

    def load_file(self):
        filepath = any_to_string(self.path_input())
        self.load_file_with_path(filepath)

    def execute(self):
        self.sample_rate_out.send(int(self.sample_rate))
        self.output.send(self.waveform)


class TorchAudioPlaySoundNode(TorchNode):
    mixer = None
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioPlaySoundNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if AudioMixerNode.audio_mixer is None:
            AudioMixerNode.audio_mixer = AudioMixer()

        self.trigger_input = self.add_input('trigger', widget_type='button', callback=self.play)
        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.path_input = self.add_input('path in', callback=self.load_file)
        self.load_file = self.add_input('load file', widget_type='button', callback=self.request_load_file)
        self.file_name = self.add_label('')
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop)
        self.last_sound_id = None

    def request_load_file(self):
        loader = LoadDialog(self, self.load_file_callback, extensions=['.aif', '.wav', '.mp3'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found at: {filepath}")
            return None, None

            # torchaudio.load returns a tuple of (waveform, sample_rate)
        self.waveform, self.sample_rate = torchaudio.load(filepath)
        if not self.waveform.is_cpu:
            self.waveform = self.waveform.cpu()
        if self.waveform.ndim == 1:
            # If mono, add a channel dimension
            self.waveform = self.waveform.unsqueeze(1)
        elif self.waveform.shape[1] > self.waveform.shape[0]:
            self.waveform = self.waveform.T
        self.waveform_np = self.waveform.numpy()
        self.stored_waveform_np = self.waveform_np.copy()
        file_name = filepath.split('/')[-1]
        self.file_name.set(file_name)

    def load_file(self):
        filepath = self.path_input()
        self.load_file_with_path(filepath)

    def stop(self):
        if self.last_sound_id is not None:
            AudioMixerNode.audio_mixer.stop(self.last_sound_id)
            self.last_sound_id = None

    def play(self):
        if self.stored_waveform_np is not None:
            try:
                if AudioMixerNode.audio_mixer is not None:
                    self.last_sound_id = AudioMixerNode.audio_mixer.play(self.stored_waveform_np)
            except Exception as e:
                print(f"Error sending sound to mixer: {e}")
            return self.last_sound_id

    def execute(self):
        waveform = self.input_to_tensor()
        if not waveform.is_cpu:
            waveform = waveform.cpu()
        if waveform.ndim == 1:
            # If mono, add a channel dimension
            waveform = waveform.unsqueeze(1)
        elif waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.T
        self.waveform_np = waveform.numpy()
        self.stored_waveform_np = self.waveform_np.copy()
        try:
            if AudioMixerNode.audio_mixer is not None:
                self.last_sound_id = AudioMixerNode.audio_mixer.play(self.waveform_np)
        except Exception as e:
            print(f"Error sending sound to mixer: {e}")
        return self.last_sound_id


class TorchAudioMultiPlayerNode(TorchNode):
    mixer = None
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioMultiPlayerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if AudioMixerNode.audio_mixer is None:
            AudioMixerNode.audio_mixer = AudioMixer()

        self.trigger_input = self.add_input('trigger', widget_type='button', callback=self.play)
        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.path_input = self.add_input('path in', callback=self.load_file)
        self.load_file = self.add_input('load file', widget_type='button', callback=self.request_load_file)
        self.file_name = self.add_label('')
        self.remove = self.add_input('remove wave', callback=self.remove_wave)
        self.clear_button = self.add_input('clear waves', callback=self.clear_waves)
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop)
        self.last_sound_id = None
        self.waves = {}
        self.player_ids = {}
        self.last_loaded = None

    def request_load_file(self):
        loader = LoadDialog(self, self.load_file_callback, extensions=['.aif', '.wav', '.mp3'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found at: {filepath}")
            return None, None

            # torchaudio.load returns a tuple of (waveform, sample_rate)
        self.waveform, self.sample_rate = torchaudio.load(filepath)
        if not self.waveform.is_cpu:
            self.waveform = self.waveform.cpu()
        if self.waveform.ndim == 1:
            # If mono, add a channel dimension
            self.waveform = self.waveform.unsqueeze(1)
        elif self.waveform.shape[1] > self.waveform.shape[0]:
            self.waveform = self.waveform.T

        file_name = filepath.split('/')[-1]
        self.file_name.set(file_name)
        sample_name = file_name.split('.')[0]
        self.waves[sample_name] = self.waveform.numpy()
        self.last_loaded = sample_name

    def load_file(self):
        filepath = self.path_input()
        if type(filepath) == list:
            for path in filepath:
                self.load_file_with_path(path)
        else:
            self.load_file_with_path(filepath)

    def remove_wave(self):
        name = self.remove()
        if name in self.waves:
            del self.waves[name]
        if name in self.player_ids:
            del self.player_ids[name]

    def clear_waves(self):
        self.waves.clear()

    def stop(self):
        if self.last_sound_id is not None:
            AudioMixerNode.audio_mixer.stop(self.last_sound_id)
            self.last_sound_id = None

    def play(self):
        trigger = self.trigger_input()
        name = self.last_loaded
        if type(trigger) == str and trigger in self.waves:
            name = trigger
        try:
            if AudioMixerNode.audio_mixer is not None:
                self.last_sound_id = AudioMixerNode.audio_mixer.play(self.waves[name])
                self.player_ids[name] = self.last_sound_id
        except Exception as e:
            print(f"Error sending sound to mixer: {e}")
        return self.last_sound_id

    def execute(self):
        waveform = self.input_to_tensor()
        if not waveform.is_cpu:
            waveform = waveform.cpu()
        if waveform.ndim == 1:
            # If mono, add a channel dimension
            waveform = waveform.unsqueeze(1)
        elif waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.T
        self.waveform_np = waveform.numpy()
        self.stored_waveform_np = self.waveform_np.copy()
        try:
            if AudioMixerNode.audio_mixer is not None:
                self.last_sound_id = AudioMixerNode.audio_mixer.play(self.waveform_np)
        except Exception as e:
            print(f"Error sending sound to mixer: {e}")
        return self.last_sound_id


class AudioMixer:
    """
    Manages mixing and playback. Now supports re-initialization with new parameters.
    """

    def __init__(self, samplerate=44100, channels=2, blocksize=1024, device=None):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize

        if device is None:
            defaults = sd.query_hostapis()[0]
            default_output_device = defaults['default_output_device']

            all_devices = sd.query_devices()
            device = all_devices[default_output_device]
            print(device)
            self.device = device['name']
        else:
            self.device = device

        self.active_sounds = {}
        self._lock = threading.Lock()
        self.stream = None
        self.start_stream()  # Start the stream on initialization

    def start_stream(self):
        """Starts or restarts the audio output stream."""
        if self.stream and self.stream.active:
            print("Stream is already active.")
            return

        # Stop any existing stream before starting a new one
        if self.stream:
            self.stream.stop()
            self.stream.close()

        # print(f"Starting AudioMixer on device {self.device} (Sample Rate: {self.samplerate} Hz)...")
        try:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                device=self.device,
                blocksize=self.blocksize,
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stream = None

    def _audio_callback(self, outdata, frames, time, status):
        # ... (This function remains exactly the same as the previous version) ...
        if status: print(f"Stream status alert: {status}")
        outdata.fill(0)
        sounds_to_remove = []
        with self._lock:
            for sound_id, sound_data in list(self.active_sounds.items()):
                chunk = sound_data['data'][sound_data['pos']: sound_data['pos'] + frames]
                chunk_len = len(chunk)
                if chunk_len > 0 and outdata[:chunk_len].shape == chunk.shape:
                    outdata[:chunk_len] += chunk
                sound_data['pos'] += chunk_len
                if sound_data['pos'] >= len(sound_data['data']):
                    sounds_to_remove.append(sound_id)
            for sound_id in sounds_to_remove:
                del self.active_sounds[sound_id]
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def play(self, waveform_np: np.ndarray):
        # ... (This function remains exactly the same) ...
        if self.stream is None or not self.stream.active:
            self.start_stream()
        # ... (rest of the play logic)
        if not isinstance(waveform_np, np.ndarray): raise TypeError("Input must be a NumPy array.")
        if waveform_np.ndim != 2 or waveform_np.shape[1] != self.channels:
            # Simple resampling/rechanneling could be added here if needed
            print(f"Warning: Waveform shape mismatch. Expected (samples, {self.channels}), got {waveform_np.shape}.")
            return
        sound_id = dpg.generate_uuid()
        sound_data = {'data': waveform_np, 'pos': 0}
        with self._lock:
            self.active_sounds[sound_id] = sound_data
        return sound_id

    def stop(self, sound_id=None):
        """Stops the playback stream and clears all sounds."""
        if self.stream:
            if sound_id is not None:
                if sound_id in self.active_sounds:
                    del self.active_sounds[sound_id]
                    return
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self._lock:
            self.active_sounds.clear()
        # print("AudioMixer stopped.")


class AudioMixerNode(Node):
    """
    A node to control the global audio mixer settings.
    There should only be one of these in a graph.
    """
    audio_mixer = None

    @staticmethod
    def factory(name, data, args=None):
        audio_mixer = None
        node = AudioMixerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.output_devices = {}
        try:
            all_devices = sd.query_devices()
            for i, device_info in enumerate(all_devices):
                if device_info['max_output_channels'] > 0:
                    self.output_devices[device_info['name']] = i
        except Exception as e:
            print(f"Could not query audio devices: {e}")

        # --- Discover output devices ---
        if AudioMixerNode.audio_mixer is None:
            # print("No audio mixer found.")
            # Determine a safe default device
            default_device_name = "No output devices found"
            if self.output_devices:
                try:
                    default_device_index = sd.default.device['output']
                    default_device_name = sd.query_devices(default_device_index)['name']
                except (ValueError, KeyError, sd.PortAudioError):
                    default_device_name = next(iter(self.output_devices))
        else:
            default_device_name = AudioMixerNode.audio_mixer.device

        # --- UI elements to control mixer parameters ---
        self.device_choice = self.add_property('output device', widget_type='combo', width=150, default_value=default_device_name,
                                               callback=self.params_changed)
        self.device_choice.widget.combo_items = list(self.output_devices.keys())

        self.sample_rate_prop = self.add_property('sample rate', widget_type='combo', width=150, default_value='44100',
                                                  callback=self.params_changed)
        self.sample_rate_prop.widget.combo_items = ['16000', '32000', '44100', '48000']

        self.channels_prop = self.add_property('channels', widget_type='combo', width=150, default_value='2 (Stereo)',
                                               callback=self.params_changed)
        self.channels_prop.widget.combo_items = ['1 (Mono)', '2 (Stereo)']

        self.stop_input = self.add_input('stop all', widget_type='button', callback=self.stop)

    def custom_create(self, from_file):
        # Initialize the mixer for the first time
        self.params_changed()

    def stop(self):
        if self.audio_mixer is not None:
            self.audio_mixer.stop()

    def params_changed(self):
        """
        Called whenever a UI parameter is changed. This function will stop the
        old mixer and create a new one with the updated settings.
        """

        # --- Gather settings from UI ---
        selected_device_name = self.device_choice()
        if not selected_device_name or selected_device_name not in self.output_devices:
            print("Invalid or no output device selected. Mixer not started.")
            if AudioMixerNode.audio_mixer is not None:
                AudioMixerNode.audio_mixer.stop()
                AudioMixerNode.audio_mixer = None
            return

        device_id = self.output_devices.get(selected_device_name)
        samplerate = int(self.sample_rate_prop())
        channels = 1 if '1' in self.channels_prop() else 2

        # --- Check if mixer needs to be updated ---
        needs_update = False
        if AudioMixerNode.audio_mixer is None:
            needs_update = True
        elif (AudioMixerNode.audio_mixer.device != device_id or
              AudioMixerNode.audio_mixer.samplerate != samplerate or
              AudioMixerNode.audio_mixer.channels != channels):
            needs_update = True

        if needs_update:
            # print("Mixer parameters changed. Re-initializing...")

            # Stop the old mixer if it exists
            if AudioMixerNode.audio_mixer:
                AudioMixerNode.audio_mixer.stop()

            # Create and assign the new global mixer instance
            AudioMixerNode.audio_mixer = AudioMixer(
                samplerate=samplerate,
                channels=channels,
                device=device_id
            )
        # else:
        #     print("Mixer parameters unchanged.")
        # print('mixer', AudioMixerNode.audio_mixer)

    def custom_cleanup(self):
        """
        When this node is deleted, it should stop the global mixer.
        """
        if AudioMixerNode.audio_mixer:
            AudioMixerNode.audio_mixer.stop()
            AudioMixerNode.audio_mixer = None
        print("AudioMixerNode cleaned up and global mixer stopped.")