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
    Node.app.register_node('sampler_voice', SamplerVoiceNode.factory)
    Node.app.register_node('sampler_engine', SamplerEngineNode.factory)
    Node.app.register_node('multi_voice_sampler', SamplerMultiVoiceNode.factory)

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
            raise IOError('No input audio device found.')

        print(f"Initialized with default device: '{self.sources[self.device_index]}' (index {self.device_index})")

    def get_device_list(self):
        '''Returns a list of available input device names.'''
        return list(self.sources.values())

    def change_source(self, source_name):
        '''Changes the input device by its name.'''
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
        '''
        Sets the user-defined callback routine.

        IMPORTANT: The routine will receive a NumPy array, not raw bytes.
        Signature: your_callback(numpy_array, frame_count, time_info, status_flags)
        '''
        self.callback_routine = routine

    def _internal_callback(self, indata, frames, time, status):
        '''
        The internal callback passed to sounddevice. It calls the user's routine.
        `indata` is already a NumPy array.
        '''
        if self.callback_routine:
            # We call the user's routine with a compatible signature.
            # NOTE: The 'flag' from pyaudio is analogous to 'status' in sounddevice.
            self.callback_routine(indata, frames, time, status)

    def start(self):
        '''Starts the audio stream.'''
        if self.stream and self.stream.active:
            print('Stream is already active.')
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
            print(f'Error starting stream: {e}')
            self.stream = None
            return False

    def stop(self):
        '''Stops the audio stream.'''
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print('Audio stream stopped.')
        return self.stream is None  # Return True if successfully stopped (is None)

    def get_device_info(self):
        '''Gets the device info dictionary for the current device.'''
        return sd.query_devices(self.device_index)

    def get_max_input_channels(self):
        '''Gets max input channels for the current device.'''
        return self.get_device_info()['max_input_channels']

    def get_default_sample_rate(self):
        '''Gets the default sample rate for the current device.'''
        return int(self.get_device_info()['default_samplerate'])

    def check_format(self, rate, channels, data_format):
        '''Checks if a given format is supported by the current device.'''
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
            print(f'File not found at: {filepath}')
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
            print(f'File not found at: {filepath}')
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
                print(f'Error sending sound to mixer: {e}')
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
            print(f'Error sending sound to mixer: {e}')
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
            print(f'File not found at: {filepath}')
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
            print(f'Error sending sound to mixer: {e}')
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
            print(f'Error sending sound to mixer: {e}')
        return self.last_sound_id


class AudioMixer:
    '''
    Manages mixing and playback. Now supports re-initialization with new parameters.
    '''

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
        '''Starts or restarts the audio output stream.'''
        if self.stream and self.stream.active:
            print('Stream is already active.')
            return

        # Stop any existing stream before starting a new one
        if self.stream:
            self.stream.stop()
            self.stream.close()

        # print(f'Starting AudioMixer on device {self.device} (Sample Rate: {self.samplerate} Hz)...')
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
            print(f'Error starting audio stream: {e}')
            self.stream = None

    def _audio_callback(self, outdata, frames, time, status):
        # ... (This function remains exactly the same as the previous version) ...
        if status: print(f'Stream status alert: {status}')
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
        if not isinstance(waveform_np, np.ndarray): raise TypeError('Input must be a NumPy array.')
        if waveform_np.ndim != 2 or waveform_np.shape[1] != self.channels:
            # Simple resampling/rechanneling could be added here if needed
            print(f'Warning: Waveform shape mismatch. Expected (samples, {self.channels}), got {waveform_np.shape}.')
            return
        sound_id = dpg.generate_uuid()
        sound_data = {'data': waveform_np, 'pos': 0}
        with self._lock:
            self.active_sounds[sound_id] = sound_data
        return sound_id

    def stop(self, sound_id=None):
        '''Stops the playback stream and clears all sounds.'''
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
        # print('AudioMixer stopped.')


class AudioMixerNode(Node):
    '''
    A node to control the global audio mixer settings.
    There should only be one of these in a graph.
    '''
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
            print(f'Could not query audio devices: {e}')

        # --- Discover output devices ---
        if AudioMixerNode.audio_mixer is None:
            # print('No audio mixer found.')
            # Determine a safe default device
            default_device_name = 'No output devices found'
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
        '''
        Called whenever a UI parameter is changed. This function will stop the
        old mixer and create a new one with the updated settings.
        '''

        # --- Gather settings from UI ---
        selected_device_name = self.device_choice()
        if not selected_device_name or selected_device_name not in self.output_devices:
            print('Invalid or no output device selected. Mixer not started.')
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
            # print('Mixer parameters changed. Re-initializing...')

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
        #     print('Mixer parameters unchanged.')
        # print('mixer', AudioMixerNode.audio_mixer)

    def custom_cleanup(self):
        '''
        When this node is deleted, it should stop the global mixer.
        '''
        if AudioMixerNode.audio_mixer:
            AudioMixerNode.audio_mixer.stop()
            AudioMixerNode.audio_mixer = None
        print('AudioMixerNode cleaned up and global mixer stopped.')



import soundfile as sf


class Sample:
    def __init__(self, data, volume=1.0, loop=False, loop_start=0, loop_end=-1, crossfade_frames=0, pitch=1.0):
        # Allow passing path or array
        if isinstance(data, str):
            # Load from file (placeholder if we had file IO, but user only used arrays so far in demos)
            # implementation for future if needed:
            # self.data, _ = sf.read(data, dtype='float32')
            pass
        else:
            self.data = data

        self.default_volume = volume
        self.loop = loop
        self.loop_start = loop_start

        # If loop_end is -1 or greater than length, set to length
        if loop_end < 0 or loop_end > len(self.data):
            self.loop_end = len(self.data)
        else:
            self.loop_end = loop_end

        self.crossfade_frames = min(crossfade_frames, self.loop_end - self.loop_start)
        self.default_pitch = pitch


class Voice:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.active = False
        self.sample = None
        self.position = 0.0
        self.looping = False
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.loop_start = 0
        self.loop_end = 0
        self.crossfade_frames = 0
        self.pitch = 1.0
        self.zero_vol_frames = 0

    def trigger(self, sample, volume=None, pitch=None):
        self.sample = sample
        # Use simple alias for speed access during process?
        # self.sample_data = sample.data

        self.looping = sample.loop

        # Allow overrides if provided, else defaults
        start_vol = volume if volume is not None else sample.default_volume
        self.target_volume = start_vol

        # If voice was inactive, loop starts from 0 volume to avoid pops unless instant attack is desired.
        # For a sampler, usually we want instant attack, but let's set current to target to snap.
        self.current_volume = start_vol
        self.active = True
        self.zero_vol_frames = 0

        self.pitch = pitch if pitch is not None else sample.default_pitch

        self.loop_start = sample.loop_start
        self.loop_end = sample.loop_end
        self.crossfade_frames = sample.crossfade_frames

        self.position = float(self.loop_start) if self.looping else 0.0
        self.position = 0.0

    def set_volume(self, volume):
        self.target_volume = volume

    def set_pitch(self, pitch):
        self.pitch = pitch

    def set_loop_window(self, loop_start, loop_end, crossfade_frames):
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.crossfade_frames = crossfade_frames

    def process(self, frames, channels):
        if not self.active or self.sample is None:
            return np.zeros((frames, channels), dtype=np.float32)
        # Auto-stop on silence
        # Threshold for silence
        if self.target_volume <= 0.0001 and self.current_volume <= 0.0001:
            self.zero_vol_frames += frames
            if self.zero_vol_frames > self.sample_rate:  # 1 second
                self.active = False
                return np.zeros((frames, channels), dtype=np.float32)
        else:
            self.zero_vol_frames = 0
        output = np.zeros((frames, channels), dtype=np.float32)
        frames_processed = 0

        # Local refs for speed/clarity
        sample_data = self.sample.data

        # Calculate volume ramp for this block
        if self.current_volume != self.target_volume:
            vol_curve = np.linspace(self.current_volume, self.target_volume, frames)
            self.current_volume = self.target_volume
        else:
            vol_curve = self.current_volume
        # Ensure pitch avoids divide by zero or negative infinite loops
        # We only support positive pitch for now to simplify logic
        local_pitch = max(0.001, self.pitch)
        while frames_processed < frames:
            remaining_frames = frames - frames_processed
            # If looping, we are bound by loop_end, else sample_len
            effective_end = self.loop_end if self.looping else len(sample_data)

            # How many frames can we generate before hitting effective_end?
            # dist = (end - current_pos) / pitch
            dist_samples = effective_end - self.position

            # If we are already past end (sanity), handle wrap/stop
            if dist_samples <= 0:
                if self.looping:
                    self.position = float(self.loop_start) + (self.position - self.loop_end)
                    # Sanity check if loop is tiny or position is way off
                    if self.position >= effective_end: self.position = float(self.loop_start)
                    dist_samples = effective_end - self.position
                else:
                    self.active = False
                    break
            # Calculate max source frames we can consume
            # We want to fill 'remaining_frames' of output
            # Output frames consumed = frames * pitch
            # So frames_needed_source = remaining_frames * pitch

            # We are limited by source frames available (dist_samples)
            # max_output_frames_from_source = dist_samples / pitch

            chunk_len = int(dist_samples / local_pitch)
            # If chunk_len is 0 (less than 1 frame), we might still need to output something if dist > 0?
            # If dist_samples < pitch, we can't produce a full frame from this segment?
            # Actually we can produce fractional frame? No, output is discrete.
            # If we need 1 output frame, we consume `pitch` source frames.
            # If available < pitch, we need to wrap/stop in middle of frame computation?
            # That's too complex. Let's clamp chunk_len.
            # If chunk_len == 0 but we have space, it means we hit end very soon.
            # We should probably process 1 frame at least if possible, handling wrap manually?
            # Or just rely on loop wrapping to handle it.
            # Simplification: If chunk_len < 1, force wrap immediately?

            if chunk_len < 1:
                # Less than one output frame left in this segment.
                # Force wrap logic immediately.
                if self.looping:
                    self.position = float(self.loop_start) + (self.position - self.loop_end)
                    if self.position >= effective_end: self.position = float(self.loop_start)
                    continue
                else:
                    self.active = False
                    break

            chunk_len = min(remaining_frames, chunk_len)

            # Generate indices
            # indices = position + i * pitch
            out_indices = np.arange(chunk_len)
            sample_indices_float = self.position + out_indices * local_pitch

            # Linear Interpolation
            idx_int = sample_indices_float.astype(int)  # floor
            # Be careful with upper bound for interpolation (idx_int + 1)
            # We know idx_int < effective_end.  (because of chunk_len calculation)
            # But idx_int + 1 might be == effective_end.
            # If effective_end == len(sample_data), this is out of bounds.
            # We need safe access.

            alpha = sample_indices_float - idx_int
            if sample_data.ndim > 1: alpha = alpha[:, np.newaxis]

            # Safeguard access
            # Current sample
            # idx_int is safe.
            s0 = sample_data[idx_int]

            # Next sample
            idx_next = idx_int + 1
            # Handle boundary
            # If idx_next >= len(sample_data), what do?
            # If looping, strictly we should look at loop_start if idx_next == loop_end?
            # Yes, for perfect loop.

            # Vectorize boundary check
            # We need a safe gather
            # Create a mask for boundary
            mask_over = idx_next >= len(sample_data)
            if self.looping:
                # If idx_next hits loop_end (or sample limit), wrap to loop_start
                # NOTE: effective_end might be smaller than sample len.
                mask_loop = idx_next >= self.loop_end
                # If mask_loop is true, we should probably wrap.
                # But we calculated `chunk_len` so we wouldn't cross `effective_end`.
                # So `sample_indices_float` max is `position + (chunk_len-1)*pitch`
                # < `position + (dist/pitch - 1/pitch)*pitch` = `position + dist - 1` = `effective_end - 1`.
                # So `idx_int` max is `effective_end - 1` (or less).
                # So `idx_next` max is `effective_end`.
                # So yes, we might hit effective_end.

                # If we hit loop_end, we should wrap to loop_start?
                # Yes ideally.
                idx_next[mask_loop] = self.loop_start
            else:
                # If not looping, we just clamp or use 0?
                idx_next[mask_over] = len(sample_data) - 1  # Clamp to last

            # Safe access
            s1 = sample_data[idx_next]

            # Mix
            raw_chunk = s0 * (1.0 - alpha) + s1 * alpha

            # Apply Crossfade if looping
            # This logic needs to be adapted for interpolated reading.
            # Crossfade zone is defined in terms of sample indices: [loop_end - x, loop_end]
            # We have sample_indices_float.
            # Check if sample_indices_float >= loop_end - crossfade_frames
            if self.looping and self.crossfade_frames > 0:
                cf_start = self.loop_end - self.crossfade_frames

                # Create mask
                in_fade_mask = sample_indices_float >= cf_start

                if np.any(in_fade_mask):
                    indices_in_fade = sample_indices_float[in_fade_mask]

                    # fade_pos relative to start of fade
                    fade_pos = indices_in_fade - cf_start

                    # Alpha for crossfade
                    cf_alpha = fade_pos / float(self.crossfade_frames)
                    if sample_data.ndim > 1: cf_alpha = cf_alpha[:, np.newaxis]

                    # Source B: read from loop_start + fade_pos
                    # WE MUST INTERPOLATE SOURCE B TOO!
                    # pos_b = loop_start + fade_pos
                    pos_b = self.loop_start + fade_pos

                    idx_b_int = pos_b.astype(int)
                    alpha_b = pos_b - idx_b_int
                    if sample_data.ndim > 1: alpha_b = alpha_b[:, np.newaxis]

                    # Safe access for Source B
                    # idx_b_int starts at loop_start, goes up.
                    # Should stay within valid range usually.
                    sb0 = sample_data[idx_b_int]
                    # for sb1, check bounds
                    idx_b_next = idx_b_int + 1
                    # Wrap or clamp?
                    # usually fine unless loop_start + crossfade > loop_end ?? (User error)
                    # We clamped crossfade_frames earlier.

                    # Just in case:
                    idx_b_next = np.minimum(idx_b_next, len(sample_data) - 1)

                    sb1 = sample_data[idx_b_next]

                    source_b = sb0 * (1.0 - alpha_b) + sb1 * alpha_b

                    source_a = raw_chunk[in_fade_mask]

                    # Mix
                    mixed = (1.0 - cf_alpha) * source_a + cf_alpha * source_b

                    # Write back
                    # This is tricky with view/copy.
                    # 'raw_chunk' is a new array created by expression? Yes (s0*.. + s1*..).
                    # So we can assign slice.
                    # in_fade_mask is bool array of length chunk_len.
                    # Assigning to raw_chunk[in_fade_mask] works.
                    raw_chunk[in_fade_mask] = mixed
            # Apply volume
            # print(f"DEBUG: raw_chunk shape {raw_chunk.shape}, vol_curve type {type(vol_curve)}, vol_curve {vol_curve}")
            if isinstance(vol_curve, np.ndarray):
                vol_chunk = vol_curve[frames_processed:frames_processed + chunk_len]
                vol_chunk = vol_chunk[:, np.newaxis]  # Ensure (N, 1)
                output[frames_processed:frames_processed + chunk_len] = raw_chunk * vol_chunk
            else:
                # Ensure scalar
                output[frames_processed:frames_processed + chunk_len] = raw_chunk * float(vol_curve)

            # Advance position
            self.position += chunk_len * local_pitch
            frames_processed += chunk_len

            # Handle wrapping (loop is handled by `dist_samples` logic mostly, but if we land exactly on end)
            if self.looping and self.position >= self.loop_end:
                self.position = float(self.loop_start) + (self.position - self.loop_end)
            elif not self.looping and self.position >= effective_end:
                self.active = False
        return output


class SamplerEngine:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.voices = [Voice(sample_rate=sample_rate) for _ in range(128)]
        self.lock = threading.Lock()  # To protect voice state updates from main thread vs audio thread
        self.stream = None
        self.active = True  # Master active flag
        self.master_volume = 1.0

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=512  # Low latency
        )
        self.stream.start()
        print("Sampler Engine Started")

    def stop(self):
        if self.stream:
            # Graceful fade out
            print("Stopping engine with fade out...")
            for i in range(50):
                self.master_volume = 1.0 - ((i + 1) / 50.0)  # Reach 0.0
                sd.sleep(10)  # 10ms * 50 = 500ms fade

            # Ensure 0
            self.master_volume = 0.0
            sd.sleep(100)  # Wait for buffer to clear

            self.active = False
            self.stream.stop()
            self.stream.close()

    def play_voice(self, voice_index, sample, volume=None, pitch=None):
        if 0 <= voice_index < 128:
            # Check channels match
            # sample.data check?
            # We trust Sample object creation to handle data appropriately or checks inside Voice

            # Simple channel fixup if needed for 1-channel data on 2-channel engine
            if sample.data.ndim == 1 and self.channels == 2:
                # This modifies the sample object's data potentially shared!
                # Better to do this in Sample init.
                # But let's assume Sample is clean.
                pass
            with self.lock:
                self.voices[voice_index].trigger(sample, volume, pitch)

    def set_voice_pitch(self, voice_index, pitch):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_pitch(pitch)

    def set_voice_volume(self, voice_index, volume):
        if 0 <= voice_index < 128:
            # We don't strictly need lock for atomic float write but good practice if logic gets complex
            self.voices[voice_index].set_volume(volume)

    def set_voice_loop_window(self, voice_index, loop_start, loop_end, crossfade_frames):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_loop_window(loop_start, loop_end, crossfade_frames)

    def audio_callback(self, outdata, frames, time, status):
        # buffer clears automatically? SD docs say 'The output buffer is not initialized' so we must write everything.
        outdata.fill(0)

        # We need a temporary buffer to sum voices because outdata needs to be written to directly or copied
        # Summing directly into outdata is efficient

        # Note: processing 128 voices in python callback might be heavy.
        # Optimization: only process active voices.

        # To avoid holding the lock for the entire processing time (which causes dropouts),
        # we might want to just grab references. But Voice state changes in `process`.
        # Python GIL is the main bottleneck here.
        # Let's try simple locking first.

        # Let's just use the lock. If it glitches, we optimize.

        # OPTIMIZATION: Check active flag first without lock? strict boolean read is atomic.

        # A mix buffer
        mix = np.zeros((frames, self.channels), dtype=np.float32)

        active_count = 0

        # We assume `voices` list itself doesn't change structure, just contents of Voice objects.
        # But `process` changes `position` and `active`.
        # If PlayVoice is called during callback, it acquires lock.

        # We will try to NOT lock the whole loop if possible, or lock short.
        # ideally `process` is the heavy part.

        # Let's acquire lock, copy active voices or snapshots? No, too slow.
        # Real-time audio in Python is risky with locks in callback.
        # Let's just iterate. The race condition is `trigger` overwriting data while `process` reads it.
        # `trigger` is rare compared to callback.

        # Let's just use the lock. If it glitches, we optimize.

        # OPTIMIZATION: Check active flag first without lock? strict boolean read is atomic.

        for v in self.voices:
            if v.active:
                # process returns a new array, which causes allocation.
                # Ideally we pass a buffer to add to.
                # But let's stick to the plan: process() returns data.
                voice_out = v.process(frames, self.channels)
                mix += voice_out
                active_count += 1

        # Custom master volume/fade
        mix *= self.master_volume

        # Hard clipper to avoid nasty digital distortion
        np.clip(mix, -1.0, 1.0, out=mix)

        if not self.active:
            mix.fill(0)
        outdata[:] = mix


class SamplerEngineNode(Node):
    """
    Manages the global SamplerEngine instance.
    Similar to AudioMixerNode, ensures only one engine runs.
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

        # Ensure engine is initialized even if SamplerEngineNode isn't present,
        # though usually one should exist for global control.
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
        LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
        try:
            # torchaudio load
            waveform, sample_rate = torchaudio.load(filepath)
            if not waveform.is_cpu:
                waveform = waveform.cpu()

            # Convert (C, N) to (N, C) for SamplerEngine
            arr = waveform.numpy().T

            # Create Sample object
            self.sample = Sample(arr)
            # Update label
            self.file_label.set(os.path.basename(filepath))
            self.length_label.set(f"Length: {len(arr)} samples")

        except Exception as e:
            print(f"Error loading file {filepath}: {e}")

    def load_file(self):
        val = self.path_input()
        if val == "bang":
            self.request_load_file()
        elif val:
            self.load_file_with_path(val)

    def update_params(self):
        if SamplerEngineNode.engine:
            # Update active voice params immediately
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

            # Handle default loop_end (-1)
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
            # Gather params
            idx = int(self.voice_idx_input())
            vol = float(self.volume_input())
            pitch = float(self.pitch_input())
            loop = bool(self.loop_input())
            l_start = int(self.loop_start_input())
            l_end = int(self.loop_end_input())
            xf = int(self.crossfade_input())

            # Handle default loop_end (-1)
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
        # SamplerEngine currently has set_voice_volume to 0, but no explicit stop_voice method exposed well?
        # setting volume to 0 effectively mutes it, but doesn't deactivate it in the engine unless we add logic.
        # Actually SamplerEngine does not have a 'stop_voice' method that sets active=False directly.
        # But setting volume to 0 is a good proxy.
        # Impl: set volume to 0.
        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.set_voice_volume(self.last_voice_idx, 0.0)

    def execute(self):
        # Support triggering via tensor input or signal if needed
        pass


class SamplerMultiVoiceNode(Node):
    """
    A node to control ALL voices of the SamplerEngine.
    Maintains state for each voice and updates UI based on selected voice index.
    """

    @staticmethod
    def factory(name, data, args=None):
        node = SamplerMultiVoiceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Initialize state for 128 voices
        self.voices_state = []
        for i in range(128):
            self.voices_state.append({
                "sample": None,
                "path": "",
                "volume": 1.0,
                "pitch": 1.0,
                "loop": False,
                "loop_start": 0,
                "loop_end": -1,
                "crossfade": 0,
                "playing": False
            })

        self.current_idx = 0
        self.ignore_updates = False  # Flag to prevent callbacks during UI sync

        # Ensure engine
        if SamplerEngineNode.engine is None:
            SamplerEngineNode.engine = SamplerEngine()
            SamplerEngineNode.engine.start()
        # Inputs
        # Voice Index (Callback to switch context)
        self.voice_idx_input = self.add_input('voice index', widget_type='drag_int', default_value=0, min=0, max=127,
                                              callback=self.on_voice_idx_change)

        self.play_toggle = self.add_input('play', widget_type='checkbox', default_value=False,
                                          callback=self.toggle_play)

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

        self.pos_output = self.add_output('position')
        # Initial sync
        # self.sync_ui_to_state() # Do not sync here, wait for custom_create

    def custom_create(self, from_file):
        self.sync_ui_to_state()

    def request_load_file(self, *args):
        LoadDialog(self, self.load_file_callback, extensions=['.wav', '.mp3', '.aif', '.flac'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self.load_file_with_path(path)

    def load_file_with_path(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
        try:
            # torchaudio load
            waveform, sample_rate = torchaudio.load(filepath)
            if not waveform.is_cpu:
                waveform = waveform.cpu()

            arr = waveform.numpy().T
            sample = Sample(arr)

            # Update state
            state = self.voices_state[self.current_idx]
            state["sample"] = sample
            state["path"] = filepath
            state[
                "loop_end"] = -1  # Reset loop end on new file load? Or keep? usually reset logic takes over in trigger
            # User might expect loop end to persist if manually set, but if new file is shorter/longer?
            # Let's reset loop points to default for new file for safety
            state["loop_start"] = 0
            state["loop_end"] = -1

            # Update labels
            self.file_label.set(os.path.basename(filepath))
            self.length_label.set(f"Length: {len(arr)} samples")

            # Sync UI to show new defaults
            # But we are in load_file which might be triggered by UI.
            # We should update inputs.
            self.ignore_updates = True
            self.loop_start_input.set(0)
            self.loop_end_input.set(-1)
            self.ignore_updates = False

            if state["playing"]:
                self.trigger_voice(self.current_idx)

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

        # Sync playing state from Engine truth
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
        self.volume_input.set(state["volume"])
        self.pitch_input.set(state["pitch"])
        self.loop_input.set(state["loop"])
        self.loop_start_input.set(state["loop_start"])
        self.loop_end_input.set(state["loop_end"])
        self.crossfade_input.set(state["crossfade"])

        self.ignore_updates = False

    def frame_task(self):
        if SamplerEngineNode.engine:
            voice = SamplerEngineNode.engine.voices[self.current_idx]

            # Update position
            self.pos_output.send(voice.position)

            # Sync Toggle State with Voice Active State
            ui_playing = self.voices_state[self.current_idx]["playing"]

            if voice.active and not ui_playing:
                # Voice started externally (e.g. via list trigger)
                self.voices_state[self.current_idx]["playing"] = True
                self.ignore_updates = True
                self.play_toggle.set(True)
                self.ignore_updates = False

            elif not voice.active and ui_playing:
                # Voice stopped (e.g. auto-stop or finished)
                self.voices_state[self.current_idx]["playing"] = False
                self.ignore_updates = True
                self.play_toggle.set(False)
                self.ignore_updates = False

                # Should we remove frame task?
                # If we are just viewing this voice and it stops, we stop frame task.
                self.remove_frame_tasks()
            # If voice is not active and we are not playing, we should probably stop frame task
            # BUT: frame_task is added based on UI state or trigger.
            # If we are here, we are active.

    def update_params(self, *args):
        if self.ignore_updates: return

        state = self.voices_state[self.current_idx]
        state["volume"] = float(self.volume_input())
        state["pitch"] = float(self.pitch_input())

        if SamplerEngineNode.engine:
            SamplerEngineNode.engine.set_voice_volume(self.current_idx, state["volume"])
            SamplerEngineNode.engine.set_voice_pitch(self.current_idx, state["pitch"])
            # Check if volume change implies play? (User hint)
            # If voice became active due to volume change (unlikely in this engine but per user note):
            if SamplerEngineNode.engine.voices[self.current_idx].active:
                if not state["playing"]:
                    state["playing"] = True
                    self.ignore_updates = True
                    self.play_toggle.set(True)
                    self.ignore_updates = False
                    self.add_frame_task()

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
        # Handle input value
        try:
            val = self.play_toggle()

            # Check for list input (Advanced Triggering)
            if isinstance(val, (list, tuple)):
                if not SamplerEngineNode.engine: return

                # Helper to process voice update
                def update_voice(idx, vol):
                    if not (0 <= idx < 128): return

                    # Update Engine
                    SamplerEngineNode.engine.set_voice_volume(idx, vol)

                    # Update Internal State
                    self.voices_state[idx]["volume"] = vol

                    voice = SamplerEngineNode.engine.voices[idx]

                    # Logic: If volume > 0 and not active, TRIGGER
                    if not voice.active and vol > 0:
                        self.trigger_voice(idx)
                        self.voices_state[idx]["playing"] = True

                        # If this is the current voice visible in UI
                        if idx == self.current_idx:
                            self.ignore_updates = True
                            self.play_toggle.set(True)
                            self.volume_input.set(vol)
                            self.ignore_updates = False
                            self.add_frame_task()

                    # Logic: If volume is 0, we let auto-stop handle it OR we assume it might stop?
                    # But if we are adjusting volume of CURRENT voice, we must update UI widget
                    elif idx == self.current_idx:
                        self.ignore_updates = True
                        self.volume_input.set(vol)
                        self.ignore_updates = False

                        # If it is active (even if we just set vol to 0, it takes 1s to stop)
                        # Ensure frame task is running to catch the auto-stop event
                        if voice.active:
                            self.add_frame_task()

                # Heuristic: List of lists or flat list?
                # Check first element
                if len(val) > 0:
                    first = val[0]
                    if isinstance(first, (list, tuple)):
                        # Method 1: [[idx, vol], ...]
                        for item in val:
                            if len(item) >= 2:
                                update_voice(int(item[0]), float(item[1]))
                    else:
                        # Method 2: Flat list [vol0, vol1, ...]
                        for i, v_vol in enumerate(val):
                            update_voice(i, float(v_vol))

                # Do not proceed with standard toggle logic if list received
                return
        except Exception as e:
            # Not a list, or other error, proceed to standard logic
            # print(f"Error in toggle_play: {e}")
            pass

        if self.ignore_updates: return

        state = self.voices_state[self.current_idx]
        is_playing = bool(self.play_toggle())
        state["playing"] = is_playing

        if is_playing:
            self.trigger_voice(self.current_idx)
            self.add_frame_task()
        else:
            if SamplerEngineNode.engine:
                SamplerEngineNode.engine.set_voice_volume(self.current_idx, 0.0)
            self.remove_frame_tasks()

    def trigger_voice(self, idx):
        if SamplerEngineNode.engine:
            state = self.voices_state[idx]
            if state["sample"]:
                s = state["sample"]
                s.default_volume = state["volume"]
                s.default_pitch = state["pitch"]
                s.loop = state["loop"]
                s.loop_start = state["loop_start"]
                le = state["loop_end"]
                if le < 0: le = len(s.data)
                s.loop_end = le
                s.crossfade_frames = state["crossfade"]

                SamplerEngineNode.engine.play_voice(idx, s, volume=state["volume"], pitch=state["pitch"])

    def execute(self):
        # Frame task handles updates now
        pass
