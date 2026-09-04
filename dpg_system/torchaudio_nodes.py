from dpg_system.torch_base_nodes import *
import torch
import torchaudio
from dpg_system.node import LoadDialog

import time
import os

# Input streams and rate conversion are shared with whisper via audio_io;
# the only output stream is the sampler engine's (see MixerBridge below).
from dpg_system.audio_io import AudioSource


def register_torchaudio_nodes():
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
    Node.app.register_node('t.audio.file_stream', TorchAudioFileStreamNode.factory)

    # Node.app.register_node('ta.vad', TorchAudioVADNode.factory) - does not seem to do anything


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
        self.source = AudioSource()
        self.source_name = self.source.sources.get(self.source.device_index, 'none')
        self.source.set_callback(self.audio_callback)

        self.stream_input = self.add_input('stream', widget_type='checkbox', default_value=self.streaming,
                                           callback=self.stream_on_off)
        self.source_choice = self.add_property('source', widget_type='combo', width=180, default_value=self.source_name, callback=self.source_params_changed)
        self.source_choice.widget.combo_items = self.source.get_device_list() or ['none']
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
        if sample_rate != self.source.samplerate:
            changed = True
        dtype_str = self.format_dict.get(self.format(), 'float32')
        if dtype_str != self.source.dtype:
            changed = True

        chunk = self.chunk_size()
        if chunk != self.source.blocksize:
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
                self.source.samplerate = sample_rate
                self.source.dtype = dtype_str
                self.source.blocksize = chunk
                if streaming:
                    self.streaming = self.source.start()
            else:
                sample_rate = self.source.get_default_sample_rate()
                if self.source.check_format(sample_rate, channels, dtype_str):
                    self.source.stop()
                    self.source.channels = channels
                    self.source.samplerate = sample_rate
                    self.sample_rate.set(self.source.samplerate)
                    self.source.dtype = dtype_str
                    self.source.blocksize = chunk
                    if streaming:
                        self.streaming = self.source.start()
                else:
                    print('Audio Source format invalid: channels =', channels, 'rate =', sample_rate, 'format =', dtype_str)

    def audio_callback(self, indata, frame_count, time_info, flag):
        # This runs on the sounddevice PortAudio thread. An uncaught
        # exception here can silently kill the input stream — catch it
        # broadly so the next callback still fires and the user sees the
        # diagnostic instead of a stream that just stopped working.
        try:
            torch_ready_numpy = indata.T
            torch_audio_data = torch.from_numpy(torch_ready_numpy)
            self.output.send(torch_audio_data)
        except Exception as e:
            print(f'{self.label}: audio_callback: {type(e).__name__}: {e}')
            traceback.print_exc()

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
            self.source = None


class TorchAudioKaldiPitchNode(TorchNode):
    # torchaudio removed compute_kaldi_pitch in 2.1. The node stays registered
    # so saved patches and the help patch still load; on such installs it
    # explains itself once and passes nothing rather than raising per call.
    _available = hasattr(torchaudio.functional, 'compute_kaldi_pitch')
    _warned = False

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
        if not TorchAudioKaldiPitchNode._available:
            if not TorchAudioKaldiPitchNode._warned:
                TorchAudioKaldiPitchNode._warned = True
                print(f'{self.label}: torchaudio {torchaudio.__version__} no longer has '
                      'compute_kaldi_pitch (removed in 2.1). Use speech_pitch '
                      '(parselmouth or pyin backend) instead.')
            return
        data = self.input_to_tensor()
        if data is None:
            return
        try:
            pitch_feature = torchaudio.functional.compute_kaldi_pitch(data, self.rate())
            nccf, pitch = pitch_feature[..., 0], pitch_feature[..., 1]
            self.nccf_output.send(nccf)
            self.pitch_output.send(pitch)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        if data is None:
            return
        try:
            active_audio = torchaudio.functional.vad(data, self.rate(), trigger_level=self.trigger_level(), noise_reduction_amount=self.noise_reduction())
            self.vad_output.send(active_audio)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        if data is None:
            return
        try:
            active_audio = torchaudio.functional.gain(data, self.gain())
            self.output.send(active_audio)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        if data is None:
            return
        try:
            active_audio = torchaudio.functional.contrast(data, self.contrast())
            self.output.send(active_audio)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        if data is None:
            return
        if len(data.shape) < 2:
            # Was: data.unsqueeze(dim=0) — not in-place, the result was
            # discarded and data stayed 1D. Rebind so a 1D input is
            # actually promoted to [1, N] before the loudness call.
            data = data.unsqueeze(dim=0)
        if data.shape[-1] < 6400:
            print(self.label, 'too few samples to calculate loudness (min 6400)')
            return
        try:
            active_audio = torchaudio.functional.loudness(data, self.rate())
            self.loudness_output.send(active_audio)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        if data is None:
            return
        try:
            overdriven_audio = torchaudio.functional.overdrive(data, self.gain(), self.colour())
            self.output.send(overdriven_audio)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


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
        # Use a separate attribute for the button so the load_file() method
        # defined below stays reachable via self.load_file. The previous
        # `self.load_file = self.add_input(...)` shadowed the method on the
        # instance; the button still worked (its callback was request_load_file,
        # not load_file), but any later self.load_file() call would invoke the
        # NodeInput instead of the method.
        self.load_button = self.add_input('load file', widget_type='button', callback=self.request_load_file)
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
        # No file loaded yet — sample_rate / waveform are still None and
        # int(None) / output.send(None) both crash. Bail rather than tear down.
        if self.waveform is None or self.sample_rate is None:
            return
        try:
            self.sample_rate_out.send(int(self.sample_rate))
            self.output.send(self.waveform)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchAudioPlaySoundNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioPlaySoundNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_input = self.add_input('trigger', widget_type='button', callback=self.play)
        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.path_input = self.add_input('path in', callback=self.load_file)
        # Separate attribute for the button so the load_file() method below
        # isn't shadowed by the NodeInput instance.
        self.load_button = self.add_input('load file', widget_type='button', callback=self.request_load_file)
        self.file_name = self.add_label('')
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop)
        # Rate of audio arriving on the tensor inlet; files carry their own.
        self.rate_input = self.add_input('sample_rate', widget_type='drag_int', default_value=44100)
        self.last_sound_id = None
        # Pre-declare the per-instance waveform buffers so play()'s
        # `stored_waveform_np is not None` guard has an attribute to test
        # against — previously hitting the play button before any audio had
        # been loaded raised AttributeError.
        self.waveform_np = None
        self.stored_waveform_np = None
        self.stored_rate = 44100

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
        self.stored_rate = int(self.sample_rate)
        file_name = filepath.split('/')[-1]
        self.file_name.set(file_name)

    def load_file(self):
        filepath = self.path_input()
        self.load_file_with_path(filepath)

    def stop(self):
        if self.last_sound_id is not None:
            MixerBridge.stop(self.last_sound_id)
            self.last_sound_id = None

    def custom_cleanup(self):
        self.stop()

    def play(self):
        if self.stored_waveform_np is None:
            return None
        try:
            self.last_sound_id = MixerBridge.play(self.stored_waveform_np, self.stored_rate)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
        return self.last_sound_id

    def execute(self):
        data = self.input_to_tensor()
        if data is None:
            return
        try:
            waveform = data.detach().cpu()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(1)
            elif waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.T
            self.stored_waveform_np = waveform.numpy().copy()
            self.stored_rate = int(self.rate_input())
            self.play()
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
        return self.last_sound_id


class TorchAudioMultiPlayerNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioMultiPlayerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_input = self.add_input('trigger', widget_type='button', callback=self.play)
        self.input = self.add_input('audio tensor in', triggers_execution=True)
        self.path_input = self.add_input('path in', callback=self.load_file)
        # Separate attribute for the button so the load_file() method below
        # isn't shadowed by the NodeInput instance.
        self.load_button = self.add_input('load file', widget_type='button', callback=self.request_load_file)
        self.file_name = self.add_label('')
        self.remove = self.add_input('remove wave', callback=self.remove_wave)
        self.clear_button = self.add_input('clear waves', callback=self.clear_waves)
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop)
        # Rate of audio arriving on the tensor inlet; files carry their own.
        self.rate_input = self.add_input('sample_rate', widget_type='drag_int', default_value=44100)
        self.last_sound_id = None
        self.waves = {}
        self.player_ids = {}
        self.last_loaded = None
        # Pre-declared so execute()'s in-place numpy buffer pattern doesn't
        # silently assume these attributes exist on first call.
        self.waveform_np = None
        self.stored_waveform_np = None
        self.stored_rate = 44100

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
        self.waves[sample_name] = (self.waveform.numpy(), int(self.sample_rate))
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
            MixerBridge.stop(self.last_sound_id)
            self.last_sound_id = None

    def custom_cleanup(self):
        self.stop()

    def play(self):
        trigger = self.trigger_input()
        name = self.last_loaded
        if type(trigger) == str and trigger in self.waves:
            name = trigger
        if name is None or name not in self.waves:
            return None
        try:
            array, rate = self.waves[name]
            self.last_sound_id = MixerBridge.play(array, rate)
            self.player_ids[name] = self.last_sound_id
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
        return self.last_sound_id

    def execute(self):
        waveform = self.input_to_tensor()
        if waveform is None:
            return
        try:
            waveform = waveform.detach().cpu()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(1)
            elif waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.T
            self.stored_waveform_np = waveform.numpy().copy()
            self.stored_rate = int(self.rate_input())
            self.last_sound_id = MixerBridge.play(self.stored_waveform_np, self.stored_rate)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
        return self.last_sound_id


class MixerBridge:
    """Plays arrays through the sampler engine, on voices reserved for it.

    This replaces a second PortAudio output stream (the old AudioMixer) that
    t.audio.play and t.audio.multiplayer used to open beside the sampler and
    synth stream, with all the device contention that implied. A voice is
    just varispeed playback of a Sample built from the array, at a pitch of
    the array's rate over the engine's -- so a 16 kHz tensor and a 48 kHz
    file both play at the right speed, which the old mixer never did.

    The pool sits at the top of the engine's 128 voices, clear of
    polyphonic_sampler's default range (64-80). audio_mixer moves it.
    """

    first_voice = 112
    voice_count = 16
    # voice index -> when it was last triggered; the oldest is stolen when
    # the pool is full, and a just-triggered voice is not re-used before its
    # command has had a chance to drain on the audio thread.
    _started = {}
    GRACE_SECONDS = 0.2

    @staticmethod
    def engine():
        try:
            from dpg_system.sampler_nodes import SamplerEngineNode
            from dpg_system.sampler import SamplerEngine
        except Exception as error:
            print(f'MixerBridge: sampler engine unavailable ({error})')
            return None
        engine = SamplerEngineNode.engine
        if engine is None:
            engine = SamplerEngine()
            if not engine.start():
                return None
            SamplerEngineNode.engine = engine
        return engine

    @classmethod
    def pool(cls):
        first = max(0, min(127, int(cls.first_voice)))
        last = min(128, first + max(1, int(cls.voice_count)))
        return range(first, last)

    @classmethod
    def _allocate(cls, engine):
        now = time.monotonic()
        oldest, oldest_time = None, None
        for index in cls.pool():
            started = cls._started.get(index, 0.0)
            if not engine.voices[index].active and now - started > cls.GRACE_SECONDS:
                return index
            if oldest_time is None or started < oldest_time:
                oldest, oldest_time = index, started
        return oldest

    @classmethod
    def play(cls, array, sample_rate, volume=1.0):
        """Start `array` playing; returns the voice index as a sound id."""
        engine = cls.engine()
        if engine is None:
            return None
        from dpg_system.sampler import Sample
        data = np.asarray(array, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        elif data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T                     # (channels, frames) -> (frames, channels)
        elif data.ndim != 2:
            print(f'MixerBridge: cannot play an array of shape {data.shape}')
            return None
        if data.shape[1] > 2:
            data = data[:, :2]                # voices are at most stereo
        if data.shape[0] < 2:
            return None
        sample = Sample(np.ascontiguousarray(data))
        rate = float(sample_rate) if sample_rate else float(engine.sample_rate)
        pitch = rate / float(engine.sample_rate)
        index = cls._allocate(engine)
        if index is None:
            return None
        voice = engine.voices[index]
        voice.set_envelope(0.0, 0.005)        # 5 ms tail so a stop does not click
        voice.trigger(sample, volume=float(volume), pitch=pitch, mode='normal')
        cls._started[index] = time.monotonic()
        return index

    @classmethod
    def stop(cls, sound_id):
        engine = cls.engine()
        if engine is not None and sound_id is not None:
            engine.stop_voice(int(sound_id))

    @classmethod
    def stop_all(cls):
        engine = cls.engine()
        if engine is not None:
            for index in cls.pool():
                engine.stop_voice(index)


class AudioMixerNode(Node):
    '''
    Settings for array playback through the shared audio engine: which
    output device the engine uses (engine-wide, the same choice audio_out~
    offers) and which voices t.audio.play / t.audio.multiplayer draw on.
    Optional -- the players work without one.
    '''

    @staticmethod
    def factory(name, data, args=None):
        node = AudioMixerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self._devices = []
        try:
            from dpg_system.sampler import output_devices
            self._devices = [('%s (%d ch)' % (name, count), index, count)
                             for index, name, count in output_devices()]
        except Exception as e:
            print(f'audio_mixer: could not list output devices ({e})')
        self._device_pending = False

        self.device_choice = self.add_property('output device', widget_type='combo', width=220,
                                               default_value='', callback=self.device_chosen)
        self.device_choice.widget.combo_items = [d[0] for d in self._devices]
        self.first_voice_prop = self.add_property('first voice', widget_type='drag_int',
                                                  default_value=MixerBridge.first_voice,
                                                  min=0, max=127, callback=self.pool_changed)
        self.voice_count_prop = self.add_property('voice count', widget_type='drag_int',
                                                  default_value=MixerBridge.voice_count,
                                                  min=1, max=128, callback=self.pool_changed)
        self.stop_all_button = self.add_input('stop all', widget_type='button', callback=self.stop_all)
        self.add_frame_task()

    def pool_changed(self):
        MixerBridge.first_voice = int(self.first_voice_prop())
        MixerBridge.voice_count = int(self.voice_count_prop())

    def device_chosen(self):
        # Deferred to the frame task: reopening the stream stalls ~100 ms.
        self._device_pending = True

    def update_parameters_from_widgets(self):
        self.pool_changed()
        if str(self.device_choice()).strip():
            self._device_pending = True

    def frame_task(self):
        if not self._device_pending:
            return
        self._device_pending = False
        chosen = str(self.device_choice()).strip()
        for display, index, _count in self._devices:
            if display == chosen:
                engine = MixerBridge.engine()
                if engine is None:
                    print('audio_mixer: no audio engine')
                    return
                ok, message = engine.set_device(index)
                if not ok:
                    print('audio_mixer: ' + message)
                return

    def stop_all(self):
        MixerBridge.stop_all()


class TorchAudioFileStreamNode(TorchNode):
    """
    Streams an audio file in real-time chunks, matching the output format
    of t.audio_source (torch tensors of shape (channels, chunk_size)).

    Supports play/pause/stop, looping, speed control, and clip start/end.
    Connect to whisper.audio_in, speech_pitch, or any node expecting
    streaming audio tensors.
    """

    @staticmethod
    def factory(name, data, args=None):
        return TorchAudioFileStreamNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.waveform = None       # (channels, total_samples) torch tensor
        self.file_sample_rate = None
        self.playing = False
        self.play_pos = 0          # current sample position in waveform
        self.last_emit_time = 0.0

        # ── Inputs ──
        self.play_input = self.add_input('play', widget_type='checkbox',
                                         default_value=False,
                                         callback=self.play_toggle)
        self.path_input = self.add_input('path in', callback=self.load_from_input)
        self.load_button = self.add_input('load file', widget_type='button',
                                          callback=self.request_load_file)
        self.file_label = self.add_label('')
        self.rewind_button = self.add_input('rewind', widget_type='button',
                                           callback=self.rewind)

        # ── Properties ──
        self.sample_rate_prop = self.add_input('sample_rate', widget_type='drag_int',
                                               default_value=16000)
        self.chunk_size_prop = self.add_input('chunk_size', widget_type='drag_int',
                                              default_value=1024)
        self.channels_prop = self.add_input('channels', widget_type='input_int',
                                            default_value=1)
        self.speed_prop = self.add_input('speed', widget_type='drag_float',
                                         default_value=1.0)
        self.loop_prop = self.add_input('loop', widget_type='checkbox',
                                        default_value=False)
        self.clip_start_prop = self.add_input('clip_start', widget_type='drag_float',
                                              default_value=0.0)
        self.clip_end_prop = self.add_input('clip_end', widget_type='drag_float',
                                            default_value=0.0)

        # ── Outputs ──
        self.output = self.add_output('audio tensors')
        self.sample_rate_out = self.add_output('sample_rate')
        self.position_out = self.add_output('position')
        self.done_out = self.add_output('done')

    # ── File loading ──

    def request_load_file(self):
        LoadDialog(self, self.load_file_callback, extensions=['.aif', '.wav', '.mp3', '.flac', '.ogg'])

    def load_file_callback(self, path):
        self.path_input.set(path)
        self._load_file(path)

    def load_from_input(self):
        path = any_to_string(self.path_input())
        if path:
            self._load_file(path)

    def _load_file(self, filepath):
        if not os.path.exists(filepath):
            print(f'File not found: {filepath}')
            return
        try:
            waveform, sr = torchaudio.load(filepath)
            # waveform shape: (channels, samples)
            if not waveform.is_cpu:
                waveform = waveform.cpu()

            self.file_sample_rate = sr

            # Resample to the desired output sample rate if different
            out_sr = int(self.sample_rate_prop())
            if sr != out_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=out_sr)
                waveform = resampler(waveform)

            # Convert to desired channel count
            out_channels = max(1, int(self.channels_prop()))
            if waveform.shape[0] > out_channels:
                # Downmix: take first N channels (or average for mono)
                if out_channels == 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:
                    waveform = waveform[:out_channels]
            elif waveform.shape[0] < out_channels:
                # Upmix: repeat channels
                repeats = out_channels // waveform.shape[0] + 1
                waveform = waveform.repeat(repeats, 1)[:out_channels]

            self.waveform = waveform
            self.play_pos = 0

            file_name = os.path.basename(filepath)
            duration = waveform.shape[1] / out_sr
            self.file_label.set(f'{file_name} ({duration:.1f}s)')
            self.sample_rate_out.send(out_sr)

            print(f'Loaded: {file_name} → {waveform.shape[0]}ch, {out_sr}Hz, {duration:.1f}s')
        except Exception as e:
            # traceback is already in scope via the conversion_utils star
            # import; the local `import traceback` here was redundant.
            print(f'{self.label}: error loading audio file: {type(e).__name__}: {e}')
            traceback.print_exc()

    # ── Playback control ──

    def play_toggle(self):
        should_play = self.play_input()
        if should_play:
            self._start_playback()
        else:
            self._stop_playback()

    def _start_playback(self):
        if self.waveform is None:
            print('No audio file loaded')
            return
        self.playing = True
        self.last_emit_time = time.time()
        # Apply clip start
        out_sr = int(self.sample_rate_prop())
        clip_start = float(self.clip_start_prop())
        if clip_start > 0:
            self.play_pos = int(clip_start * out_sr)
        self.add_frame_task()

    def _stop_playback(self):
        self.playing = False
        self.remove_frame_tasks()

    def rewind(self):
        out_sr = int(self.sample_rate_prop())
        clip_start = float(self.clip_start_prop())
        self.play_pos = int(clip_start * out_sr) if clip_start > 0 else 0

    # ── Streaming via frame_task ──

    def frame_task(self):
        if not self.playing or self.waveform is None:
            return
        # frame_task runs every frame, so an unhandled exception here would
        # both flood the console and leave self.playing in an inconsistent
        # state. Wrap the whole body, surface the error once per frame, and
        # stop playback so the user can recover.
        try:
            self._emit_chunks_for_frame()
        except Exception as e:
            print(f'{self.label}: frame_task: {type(e).__name__}: {e}')
            traceback.print_exc()
            self.playing = False
            self.remove_frame_tasks()

    def _emit_chunks_for_frame(self):
        now = time.time()
        out_sr = int(self.sample_rate_prop())
        chunk_size = int(self.chunk_size_prop())
        speed = max(0.01, float(self.speed_prop()))

        # Calculate how many samples we should have emitted since last frame
        dt = now - self.last_emit_time
        samples_due = int(dt * out_sr * speed)

        if samples_due < chunk_size:
            return  # not time for a chunk yet

        self.last_emit_time = now

        total_samples = self.waveform.shape[1]

        # Apply clip boundaries
        clip_start_samp = 0
        clip_end_samp = total_samples
        clip_start = float(self.clip_start_prop())
        clip_end = float(self.clip_end_prop())
        if clip_start > 0:
            clip_start_samp = min(int(clip_start * out_sr), total_samples)
        if clip_end > 0:
            clip_end_samp = min(int(clip_end * out_sr), total_samples)
        if clip_end_samp <= clip_start_samp:
            clip_end_samp = total_samples

        # Emit as many full chunks as are due
        while samples_due >= chunk_size:
            end_pos = self.play_pos + chunk_size

            if end_pos <= clip_end_samp:
                chunk = self.waveform[:, self.play_pos:end_pos]
                self.play_pos = end_pos
            else:
                # Partial chunk at end
                remaining = clip_end_samp - self.play_pos
                if remaining > 0:
                    partial = self.waveform[:, self.play_pos:clip_end_samp]
                    if self.loop_prop():
                        # Wrap around for seamless loop
                        wrap_needed = chunk_size - remaining
                        wrap = self.waveform[:, clip_start_samp:clip_start_samp + wrap_needed]
                        chunk = torch.cat([partial, wrap], dim=1)
                        self.play_pos = clip_start_samp + wrap_needed
                    else:
                        # Pad with zeros for the final chunk
                        padding = torch.zeros(self.waveform.shape[0],
                                              chunk_size - remaining)
                        chunk = torch.cat([partial, padding], dim=1)
                        self.play_pos = clip_end_samp
                else:
                    if self.loop_prop():
                        self.play_pos = clip_start_samp
                        chunk = self.waveform[:, self.play_pos:self.play_pos + chunk_size]
                        self.play_pos += chunk_size
                    else:
                        # Playback complete
                        self.playing = False
                        self.play_input.set(False)
                        self.done_out.send(1)
                        self.remove_frame_tasks()
                        return

            # Ensure chunk is exactly chunk_size
            if chunk.shape[1] < chunk_size:
                padding = torch.zeros(chunk.shape[0], chunk_size - chunk.shape[1])
                chunk = torch.cat([chunk, padding], dim=1)

            self.output.send(chunk)
            samples_due -= chunk_size

        # Output position as fraction (0..1)
        playable = clip_end_samp - clip_start_samp
        if playable > 0:
            pos_frac = (self.play_pos - clip_start_samp) / playable
            self.position_out.send(float(pos_frac))

        # Check if we've reached the end
        if self.play_pos >= clip_end_samp and not self.loop_prop():
            self.playing = False
            self.play_input.set(False)
            self.done_out.send(1)
            self.remove_frame_tasks()

    def custom_cleanup(self):
        self.playing = False
        self.remove_frame_tasks()
        self.waveform = None
