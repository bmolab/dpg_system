from dpg_system.torch_base_nodes import *
import torch
import torchaudio
pyaudio_available = True
try:
    import pyaudio
except Exception as e:
    print('pyaudio not available')
    pyaudio_available = False


def register_torchaudio_nodes():
    global pyaudio_available
    if pyaudio_available:
        Node.app.register_node('t.audio_source', TorchAudioSourceNode.factory)
    Node.app.register_node('ta.kaldi_pitch', TorchAudioKaldiPitchNode.factory)
    Node.app.register_node('t.audio.gain', TorchAudioGainNode.factory)
    Node.app.register_node('t.audio.contrast', TorchAudioContrastNode.factory)
    Node.app.register_node('t.audio.loudness', TorchAudioLoudnessNode.factory)
    Node.app.register_node('t.audio.overdrive', TorchAudioOverdriveNode.factory)
    # Node.app.register_node('ta.vad', TorchAudioVADNode.factory) - does not seem to do anything


class AudioSource:
    audio = None

    def __init__(self, channels=1, rate=16000, chunk=1024, data_format=pyaudio.paFloat32):
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        self.stream = None
        self.callback_routine = None
        self.format = data_format
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.sources = {}
        self.device_info = self.audio.get_default_input_device_info()
        self.device_index = self.device_info['index']
        self.device_count = self.audio.get_device_count()
        for i in range(self.device_count):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                self.sources[i] = device_info['name']

    def get_device_list(self):
        dev_list = []
        for index in self.sources:
            dev_list.append(self.sources[index])
        return dev_list

    def change_source(self, source_name):
        for index in self.sources:
            name = self.sources[index]
            if name == source_name:
                self.device_index = index
                self.device_info = self.audio.get_device_info_by_index(index)
                print(self.device_index, self.device_info)

    def set_callback(self, routine):
        self.callback_routine = routine

    def callback(self, input_data, frame_count, time_info, flag):
        if self.callback_routine:
            return self.callback_routine(input_data, frame_count, time_info, flag)

    def start(self):
        if self.stream is None:
            self.stream = self.audio.open(rate=self.rate, channels=self.channels, input_device_index=self.device_index, format=self.format, input=True, frames_per_buffer=self.chunk, stream_callback=self.callback)
        return self.stream is not None

    def stop(self):
        if self.stream is not None:
            if self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            self.stream = None
        return self.stream is not None

    def get_max_input_channels(self):
        return self.device_info['maxInputChannels']

    def get_default_sample_rate(self):
        return int(self.device_info['defaultSampleRate'])

    def check_format(self, rate, channels, data_format):
        # print('check format', rate, channels, data_format)
        try:
            supported = self.audio.is_format_supported(rate=rate, input_device=self.device_index, input_channels=channels, input_format=data_format)
            return supported
        except Exception as ex:
            if ex == ValueError:
                print('unsupported audio source parameters')
        return False




class TorchAudioSourceNode(TorchNode):
    format_dict = {
        'float': pyaudio.paFloat32,
        'int32': pyaudio.paInt32,
        'int24': pyaudio.paInt24,
        'int16': pyaudio.paInt16
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAudioSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.streaming = False
        self.dtype = torch.float32
        self.source = AudioSource()
        self.source_name = self.source.device_info['name']
        self.source.set_callback(self.audio_callback)
        self.stream_input = self.add_input('stream', widget_type='checkbox', default_value=self.streaming,
                                           callback=self.stream_on_off)
        self.source_choice = self.add_property('source', widget_type='combo', width=180, default_value=self.source_name, callback=self.source_params_changed)
        self.source_choice.widget.combo_items = self.source.get_device_list()
        self.channels = self.add_input('channels', widget_type='input_int', default_value=1, callback=self.source_params_changed)
        self.sample_rate = self.add_input('sample_rate', widget_type='drag_int', default_value=16000, callback=self.source_params_changed)
        self.format = self.add_property('sample format', widget_type='combo', default_value='float', callback=self.source_params_changed)
        self.format.widget.combo_items = ['float', 'int32', 'int24', 'int16']
        self.chunk_size = self.add_input('chunk_size', widget_type='drag_int', default_value=1024, callback=self.source_params_changed)
        self.output = self.add_output('audio tensors')

    def source_params_changed(self):
        changed = False
        source_changed = False
        source = self.source_choice()
        if source != self.source_name:
            source_changed = True
            self.source_name = source

        channels = self.channels()
        if channels != self.source.channels:
            changed = True
        sample_rate = self.sample_rate()
        if sample_rate != self.source.rate:
            changed = True
        data_format = self.format()
        if data_format in self.format_dict:
            data_format = self.format_dict[data_format]
            if data_format != self.source.format:
                changed = True
        else:
            data_format = self.source.format
        chunk = self.chunk_size()
        if chunk != self.source.chunk:
            changed = True

        streaming = self.streaming
        if changed or source_changed:
            self.source.change_source(source)
            maxChannels = self.source.get_max_input_channels()
            if channels > maxChannels:
                channels = maxChannels
                self.channels.set(channels)

            if self.source.check_format(sample_rate, channels, data_format):
                self.source.stop()
                self.source.channels = channels
                self.source.rate = sample_rate
                self.source.format = data_format
                self.source.chunk = chunk
                if streaming:
                    self.streaming = self.source.start()
            else:
                sample_rate = self.source.get_default_sample_rate()
                if self.source.check_format(sample_rate, channels, data_format):
                    self.source.stop()
                    self.source.channels = channels
                    self.source.rate = sample_rate
                    self.sample_rate.set(self.source.rate)
                    self.source.format = data_format
                    self.source.chunk = chunk
                    if streaming:
                        self.streaming = self.source.start()
                else:
                    print('Audio Source format invalid: channels =', channels, 'rate =', sample_rate, 'format =', data_format)

    def audio_callback(self, data, frame_count, time_info, flag):
        numpy_audio_data = np.frombuffer(buffer=data, dtype=np.float32).copy()
        chans = self.source.channels
        if chans > 1:
            numpy_audio_data = numpy_audio_data.reshape(-1, chans).swapaxes(0, 1)
        torch_audio_data = torch.from_numpy(numpy_audio_data)
        self.output.send(torch_audio_data)
        return data, pyaudio.paContinue

    def stream_on_off(self):
        if self.stream_input():
            if not self.streaming:
                self.streaming = self.source.start()
        else:
            if self.streaming:
                self.streaming = self.source.stop()

    def custom_cleanup(self):
        if self.streaming:
            self.source.stop()
            self.source.audio.terminate()
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


# class TorchAudioSpectrogramNode(TorchNode):
#     window_function_dict = {
#         'blackman': torch.blackman_window,
#         'bartlett': torch.bartlett_window,
#         'hann': torch.hann_window,
#         'hamming': torch.hamming_window,
#         'nutall': torch.kaiser_window
#     }
#     @staticmethod
#     def factory(name, data, args=None):
#         node = TorchAudioSpectrogramNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#
#         self.transform = torchaudio.transforms.Spectrogram(n_fft=400)
#         self.input = self.add_input('audio tensor in', triggers_execution=True)
#         self.n_fft = self.add_input('n_fft', widget_type='drag_int', default_value=400, callback=self.params_changed)
#         self.window_length = self.add_input('window_length', widget_type='drag_int', default_value=400, callback=self.params_changed)
#         self.hop_length = self.add_input('hop_length', widget_type='drag_int', default_value=200, callback=self.params_changed)
#         self.window_function = self.add_input('window', widget_type='combo', default_value='hann', callback=self.params_changed)
#         self.window_function.widget.combo_items = ['blackman', 'bartlett', 'hamming', 'hann', 'kaiser']
#         self.power = self.add_input('power', widget_type='drag_float', default_value=2.0, callback=self.params_changed)
#         self.normalized = self.add_input('norm', widget_type='combo', default_value='frame_length', callback=self.params_changed)
#         self.normalized.widget.combo_items = ['frame_length', 'window']
#         self.one_sided = self.add_input('one-sided', widget_type='checkbox', default_value=True, callback=self.params_changed)
#         self.output = self.add_output('spectrogram out')
#
#
#     def params_changed(self):
#         win_f = torch.hann_window
#         if self.window_function() in self.window_function_dict:
#             win_f = self.window_function_dict[self.window_function()]
#         self.transform = torchaudio.transforms.Spectrogram(
#             n_fft=self.n_fft(),
#             win_length=self.window_length(),
#             hop_length=self.hop_length(),
#             window_fn=win_f,
#             power = self.power(),
#             normalized=self.normalized(),
#             onesided=self.one_sided()
#         )
#
#     def execute(self):
#         data = self.input_to_tensor()
#         if data is not None:
#             spectrogram = self.transform(data)
#             self.output.send(spectrogram)





