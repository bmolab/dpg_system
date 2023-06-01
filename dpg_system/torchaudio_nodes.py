from dpg_system.torch_base_nodes import *
import torch
import torchaudio
import pyaudio


def register_torchaudio_nodes():
    Node.app.register_node('t.audio_source', TorchAudioSourceNode.factory)
    Node.app.register_node('ta.kaldi_pitch', TorchAudioKaldiPitchNode.factory)
    Node.app.register_node('t.audio.gain', TorchAudioGainNode.factory)
    Node.app.register_node('t.audio.contrast', TorchAudioContrastNode.factory)
    Node.app.register_node('t.audio.loudness', TorchAudioLoudnessNode.factory)
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

    def check_format(self, rate, channels, data_format):
        print('check format', rate, channels, data_format)
        return self.audio.is_format_supported(rate=rate, input_device=self.device_index, input_channels=channels, input_format=data_format)


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
        self.channels = self.add_property('channels', widget_type='input_int', default_value=1, callback=self.source_params_changed)
        self.sample_rate = self.add_property('sample_rate', widget_type='drag_int', default_value=16000, callback=self.source_params_changed)
        self.format = self.add_property('sample format', widget_type='combo', default_value='float', callback=self.source_params_changed)
        self.format.widget.combo_items = ['float', 'int32', 'int24', 'int16']
        self.chunk_size = self.add_property('chunk_size', widget_type='drag_int', default_value=1024, callback=self.source_params_changed)
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
            if source_changed:
                self.source.change_source(source)
            if self.source.check_format(sample_rate, channels, data_format) or source_changed:
                self.source.stop()
                self.source.channels = channels
                self.source.rate = sample_rate
                self.source.format = data_format
                self.source.chunk = chunk
                if streaming:
                    self.streaming = self.source.start()
            else:
                print('Audio Source format invalid: channels =', channels, 'rate =', sample_rate, 'format =', data_format)

    def audio_callback(self, data, frame_count, time_info, flag):
        numpy_audio_data = np.frombuffer(buffer=data, dtype=np.float32).copy()
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




