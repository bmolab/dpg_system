"""t.audio_source: the microphone as torch tensors.

Kept for the patches that feed live input to torch analysis (t.rfft, t.cwt,
t.energy). Sound between nodes is otherwise a ~ signal; the general route
into torch is now adc~ -> capture~ with format 'torch cpu' or 'torch mps',
which takes any ~ signal and never waits on a GUI frame for its audio.

The file players and effects that used to live here (t.audio.file, .play,
.multiplayer, .file_stream, .gain, .contrast, .overdrive, .loudness,
.kaldi_pitch, ta.vad) and audio_mixer were retired 2026-10-01 in favour of
sampler_osc~, the samplers, vca~, distort~ and shaper~.
"""
from dpg_system.torch_base_nodes import *
import torch

import queue

# The input stream is shared with whisper and nemotron via audio_io.
from dpg_system.audio_io import AudioSource


def register_torchaudio_nodes():
    Node.app.register_node('t.audio_source', TorchAudioSourceNode.factory)


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
        # The rate the stream really opened at; patch into whatever analyses it.
        self.rate_output = self.add_output('sample_rate')
        self.dropped_output = self.add_output('dropped')

        # Blocks cross from PortAudio's thread to the main thread through
        # this queue: the callback only copies and enqueues, and frame_task
        # sends. Running the downstream graph on the audio thread meant any
        # heavy node dropped device buffers and any widget-touching node
        # was on the wrong thread.
        self._chunks = queue.SimpleQueue()
        self._dropped = 0
        self._reported_dropped = 0
        self.add_frame_task()

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

        if changed or source_changed:
            self.rate_output.send(int(self.source.samplerate))

    # A stalled patch must not grow the queue without bound: past this many
    # waiting blocks (about 4 s at the default 1024 @ 16 kHz) new ones are
    # dropped and counted. And one frame must not dump an unbounded burst.
    MAX_QUEUED = 64
    MAX_PER_FRAME = 16

    def audio_callback(self, indata, frame_count, time_info, flag):
        # PortAudio's thread: copy and queue, nothing else. The buffer is
        # PortAudio's and is reused, so the copy is not optional. An uncaught
        # exception here silently kills the stream, hence the broad catch.
        try:
            if self._chunks.qsize() >= TorchAudioSourceNode.MAX_QUEUED:
                self._dropped += 1
                return
            self._chunks.put(indata.T.copy())
        except Exception as e:
            print(f'{self.label}: audio_callback: {type(e).__name__}: {e}')
            traceback.print_exc()

    def frame_task(self):
        sent = 0
        while sent < TorchAudioSourceNode.MAX_PER_FRAME:
            try:
                chunk = self._chunks.get_nowait()
            except queue.Empty:
                break
            self.output.send(torch.from_numpy(chunk))
            sent += 1
        if self._dropped != self._reported_dropped:
            self._reported_dropped = self._dropped
            self.dropped_output.send(self._dropped)

    def stream_on_off(self):
        if self.stream_input():
            if not self.streaming:
                self.streaming = self.source.start()
                if self.streaming:
                    self.rate_output.send(int(self.source.samplerate))
        else:
            if self.streaming:
                is_stopped = self.source.stop()
                if is_stopped:
                    self.streaming = False

    def custom_cleanup(self):
        self.remove_frame_tasks()
        if self.streaming:
            self.source.stop()
            self.source = None
