import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from elevenlabs.types import VoiceSettings
from elevenlabs.client import ElevenLabs
from queue import Queue, Empty, Full
import threading
import traceback
import time
from typing import Iterator

# create a file called elevenlabs_key.py and put in
# api_key = 'xxxxxxxx....'
# (your api key)

from dpg_system.elevenlabs_key import api_key

def register_elevenlabs_nodes():
    Node.app.register_node("eleven_labs", ElevenLabsNode.factory)


class PcmStreamer:
    """Speech from the API into the shared audio engine, as it arrives.

    The API is asked for raw 24 kHz PCM rather than MP3, so there is nothing
    to decode and no external player: each chunk becomes float samples and
    goes into a StreamUnit (the same ring stream~ uses) that the engine
    mixes beside its voices and the synth graph. One stream, one device,
    and a device chosen on audio_out~ or audio_mixer applies here too.

    Every chunk is also handed to `chunk_listener`, which the node uses for
    its 'audio' outlet, so speech can be patched into stream~, the speech
    analysis nodes, or anything else that takes audio.
    """

    RATE = 24000
    LATENCY = 0.1        # seconds held before a phrase starts sounding

    def __init__(self):
        self.force_stop = False
        self.play = True
        self.level = 1.0
        self.unit = None
        self.engine = None
        self.chunk_listener = None
        self._pending = b''

    def _ensure_engine(self):
        if self.engine is not None:
            return self.engine
        try:
            from dpg_system.sampler_nodes import SamplerEngineNode
            from dpg_system.sampler import SamplerEngine
            from dpg_system.synth_core import StreamUnit
        except Exception as error:
            print(f'ElevenLabs: audio engine unavailable ({error})')
            return None
        engine = SamplerEngineNode.engine
        if engine is None:
            engine = SamplerEngine()
            if not engine.start():
                return None
            SamplerEngineNode.engine = engine
        self.unit = StreamUnit(engine.sample_rate)
        self.unit.source_rate = float(PcmStreamer.RATE)
        self.unit.latency = PcmStreamer.LATENCY
        # Speech arrives faster than real time and must all be heard.
        self.unit.max_backlog = None
        engine.add_renderer(self)
        self.engine = engine
        return engine

    # -- audio thread --

    def render_into(self, mix, frames):
        unit = self.unit
        unit.render(frames)
        out = unit.out
        if out.constant and out.value == 0.0:
            return
        block = out.array(frames)
        if self.level != 1.0:
            block = block * self.level
        mix[:, 0] += block
        if mix.shape[1] > 1:
            mix[:, 1] += block

    # -- service thread --

    def stream(self, audio_stream: Iterator[bytes]):
        self.force_stop = False
        engine = self._ensure_engine() if self.play else self.engine
        self._pending = b''
        for chunk in audio_stream:
            if self.force_stop:
                break
            if not chunk:
                continue
            data = self._pending + chunk
            usable = len(data) - (len(data) % 2)      # whole 16-bit samples only
            self._pending = data[usable:]
            if usable == 0:
                continue
            samples = np.frombuffer(data[:usable], dtype='<i2').astype(np.float32) / 32768.0
            if self.play and self.unit is not None:
                self.unit.push(samples)
            if self.chunk_listener is not None:
                try:
                    self.chunk_listener(samples)
                except Exception as error:
                    print(f'ElevenLabs: audio outlet error ({error})')
        if self.force_stop and self.unit is not None:
            self.unit.deactivate()
        self.force_stop = False

    def speaking(self):
        """True while queued speech is still sounding after the API is done."""
        return self.unit is not None and self.unit.backlog > 0

    def do_stop(self):
        self.force_stop = True
        if self.unit is not None:
            self.unit.deactivate()

    def hard_stop(self):
        self.do_stop()

    def close(self):
        self.do_stop()
        if self.engine is not None:
            self.engine.remove_renderer(self)
            self.engine = None


def service_eleven_labs():
    while not ElevenLabsNode._stop_event.is_set():
        # Snapshot to avoid mutation-during-iteration if a node is destroyed mid-loop
        for instance in list(ElevenLabsNode.instances):
            try:
                instance.service_queue()
            except Exception as e:
                print('service_eleven_labs:', e)
                traceback.print_exception(e)
        ElevenLabsNode._stop_event.wait(0.1)


class ElevenLabsNode(Node):
    instances = []
    _stop_event = threading.Event()
    _service_thread = None
    _service_thread_lock = threading.Lock()

    @classmethod
    def _ensure_service_thread(cls):
        with cls._service_thread_lock:
            if cls._service_thread is None or not cls._service_thread.is_alive():
                cls._stop_event.clear()
                cls._service_thread = threading.Thread(
                    target=service_eleven_labs, daemon=True
                )
                cls._service_thread.start()

    @staticmethod
    def factory(name, data, args=None):
        node = ElevenLabsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.text_input = self.add_input('text to speak', triggers_execution=True)
        self.streamer = PcmStreamer()
        self.streamer.chunk_listener = self._chunk_arrived

        try:
            self.client = ElevenLabs(api_key=api_key)

            self.voices = self.client.voices.get_all()
            self.models = self.client.models.list()
            self.voice_name = 'David'
            self.active = False
            self.audio_stream = None
            if len(args) > 0:
                voice_name = any_to_string(args[0])
            else:
                voice_name = self.voice_name
            self.voice_id = None
            for voice in self.voices.voices:
                if voice.name == voice_name:
                    self.voice_name = voice_name
                    self.voice_id = voice.voice_id
                    break

            self.voice_dict = {}
            for voice in self.voices.voices:
                name = voice.name
                id = voice.voice_id
                self.voice_dict[name] = id
            self.model_dict = {}
            for model in self.models:
                name = model.name
                self.model_dict[name] = model.model_id

            voice_names = list(self.voice_dict.keys())
        except Exception as e:
            self.client = None
            self.voices = None
            self.models = None
            voice_names = []
            self.voice_name = ''
            self.active = False
            self.audio_stream = None
            self.voice_dict = {}
            self.model_dict = {}
            self.voice_id = None

        self.voice_name_input = self.add_input('voice', widget_type='combo', default_value=self.voice_name, callback=self.voice_changed)
        self.voice_name_input.widget.combo_items = voice_names
        self.model_choice = self.add_input('model', widget_type='combo', widget_width=250, default_value="Eleven Turbo v2.5")
        if self.client is not None:
            self.model_choice.widget.combo_items = list(self.model_dict.keys())
        self.speed = self.add_input('speed', widget_type='drag_float', max=1.2, min=0.7, default_value=1.0)
        self.stability = self.add_input('stability', widget_type='drag_float', default_value=.02, callback=self.voice_changed)
        self.similarity_boost = self.add_input('similarity_boost', widget_type='drag_float', default_value=1.0, callback=self.voice_changed)
        self.style = self.add_input('style exaggeration', widget_type='drag_float', default_value=0.5, callback=self.voice_changed)
        self.latency = self.add_input('latency', widget_type='combo', default_value='0')
        self.latency.widget.combo_items = ['0', '1', '2', '3', '4', '5']
        self.stop_streaming_input = self.add_input('stop', widget_type='button', callback=self.stop_streaming)
        self.hard_stop_input = self.add_input('hard stop', widget_type='button', callback=self.hard_stop_streaming)
        self.accept_input = self.add_input('accept input', widget_type='checkbox', default_value=True)
        self.play_input = self.add_input('play', widget_type='checkbox', default_value=True,
                                         callback=self.play_settings_changed)
        self.level_input = self.add_input('level', widget_type='drag_float', default_value=1.0,
                                          min=0.0, max=2.0, callback=self.play_settings_changed)
        self.active_output = self.add_output('speaking')
        self.backlog_out = self.add_output('backlog')
        # The speech itself, as float32 chunks at 24 kHz, for stream~, the
        # speech analysis nodes, or recording.
        self.audio_out = self.add_output('audio')
        self.rate_out = self.add_output('sample_rate')

        self.voice_record = None
        self.previously_active = False
        self.backlog = False
        self.voice_settings = None
        self.phrase_queue = Queue(16)
        ElevenLabsNode.instances.append(self)
        ElevenLabsNode._ensure_service_thread()

    def custom_cleanup(self):
        # Stop any in-flight playback and drain the queue
        try:
            self.streamer.close()
        except Exception:
            pass
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except Empty:
                break
        if self in ElevenLabsNode.instances:
            ElevenLabsNode.instances.remove(self)

    def play_settings_changed(self):
        self.streamer.play = bool(self.play_input())
        self.streamer.level = max(0.0, any_to_float(self.level_input()))

    def update_parameters_from_widgets(self):
        self.play_settings_changed()

    def _chunk_arrived(self, samples):
        self.audio_out.send(samples)

    def voice_changed(self):
        current_voice_name = self.voice_name_input()
        if current_voice_name in self.voice_dict:
            if self.client is not None:
                self.voice_id = self.voice_dict[current_voice_name]
                self.voice_settings = VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style())

    def post_creation_callback(self):
        if self.client is not None and self.voice_id is not None:
            self.voice_settings = VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style())

    def execute(self):
        if self.accept_input():
            self.text_to_speak = any_to_string(self.text_input())
            if len(self.text_to_speak) > 0:
                self.add_frame_task()
                try:
                    self.phrase_queue.put_nowait(self.text_to_speak)
                except Full:
                    print('ElevenLabs: phrase queue full, dropping text')
                    return
                if self.phrase_queue.qsize() > 1:
                    self.backlog = True
                    self.backlog_out.send(self.backlog)
                else:
                    self.backlog = False
                    self.backlog_out.send(self.backlog)


    def hard_stop_streaming(self):
        self.streamer.hard_stop()
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except Empty:
                break
            self.phrase_queue.task_done()

    def stop_streaming(self):
        self.streamer.do_stop()
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except Empty:
                break
            self.phrase_queue.task_done()

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty() and self.client is not None and self.voice_id is not None:
            self.active = True
            try:
                text = self.phrase_queue.get()
                if self.phrase_queue.qsize() == 0:
                    self.backlog = False
                    self.backlog_out.send(False)
                model_name = self.model_choice()
                model = self.model_dict.get(model_name, 'eleven_turbo_v2_5')
                latency = int(self.latency())
                settings = VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style(), speed=self.speed())

                try:
                    self.audio_stream = self.client.text_to_speech.stream(
                        voice_id=self.voice_id,
                        text=text,
                        model_id=model,
                        voice_settings=settings,
                        optimize_streaming_latency=latency,
                        output_format='pcm_%d' % PcmStreamer.RATE,
                    )
                except Exception as e:
                    print('ElevenLabs API error:', e)
                    return

                try:
                    self.rate_out.send(PcmStreamer.RATE)
                    self.streamer.stream(self.audio_stream)
                except Exception as e:
                    print('ElevenLabs stream error:', e)
            finally:
                self.active = False

    def frame_task(self):
        # 'speaking' covers the audible tail: the API can finish sending
        # while the engine still has the end of the phrase to play.
        speaking = self.active or self.streamer.speaking()
        if speaking != self.previously_active:
            self.active_output.send(speaking)
            self.previously_active = speaking
