import numpy as np
from collections import deque
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.synth_nodes import SynthNode, synth_graph
from dpg_system.synth_core import StreamUnit
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


class ElevenLabsNode(SynthNode):
    """Text in, speech out -- as a signal, like any other source.

    The API is asked for raw 24 kHz PCM rather than MP3, so there is nothing
    to decode and no external player: each chunk becomes float samples and
    goes into a StreamUnit, the same ring stream~ uses, and the speech comes
    out of 'left out' / 'right out' for fader_out~, audio_out~, vocoder~,
    vst~ or anything else that takes a signal. Nothing sounds until one of
    those is patched, which is how every other ~ object behaves too.

    A phrase arrives from the service in a fraction of its own length, so
    the unit is set to keep everything rather than skip a backlog, and holds
    a tenth of a second before a phrase starts sounding so a slow first
    chunk does not stutter.

    The service thread only ever pushes samples into the ring, which is
    single-producer / single-consumer and needs no lock. Everything the
    node says to the patch -- 'speaking', 'sounding', 'backlog' and the
    'phrase samples' chunks -- is sent from the frame task, on the main
    thread, so a node patched to an outlet runs where nodes are meant to run.

    'speaking' means busy, from the moment a phrase is handed to the service
    until the last of it has played, and is what to gate new text on.
    'sounding' means audio is leaving the outlets right now, and is what to
    hand a listener (whisper) that should not hear the node talk, or a face
    that should move with it. 'phrase samples' is the phrase as data, 24 kHz
    float32, delivered as it arrives -- the service sends a whole phrase in
    a fraction of a second, so it is all out long before the sound -- for
    recording or whole-phrase analysis; anything that should line up with
    what is heard wants capture~ on the signal instead.
    """

    instances = []
    _stop_event = threading.Event()
    _service_thread = None
    _service_thread_lock = threading.Lock()

    RATE = 24000
    LATENCY = 0.1        # seconds held before a phrase starts sounding

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
        self.unit = StreamUnit(synth_graph.sample_rate)
        self.unit.source_rate = float(ElevenLabsNode.RATE)
        self.unit.latency = ElevenLabsNode.LATENCY
        self.unit.max_backlog = None     # speech must all be heard, in order

        self.text_input = self.add_input('text to speak', triggers_execution=True)

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
        self.add_modulation_input('level', self.unit.level_in, default_value=1.0,
                                  minimum=0.0, maximum=2.0, speed=0.01,
                                  attenuverter=False)

        self.left_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
        self.active_output = self.add_output('speaking')
        # 'speaking' is busy: a phrase has been sent off, or is still queued
        # in the ring. 'sounding' is narrower -- audio is leaving the outlets
        # this instant -- which is what a listener such as whisper wants to
        # be told to ignore, and what an animated mouth wants.
        self.sounding_out = self.add_output('sounding')
        self.backlog_out = self.add_output('backlog')
        # The phrase as data: float32 chunks at 24 kHz, delivered as they
        # arrive from the service (all of it within a fraction of a second),
        # for recording or whole-phrase analysis. Named so that nobody
        # patches it to hear the voice -- that is what the signal pair is.
        self.audio_out = self.add_output('phrase samples')
        self.audio_out.name_archive.append('audio')

        self.voice_record = None
        self.previously_active = False
        self.previously_sounding = False
        self.backlog = False
        self.voice_settings = None
        self.phrase_queue = Queue(16)
        # (outlet, value) pairs queued by the service thread, sent from the
        # frame task. deque.append / popleft are atomic under the GIL.
        self._pending_sends = deque()
        self.force_stop = False
        self._pending = b''
        ElevenLabsNode.instances.append(self)
        ElevenLabsNode._ensure_service_thread()

        self.add_switch()
        self.finish_synth_node()

    def custom_cleanup(self):
        super().custom_cleanup()
        self.force_stop = True
        self.unit.deactivate()
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except Empty:
                break
        if self in ElevenLabsNode.instances:
            ElevenLabsNode.instances.remove(self)

    @staticmethod
    def _clean(value, low, high):
        # The widgets hold 32-bit floats: 0.7 comes back as 0.69999998 and
        # 1.2 as 1.20000005, and the service checks its limits exactly, so
        # both ends of the speed range were refused. Three decimals is finer
        # than anything the settings respond to.
        return min(high, max(low, round(any_to_float(value), 3)))

    def _voice_settings(self, with_speed=False):
        settings = dict(stability=self._clean(self.stability(), 0.0, 1.0),
                        similarity_boost=self._clean(self.similarity_boost(), 0.0, 1.0),
                        style=self._clean(self.style(), 0.0, 1.0))
        if with_speed:
            settings['speed'] = self._clean(self.speed(), 0.7, 1.2)
        return VoiceSettings(**settings)

    def voice_changed(self):
        current_voice_name = self.voice_name_input()
        if current_voice_name in self.voice_dict:
            if self.client is not None:
                self.voice_id = self.voice_dict[current_voice_name]
                self.voice_settings = self._voice_settings()

    def post_creation_callback(self):
        if self.client is not None and self.voice_id is not None:
            self.voice_settings = self._voice_settings()

    def execute(self):
        if self.accept_input():
            self.text_to_speak = any_to_string(self.text_input())
            if len(self.text_to_speak) > 0:
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

    def speaking(self):
        """True while queued speech is still sounding after the API is done."""
        return self.unit.backlog > 0

    def hard_stop_streaming(self):
        self.stop_streaming()

    def stop_streaming(self):
        self.force_stop = True
        self.unit.deactivate()
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except Empty:
                break
            self.phrase_queue.task_done()

    # -- service thread ---------------------------------------------------

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty() and self.client is not None and self.voice_id is not None:
            self.active = True
            try:
                text = self.phrase_queue.get()
                if self.phrase_queue.qsize() == 0:
                    self.backlog = False
                    self._pending_sends.append((self.backlog_out, False))
                model_name = self.model_choice()
                model = self.model_dict.get(model_name, 'eleven_turbo_v2_5')
                settings = self._voice_settings(with_speed=True)
                extra = {}
                latency = int(self.latency())
                if latency > 0:
                    # The v3 models refuse the request outright if this is
                    # present, at any value -- and 0 is the default anyway,
                    # so it is only ever sent when it says something.
                    if model.startswith('eleven_v3'):
                        print("ElevenLabs: 'latency' ignored -- " + model + ' does not take optimize_streaming_latency')
                    else:
                        extra['optimize_streaming_latency'] = latency

                try:
                    self.audio_stream = self.client.text_to_speech.stream(
                        voice_id=self.voice_id,
                        text=text,
                        model_id=model,
                        voice_settings=settings,
                        output_format='pcm_%d' % ElevenLabsNode.RATE,
                        **extra,
                    )
                except Exception as e:
                    print('ElevenLabs API error:', self._api_error_text(e))
                    return

                try:
                    self.stream(self.audio_stream)
                except Exception as e:
                    print('ElevenLabs stream error:', self._api_error_text(e))
            finally:
                self.active = False

    @staticmethod
    def _api_error_text(error):
        """The service's own sentence, when there is one, rather than the
        SDK's dump of every response header around it."""
        body = getattr(error, 'body', None)
        if isinstance(body, dict):
            detail = body.get('detail', body)
            if isinstance(detail, dict) and detail.get('message'):
                return str(detail['message'])
            return str(detail)
        return str(error)

    def stream(self, audio_stream: Iterator[bytes]):
        self.force_stop = False
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
            self.unit.push(samples)
            self._pending_sends.append((self.audio_out, samples))
        if self.force_stop:
            self.unit.deactivate()
        self.force_stop = False

    # -- main thread ------------------------------------------------------

    def synth_frame_task(self):
        while self._pending_sends:
            outlet, value = self._pending_sends.popleft()
            outlet.send(value)
        # 'speaking' covers the audible tail: the API can finish sending
        # while the engine still has the end of the phrase to play.
        speaking = self.active or self.speaking()
        if speaking != self.previously_active:
            self.active_output.send(speaking)
            self.previously_active = speaking
        sounding = bool(self.unit.enabled and self.unit.playing)
        if sounding != self.previously_sounding:
            self.sounding_out.send(sounding)
            self.previously_sounding = sounding
