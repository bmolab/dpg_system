import requests  # you have to install this library, with pip for example
import io
from pydub import AudioSegment
from pydub.playback import play
import dearpygui.dearpygui as dpg
import math
import numpy as np
import json
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from elevenlabs import client, stream, generate, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from queue import Queue
import threading
import time
api_key = 'be1eae804441ec11f0fe872f82ad44f3'


def register_elevenlab_nodes():
    Node.app.register_node("eleven_lab", ElevenLabNode.factory)
    Node.app.register_node("eleven_labs", ElevenLabsNode.factory)


class ElevenLabNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ElevenLabNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.text_input = self.add_input('text to speak', triggers_execution=True)

        self.headers = {
            'xi-api-key': api_key,
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        self.url = 'https://api.elevenlabs.io/v1/text-to-speech/'

        voices_url = 'https://api.elevenlabs.io/v1/voices'
        response = requests.get(voices_url)
        response_string = response.content.decode('utf-8')
        rj = json.loads(response_string)
        voice_count = len(rj['voices'])
        self.voices = {}
        for i in range(voice_count):
            name = rj['voices'][i]['name']
            id = rj['voices'][i]['voice_id']
            self.voices[name] = id
        self.voices['David'] = 'p1NETszyIlYTrbi495Pf'

        voice_names = list(self.voices.keys())
        self.voice = self.add_input('voice', widget_type='combo')
        self.voice.widget.combo_items = voice_names

        self.stability = self.add_input('stability', widget_type='drag_float', default_value=.02)
        self.similarity_boost = self.add_input('similarity_boost', widget_type='drag_float', default_value=1.0)

    def execute(self):
        text_to_speak = any_to_string(self.text_input())
        current_voice = self.voice()
        data = {
            'text': text_to_speak,
            'voice_settings':
            {
                'stability': self.stability(),
                'similarity_boost': self.similarity_boost()
            }
        }
        response = requests.post(self.url + self.voices[current_voice], json=data, headers=self.headers)
        audio_data = response.content
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        play(audio_segment)


def service_eleven_labs():
    while True:
        for instance in ElevenLabsNode.instances:
            try:
                instance.service_queue()
            except Exception as e:
                print('service_eleven_labs', e)
        time.sleep(0.1)


class ElevenLabsNode(Node):
    instances = []
    @staticmethod
    def factory(name, data, args=None):
        node = ElevenLabsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.text_input = self.add_input('text to speak', triggers_execution=True)

        self.client = ElevenLabs(api_key=api_key)
        self.voices = self.client.voices.get_all()
        self.models = self.client.models.get_all()
        self.voice_name = 'David'
        self.active = False
        self.audio_stream = None
        if len(args) > 0:
            voice_name = any_to_string(args[0])
        else:
            voice_name = self.voice_name
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
        self.voice_name_input = self.add_input('voice', widget_type='combo', default_value=self.voice_name, callback=self.voice_changed)
        self.voice_name_input.widget.combo_items = voice_names
        self.model_choice = self.add_input('model', widget_type='combo', widget_width=250, default_value="Eleven Turbo v2")
        self.model_choice.widget.combo_items = list(self.model_dict.keys())
        self.stability = self.add_input('stability', widget_type='drag_float', default_value=.02, callback=self.voice_changed)
        self.similarity_boost = self.add_input('similarity_boost', widget_type='drag_float', default_value=1.0, callback=self.voice_changed)
        self.style = self.add_input('style exaggeration', widget_type='drag_float', default_value=0.5, callback=self.voice_changed)
        self.latency = self.add_input('latency', widget_type='combo', default_value='0')
        self.latency.widget.combo_items = ['0', '1', '2', '3', '4']
        self.stop_streaming_input = self.add_input('stop', widget_type='button', callback=self.stop_streaming)
        self.accept_input = self.add_input('accept input', widget_type='checkbox')
        self.active_output = self.add_output('speaking')
        self.voice_record = None
        self.previously_active = False
        ElevenLabsNode.instances.append(self)
        self.phrase_queue = Queue(16)
        self.thread = threading.Thread(target=service_eleven_labs)
        self.thread.start()

    def voice_changed(self):
        self.voice_record = None
        current_voice_name = self.voice_name_input()
        if current_voice_name in self.voice_dict:
            self.voice_id = self.voice_dict[current_voice_name]
        self.voice_record = Voice(voice_id=self.voice_id, settings=VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style()))

    def post_creation_callback(self):
        self.voice_record = Voice(voice_id=self.voice_id, settings=VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style()))

    def execute(self):
        if self.accept_input():
            self.text_to_speak = any_to_string(self.text_input())
            self.add_frame_task()
            self.phrase_queue.put(self.text_to_speak)

    def stop_streaming(self):
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get(block=False)
            except Exception as e:
                continue
            self.phrase_queue.task_done()

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty():
            self.active = True
            text = self.phrase_queue.get()
            model = self.model_dict[self.model_choice()]
            latency = int(self.latency())
            self.audio_stream = generate(api_key=api_key, text=text, voice=self.voice_record, model=model, stream=True, latency=latency)
            audio = stream(self.audio_stream)
            self.active = False


    def frame_task(self):
        if self.active != self.previously_active:
            self.active_output.send(self.active)
            self.previously_active = self.active
