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
api_key = 'be1eae804441ec11f0fe872f82ad44f3'


def register_elevenlab_nodes():
    Node.app.register_node("eleven_lab", ElevenLabNode.factory)


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
        self.voices['David']= 'p1NETszyIlYTrbi495Pf'

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

