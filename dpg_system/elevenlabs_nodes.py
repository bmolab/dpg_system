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
from elevenlabs import client, stream, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from queue import Queue
import threading
import traceback
import time
import shutil
import subprocess
from typing import Iterator, Union
api_key = 'be1eae804441ec11f0fe872f82ad44f3'

def register_elevenlabs_nodes():
    Node.app.register_node("eleven_labs", ElevenLabsNode.factory)


def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True


class Streamer:
    def __init__(self):
        self.force_stop = False
        self.process = None

    def do_stop(self):
        self.force_stop = True

    def hard_stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception as e:
                print('ElevenLabs Streamer error', e)

    def stream(self, audio_stream: Iterator[bytes]) -> bytes:
        if not is_installed("mpv"):
            message = (
                "mpv not found, necessary to stream audio. "
                "On mac you can install it with 'brew install mpv'. "
                "On linux and windows you can install it from https://mpv.io/"
            )
            raise ValueError(message)

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.process = mpv_process

        audio = b""

        for chunk in audio_stream:
            if self.force_stop:
                audio = b""
                break
            if chunk is not None:
                if self.force_stop:
                    audio = b""
                    break
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore

                audio += chunk
                if self.force_stop:
                    audio = b""
                    break

        if mpv_process.stdin:
            mpv_process.stdin.close()
        if self.force_stop:
             mpv_process.terminate()
             self.force_stop = False
        else:
            mpv_process.wait()
        self.force_stop = False
        return audio

def service_eleven_labs():
    while True:
        for instance in ElevenLabsNode.instances:
            try:
                instance.service_queue()
            except Exception as e:
                print('service_eleven_labs:', e)
                traceback.print_exception(e)

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

        try:
            self.client = ElevenLabs(api_key=api_key)
            print(response)

            self.voices = self.client.voices.get_all()
            self.models = self.client.models.get_all()
            self.voice_name = 'David'
            self.active = False
            self.streamer = Streamer()
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
        except Exception as e:
            self.client = None
            self.voices = None
            self.models = None
            voice_names = []
            self.voice_name = ''
            self.active = False
            self.audio_stream = None
            self.voice_dict = {}

        self.voice_name_input = self.add_input('voice', widget_type='combo', default_value=self.voice_name, callback=self.voice_changed)
        self.voice_name_input.widget.combo_items = voice_names
        self.model_choice = self.add_input('model', widget_type='combo', widget_width=250, default_value="Eleven Turbo v2")
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
        self.active_output = self.add_output('speaking')
        self.backlog_out = self.add_output('backlog')

        self.voice_record = None
        self.previously_active = False
        self.backlog = False
        self.voice_settings = VoiceSettings()
        ElevenLabsNode.instances.append(self)
        self.phrase_queue = Queue(16)
        self.thread = threading.Thread(target=service_eleven_labs)
        self.thread.start()

    def voice_changed(self):
        self.voice_record = None
        current_voice_name = self.voice_name_input()
        if current_voice_name in self.voice_dict:
            if self.client is not None:
                self.voice_id = self.voice_dict[current_voice_name]
                self.voice_record = Voice(voice_id=self.voice_id, settings=VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style()))

    def post_creation_callback(self):
        if self.client is not None:
            self.voice_record = Voice(voice_id=self.voice_id, settings=VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style()))

    def execute(self):
        if self.accept_input():
            self.text_to_speak = any_to_string(self.text_input())
            if len(self.text_to_speak) > 0:
                self.add_frame_task()
                self.phrase_queue.put(self.text_to_speak)
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
                self.phrase_queue.get(block=False)
            except Exception as e:
                continue
            self.phrase_queue.task_done()

    def stop_streaming(self):
        self.streamer.do_stop()
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get(block=False)
            except Exception as e:
                continue
            self.phrase_queue.task_done()

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty() and self.client is not None:
            self.active = True
            text = self.phrase_queue.get()
            if self.phrase_queue.qsize() == 0:
                self.backlog = False
                self.backlog_out.send(False)
            model = self.model_dict[self.model_choice()]
            latency = int(self.latency())
            settings = VoiceSettings(stability=self.stability(), similarity_boost=self.similarity_boost(), style=self.style(), speed=self.speed(), latency=self.latency())

            self.audio_stream = self.client.generate(text=text, voice=self.voice_record, model=model, stream=True, optimize_streaming_latency=latency, voice_settings=settings)
            try:
                audio = self.streamer.stream(self.audio_stream)
            except Exception as e:
                print(e)
            self.active = False

    def frame_task(self):
        if self.active != self.previously_active:
            self.active_output.send(self.active)
            self.previously_active = self.active
