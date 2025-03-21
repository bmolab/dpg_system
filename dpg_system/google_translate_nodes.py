import dearpygui.dearpygui as dpg
from dpg_system.conversion_utils import *
from dpg_system.node import Node

import concurrent.futures
import requests
import re
import os
import html
import urllib.parse
import threading
from queue import Queue
import time
from google.cloud import translate_v2 as translate
# from google.cloud import storage
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError

def register_google_translate_nodes():
    Node.app.register_node('translate', GoogleTranslateNode.factory)
    Node.app.register_node('translate_api', GoogleTranslateAPINode.factory)



class EasyGoogleTranslate:

    '''
        Unofficial Google Translate API.

        This library does not need an api key or something else to use, it's free and simple.
        You can either use a string or a file to translate but the text must be equal to or less than 5000 character.
        You can split your text into 5000 characters to translate more.

        Google Translate supports 108 different languages. You can use any of them as source and target language in this application.
        If source language is not specified, it will detect source language automatically.
        This application supports multi thread translation, you can use it to translate multiple languages at once.
        Detailed language list can be found here:  https://cloud.google.com/translate/docs/languages


        Examples:
            #1: Specify default source and target language at beginning and use it any time.
                translator = GoogleTranslateRequest(
                    source_language='en',
                    target_language='de',
                    timeout=10
                )
                result = translator.translate('This is an example.')
                print(result)

            #2: Don't specify default parameters.
                translator = GoogleTranslateRequest()
                result = translator.translate('This is an example.', target_language='tr')
                print(result)

            #2: Override default parameters.
                translator = GoogleTranslateRequest(target_language='tr')
                result = translator.translate('This is an example.', target_language='fr')
                print(result)

            #4: Translate a text in multiple languages at once via multi-threading.
                translator = GoogleTranslateRequest()
                result = translator.translate(text='This is an example.', target_language=['tr', 'fr', 'de'])
                print(result)

            #5: Translate a file in multiple languages at once via multi-threading.
                translator = GoogleTranslateRequest()
                result = translator.translate_file(file_path='text.txt', target_language=['tr', 'fr', 'de'])
                print(result)

    '''

    def __init__(self, source_language='auto', target_language='tr', timeout=5):
        self.source_language = source_language
        self.target_language = target_language
        self.timeout = timeout
        self.pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'

    def make_request(self, target_language, source_language, text, timeout):
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = 'https://translate.google.com/m?tl=%s&sl=%s&q=%s'%(target_language, source_language, escaped_text)
        result = None
        try:
            response = requests.get(url, timeout=timeout)
            result = response.text.encode('utf8').decode('utf8')
            result = re.findall(self.pattern, result)
        except Exception as e:
            print('translate request', e)

        if not result:
            # print('\nError: Unknown error.')
            # f = open('error.txt')
            # f.write(response.text)
            # f.close()
            # exit(0)
            return None
        return html.unescape(result[0])

    def translate(self, text, target_language='', source_language='', timeout=''):
        if not target_language:
            target_language = self.target_language
        if not source_language:
            source_language = self.source_language
        if not timeout:
            timeout = self.timeout
        if len(text) > 5000:
            print('\nError: It can only detect 5000 characters at once. (%d characters found.)'%(len(text)))
            exit(0)
        if type(target_language) is list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.make_request, target, source_language, text, timeout) for target in target_language]
                return_value = [f.result() for f in futures]
                return return_value
        return self.make_request(target_language, source_language, text, timeout)

    def translate_file(self, file_path, target_language='', source_language='', timeout=''):
        if not os.path.isfile(file_path):
            print('\nError: The file or path is incorrect.')
            exit(0)
        f = open(file_path)
        text = self.translate(f.read(), target_language, source_language, timeout)
        f.close()
        return text

languages = {
'auto':'auto',
'Afrikaans':'af',
'Albanian':'sq',
'Amharic':'am',
'Arabic':'ar',
'Armenian':'hy',
'Assamese':'as',
'Aymara':'ay',
'Azerbaijani':'az',
'Bambara':'bm',
'Basque':'eu',
'Belarusian':'be',
'Bengali':'bn',
'Bhojpuri':'bho',
'Bosnian':'bs',
'Bulgarian':'bg',
'Catalan':'ca',
'Cebuano':'ceb',
'Chinese (Simplified)':'zh-CN',
'Chinese (Traditional)':'zh-TW',
'Corsican':'co',
'Croatian':'hr',
'Czech':'cs',
'Danish':'da',
'Dhivehi':'dv',
'Dogri':'doi',
'Dutch':'nl',
'English':'en',
'Esperanto':'eo',
'Estonian':'et',
'Ewe':'ee',
'Filipino (Tagalog)':'fil',
'Finnish':'fi',
'French':'fr',
'Frisian':'fy',
'Galician':'gl',
'Georgian':'ka',
'German':'de',
'Greek':'el',
'Guarani':'gn',
'Gujarati':'gu',
'Haitian Creole':'ht',
'Hausa':'ha',
'Hawaiian':'haw',
'Hebrew':'he or iw',
'Hindi':'hi',
'Hmong':'hmn',
'Hungarian':'hu',
'Icelandic':'is',
'Igbo':'ig',
'Ilocano':'ilo',
'Indonesian':'id',
'Irish':'ga',
'Italian':'it',
'Japanese':'ja',
'Javanese':'jv or jw',
'Kannada':'kn',
'Kazakh':'kk',
'Khmer':'km',
'Kinyarwanda':'rw',
'Konkani':'gom',
'Korean':'ko',
'Krio':'kri',
'Kurdish':'ku',
'Kurdish (Sorani)':'ckb',
'Kyrgyz':'ky',
'Lao':'lo',
'Latin':'la',
'Latvian':'lv',
'Lingala':'ln',
'Lithuanian':'lt',
'Luganda':'lg',
'Luxembourgish':'lb',
'Macedonian':'mk',
'Maithili':'mai',
'Malagasy':'mg',
'Malay':'ms',
'Malayalam':'ml',
'Maltese':'mt',
'Maori':'mi',
'Marathi':'mr',
'Meiteilon (Manipuri)':'mni-Mtei',
'Mizo':'lus',
'Mongolian':'mn',
'Myanmar (Burmese)':'my',
'Nepali':'ne',
'Norwegian':'no',
'Nyanja (Chichewa)':'ny',
'Odia (Oriya)':'or',
'Oromo':'om',
'Pashto':'ps',
'Persian':'fa',
'Polish':'pl',
'Portuguese (Portugal, Brazil)':'pt',
'Punjabi':'pa',
'Quechua':'qu',
'Romanian':'ro',
'Russian':'ru',
'Samoan':'sm',
'Sanskrit':'sa',
'Scots Gaelic':'gd',
'Sepedi':'nso',
'Serbian':'sr',
'Sesotho':'st',
'Shona':'sn',
'Sindhi':'sd',
'Sinhala (Sinhalese)':'si',
'Slovak':'sk',
'Slovenian':'sl',
'Somali':'so',
'Spanish':'es',
'Sundanese':'su',
'Swahili':'sw',
'Swedish':'sv',
'Tagalog (Filipino)':'tl',
'Tajik':'tg',
'Tamil':'ta',
'Tatar':'tt',
'Telugu':'te',
'Thai':'th',
'Tigrinya':'ti',
'Tsonga':'ts',
'Turkish':'tr',
'Turkmen':'tk',
'Twi (Akan)':'ak',
'Ukrainian':'uk',
'Urdu':'ur',
'Uyghur':'ug',
'Uzbek':'uz',
'Vietnamese':'vi',
'Welsh':'cy',
'Xhosa':'xh',
'Yiddish':'yi',
'Yoruba':'yo',
'Zulu':'zu'
}

def service_google_translate():
    while True:
        for instance in GoogleTranslateNode.instances:
            try:
                instance.service_queue()
            except Exception as e:
                print('service_google_translate')
                traceback.print_exception(e)

        time.sleep(0.1)


class GoogleTranslateNode(Node):
    instances = []
    @staticmethod
    def factory(name, data, args=None):
        node = GoogleTranslateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.translator = None
        self.source_language = 'en'
        self.target_language = 'fr'

        if len(args) > 0:
            self.source_language = args[0]
        if len(args) > 1:
            self.target_language = args[1]
        self.translator = EasyGoogleTranslate(source_language=self.source_language, target_language=self.target_language, timeout=10)
        self.input = self.add_input('text in', triggers_execution=True)

        self.source_language_input = self.add_input('source language', widget_type='combo', default_value='English', callback=self.language_changed)
        self.source_language_input.widget.combo_items = list(languages.keys())
        self.dest_language_input = self.add_input('dest language', widget_type='combo', default_value='French', callback=self.language_changed)
        self.dest_language_input.widget.combo_items = list(languages.keys())

        self.queue_input = self.add_input('use queue', widget_type='checkbox', default_value=True)
        self.timeout = self.add_input('time out', widget_type='drag_float', default_value=5.0)
        self.output = self.add_output('translation out')

        GoogleTranslateNode.instances.append(self)
        self.active = False
        self.phrase_queue = Queue(16)
        self.thread = threading.Thread(target=service_google_translate)
        self.thread.start()

    def language_changed(self):
        src = self.source_language_input()
        dst = self.dest_language_input()
        self.source_language = languages[src]
        self.target_language = languages[dst]
        self.translator = None
        self.translator = EasyGoogleTranslate(source_language=self.source_language,
                                              target_language=self.target_language, timeout=10)

    def execute(self):
        self.text_to_translate = any_to_string(self.input())
        if len(self.text_to_translate) > 0:
            self.phrase_queue.put(self.text_to_translate)

        # if self.translator is not None:
        #     result = self.translator.translate(any_to_string(self.input()))
        #     self.output.send(result)

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty() and self.translator is not None:
            self.active = True
            timeout = self.timeout()
            self.translator.timeout = timeout
            if self.queue_input():
                text = self.phrase_queue.get()
                timeout = self.timeout()
                self.translator.timeout = timeout
                result = self.translator.translate(text)
                if result is not None:
                    self.output.send(result)
            else:
                text = ''
                size = self.phrase_queue.qsize()
                for i in range(size):
                    text = self.phrase_queue.get()
                result = self.translator.translate(text)
                if result is not None:
                    self.output.send(result)
            self.active = False


#
# translator = EasyGoogleTranslate(source_language='ko', target_language='en', timeout=10)
# print('start')
# result = translator.translate('사과의 의미로 내가 쏠게 피자랑 치킨도 사줘 그래야 용서해 줄 거야 알겠어 피자랑 치킨도 살게 유진 너는 별일 없어 아 맞다 나 이직했어 명함 줄게 우와 여기로 옮긴 거야 너 전부터 여기서 일하고 싶다고 했었잖아 축하해 고마워 나 집도 큰 데로 옮겼어 그러니까 서울 오면 우리 집에서 자 알겠어 그렇게 할게')
# print(result)

def service_google_translate_api():
    while True:
        for instance in GoogleTranslateAPINode.instances:
            try:
                instance.service_queue()
            except Exception as e:
                print('service_google_translate_api')
                traceback.print_exception(e)

        time.sleep(0.1)

class GoogleTranslateAPINode(Node):
    instances = []
    @staticmethod
    def factory(name, data, args=None):
        node = GoogleTranslateAPINode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.translator = None
        self.source_language = 'ko'
        self.target_language = 'en'

        if len(args) > 0:
            self.source_language = args[0]
        if len(args) > 1:
            self.target_language = args[1]

        self.translate_client = translate.Client()

        self.input = self.add_input('text in', triggers_execution=True)

        self.source_language_input = self.add_input('source language', widget_type='combo', default_value='English')
        self.source_language_input.widget.combo_items = list(languages.keys())
        self.dest_language_input = self.add_input('dest language', widget_type='combo', default_value='French')
        self.dest_language_input.widget.combo_items = list(languages.keys())

        self.queue_input = self.add_input('use queue', widget_type='checkbox', default_value=True)

        self.output = self.add_output('translation out')

        GoogleTranslateAPINode.instances.append(self)
        self.active = False
        self.phrase_queue = Queue(16)
        self.thread = threading.Thread(target=service_google_translate_api)
        self.thread.start()

    def execute(self):
        self.text_to_translate = any_to_string(self.input())
        if len(self.text_to_translate) > 0:
            self.phrase_queue.put(self.text_to_translate)

    def translate(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        src = self.source_language_input()
        dst = self.dest_language_input()
        self.source_language = languages[src]
        self.target_language = languages[dst]
        if self.source_language == 'auto':
            result = self.translate_client.translate(text, target_language=self.target_language)
        else:
            result = self.translate_client.translate(text, target_language=self.target_language, source_language=self.source_language)
        return result["translatedText"]

    def service_queue(self):
        if not self.active and not self.phrase_queue.empty() and self.translate_client is not None:
            self.active = True
            if self.queue_input():
                text = self.phrase_queue.get()
                result = self.translate(text)

                if result is not None:
                    self.output.send(result)
            else:
                text = ''
                size = self.phrase_queue.qsize()
                for i in range(size):
                    text = self.phrase_queue.get()
                result = self.translate(text)
                if result is not None:
                    self.output.send(result)
            self.active = False


