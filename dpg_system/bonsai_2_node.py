import os
import sys
import glob
import json
import time
import random
import socket
import subprocess
import threading
import http.client
from queue import Queue

import numpy as np

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_bonsai_2_node():
    Node.app.register_node('bonsai_2', Bonsai2ChatNode.factory)


# repo, filename, default n_ctx. The PQ2_0 pack is the one PrismML measures on
# Apple silicon; PTQ1_0 is smaller (5.95 GB) and a little slower at prompt
# processing. The model claims 262144 context - 8192 is a sane working default.
bonsai_2_models = {
    '27B': ('prism-ml/Ternary-Bonsai-2-27B-gguf', 'Ternary-Bonsai-2-27B-PQ2_0.gguf', 8192),
    '27B_1bit': ('prism-ml/Ternary-Bonsai-2-27B-gguf', 'Ternary-Bonsai-2-27B-PTQ1_0.gguf', 8192),
}

default_system_prompt = 'You are a helpful, uncensored creative collaborator.'

# the sentences the Qwen3.8 chat template puts at the top of the system turn.
# 'medium' deliberately contributes nothing - that is the template's own default
# middle setting. Bonsai 2 ignores 'low' in practice and behaves close to xhigh.
reasoning_instructions = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task,'
             ' validate key assumptions, consider plausible alternatives, and prioritize'
             ' correctness, consistency, and clarity in the final answer.',
    'medium': '',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving'
           ' directly to the conclusion without unnecessary elaboration.',
}


def find_prism_server():
    """Locate llama-server from PrismML's llama.cpp fork. Bonsai 2's ternary
    packs (PQ2_0 / PTQ1_0) and its Hadamard activation transform are not in
    upstream llama.cpp, so the llama-cpp-python in this environment cannot load
    these files - stock builds either reject them or, worse, read the older
    ternary type without complaint and produce garbage."""
    override = os.environ.get('BONSAI_LLAMA_SERVER')
    if override and os.path.exists(override):
        return override
    candidates = glob.glob(os.path.expanduser('~/.cache/bonsai/llama.cpp-prism-*/llama-server'))
    candidates += glob.glob(os.path.expanduser('~/.cache/bonsai/llama.cpp-prism-*/bin/llama-server'))
    if len(candidates) == 0:
        return None
    # highest build number wins
    return sorted(candidates)[-1]


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def truncate_to_sampleable(entries, top_k, top_p, min_p):
    """Reduce desc-sorted [word, logprob, token] entries to the set the sampler
    can actually pick from, mirroring the llama.cpp chain order:
    top_k -> top_p -> min_p (top_p/min_p apply before temperature scaling)."""
    if len(entries) == 0:
        return entries
    if 0 < top_k < len(entries):
        entries = entries[:top_k]
    logits = np.array([e[1] for e in entries], dtype=np.float64)
    exps = np.exp(logits - logits[0])
    probs = exps / exps.sum()
    if 0.0 < top_p < 1.0:
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, top_p)) + 1
        entries = entries[:cut]
        probs = probs[:cut] / probs[:cut].sum()
    if min_p > 0.0:
        keep = probs >= min_p * probs[0]
        entries = [e for e, k in zip(entries, keep) if k]
    return entries


class BonsaiServer:
    """A llama-server child process holding one model, shared by every bonsai_2
    node that asks for the same file. The model lives out of process because it
    needs the fork's binary; the node drives it token by token over HTTP on
    localhost. One token per request sounds wasteful but cache_prompt means the
    server re-uses the whole KV prefix, so the only added cost is a loopback
    round trip against a model that generates at tens of milliseconds a token."""

    servers = {}
    servers_lock = threading.Lock()

    @classmethod
    def acquire(cls, model_path, n_ctx, n_gpu_layers, verbose=False):
        key = (model_path, int(n_ctx), int(n_gpu_layers))
        with cls.servers_lock:
            server = cls.servers.get(key)
            if server is None:
                server = BonsaiServer(model_path, int(n_ctx), int(n_gpu_layers), verbose)
                cls.servers[key] = server
            server.users += 1
        return server

    @classmethod
    def release(cls, server):
        if server is None:
            return
        with cls.servers_lock:
            server.users -= 1
            # the process is left running: reloading 7 GB costs a minute and
            # patches routinely delete and rebuild nodes. shutdown_all() at
            # app exit is what actually stops it.

    @classmethod
    def shutdown_all(cls):
        with cls.servers_lock:
            for server in cls.servers.values():
                server.stop()
            cls.servers.clear()

    def __init__(self, model_path, n_ctx, n_gpu_layers, verbose=False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.users = 0
        self.verbose = verbose
        self.port = free_port()
        self.process = None
        self.connection = None
        self.connection_lock = threading.Lock()
        self.ready = False
        self.token_pieces = {}
        self.start()

    # ------------------------------------------------------------- process

    def start(self):
        binary = find_prism_server()
        if binary is None:
            raise RuntimeError(
                'bonsai_2: PrismML llama-server not found. Bonsai 2 needs the fork'
                ' at https://github.com/PrismML-Eng/llama.cpp - unpack a release into'
                ' ~/.cache/bonsai/llama.cpp-prism-<build>/ or set BONSAI_LLAMA_SERVER')
        command = [
            binary,
            '-m', self.model_path,
            '-c', str(self.n_ctx),
            '-ngl', str(self.n_gpu_layers_arg()),
            '-fa', 'on',
            '--host', '127.0.0.1',
            '--port', str(self.port),
            '-np', '1',
            '--no-webui',
        ]
        print('bonsai_2: starting', os.path.basename(binary), 'on port', self.port)
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self.drain_log, daemon=True).start()
        self.wait_until_ready()

    def n_gpu_layers_arg(self):
        return 99 if self.n_gpu_layers < 0 else self.n_gpu_layers

    def drain_log(self):
        # the loading lines are worth seeing; once the model is up the server
        # logs a line per request, which would swamp the console
        for line in self.process.stderr:
            if self.verbose or not self.ready:
                sys.stdout.write('bonsai_2 server: ' + line)

    def wait_until_ready(self, timeout=600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('bonsai_2: llama-server exited during startup'
                                   ' (code %s)' % self.process.returncode)
            try:
                health = self.request('GET', '/health', None)
                if health.get('status') == 'ok':
                    self.ready = True
                    print('bonsai_2: server ready')
                    return
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError('bonsai_2: llama-server did not become ready in time')

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.close_connection()
        self.process = None

    # ---------------------------------------------------------------- http

    def close_connection(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

    def request(self, method, path, payload, timeout=600):
        body = None if payload is None else json.dumps(payload)
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        with self.connection_lock:
            for attempt in (0, 1):
                try:
                    if self.connection is None:
                        self.connection = http.client.HTTPConnection('127.0.0.1', self.port,
                                                                     timeout=timeout)
                    self.connection.request(method, path, body=body, headers=headers)
                    response = self.connection.getresponse()
                    data = response.read()
                    if response.status != 200:
                        raise RuntimeError('bonsai_2: server returned %d: %s'
                                           % (response.status, data[:200]))
                    return json.loads(data)
                except (http.client.HTTPException, ConnectionError, OSError):
                    # a kept-alive connection can be dropped between tokens
                    self.close_connection()
                    if attempt == 1:
                        raise

    def tokenize(self, text, add_special=False):
        result = self.request('POST', '/tokenize',
                              {'content': text, 'add_special': add_special})
        return result.get('tokens', [])

    def detokenize(self, tokens):
        result = self.request('POST', '/detokenize', {'tokens': [int(t) for t in tokens]})
        return result.get('content', '')

    def token_piece(self, token):
        # only used for display lists; the completion response already carries
        # the text of every token it generates
        token = int(token)
        piece = self.token_pieces.get(token)
        if piece is None:
            piece = self.detokenize([token])
            self.token_pieces[token] = piece
        return piece

    def completion(self, payload):
        return self.request('POST', '/completion', payload)

    def stream_completion(self, payload, timeout=600):
        """Yield one decoded event per generated token. This gets its own
        connection rather than the shared keep-alive one, so abandoning a
        generation mid-flight can just drop the socket - which is also how the
        server is told to stop - without leaving the shared connection in a
        half-read state."""
        payload = dict(payload)
        payload['stream'] = True
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        try:
            connection.request('POST', '/completion', body=json.dumps(payload),
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError('bonsai_2: server returned %d on stream'
                                   % response.status)
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data: '):
                    continue
                yield json.loads(line[6:])
        finally:
            try:
                connection.close()
            except Exception:
                pass



class Bonsai2ChatNode(Node):
    bonsai_nodes = []

    @staticmethod
    def factory(name, data, args=None):
        node = Bonsai2ChatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.server = None
        self.model_key = '27B'
        # an optional arg can pick a pack or point straight at a .gguf
        self.model_path = None
        if args is not None and len(args) > 0:
            candidate = any_to_string(args[0])
            if candidate.endswith('.gguf'):
                self.model_path = candidate
            elif candidate in bonsai_2_models:
                self.model_key = candidate

        self.system_prompt = default_system_prompt
        self.prompt = ''
        self.new_system_prompt = True
        self.do_reset = False
        self.preprompt = ''
        self.streaming_prompt = ''
        self.seed = 3701
        self.thread = None
        self.last_generated_token = None

        # token ids resolved once the server is up
        self.eos_token = -1
        self.end_of_turn_token = -1
        self.think_open_token = -1
        self.think_close_token = -1
        self.stop_tokens = []
        self.return_token = -1
        self.double_return_token = -1

        # our mirror of the server's kv cache: every request sends this list as
        # the prompt and cache_prompt re-uses the matching prefix
        self.context_tokens = []
        self.prompt_tokens = []
        self.response_start = 0
        self.answer_start = 0

        # a user turn opened before the words have all arrived
        self.spoken_turn_open = False
        self.spoken_turn_start = 0
        self.spoken_text_start = 0
        self.spoken_turn_new_system = True
        self.prefill_lock = threading.Lock()
        self.prefill_wanted = threading.Event()
        self.prefill_thread = None
        self.prefill_stop = False
        self.starting_server = False

        self.on_off = self.add_input('on / off', widget_type='checkbox', default_value=False, triggers_execution=True)
        self.step_input = self.add_input('step (+-1)', callback=self.initiate_step)
        self.choice_input = self.add_input('choice (+-1)', callback=self.choose_from_possibilities)

        self.reset_input = self.add_input('reset', widget_type='button', callback=self.set_do_reset)
        self.system_prompt_input = self.add_input('system_prompt', default_value=self.system_prompt,
                                                  callback=self.system_prompt_received)
        self.prompt_input = self.add_input('prompt', callback=self.prompt_received)
        self.pre_prompt_input = self.add_input('pre-prompt', callback=self.preprompt_received)
        self.streaming_prompt_input = self.add_input('streaming_prompt', default_value='', callback=self.streaming_prompt_received)
        self.display_ui_input = self.add_input('ui from window', callback=self.handle_ui)
        self.force_stop = self.add_input('hard interrupt', widget_type='button', callback=self.hard_stop)
        self.interrupt_input = self.add_input('interrupt', widget_type='button', callback=self.stop)
        self.polite_stop_input = self.add_input('polite_stop', widget_type='button', callback=self.polite_stop)
        self.seed_input = self.add_input('seed', widget_type='drag_int', default_value=self.seed, callback=self.set_seed)
        self.temperature = self.add_input('temperature', widget_type='slider_float', widget_width=150, min=0.0, max=10.0, default_value=1.0, callback=self.temp_changed)
        self.top_k = self.add_input('top_k', widget_type='drag_int', default_value=20)
        self.top_p = self.add_input('top_p', widget_type='drag_float', default_value=0.95)
        self.min_p = self.add_input('min_p', widget_type='drag_float', default_value=0.0)
        self.repeat_penalty = self.add_input('repeat_penalty', widget_type='slider_float', default_value=1.0)
        self.max_tokens = self.add_input('max_tokens', widget_type='drag_int', default_value=-1)
        self.target_length = self.add_input('target_length', widget_type='drag_int', default_value=50)
        self.slow_down = self.add_input('slowdown', widget_type='slider_float', widget_width=150, min=0, max=1.0, default_value=0)

        # off by default, unlike gemma_4: this model writes markdown, and an
        # asterisk in its output is far more often bold or a bullet than a
        # roleplay *action*
        self.separate_actions = self.add_input('separate_actions', widget_type='checkbox', default_value=False)
        self.display_mode = self.add_input('display_mode', widget_type='combo', default_value='temperature')
        self.display_mode.widget.combo_items = ['temperature', 'entropy', 'probability', 'unnormed_probability']
        self.sigmoid_scaler = self.add_input('sigmoid scaler', widget_type='drag_float', default_value=0.2)
        self.sigmoid_offset = self.add_input('sigmoid offset', widget_type='drag_float', default_value=0)
        # eval incoming text token-by-token, scoring each token against the
        # model's standing prediction (in/out-of-distribution measure)
        self.score_input = self.add_input('score_incoming_text', widget_type='checkbox', default_value=False)
        self.reset_score_input = self.add_input('reset_input_score', widget_type='button', callback=self.reset_input_score)

        self.show_probs = self.add_input('show_probs', widget_type='checkbox', default_value=False, callback=self.show_probs_changed)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_text)

        default_ctx = bonsai_2_models[self.model_key][2] if self.model_path is None else 8192
        self.n_ctx = self.add_option('n_ctx', widget_type='drag_int', default_value=default_ctx)
        self.n_gpu_layers = self.add_option('n_gpu_layers', widget_type='drag_int', default_value=-1)
        self.thinking = self.add_option('thinking', widget_type='checkbox', default_value=False,
                                        callback=self.thinking_changed)
        self.reasoning_effort = self.add_option('reasoning_effort', widget_type='combo',
                                                default_value='xhigh', callback=self.thinking_changed)
        self.reasoning_effort.widget.combo_items = ['xhigh', 'medium', 'low']
        # off: layout token list stays aligned with context tokens only for the
        # visible text, so back-stepping will pass through invisible thought tokens
        self.thinking_in_layout = self.add_option('thinking in layout', widget_type='checkbox', default_value=True)
        # how hard the end-of-turn ramp pushes, in log-probability units. the
        # in-process gemma node scales this against the top logit; over http we
        # only see log-probabilities, so the push is an absolute amount
        self.stop_bias = self.add_option('stop_bias', widget_type='drag_float', default_value=8.0)
        self.server_verbose = self.add_option('server_log', widget_type='checkbox', default_value=False)
        # tokens taken per request while free-running; 1 asks one at a time,
        # which is about 12% slower but re-reads the sampling controls every token
        self.stream_chunk = self.add_option('stream_chunk', widget_type='drag_int', default_value=8)
        self.keep_thinking = self.add_option('keep thinking in context', widget_type='checkbox',
                                             default_value=False)
        # read the prompt into the model as the words arrive rather than all at
        # once on submit. Nothing happens until the node is switched on.
        self.preread = self.add_option('read prompt as it arrives', widget_type='checkbox',
                                       default_value=True)

        self.output = self.add_output('output')
        self.thinking_out = self.add_output('thinking')
        self.output_end_of_text = self.add_output('end')
        self.token_out = self.add_output('token_out')
        self.layout_out = self.add_output('layout_out')
        self.actions_out = self.add_output('actions_out')
        self.active_out = self.add_output('active')
        self.input_score_out = self.add_output('input_token_score')
        self.input_cumulative_out = self.add_output('input_cumulative_score')

        self.new_response = True
        self.queue = Queue(maxsize=16)
        self.active = False
        self.stepping = False
        self.in_step = False
        self.last_key_was_enter = False
        self.suppress_prompt_layout = False
        self.in_thinking = False
        self.in_action = False
        self.pending_utf8 = b''
        self.take_step = 0
        self.stopping = False
        self.force_stop_now = False
        self.force_stop_at_next = False
        self.possible_stop = False

        self.choices = []
        self.poss_dict = []
        self.chosen_index = 0
        self.entropy = 0.1
        self.probability = 0.0
        self.unnormed_probability = 0.0
        self.input_logprob_sum = 0.0
        self.input_unnormed_sum = 0.0
        self.input_token_count = 0

        Bonsai2ChatNode.bonsai_nodes.append(self)
        self.add_frame_task()

    # --------------------------------------------------------------- model

    def ensure_server(self):
        if self.server is not None:
            return self.server
        path = self.model_path
        if path is None or not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            repo_id, filename = bonsai_2_models[self.model_key][:2]
            print('bonsai_2: fetching', filename, '(cached after first download)')
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        server = BonsaiServer.acquire(path, any_to_int(self.n_ctx()),
                                      any_to_int(self.n_gpu_layers()),
                                      verbose=self.server_verbose())
        self.server = server
        if self.eos_token == -1:
            # Bonsai 2 keeps Qwen3.8's turn structure:
            #   <|im_start|>role\n ... <|im_end|>\n
            # with thinking in a <think> ... </think> block at the head of the
            # model's turn - so the block is opened by the generation prompt,
            # not by the model
            self.end_of_turn_token = server.tokenize('<|im_end|>', add_special=False)[-1]
            self.eos_token = server.tokenize('<|endoftext|>', add_special=False)[-1]
            self.think_open_token = server.tokenize('<think>', add_special=False)[-1]
            self.think_close_token = server.tokenize('</think>', add_special=False)[-1]
            self.stop_tokens = [self.eos_token, self.end_of_turn_token]
            self.return_token = server.tokenize('\n', add_special=False)[-1]
            self.double_return_token = server.tokenize('\n\n', add_special=False)[-1]
        return server

    # ------------------------------------------------------------------ ui

    def handle_ui(self):
        input_data = self.display_ui_input()
        if type(input_data) == list:
            if input_data[0] == 'key':
                key = any_to_int(input_data[1])
                if key in (257, 335):   # return / numpad enter
                    if self.last_key_was_enter:
                        self.last_key_was_enter = False
                        self.submit_streaming_prompt()
                    else:
                        self.last_key_was_enter = True
                        self.handle_streaming_prompt('\n')
                    return
                self.last_key_was_enter = False
                if key < 256:
                    # space is the pause gesture only when no text is being composed
                    if (key == 32 and self.on_off() and self.active
                            and len(self.streaming_prompt) == 0):
                        self.on_off.set(0)
                        return
                    self.handle_streaming_prompt(chr(key))
                if key == 262:      # right arrow - step forward
                    if self.on_off():
                        self.on_off.set(0)
                    self.stepping = True
                    self.take_step = 1
                elif key == 263:    # left arrow - step back
                    if self.on_off():
                        self.on_off.set(0)
                    self.stepping = True
                    self.take_step = -1
                elif key == 264:    # down arrow - previous possibility / scroll
                    if self.show_probs():
                        self.chosen_index -= 1
                        self.choose_possibility(self.chosen_index)
                    else:
                        self.layout_out.send(['scroll_down'])
                elif key == 265:    # up arrow - next possibility / scroll
                    if self.show_probs():
                        self.chosen_index += 1
                        self.choose_possibility(self.chosen_index)
                    else:
                        self.layout_out.send(['scroll_up'])
                elif key == 259:    # backspace
                    if len(self.streaming_prompt) > 0:
                        # editing composed text: not a step-back gesture
                        self.streaming_prompt = self.streaming_prompt[:-1]
                        self.layout_out.send(['backspace_streaming_prompt'])
                    else:
                        if self.on_off():
                            self.on_off.set(0)
                        if self.in_step:
                            self.take_step = -1
                        else:
                            self.back_step(1)

    def choose_possibility(self, poss_index):
        self.chosen_index = poss_index
        if self.chosen_index < 0:
            self.chosen_index = 0
        max_choice = len(self.poss_dict)
        if max_choice == 0:
            return
        if self.chosen_index >= max_choice:
            self.chosen_index = max_choice - 1
        self.layout_out.send(['choose', self.chosen_index])
        self.output.send('<backspace>')
        self.output.send(self.poss_dict[self.chosen_index][0])
        self.swap_last_token(self.poss_dict[self.chosen_index][2])

    def choose_from_possibilities(self):
        chooser = self.choice_input()
        if chooser == 0 or len(self.poss_dict) == 0:
            return
        if chooser > 0:
            self.chosen_index += 1
            if self.chosen_index >= len(self.poss_dict):
                self.chosen_index = len(self.poss_dict) - 1
        elif chooser < 0:
            self.chosen_index -= 1
            if self.chosen_index < 0:
                self.chosen_index = 0
        self.layout_out.send(['choose', self.chosen_index])
        self.output.send('<backspace>')
        self.output.send(self.poss_dict[self.chosen_index][0])
        self.swap_last_token(self.poss_dict[self.chosen_index][2])

    def swap_last_token(self, token):
        # picking a different candidate rewrites the tail of our token list;
        # the next request re-uses everything before it from the server's cache
        self.last_generated_token = token
        if len(self.context_tokens) > 0:
            self.context_tokens[-1] = int(token)

    # ------------------------------------------------------------ execution

    def execute(self):
        if self.active_input != self.on_off:
            if not self.active:
                self.process_next_prompt()
        else:
            if self.on_off():
                self.stepping = False
                self.start_server_early()
            else:
                self.stepping = True

    def start_server_early(self):
        """Switching on loads the model, rather than leaving it until the first
        prompt arrives. That is what the toggle is for, and it is also what lets
        the FIRST utterance be read as it arrives - there is nothing to read
        into until the server exists."""
        if self.server is not None or self.starting_server:
            return
        self.starting_server = True

        def load():
            try:
                self.ensure_server()
            except Exception as e:
                print('bonsai_2:', e)
            finally:
                self.starting_server = False

        threading.Thread(target=load, daemon=True).start()

    def frame_task(self):
        if self.do_reset and not self.active:
            self.reset()
        self.process_next_prompt()

    def process_next_prompt(self):
        if not self.active and not self.queue.empty():
            # never wait on the lock here: this runs every frame, and a read in
            # flight would stall the editor. Starting one frame later is free.
            if not self.prefill_lock.acquire(blocking=False):
                return
            try:
                self.prompt = self.queue.get()
                self.active = True
            finally:
                self.prefill_lock.release()
            self.active_out.send(True)
            self.initiate_generation()

    def initiate_step(self):
        value = self.step_input()
        if type(value) == int:
            self.stepping = True
            self.take_step = 1 if value > 0 else -1

    def initiate_generation(self):
        self.thread = threading.Thread(target=self.generate)
        self.thread.start()

    def generate(self):
        preferred_id = None
        self.pending_utf8 = b''
        self.in_action = False
        self.force_stop_now = False
        self.force_stop_at_next = False
        self.possible_stop = False
        try:
            self.ensure_server()
            with self.prefill_lock:
                self.prepare_prompt()
            self.response_start = len(self.context_tokens)
            self.answer_start = self.response_start
            while True:
                if self.active == False or self.do_reset:
                    break

                budget = self.stream_budget()
                if budget > 0:
                    if self.run_stream(budget):
                        break
                else:
                    token_string, token_id = self.sample_token(preferred_id)
                    preferred_id = None
                    if token_string is None:
                        break
                    self.emit_token(token_id, token_string)

                if not self.on_off():
                    if self.stepping:
                        while self.take_step == 0 and not self.on_off() and not self.do_reset:
                            self.in_step = True
                            time.sleep(0.005)
                        self.in_step = False
                        if self.take_step == 1 and len(self.streaming_prompt) > 0:
                            streaming_tokens = self.server.tokenize(self.streaming_prompt)
                            pairs = []
                            for token in streaming_tokens:
                                pairs += [[token, self.server.token_piece(token)]]
                            if self.score_input():
                                self.eval_and_score(streaming_tokens)
                            else:
                                self.context_tokens += [int(t) for t in streaming_tokens]
                            self.output.send(self.streaming_prompt)
                            self.layout_out.send(['streaming_prompt', pairs])
                            self.layout_out.send(['accept_streamed_prompt'])
                            self.streaming_prompt = ''
                        if self.take_step == -1:
                            preferred_id = self.back_step(2)
                        self.take_step = 0
                    else:
                        break
                else:
                    sleeper = self.slow_down()
                    if sleeper > 0:
                        time.sleep(sleeper)

                if 0 < any_to_int(self.max_tokens()) <= len(self.context_tokens) - self.response_start:
                    self.end_of_generation()
                    break

            self.warm_cache_after_turn()

        except Exception as e:
            print('bonsai_2:', e)
        self.active = False
        self.active_out.send(self.active)
        sys.exit()

    def warm_cache_after_turn(self):
        """Removing a thought from the context means the server no longer has a
        matching prefix, so it has to re-read everything after the gap - about a
        whole answer's worth, a second or two of it. Spend that here, in the
        pause after an answer, instead of making the next prompt wait for it: one
        throwaway token is enough to put the shortened context back in the
        server's cache.

        Done before 'active' is cleared, so a queued prompt cannot start a second
        conversation with the server while this is in flight."""
        if self.keep_thinking() or len(self.context_tokens) == 0 or self.do_reset:
            return
        before = len(self.context_tokens)
        self.strip_previous_thinking()
        if len(self.context_tokens) == before:
            return              # nothing was stripped, the cache still matches
        try:
            payload = self.sampling_payload(1, with_ramp=False)
            payload['n_predict'] = 1
            self.server.completion(payload)
        except Exception:
            pass                # a cold cache costs time, never correctness

    def emit_token(self, token_id, token_string):
        """Route one token: thinking to its own outlet, an unspoken aside to the
        action outlet, everything else to the text."""
        self.token_out.send(token_id)

        if token_id == self.think_close_token:
            self.in_thinking = False
            # the answer starts here, and the full stops inside the thought are
            # not sentence ends in it
            self.answer_start = len(self.context_tokens)
            self.possible_stop = False
            if self.thinking_in_layout():
                self.send_to_layout(token_id, token_string)
            return
        if self.in_thinking:
            self.thinking_out.send(token_string)
            if self.thinking_in_layout():
                self.send_to_layout(token_id, token_string)
            return

        if self.separate_actions():
            if self.in_action:
                self.actions_out.send(token_string)
                if self.ends_action(token_string):
                    self.in_action = False
                    self.actions_out.send('\n')
                return
            if self.starts_action(token_string):
                self.in_action = True
                self.actions_out.send(token_string)
                return

        self.output.send(token_string)
        self.send_to_layout(token_id, token_string)

    def starts_action(self, token_string):
        """Does this token open an unspoken aside?

        The model marks an aside with a single asterisk at the start of the
        phrase - *Looks around the room.* - whether that phrase begins a line or
        sits inside one, so any single asterisk opens one. Doubled asterisks are
        bold or a heading, which this model uses for emphasis it IS speaking, and
        never open an aside.

        What this cannot tell apart is single-asterisk *italic* emphasis on a
        spoken word, which looks identical to a one-word aside and so goes
        unspoken. This model reaches for **bold** far more often, which is why
        the trade falls this way. It also means a markdown bulleted list is
        treated as asides, one line at a time - the end of a line closes one.
        """
        return '*' in token_string and '**' not in token_string

    def ends_action(self, token_string):
        """Any single asterisk closes an open aside, including the fused tokens
        this tokeniser produces - the closing mark usually arrives as '.*' or
        '*.' rather than alone. The end of the line closes one too, so an aside
        the model never closes cannot swallow the speech that follows it."""
        if '*' in token_string and '**' not in token_string:
            return True
        return '\n' in token_string

    # ------------------------------------------------------- streamed run

    def stream_budget(self):
        """How many tokens may be taken in one streamed request, 0 meaning ask
        for them one at a time.

        One request per token costs about 5 ms of round trip each, near enough
        12% of the generation rate. Taking a short run of tokens per request
        recovers effectively all of that (a chunk of 8 measured within 1% of one
        long stream), while still re-reading the sampling controls often enough
        that moving a slider mid-generation is felt within about half a second.

        It only applies while free-running. Stepping wants one token at a time
        by definition; 'slowdown' is already spending far more than the round
        trip; and the end-of-turn ramp changes its logit bias per token, which a
        single request cannot do - so the last stretch before target_length goes
        back to one at a time, and the nudge works exactly as before."""
        chunk = any_to_int(self.stream_chunk())
        if chunk < 2:
            return 0
        if not self.on_off() or self.stepping or self.take_step != 0:
            return 0
        if float(self.slow_down()) > 0:
            return 0
        if self.force_stop_now or self.force_stop_at_next:
            return 0
        if self.in_thinking:
            return chunk          # no bias applies while reasoning: stream it
        produced = self.answer_produced()
        target = max(any_to_int(self.target_length()), 2)
        budget = int(target * 0.8) - produced       # stop short of the ramp
        hard = any_to_int(self.max_tokens())
        if hard > 0:
            budget = min(budget, hard - produced)
        if budget < 2:
            return 0
        return min(budget, chunk)

    def run_stream(self, budget):
        """Take up to budget tokens from one request. Returns True if the turn
        ended. Abandoning the loop closes the socket, which stops the server."""
        payload = self.sampling_payload(40 if self.wants_candidates() else 1)
        payload['n_predict'] = budget
        finished = False
        for event in self.server.stream_completion(payload):
            probs = event.get('completion_probabilities') or []
            if len(probs) > 0:
                entry = probs[0]
                token = int(entry.get('id'))
                self.choices = entry.get('top_logprobs') or []
                if token in self.stop_tokens:
                    self.end_of_generation()
                    finished = True
                    break
                self.context_tokens.append(token)
                self.last_generated_token = token
                token_string = self.stream_piece(entry, token)
                if token_string in ['.', '?', '!']:
                    self.possible_stop = True
                self.update_measures(token)
                self.emit_token(token, token_string)
                if self.force_stop_at_next and self.possible_stop:
                    break       # polite stop: leave the run at a sentence end
            if event.get('stop'):
                if event.get('stop_type') == 'eos':
                    self.end_of_generation()
                    finished = True
                break
            if (not self.active or self.do_reset or not self.on_off()
                    or self.force_stop_now or self.stepping or self.take_step != 0):
                break
        return finished

    def send_to_layout(self, token_id, token_string):
        if self.show_probs():
            self.chosen_index = self.build_poss_dict(token_id)
            self.layout_out.send(['choice_list', self.poss_dict, self.chosen_index])
        mode = self.display_mode()
        toner = 255
        if mode == 'temperature':
            toner = self.temperature()
        elif mode == 'entropy':
            toner = self.entropy
        elif mode == 'probability':
            toner = self.probability
        elif mode == 'unnormed_probability':
            toner = self.unnormed_probability
        self.layout_out.send(['add', int(token_id), token_string, toner])

    # ----------------------------------------------------------- token ops

    def strip_previous_thinking(self):
        """Drop earlier turns' reasoning from the context before a new turn is
        built. This is what the model's own chat template does by default, and
        it is not only a saving of context: several turns of visible
        deliberation give the model its own earlier reasoning to imitate, and it
        starts restating the previous answer instead of writing a new one.

        A turn that ended inside its own thought is dropped whole - there is no
        answer in it to keep."""
        if self.keep_thinking():
            return
        tokens = self.context_tokens
        spans = []
        opened_at = None
        for index, token in enumerate(tokens):
            if token == self.think_open_token:
                opened_at = index
            elif token == self.think_close_token and opened_at is not None:
                end = index + 1
                if end < len(tokens) and tokens[end] == self.double_return_token:
                    end += 1
                if index > opened_at + 2:
                    # an empty block - what the prompt carries when thinking is
                    # off - holds no reasoning to strip, and removing it would
                    # cost a re-read of everything after it for nothing
                    spans.append((opened_at, end))
                opened_at = None
        if opened_at is not None:
            spans.append((opened_at, len(tokens)))
        for start, end in reversed(spans):
            del tokens[start:end]

    def user_turn_prefix(self):
        """Everything that opens a user turn: close the model's previous turn,
        the system block when one is due, then the user header. Called once per
        turn, because it consumes the pending system prompt."""
        prefix = '' if len(self.context_tokens) == 0 else '<|im_end|>\n'
        sys_block = ''
        if self.new_system_prompt:
            self.new_system_prompt = False
            instructions = reasoning_instructions.get(self.reasoning_effort(), '') if self.thinking() else ''
            body = self.system_prompt.strip()
            if instructions:
                body = instructions + '\n\n' + body
            if body:
                sys_block = '<|im_start|>system\n' + body + '<|im_end|>\n'
        return prefix + sys_block + '<|im_start|>user\n'

    def generation_prompt(self):
        """Closes the user turn and opens the model's, with the thought block
        either opened for it to reason into or already closed."""
        gen_prompt = '<|im_start|>assistant\n'
        if self.thinking():
            gen_prompt += '<think>\n'
            self.in_thinking = True
        else:
            # an already-closed thought block makes the model answer directly
            gen_prompt += '<think>\n\n</think>\n\n'
            self.in_thinking = False
        return '<|im_end|>\n' + gen_prompt

    def prepare_prompt(self):
        """Either the words were read as they arrived, leaving only the turn's
        closing to do, or the whole prompt is built here."""
        if self.spoken_turn_open:
            self.feed_spoken_text(self.prompt)
            self.close_spoken_turn()
            self.send_prompt_to_layout(self.prompt)
        else:
            self.format_and_tokenize_prompt()

    def format_and_tokenize_prompt(self, prompt=None):
        server = self.server
        if prompt is None:
            prompt = self.prompt
        self.strip_previous_thinking()
        prefix = self.user_turn_prefix()
        if self.score_input():
            # segment the encoding so the user's words can be scored token by
            # token; concatenated segments can differ from a whole-string
            # encoding by a merge at the two text boundaries
            self.context_tokens += [int(t) for t in server.tokenize(prefix)]
            self.eval_and_score(server.tokenize(prompt))
            self.context_tokens += [int(t) for t in server.tokenize(self.generation_prompt())]
        else:
            whole = prefix + prompt + self.generation_prompt()
            self.context_tokens += [int(t) for t in server.tokenize(whole)]
        self.send_prompt_to_layout(prompt)

    def send_prompt_to_layout(self, prompt):
        if self.suppress_prompt_layout:
            self.suppress_prompt_layout = False
            return
        for token in self.server.tokenize(prompt):
            self.layout_out.send(['prompt', token, self.server.token_piece(token)])

    # ------------------------------------------- reading words as they arrive

    def request_prefill(self):
        """Ask the worker to read whatever has arrived so far. A no-op until the
        node has been switched on - typing a character should not be what pulls
        seven gigabytes of weights into memory."""
        if not self.preread() or self.server is None:
            return
        if self.prefill_thread is None:
            self.prefill_stop = False
            self.prefill_thread = threading.Thread(target=self.prefill_worker, daemon=True)
            self.prefill_thread.start()
        self.prefill_wanted.set()

    def prefill_worker(self):
        """Off the main thread: a read costs a couple of hundred milliseconds and
        the editor draws every frame. Coalescing is automatic - the worker always
        reads the CURRENT text, so a burst of words becomes one read."""
        while not self.prefill_stop:
            if not self.prefill_wanted.wait(0.25):
                continue
            self.prefill_wanted.clear()
            if self.prefill_stop:
                return
            text = self.streaming_prompt
            with self.prefill_lock:
                if self.active or self.do_reset or self.server is None:
                    continue
                try:
                    if len(text.strip()) == 0:
                        # an empty box with something queued means it was just
                        # submitted, and that turn is about to be generated
                        if self.queue.empty():
                            self.abandon_spoken_turn()
                        continue
                    self.open_spoken_turn()
                    self.feed_spoken_text(text)
                except Exception as e:
                    print('bonsai_2 pre-read:', e)
                    self.abandon_spoken_turn()

    def open_spoken_turn(self):
        if self.spoken_turn_open:
            return
        self.strip_previous_thinking()
        self.spoken_turn_start = len(self.context_tokens)
        self.spoken_turn_new_system = self.new_system_prompt
        self.context_tokens += [int(t) for t in self.server.tokenize(self.user_turn_prefix())]
        self.spoken_text_start = len(self.context_tokens)
        self.spoken_turn_open = True

    def feed_spoken_text(self, text):
        """Bring the context up to date with the words heard so far.

        The whole utterance is re-tokenised each time rather than just the new
        words: a byte-pair merge can reach across a word boundary, so appending
        fragment encodings could drift from what the model would see if the
        sentence arrived whole. Re-tokenising and keeping only the matching
        prefix cannot drift, and costs nothing when nothing changed - measured
        byte-identical to the whole string, feeding five words at a time and one
        word at a time.

        It also handles a recogniser that revises what it already said: the
        tokens back to the first difference are dropped and re-read.
        """
        server = self.server
        wanted = [int(t) for t in server.tokenize(text)]
        already = self.context_tokens[self.spoken_text_start:]
        shared = 0
        while (shared < len(already) and shared < len(wanted)
               and already[shared] == wanted[shared]):
            shared += 1
        if shared < len(already):
            del self.context_tokens[self.spoken_text_start + shared:]
        fresh = wanted[shared:]
        if len(fresh) == 0:
            return
        if self.score_input():
            # reads and scores at once: every word gets a log-probability as it
            # arrives, which is the live measure of how expected the speech is
            self.eval_and_score(fresh, opens_statement=(len(already) == 0))
        else:
            self.context_tokens += fresh
            payload = self.sampling_payload(1, with_ramp=False)
            payload['n_predict'] = 0        # read it, generate nothing
            server.completion(payload)

    def close_spoken_turn(self):
        self.context_tokens += [int(t) for t in self.server.tokenize(self.generation_prompt())]
        self.spoken_turn_open = False

    def abandon_spoken_turn(self):
        """The utterance was dropped before being submitted: take the opened turn
        back out, including the system block if this turn carried it - putting a
        system turn in the middle of a conversation is exactly what the model's
        template forbids."""
        if not self.spoken_turn_open:
            return
        del self.context_tokens[self.spoken_turn_start:]
        self.spoken_turn_open = False
        self.new_system_prompt = self.spoken_turn_new_system

    def sampling_payload(self, n_probs, extra_bias=None, with_ramp=True):
        payload = {
            'prompt': self.context_tokens,
            'n_predict': 1,
            'cache_prompt': True,
            'return_tokens': True,
            'post_sampling_probs': False,
            'n_probs': n_probs,
            'temperature': float(self.temperature()),
            'top_k': any_to_int(self.top_k()),
            'top_p': float(self.top_p()),
            'min_p': float(self.min_p()),
            'repeat_penalty': float(self.repeat_penalty()),
            'seed': int(self.seed),
        }
        bias = []
        if extra_bias is not None:
            bias += extra_bias
        ramp = self.stop_ramp() if with_ramp else 0.0
        if ramp > 0.0:
            bias.append([int(self.end_of_turn_token), ramp * float(self.stop_bias())])
        if len(bias) > 0:
            payload['logit_bias'] = bias
        return payload

    def answer_produced(self):
        """Tokens of ANSWER so far. Reasoning does not count: 'target_length'
        shapes what gets said, not how long the model thinks about it, and with
        xhigh effort the thought alone runs past any modest target."""
        return len(self.context_tokens) - self.answer_start

    def stop_ramp(self):
        """The end-of-turn nudge, as a 0..1 ramp. It only applies right after a
        sentence ends, so the model is pushed to stop at a sentence boundary
        rather than mid-clause."""
        if self.in_thinking:
            # a thought is full of full stops, and ending the turn inside one
            # loses the answer entirely - the model reasons and then says nothing
            self.possible_stop = False
            return 0.0
        if self.force_stop_now:
            return 12.0 / max(float(self.stop_bias()), 1e-6)
        if not self.possible_stop:
            return 0.0
        self.possible_stop = False
        if self.force_stop_at_next:
            self.force_stop_at_next = False
            return 12.0 / max(float(self.stop_bias()), 1e-6)
        target = max(any_to_int(self.target_length()), 2)
        produced = self.answer_produced()
        start_ramp = target * 0.8
        if target <= start_ramp:
            return 1.0
        ratio = (produced - start_ramp) / (target - start_ramp)
        return min(max(ratio, 0.0), 1.0)

    def wants_candidates(self):
        # the candidate list feeds the display measures as well as the visible
        # choice list, so fetch it whenever either needs it
        return self.show_probs() or self.display_mode() in (
            'entropy', 'probability', 'unnormed_probability')

    def sample_token(self, preferred_choice=None):
        server = self.server
        n_probs = 40 if self.wants_candidates() else 1
        result = server.completion(self.sampling_payload(n_probs))
        probs = result.get('completion_probabilities') or []
        if len(probs) == 0:
            return None, None
        entry = probs[0]
        out_token = int(entry.get('id'))
        self.choices = entry.get('top_logprobs') or []
        if preferred_choice is not None and preferred_choice >= 0:
            out_token = int(preferred_choice)

        self.last_generated_token = out_token
        if out_token in self.stop_tokens:
            self.end_of_generation()
            return None, None

        self.context_tokens.append(out_token)
        self.new_response = False
        out_string = self.stream_piece(entry, out_token)

        if out_string in ['.', '?', '!']:
            self.possible_stop = True

        self.update_measures(out_token)
        return out_string, out_token

    def stream_piece(self, entry, out_token):
        """Sequential decode: hold an incomplete utf-8 tail until the next token
        completes the character, instead of emitting mojibake per token."""
        raw = entry.get('bytes')
        if raw is not None and int(entry.get('id')) == out_token:
            piece = bytes(raw)
        else:
            piece = self.server.token_piece(out_token).encode('utf-8')
        buf = self.pending_utf8 + piece
        try:
            out = buf.decode('utf-8')
            self.pending_utf8 = b''
            return out
        except UnicodeDecodeError:
            if len(buf) < 8:
                self.pending_utf8 = buf
                return ''
            out = buf.decode('utf-8', errors='replace')
            self.pending_utf8 = b''
            return out

    def update_measures(self, out_token):
        """Entropy, probability and the unnormalised measure, computed over the
        candidates the server returned. Unlike the in-process gemma node these
        cover the returned top-N rather than the whole vocabulary, so entropy is
        a floor rather than the exact figure - with a peaked distribution the
        two agree closely, with a flat one the tail this misses is real."""
        if len(self.choices) == 0:
            return
        logprobs = np.array([c.get('logprob', -100.0) for c in self.choices], dtype=np.float64)
        temp = max(float(self.temperature()), 1e-6)
        scaled = logprobs / temp
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights = weights / weights.sum()
        self.entropy = float(-(weights * np.log(weights + 1e-12)).sum())
        self.probability = 0.0
        self.unnormed_probability = 0.0
        for choice, weight in zip(self.choices, weights):
            if int(choice.get('id', -1)) == int(out_token):
                self.probability = float(weight)
                self.unnormed_probability = self.unnormalized_probability(
                    float(choice.get('logprob', -100.0)))
                break

    def unnormalized_probability(self, logprob):
        # the same sigmoid gemma_4 uses, but fed a log-probability rather than a
        # raw logit, so it sits in a different range - a near-certain token reads
        # about 0.5 here. 'sigmoid scaler' and 'sigmoid offset' are what move it
        # back into a useful spread for a given patch

        return float(1 / (1 + np.exp(self.sigmoid_scaler() * (logprob + self.sigmoid_offset()))))

    def eval_and_score(self, tokens, opens_statement=True):
        """Consume incoming tokens one at a time so each can be scored against
        the prediction the model held before consuming it. The forced token is
        pushed by a logit bias, while the reported log-probability is read
        before the sampler chain runs, so the bias does not colour the score.
        The statement's first token is reported but kept out of the cumulative -
        the model has no settled prediction for an opening word, so it scores as
        artificially surprising.

        'opens_statement' is false when this is a later batch of an utterance
        still arriving: only the utterance's own first word is an opening, and
        excluding the first word of every batch would quietly drop most of the
        speech from the running average."""
        for i, token in enumerate(tokens):
            token = int(token)
            payload = self.sampling_payload(40, extra_bias=[[token, 100.0]], with_ramp=False)
            result = self.server.completion(payload)
            probs = result.get('completion_probabilities') or []
            logprob = None
            if len(probs) > 0:
                entry = probs[0]
                if int(entry.get('id', -1)) == token:
                    logprob = float(entry.get('logprob'))
                else:
                    for choice in (entry.get('top_logprobs') or []):
                        if int(choice.get('id', -1)) == token:
                            logprob = float(choice.get('logprob'))
                            break
            self.context_tokens.append(token)
            if logprob is not None:
                self.score_incoming_token(token, logprob,
                                          accumulate=(i > 0 or not opens_statement))

    def score_incoming_token(self, token, logprob, accumulate=True):
        unnormed = self.unnormalized_probability(logprob)
        self.input_score_out.send([self.server.token_piece(token), logprob, unnormed])
        if not accumulate:
            return
        self.input_logprob_sum += logprob
        self.input_unnormed_sum += unnormed
        self.input_token_count += 1
        count = self.input_token_count
        self.input_cumulative_out.send([self.input_logprob_sum / count,
                                        self.input_unnormed_sum / count, count])

    def reset_input_score(self):
        self.input_logprob_sum = 0.0
        self.input_unnormed_sum = 0.0
        self.input_token_count = 0

    def back_step(self, count):
        """Drop the last tokens from our mirror of the context. There is no kv
        surgery to do: the next request sends the shortened list and the server
        keeps the matching prefix of its cache."""
        n_tokens = len(self.context_tokens)
        if n_tokens >= count:
            self.last_generated_token = self.context_tokens[n_tokens - count]
            if count == 1:
                preferred_choice = self.context_tokens[n_tokens - count]
            else:
                preferred_choice = self.context_tokens[n_tokens - count + 1]
            del self.context_tokens[n_tokens - count:]
            self.layout_out.send(['step_back', count])
            for i in range(count):
                self.output.send('<backspace>')
            return preferred_choice
        return -1

    def build_poss_dict(self, chosen_token):
        chosen_index = -1
        entries = []
        for choice in self.choices:
            logprob = float(choice.get('logprob', -100.0))
            if logprob > -1000:
                token = int(choice.get('id'))
                token_word = choice.get('token', '')
                if token in self.stop_tokens:
                    token_word = '<end>'
                elif token_word in ['\n', '\r']:
                    token_word = '<return>'
                elif token_word == '\t':
                    token_word = '<tab>'
                entries.append([token_word, logprob, token])
        entries = truncate_to_sampleable(entries, any_to_int(self.top_k()),
                                         float(self.top_p()), float(self.min_p()))
        # values stay unscaled log-probabilities: llm_layout applies temperature
        # itself, driven by the ['temperature', value] message from temp_changed.
        # a log-probability is a logit shifted by a constant, and softmax ignores
        # a constant shift, so the layout's spread is unaffected
        self.poss_dict = sorted(entries, key=lambda x: x[1])
        for i, entry in enumerate(self.poss_dict):
            if entry[2] == chosen_token:
                chosen_index = i
                break
        return chosen_index

    # ------------------------------------------------------------ controls

    def end_of_generation(self):
        self.force_stop_now = False
        self.new_response = True
        self.active = False
        self.active_out.send(False)
        self.layout_out.send(['add', 0, '\n\n'])
        self.output_end_of_text.send('bang')

    def thinking_changed(self):
        # the reasoning instructions live in the system turn, so re-emit it
        self.new_system_prompt = True

    def system_prompt_received(self):
        self.system_prompt = any_to_string(self.system_prompt_input(), strip_returns=False)
        self.new_system_prompt = True

    def preprompt_received(self):
        self.preprompt = any_to_string(self.pre_prompt_input(), strip_returns=False)

    def prompt_received(self):
        prompt = any_to_string(self.prompt_input(), strip_returns=False)
        if self.preprompt != '':
            prompt = self.preprompt + prompt
            self.preprompt = ''
        self.queue.put(prompt, block=False)

    def submit_streaming_prompt(self):
        # double-Enter: drop the newline the first Enter inserted, then submit
        if self.streaming_prompt.endswith('\n'):
            self.streaming_prompt = self.streaming_prompt[:-1]
            self.layout_out.send(['backspace_streaming_prompt'])
        if len(self.streaming_prompt.strip()) == 0:
            return
        if self.active:
            # generation thread is parked in the step-wait loop: accepting is a step
            self.stepping = True
            self.take_step = 1
        else:
            # idle: the typed text becomes the next prompt; its preview is already
            # in the layout, so skip the prompt burst for this one generation
            text = self.streaming_prompt
            self.streaming_prompt = ''
            self.layout_out.send(['accept_streamed_prompt'])
            self.suppress_prompt_layout = True
            self.queue.put(text, block=False)

    def streaming_prompt_received(self):
        self.handle_streaming_prompt(any_to_string(self.streaming_prompt_input(), strip_returns=False))

    def handle_streaming_prompt(self, streaming_prompt_string):
        if streaming_prompt_string == '<backspace>':
            if len(self.streaming_prompt) > 0:
                self.streaming_prompt = self.streaming_prompt[:-1]
                self.layout_out.send(['backspace_streaming_prompt'])
            else:
                self.streaming_prompt = ''
                if self.in_step:
                    self.take_step = -1
                else:
                    self.back_step(1)
            self.request_prefill()
            return
        elif len(self.streaming_prompt) == 0:
            self.output.send('<backspace>')
            if len(streaming_prompt_string) > 0:
                if streaming_prompt_string[0] != ' ':
                    self.streaming_prompt += ' '
        self.streaming_prompt += streaming_prompt_string
        self.layout_out.send(['streaming_prompt', [[0, self.streaming_prompt]]])
        self.request_prefill()

    def set_do_reset(self):
        if not self.active:
            self.reset()
        else:
            self.do_reset = True

    def temp_changed(self):
        self.layout_out.send(['temperature', self.temperature()])

    def save_text(self):
        self.layout_out.send(['save'])

    def stop(self):
        self.stopping = True
        self.force_stop_now = True

    def polite_stop(self):
        self.force_stop_at_next = True

    def hard_stop(self):
        self.end_of_generation()

    def show_probs_changed(self):
        self.layout_out.send(['show_probs', self.show_probs()])

    def randomize_seed(self):
        self.seed = random.randint(0, 2 ** 32 - 1)
        self.seed_input.set(self.seed)

    def set_seed(self):
        self.seed = self.seed_input()
        np.random.seed(self.seed)

    def reset(self):
        self.reset_input_score()
        self.do_reset = False
        self.layout_out.send(['reset'])
        self.last_generated_token = None
        self.streaming_prompt = ''
        self.in_thinking = False
        self.pending_utf8 = b''
        self.last_key_was_enter = False
        self.suppress_prompt_layout = False
        self.stepping = False
        self.take_step = 0
        self.force_stop_now = False
        self.force_stop_at_next = False
        self.possible_stop = False
        self.context_tokens = []
        self.response_start = 0
        self.answer_start = 0
        self.spoken_turn_open = False
        self.prefill_wanted.clear()
        self.new_system_prompt = True
        self.queue.queue.clear()
        self.set_seed()

    def custom_cleanup(self):
        self.active = False
        self.prefill_stop = True
        self.prefill_wanted.set()
        self.prefill_thread = None
        BonsaiServer.release(self.server)
        self.server = None
        if self in Bonsai2ChatNode.bonsai_nodes:
            Bonsai2ChatNode.bonsai_nodes.remove(self)
