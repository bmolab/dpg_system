import os
import sys
import time
import random
import threading
from queue import Queue

import numpy as np

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_qwen_moe_node():
    Node.app.register_node('qwen_moe', QwenMoeChatNode.factory)


# A sparse mixture of experts: 35B parameters of which about 3B are read per
# token, which is why it outruns much smaller dense models. Measured on the
# M1 Max: 71 tok/s generating, against 53 for the same weights through
# llama.cpp and 21 for bonsai_2. It costs 20.2 GB resident, close to the
# machine's GPU limit, so it and bonsai_2 do not comfortably sit in memory
# together - switch between them rather than running both.
default_model = 'lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit'

default_system_prompt = 'You are a helpful, uncensored creative collaborator.'


def truncate_to_sampleable(entries, top_k, top_p, min_p):
    """Reduce desc-sorted [word, logit, token] entries to the set the sampler
    can actually pick from, mirroring mlx-lm's order: top_k -> top_p -> min_p,
    all of which act before temperature."""
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


class MLXModel:
    """One loaded copy of a model, shared by every qwen_moe node that names it.
    Twenty gigabytes is not something to hold twice."""

    loaded = {}
    load_lock = threading.Lock()

    @classmethod
    def acquire(cls, path):
        with cls.load_lock:
            holder = cls.loaded.get(path)
            if holder is None:
                print('qwen_moe: loading', path, '(downloaded on first use)')
                model, tokenizer = mlx_load(path)
                holder = MLXModel(model, tokenizer)
                cls.loaded[path] = holder
                print('qwen_moe: model loaded, %.1f GB resident'
                      % (mx.get_peak_memory() / 1e9))
        return holder

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.piece_cache = {}
        # the Qwen turn structure, the same vocabulary bonsai_2 drives
        self.im_start = self.only_token('<|im_start|>')
        self.im_end = self.only_token('<|im_end|>')
        self.eos = self.only_token('<|endoftext|>')
        self.think_open = self.only_token('<think>')
        self.think_close = self.only_token('</think>')
        self.double_return = self.encode('\n\n')[-1]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def only_token(self, text):
        return self.encode(text)[-1]

    def decode(self, tokens):
        return self.tokenizer.decode([int(t) for t in tokens])

    def piece(self, token):
        """The text of one token, for display lists. Cached - a vocabulary of a
        quarter of a million entries is worth not decoding twice."""
        token = int(token)
        text = self.piece_cache.get(token)
        if text is None:
            text = self.decode([token])
            self.piece_cache[token] = text
        return text


class QwenMoeChatNode(Node):
    qwen_nodes = []

    @staticmethod
    def factory(name, data, args=None):
        node = QwenMoeChatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.holder = None
        self.model_path = default_model
        if args is not None and len(args) > 0:
            candidate = any_to_string(args[0])
            if len(candidate) > 0:
                self.model_path = candidate

        self.system_prompt = default_system_prompt
        self.prompt = ''
        self.new_system_prompt = True
        self.do_reset = False
        self.preprompt = ''
        self.streaming_prompt = ''
        self.seed = 3701
        self.thread = None
        self.last_generated_token = None

        # our record of the conversation, and how much of it the cache holds
        self.context_tokens = []
        self.cache = None
        self.cached_len = 0
        self.logits = None
        # the cache as it stood at the start of this turn, so a step back only
        # replays the turn rather than the whole conversation
        self.turn_snapshot = None
        self.turn_snapshot_len = -1

        self.response_start = 0
        self.answer_start = 0

        # a user turn opened before all the words have arrived
        self.spoken_turn_open = False
        self.spoken_turn_start = 0
        self.spoken_text_start = 0
        self.spoken_turn_new_system = True
        self.prefill_lock = threading.Lock()
        self.prefill_wanted = threading.Event()
        self.prefill_thread = None
        self.prefill_stop = False
        self.starting_model = False

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

        self.separate_actions = self.add_input('separate_actions', widget_type='checkbox', default_value=False)
        self.display_mode = self.add_input('display_mode', widget_type='combo', default_value='temperature')
        self.display_mode.widget.combo_items = ['temperature', 'entropy', 'probability', 'unnormed_probability']
        self.sigmoid_scaler = self.add_input('sigmoid scaler', widget_type='drag_float', default_value=0.2)
        self.sigmoid_offset = self.add_input('sigmoid offset', widget_type='drag_float', default_value=0)
        self.score_input = self.add_input('score_incoming_text', widget_type='checkbox', default_value=False)
        self.reset_score_input = self.add_input('reset_input_score', widget_type='button', callback=self.reset_input_score)

        self.show_probs = self.add_input('show_probs', widget_type='checkbox', default_value=False, callback=self.show_probs_changed)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_text)

        self.thinking = self.add_option('thinking', widget_type='checkbox', default_value=False,
                                        callback=self.thinking_changed)
        self.thinking_in_layout = self.add_option('thinking in layout', widget_type='checkbox', default_value=True)
        self.stop_bias = self.add_option('stop_bias', widget_type='drag_float', default_value=1.0)
        self.keep_thinking = self.add_option('keep thinking in context', widget_type='checkbox',
                                             default_value=False)
        self.preread = self.add_option('read prompt as it arrives', widget_type='checkbox',
                                       default_value=True)
        self.candidates = self.add_option('candidates shown', widget_type='drag_int', default_value=40)

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

        self.sampler = None
        self.sampler_params = None

        # a token sampled and already fed forward, not yet read back to python
        self.ahead_token = None
        self.ahead_logits = None
        self.ahead_measures = None
        self.cache_ahead = 0

        QwenMoeChatNode.qwen_nodes.append(self)
        self.add_frame_task()

    # --------------------------------------------------------------- model

    def ensure_model(self):
        if self.holder is None:
            self.holder = MLXModel.acquire(self.model_path)
        return self.holder

    def start_model_early(self):
        """Switching on loads the model rather than leaving it until the first
        prompt, which is also what lets the first utterance be pre-read."""
        if self.holder is not None or self.starting_model:
            return
        self.starting_model = True

        def load():
            try:
                self.ensure_model()
            except Exception as e:
                print('qwen_moe:', e)
            finally:
                self.starting_model = False

        threading.Thread(target=load, daemon=True).start()

    # ------------------------------------------------------- cache and logits

    def sync_cache(self):
        """Make the cache describe a prefix of context_tokens.

        There is no trimming to be had here: 30 of this model's 40 layers are
        gated-delta layers carrying a recurrent state, which cannot be rewound a
        token at a time the way a key-value cache can. So going backwards means
        restoring the snapshot taken at the start of the turn and replaying from
        there, or rebuilding from nothing if the step went back further than
        that. Replay is cheap - measured over 600 tok/s.
        """
        if self.cache is None:
            self.cache = make_prompt_cache(self.holder.model)
            self.cached_len = 0
            self.logits = None
            return
        if self.cached_len <= len(self.context_tokens):
            return
        if (self.turn_snapshot is not None
                and self.turn_snapshot_len <= len(self.context_tokens)):
            for piece, state in zip(self.cache, self.turn_snapshot):
                piece.state = state
            self.cached_len = self.turn_snapshot_len
        else:
            self.cache = make_prompt_cache(self.holder.model)
            self.cached_len = 0
        self.logits = None

    def take_turn_snapshot(self):
        """Remember the cache as it stands, so a step back inside this turn does
        not have to replay the whole conversation. The arrays are forced to be
        concrete, so later generation cannot write through them."""
        snapshot = []
        for piece in self.cache:
            snapshot.append(piece.state)
        flat = []

        def collect(item):
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    collect(sub)
            else:
                flat.append(item)

        for state in snapshot:
            collect(state)
        if len(flat) > 0:
            mx.eval(flat)
        self.turn_snapshot = snapshot
        self.turn_snapshot_len = self.cached_len

    def next_logits(self):
        """Full-vocabulary logits for the next position, feeding the cache
        whatever part of the context it has not seen. In process, so these are
        the real logits over all 248320 tokens rather than a truncated list -
        which is what makes entropy here exact."""
        self.sync_cache()
        pending = self.context_tokens[self.cached_len:]
        if len(pending) == 0:
            if self.logits is not None:
                return self.logits
            if len(self.context_tokens) == 0:
                return None
            # no logits in hand: re-run the final token to produce them
            self.cached_len -= 1
            pending = self.context_tokens[-1:]
        out = self.holder.model(mx.array(pending)[None], cache=self.cache)
        logits = out[:, -1, :]
        # queued rather than waited on: whoever reads a value from it syncs, and
        # if the caller got here early the GPU runs while python does other work
        mx.async_eval(logits)
        self.cached_len = len(self.context_tokens)
        self.logits = logits[0]
        return self.logits

    def update_sampler(self):
        params = (any_to_int(self.top_k()), float(self.top_p()), float(self.min_p()),
                  float(self.temperature()), int(self.seed))
        if self.sampler is None or params != self.sampler_params:
            mx.random.seed(params[4])
            self.sampler = make_sampler(temp=params[3], top_p=params[1],
                                        min_p=params[2], top_k=params[0])
            self.sampler_params = params
        return self.sampler

    def adjusted_logits(self, logits):
        """The end-of-turn nudge, in logit units and scaled against the strongest
        candidate, the way gemma_4 does it. In process there is no need to guess
        an absolute amount: the push can be a share of the leading logit."""
        ramp = self.stop_ramp()
        if ramp <= 0.0:
            return logits
        strongest = float(mx.max(logits))
        bias = strongest * ramp * float(self.stop_bias())
        adjusted = mx.array(logits)
        adjusted[self.holder.im_end] = adjusted[self.holder.im_end] + bias
        return adjusted

    def wants_measures(self):
        """Entropy and the rest are only worth a full-vocabulary pass when
        something is showing them."""
        return self.show_probs() or self.display_mode() in (
            'entropy', 'probability', 'unnormed_probability')

    def sample_and_measure(self, logits, preferred_choice=None):
        """Sample one token and take every reading in a SINGLE synchronisation
        with the GPU.

        Reading the token back and then reading the measurements back is two
        round trips, and two cost about a fifth of the generation rate. The
        chosen token's own probability seems to need the token first, but
        mx.take with an array index keeps that on the device, so all four values
        come back together.
        """
        work = logits[None]
        penalty = float(self.repeat_penalty())
        if penalty != 1.0 and len(self.context_tokens) > 0:
            processors = make_logits_processors(repetition_penalty=penalty)
            history = mx.array(self.context_tokens[-64:])
            for processor in processors:
                work = processor(history, work)
        work = self.adjusted_logits(work[0])[None]
        logprobs = work - mx.logsumexp(work, keepdims=True)
        sampled = self.update_sampler()(logprobs).reshape(1)
        if preferred_choice is not None and preferred_choice >= 0:
            sampled = mx.array([int(preferred_choice)])

        if not self.wants_measures():
            mx.eval(sampled)
            return int(sampled[0]), None
        return int(sampled[0]), self.measure_arrays(logits, sampled, read=True)

    def measure_arrays(self, logits, sampled, read=False):
        """Entropy over the whole vocabulary, plus the chosen token's own
        probability gathered ON THE DEVICE with mx.take - so nothing here needs
        the token's value in python, and the whole lot can be read in one go."""
        temp = max(float(self.temperature()), 1e-6)
        scaled = logits / temp
        scaled_logprobs = scaled - mx.logsumexp(scaled)
        probs = mx.exp(scaled_logprobs)
        entropy = -mx.sum(probs * scaled_logprobs)
        chosen = mx.take(scaled_logprobs, sampled)
        raw = mx.take(logits, sampled)
        if read:
            mx.eval(sampled, entropy, chosen, raw)
            return (float(entropy), float(mx.exp(chosen)[0]), float(raw[0]))
        return (entropy, chosen, raw)

    def sample_arrays(self, logits):
        """Sample without reading anything back: the token stays a device array
        so it can be fed forward before python ever sees it."""
        work = logits[None]
        penalty = float(self.repeat_penalty())
        if penalty != 1.0 and len(self.context_tokens) > 0:
            processors = make_logits_processors(repetition_penalty=penalty)
            history = mx.array(self.context_tokens[-64:])
            for processor in processors:
                work = processor(history, work)
        work = self.adjusted_logits(work[0])[None]
        logprobs = work - mx.logsumexp(work, keepdims=True)
        sampled = self.update_sampler()(logprobs).reshape(1)
        measures = self.measure_arrays(logits, sampled) if self.wants_measures() else None
        return sampled, measures

    def run_ahead(self):
        """One token, with the model a step in front of python.

        Reading each token back before issuing the next forward pass makes the
        GPU and the CPU take turns: measured 47.8 tok/s against 68.8 for the same
        loop reading one step late. So the sampled token is fed forward as a
        device array and only then read, which lets the GPU start the next token
        while python routes this one.

        The cost is that the token is chosen before the previous one has been
        looked at, so the end-of-turn nudge notices a sentence ending one token
        later than it otherwise would - immaterial in something that ramps over
        dozens of tokens. Anything that needs real per-token control - stepping,
        choosing from the candidates, stepping back - flushes this first.
        """
        if self.ahead_token is None:
            logits = self.next_logits()
            if logits is None:
                return None, None
            sampled, measures = self.sample_arrays(logits)
            mx.async_eval(sampled)
            self.ahead_token, self.ahead_logits, self.ahead_measures = sampled, logits, measures
        # feed it forward from the device, before reading it
        out = self.holder.model(self.ahead_token[None], cache=self.cache)
        following = out[:, -1, :]
        mx.async_eval(following)
        self.cache_ahead += 1
        token = int(self.ahead_token[0])
        self.logits_for_display = self.ahead_logits
        measures = self.ahead_measures
        self.ahead_token = self.ahead_logits = self.ahead_measures = None

        self.last_generated_token = token
        if token in (self.holder.im_end, self.holder.eos):
            self.discard_ahead()
            self.end_of_generation()
            return None, None
        self.context_tokens.append(token)
        self.cached_len += 1
        self.cache_ahead -= 1
        self.logits = following[0]
        self.new_response = False
        out_string = self.stream_piece(token)
        if out_string in ['.', '?', '!']:
            self.possible_stop = True
        if measures is not None:
            entropy, chosen, raw = measures
            mx.eval(entropy, chosen, raw)
            self.entropy = float(entropy)
            self.probability = float(mx.exp(chosen)[0])
            self.unnormed_probability = self.unnormalized_probability(float(raw[0]))
        return out_string, token

    def discard_ahead(self):
        """Throw away a speculative step. The cache has already absorbed the
        token and cannot be trimmed, so it is rewound from the turn snapshot -
        which is why that snapshot is taken."""
        self.ahead_token = self.ahead_logits = self.ahead_measures = None
        if self.cache_ahead > 0:
            self.cache_ahead = 0
            self.cached_len = len(self.context_tokens) + 1   # force a rewind
            self.logits = None
            self.sync_cache()

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
                elif key == 264:    # down arrow
                    if self.show_probs():
                        self.chosen_index -= 1
                        self.choose_possibility(self.chosen_index)
                    else:
                        self.layout_out.send(['scroll_down'])
                elif key == 265:    # up arrow
                    if self.show_probs():
                        self.chosen_index += 1
                        self.choose_possibility(self.chosen_index)
                    else:
                        self.layout_out.send(['scroll_up'])
                elif key == 259:    # backspace
                    if len(self.streaming_prompt) > 0:
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
        self.discard_ahead()
        """Picking a different candidate rewrites the tail of the context. The
        cache now describes one token too many, which sync_cache will sort out on
        the next step."""
        self.last_generated_token = int(token)
        if len(self.context_tokens) > 0:
            self.context_tokens[-1] = int(token)
            self.cached_len = min(self.cached_len, len(self.context_tokens) - 1)
            self.logits = None

    # ------------------------------------------------------------ execution

    def execute(self):
        if self.active_input != self.on_off:
            if not self.active:
                self.process_next_prompt()
        else:
            if self.on_off():
                self.stepping = False
                self.start_model_early()
            else:
                self.stepping = True

    def frame_task(self):
        if self.do_reset and not self.active:
            self.reset()
        self.process_next_prompt()

    def process_next_prompt(self):
        if not self.active and not self.queue.empty():
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
            self.ensure_model()
            with self.prefill_lock:
                self.prepare_prompt()
                self.next_logits()
                self.take_turn_snapshot()
            self.response_start = len(self.context_tokens)
            self.answer_start = self.response_start
            while True:
                if self.active == False or self.do_reset:
                    break

                free_running = (self.on_off() and not self.stepping
                                and self.take_step == 0 and preferred_id is None
                                and float(self.slow_down()) == 0)
                if free_running:
                    token_string, token_id = self.run_ahead()
                else:
                    self.discard_ahead()
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
                            spoken = self.holder.encode(self.streaming_prompt)
                            pairs = [[t, self.holder.piece(t)] for t in spoken]
                            if self.score_input():
                                self.eval_and_score(spoken)
                            else:
                                self.context_tokens += [int(t) for t in spoken]
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

        except Exception as e:
            print('qwen_moe:', e)
        self.active = False
        self.active_out.send(self.active)
        sys.exit()

    def sample_token(self, preferred_choice=None):
        logits = self.next_logits()
        if logits is None:
            return None, None
        self.logits_for_display = logits
        token, measures = self.sample_and_measure(logits, preferred_choice)

        self.last_generated_token = token
        if token in (self.holder.im_end, self.holder.eos):
            self.end_of_generation()
            return None, None

        self.context_tokens.append(token)
        self.new_response = False
        out_string = self.stream_piece(token)

        if out_string in ['.', '?', '!']:
            self.possible_stop = True

        if measures is not None:
            self.entropy, self.probability, raw_logit = measures
            self.unnormed_probability = self.unnormalized_probability(raw_logit)
        return out_string, token

    def emit_token(self, token_id, token_string):
        """Route one token: thinking to its own outlet, an unspoken aside to the
        action outlet, everything else to the text."""
        self.token_out.send(token_id)

        if token_id == self.holder.think_close:
            self.in_thinking = False
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
        """A single asterisk opens an unspoken aside; doubled asterisks are bold,
        which this model uses for words it IS saying."""
        return '*' in token_string and '**' not in token_string

    def ends_action(self, token_string):
        """Any single asterisk closes one, including the fused '.*' and '*.' this
        tokeniser produces, and so does the end of the line - an aside the model
        never closes must not swallow the speech after it."""
        if '*' in token_string and '**' not in token_string:
            return True
        return '\n' in token_string

    def stream_piece(self, token):
        """Sequential decode, holding an incomplete utf-8 tail until the next
        token completes the character."""
        buf = self.pending_utf8 + self.holder.piece(token).encode('utf-8')
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

    # ---------------------------------------------------------- measurements

    def unnormalized_probability(self, logit):
        """The same sigmoid gemma_4 uses, on a real logit - so 'sigmoid scaler'
        and 'sigmoid offset' behave as they do there, rather than on the shifted
        log-probability bonsai_2 has to make do with."""
        return float(1 / (1 + np.exp(self.sigmoid_scaler() * (logit + self.sigmoid_offset()))))

    def build_poss_dict(self, chosen_token):
        chosen_index = -1
        wanted = max(any_to_int(self.candidates()), 2)
        logits = getattr(self, 'logits_for_display', None)
        if logits is None:
            return -1
        order = mx.argsort(-logits)[:wanted]
        mx.eval(order)
        entries = []
        for token in order.tolist():
            token = int(token)
            word = self.holder.piece(token)
            if token in (self.holder.im_end, self.holder.eos):
                word = '<end>'
            elif word in ['\n', '\r']:
                word = '<return>'
            elif word == '\t':
                word = '<tab>'
            entries.append([word, float(logits[token]), token])
        entries = truncate_to_sampleable(entries, any_to_int(self.top_k()),
                                         float(self.top_p()), float(self.min_p()))
        # values stay RAW LOGITS, as gemma_4 does it - llm_layout applies
        # temperature itself from the ['temperature', value] message
        self.poss_dict = sorted(entries, key=lambda x: x[1])
        for i, entry in enumerate(self.poss_dict):
            if entry[2] == chosen_token:
                chosen_index = i
                break
        return chosen_index

    # ----------------------------------------------------------- prompt work

    def user_turn_prefix(self):
        prefix = '' if len(self.context_tokens) == 0 else '<|im_end|>\n'
        sys_block = ''
        if self.new_system_prompt:
            self.new_system_prompt = False
            body = self.system_prompt.strip()
            if body:
                sys_block = '<|im_start|>system\n' + body + '<|im_end|>\n'
        return prefix + sys_block + '<|im_start|>user\n'

    def generation_prompt(self):
        gen_prompt = '<|im_start|>assistant\n'
        if self.thinking():
            gen_prompt += '<think>\n'
            self.in_thinking = True
        else:
            gen_prompt += '<think>\n\n</think>\n\n'
            self.in_thinking = False
        return '<|im_end|>\n' + gen_prompt

    def strip_previous_thinking(self):
        """Drop earlier turns' reasoning, which is what this model's own template
        does by default. Left in, it gives the model its own deliberation to
        imitate and it restates the previous answer. An empty block is left
        alone - there is nothing in it to remove."""
        if self.keep_thinking():
            return
        tokens = self.context_tokens
        spans = []
        opened_at = None
        for index, token in enumerate(tokens):
            if token == self.holder.think_open:
                opened_at = index
            elif token == self.holder.think_close and opened_at is not None:
                end = index + 1
                if end < len(tokens) and tokens[end] == self.holder.double_return:
                    end += 1
                if index > opened_at + 2:
                    spans.append((opened_at, end))
                opened_at = None
        if opened_at is not None:
            spans.append((opened_at, len(tokens)))
        for start, end in reversed(spans):
            del tokens[start:end]
        if len(spans) > 0:
            self.cached_len = min(self.cached_len, spans[0][0])
            self.turn_snapshot = None
            self.turn_snapshot_len = -1
            self.logits = None

    def prepare_prompt(self):
        if self.spoken_turn_open:
            self.feed_spoken_text(self.prompt)
            self.close_spoken_turn()
            self.send_prompt_to_layout(self.prompt)
        else:
            self.format_and_tokenize_prompt()

    def format_and_tokenize_prompt(self, prompt=None):
        if prompt is None:
            prompt = self.prompt
        self.strip_previous_thinking()
        prefix = self.user_turn_prefix()
        if self.score_input():
            self.context_tokens += [int(t) for t in self.holder.encode(prefix)]
            self.eval_and_score(self.holder.encode(prompt))
            self.context_tokens += [int(t) for t in self.holder.encode(self.generation_prompt())]
        else:
            whole = prefix + prompt + self.generation_prompt()
            self.context_tokens += [int(t) for t in self.holder.encode(whole)]
        self.send_prompt_to_layout(prompt)

    def send_prompt_to_layout(self, prompt):
        if self.suppress_prompt_layout:
            self.suppress_prompt_layout = False
            return
        for token in self.holder.encode(prompt):
            self.layout_out.send(['prompt', token, self.holder.piece(token)])

    # ------------------------------------------- reading words as they arrive

    def request_prefill(self):
        if not self.preread() or self.holder is None:
            return
        if self.prefill_thread is None:
            self.prefill_stop = False
            self.prefill_thread = threading.Thread(target=self.prefill_worker, daemon=True)
            self.prefill_thread.start()
        self.prefill_wanted.set()

    def prefill_worker(self):
        while not self.prefill_stop:
            if not self.prefill_wanted.wait(0.25):
                continue
            self.prefill_wanted.clear()
            if self.prefill_stop:
                return
            text = self.streaming_prompt
            with self.prefill_lock:
                if self.active or self.do_reset or self.holder is None:
                    continue
                try:
                    if len(text.strip()) == 0:
                        if self.queue.empty():
                            self.abandon_spoken_turn()
                        continue
                    self.open_spoken_turn()
                    self.feed_spoken_text(text)
                except Exception as e:
                    print('qwen_moe pre-read:', e)
                    self.abandon_spoken_turn()

    def open_spoken_turn(self):
        if self.spoken_turn_open:
            return
        self.strip_previous_thinking()
        self.spoken_turn_start = len(self.context_tokens)
        self.spoken_turn_new_system = self.new_system_prompt
        self.context_tokens += [int(t) for t in self.holder.encode(self.user_turn_prefix())]
        self.spoken_text_start = len(self.context_tokens)
        self.spoken_turn_open = True

    def feed_spoken_text(self, text):
        """Bring the context up to date with the words heard so far, and read
        them into the model now rather than on submit. The whole utterance is
        re-tokenised each time: a byte-pair merge can reach across a word
        boundary, so appending fragment encodings could drift from what the model
        would see if the sentence arrived whole."""
        wanted = [int(t) for t in self.holder.encode(text)]
        already = self.context_tokens[self.spoken_text_start:]
        shared = 0
        while (shared < len(already) and shared < len(wanted)
               and already[shared] == wanted[shared]):
            shared += 1
        if shared < len(already):
            del self.context_tokens[self.spoken_text_start + shared:]
            self.cached_len = min(self.cached_len, len(self.context_tokens))
            self.logits = None
        fresh = wanted[shared:]
        if len(fresh) == 0:
            return
        if self.score_input():
            self.eval_and_score(fresh, opens_statement=(len(already) == 0))
        else:
            self.context_tokens += fresh
            self.next_logits()          # read it into the cache now

    def close_spoken_turn(self):
        self.context_tokens += [int(t) for t in self.holder.encode(self.generation_prompt())]
        self.spoken_turn_open = False

    def abandon_spoken_turn(self):
        if not self.spoken_turn_open:
            return
        del self.context_tokens[self.spoken_turn_start:]
        self.cached_len = min(self.cached_len, len(self.context_tokens))
        self.logits = None
        self.spoken_turn_open = False
        self.new_system_prompt = self.spoken_turn_new_system

    # --------------------------------------------------------------- scoring

    def eval_and_score(self, tokens, opens_statement=True):
        """Score each incoming token against the prediction the model held
        before it arrived. Exact here: the log-probability comes straight from
        the full-vocabulary distribution, with no need for bonsai_2's trick of
        forcing the token with a bias and reading the value back."""
        for i, token in enumerate(tokens):
            token = int(token)
            logits = self.next_logits()
            if logits is not None:
                logprobs = logits - mx.logsumexp(logits)
                value = logprobs[token]
                raw = logits[token]
                mx.eval(value, raw)
                self.context_tokens.append(token)
                self.score_incoming_token(token, float(value), float(raw),
                                          accumulate=(i > 0 or not opens_statement))
            else:
                self.context_tokens.append(token)

    def score_incoming_token(self, token, logprob, raw_logit, accumulate=True):
        unnormed = self.unnormalized_probability(raw_logit)
        self.input_score_out.send([self.holder.piece(token), logprob, unnormed])
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

    # ------------------------------------------------------------ stepping

    def back_step(self, count):
        self.discard_ahead()
        n_tokens = len(self.context_tokens)
        if n_tokens >= count:
            self.last_generated_token = self.context_tokens[n_tokens - count]
            if count == 1:
                preferred_choice = self.context_tokens[n_tokens - count]
            else:
                preferred_choice = self.context_tokens[n_tokens - count + 1]
            del self.context_tokens[n_tokens - count:]
            self.logits = None
            self.layout_out.send(['step_back', count])
            for i in range(count):
                self.output.send('<backspace>')
            return preferred_choice
        return -1

    def answer_produced(self):
        """Tokens of ANSWER so far. Reasoning does not count towards
        target_length, and the nudge never applies while the model is thinking -
        a thought is full of full stops, and ending the turn inside one loses the
        answer entirely."""
        return len(self.context_tokens) - self.answer_start

    def stop_ramp(self):
        if self.in_thinking:
            self.possible_stop = False
            return 0.0
        if self.force_stop_now:
            return 10.0
        if not self.possible_stop:
            return 0.0
        self.possible_stop = False
        if self.force_stop_at_next:
            self.force_stop_at_next = False
            return 10.0
        target = max(any_to_int(self.target_length()), 2)
        produced = self.answer_produced()
        start_ramp = target * 0.8
        if target <= start_ramp:
            return 1.0
        ratio = (produced - start_ramp) / (target - start_ramp)
        return min(max(ratio, 0.0), 1.0)

    # ------------------------------------------------------------ controls

    def end_of_generation(self):
        self.force_stop_now = False
        self.new_response = True
        self.active = False
        self.active_out.send(False)
        self.layout_out.send(['add', 0, '\n\n'])
        self.output_end_of_text.send('bang')

    def thinking_changed(self):
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
        if self.streaming_prompt.endswith('\n'):
            self.streaming_prompt = self.streaming_prompt[:-1]
            self.layout_out.send(['backspace_streaming_prompt'])
        if len(self.streaming_prompt.strip()) == 0:
            return
        if self.active:
            self.stepping = True
            self.take_step = 1
        else:
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
        self.sampler_params = None

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
        self.sampler_params = None

    def reset(self):
        self.reset_input_score()
        self.do_reset = False
        self.layout_out.send(['reset'])
        self.last_generated_token = None
        self.streaming_prompt = ''
        self.in_thinking = False
        self.in_action = False
        self.pending_utf8 = b''
        self.last_key_was_enter = False
        self.suppress_prompt_layout = False
        self.stepping = False
        self.take_step = 0
        self.force_stop_now = False
        self.force_stop_at_next = False
        self.possible_stop = False
        self.context_tokens = []
        self.cache = None
        self.cached_len = 0
        self.logits = None
        self.turn_snapshot = None
        self.turn_snapshot_len = -1
        self.ahead_token = None
        self.ahead_logits = None
        self.ahead_measures = None
        self.cache_ahead = 0
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
        self.cache = None
        self.turn_snapshot = None
        self.logits = None
        if self in QwenMoeChatNode.qwen_nodes:
            QwenMoeChatNode.qwen_nodes.remove(self)
