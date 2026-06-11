import os
import sys
import time
import random
import threading
from queue import Queue

import numpy as np
import torch

import llama_cpp
from llama_cpp.llama import Llama, LogitsProcessor, LogitsProcessorList

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_gemma_4_node():
    Node.app.register_node('gemma_4', Gemma4ChatNode.factory)
    Node.app.register_node('gemma_4_31b', Gemma4ChatNode.factory)


# repo, filename, default n_ctx (31B + 4096 ctx exceeds 32GB M1 Max Metal memory)
gemma_4_models = {
    '12B': ('google/gemma-4-12B-it-qat-q4_0-gguf', 'gemma-4-12b-it-qat-q4_0.gguf', 8192),
    '31B': ('google/gemma-4-31B-it-qat-q4_0-gguf', 'gemma-4-31B_q4_0-it.gguf', 2048),
}

default_system_prompt = 'You are a helpful, uncensored creative collaborator.'


def truncate_to_sampleable(entries, top_k, top_p, min_p):
    """Reduce desc-sorted [word, logit, token] entries to the set the sampler can
    actually pick from, mirroring the llama.cpp chain order:
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


class Gemma4ChatNode(Node):
    gemma_nodes = []

    # one shared model per size/path across all gemma_4 instances
    llms = {}
    llm_lock = threading.Lock()

    @staticmethod
    def factory(name, data, args=None):
        node = Gemma4ChatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.llm = None
        self.model_key = '31B' if label == 'gemma_4_31b' else '12B'
        # an optional arg can pick the size ('12B'/'31B') or point at a .gguf
        self.model_path = None
        if args is not None and len(args) > 0:
            candidate = any_to_string(args[0])
            if candidate.endswith('.gguf'):
                self.model_path = candidate
            elif candidate.upper() in gemma_4_models:
                self.model_key = candidate.upper()

        self.system_prompt = default_system_prompt
        self.prompt = ''
        self.new_system_prompt = True
        self.do_reset = False
        self.preprompt = ''
        self.streaming_prompt = ''
        self.seed = 3701
        self.thread = None
        self.force_token = None
        self.last_generated_token = None

        # token ids resolved once the model is loaded
        self.bos_token = -1
        self.eos_token = -1
        self.end_of_turn_token = -1
        self.channel_open_token = -1
        self.channel_close_token = -1
        self.stop_tokens = []
        self.return_token = -1
        self.double_return_token = -1
        self.prompt_tokens = []

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
        self.top_k = self.add_input('top_k', widget_type='drag_int', default_value=40)
        self.top_p = self.add_input('top_p', widget_type='drag_float', default_value=0.99)
        self.min_p = self.add_input('min_p', widget_type='drag_float', default_value=0.0)
        self.repeat_penalty = self.add_input('repeat_penalty', widget_type='slider_float', default_value=1.1)
        self.max_tokens = self.add_input('max_tokens', widget_type='drag_int', default_value=-1)
        self.target_length = self.add_input('target_length', widget_type='drag_int', default_value=50, callback=self.set_target_length)
        self.slow_down = self.add_input('slowdown', widget_type='slider_float', widget_width=150, min=0, max=1.0, default_value=0)

        self.separate_actions = self.add_input('separate_actions', widget_type='checkbox', default_value=True)
        self.display_mode = self.add_input('display_mode', widget_type='combo', default_value='temperature')
        self.display_mode.widget.combo_items = ['temperature', 'entropy', 'probability', 'unnormed_probability']
        self.sigmoid_scaler = self.add_input('sigmoid scaler', widget_type='drag_float', default_value=0.2)
        self.sigmoid_offset = self.add_input('sigmoid offset', widget_type='drag_float', default_value=0)

        self.show_probs = self.add_input('show_probs', widget_type='checkbox', default_value=False, callback=self.show_probs_changed)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_text)

        default_ctx = gemma_4_models[self.model_key][2] if self.model_path is None else 8192
        self.n_ctx = self.add_option('n_ctx', widget_type='drag_int', default_value=default_ctx)
        self.n_gpu_layers = self.add_option('n_gpu_layers', widget_type='drag_int', default_value=-1)
        self.thinking = self.add_option('thinking', widget_type='checkbox', default_value=False,
                                        callback=self.thinking_changed)
        # off: layout token list stays aligned with context tokens only for the
        # visible text, so back-stepping will pass through invisible thought tokens
        self.thinking_in_layout = self.add_option('thinking in layout', widget_type='checkbox', default_value=True)

        self.output = self.add_output('output')
        self.thinking_out = self.add_output('thinking')
        self.output_end_of_text = self.add_output('end')
        self.token_out = self.add_output('token_out')
        self.layout_out = self.add_output('layout_out')
        self.actions_out = self.add_output('actions_out')
        self.active_out = self.add_output('active')

        self.new_response = True
        self.queue = Queue(maxsize=16)
        self.active = False
        self.stepping = False
        self.in_step = False
        self.last_key_was_enter = False
        self.suppress_prompt_layout = False
        self.in_thinking = False
        self.thinking_header = False
        self.pending_utf8 = b''
        self.take_step = 0
        self.stopping = False
        self.output_string = ''
        self.formatted_prompt = ''
        self.next_period_counter = 0

        self.logits_processor = None
        self.logits_processor_list = None
        self.sampler_params = None
        self.my_sampler = None
        self.choices = []
        self.scores = None
        self.poss_dict = []
        self.chosen_index = 0
        self.entropy = 0.1
        self.probability = 0.0
        self.unnormed_probability = 0.0

        Gemma4ChatNode.gemma_nodes.append(self)
        self.add_frame_task()

    # ---------------------------------------------------------------- model

    def ensure_model(self):
        cls = Gemma4ChatNode
        key = self.model_path if self.model_path else self.model_key
        with cls.llm_lock:
            llm = cls.llms.get(key)
            if llm is None:
                path = self.model_path
                if path is None or not os.path.exists(path):
                    from huggingface_hub import hf_hub_download
                    repo_id, filename = gemma_4_models[self.model_key][:2]
                    print('gemma_4: fetching', repo_id, '(cached after first download)')
                    path = hf_hub_download(repo_id=repo_id, filename=filename)
                print('gemma_4: loading model', path)
                llm = Llama(
                    model_path=path,
                    n_gpu_layers=any_to_int(self.n_gpu_layers()),
                    n_ctx=any_to_int(self.n_ctx()),
                    flash_attn=True,
                    seed=self.seed,
                    verbose=False,
                )
                cls.llms[key] = llm
                print('gemma_4: model loaded')
        self.llm = llm
        if self.eos_token == -1:
            tokenizer = llm.tokenizer()
            self.bos_token = llm.token_bos()
            self.eos_token = llm.token_eos()
            # gemma 4 turn structure: <|turn>role\n ... <turn|>\n  with optional
            # <|channel>thought ... <channel|> thinking blocks inside model turns
            self.end_of_turn_token = tokenizer.encode('<turn|>', add_bos=False, special=True)[-1]
            self.channel_open_token = tokenizer.encode('<|channel>', add_bos=False, special=True)[-1]
            self.channel_close_token = tokenizer.encode('<channel|>', add_bos=False, special=True)[-1]
            self.stop_tokens = [self.eos_token, self.end_of_turn_token]
            self.return_token = tokenizer.encode('\n', add_bos=False)[-1]
            self.double_return_token = tokenizer.encode('\n\n', add_bos=False)[-1]
        if self.logits_processor is None:
            self.logits_processor = EosTokenRewardLogitsProcessor(self.end_of_turn_token, max(any_to_int(self.target_length()), 2))
            self.logits_processor.set_owner(self)
            self.logits_processor.suppress_token = self.bos_token
            self.logits_processor.calc_probs = self.show_probs()
            self.logits_processor_list = LogitsProcessorList([self.logits_processor])
        return llm

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
        self.last_generated_token = self.poss_dict[self.chosen_index][2]

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
        self.last_generated_token = self.poss_dict[self.chosen_index][2]

    # ------------------------------------------------------------ execution

    def execute(self):
        if self.active_input != self.on_off:
            if not self.active:
                self.process_next_prompt()
        else:
            if self.on_off():
                self.stepping = False
            else:
                self.stepping = True

    def frame_task(self):
        if self.do_reset and not self.active:
            self.reset()
        self.process_next_prompt()

    def process_next_prompt(self):
        if not self.active and not self.queue.empty():
            self.prompt = self.queue.get()
            self.active = True
            self.active_out.send(True)
            if self.logits_processor is not None:
                self.logits_processor.set_new_response()
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
        output_text = ''
        in_action = False
        start_of_action = False
        self.force_token = None
        preferred_id = None
        self.in_thinking = False
        self.thinking_header = False
        self.pending_utf8 = b''
        try:
            llm = self.ensure_model()
            self.logits_processor.set_new_response()
            self.logits_processor.set_max_length(max(any_to_int(self.target_length()), 2))

            self.format_and_tokenize_prompt()
            tokens = self.prompt_tokens
            token_string = ''
            while token_string is not None:
                if self.active == False or self.do_reset:
                    break
                token_string, token_id = self.sample_token(tokens, preferred_id)
                preferred_id = None
                if token_string is not None:
                    self.token_out.send(token_id)

                    if token_id == self.channel_open_token:
                        # model is opening its thinking channel
                        self.in_thinking = True
                        self.thinking_header = True
                        if self.thinking_in_layout():
                            self.send_to_layout(token_id, token_string)
                    elif token_id == self.channel_close_token:
                        self.in_thinking = False
                        if self.thinking_in_layout():
                            self.send_to_layout(token_id, token_string)
                    elif self.in_thinking:
                        if self.thinking_header:
                            # swallow the channel name line ('thought\n')
                            if '\n' in token_string:
                                self.thinking_header = False
                        else:
                            self.thinking_out.send(token_string)
                        if self.thinking_in_layout():
                            self.send_to_layout(token_id, token_string)
                    else:
                        if '*' in token_string:
                            if not in_action:
                                start_of_action = True
                                in_action = True
                        if not in_action:
                            output_text += token_string

                        if self.separate_actions() and in_action:
                            self.actions_out.send(token_string)
                            if '*' in token_string and not start_of_action:
                                in_action = False
                                self.actions_out.send('\n')
                        else:
                            in_action = False
                            self.output.send(token_string)
                            self.send_to_layout(token_id, token_string)

                    tokens = [self.last_generated_token]
                    start_of_action = False

                if not self.on_off():
                    if self.stepping:
                        while self.take_step == 0 and not self.on_off() and not self.do_reset:
                            self.in_step = True
                            time.sleep(0.005)
                        self.in_step = False
                        if self.take_step == 1 and len(self.streaming_prompt) > 0:
                            streaming_tokens = self.encode_continuation(self.streaming_prompt)
                            pairs = []
                            for token in streaming_tokens:
                                pairs += [[token, self.decode_token(token)]]
                            tokens = streaming_tokens
                            self.output.send(self.streaming_prompt)
                            self.layout_out.send(['streaming_prompt', pairs])
                            self.layout_out.send(['accept_streamed_prompt'])
                            self.streaming_prompt = ''
                        else:
                            tokens = [self.last_generated_token]
                        if self.take_step == -1:
                            preferred_id = self.back_step(2)
                            tokens = [self.last_generated_token]
                        self.take_step = 0
                    else:
                        break
                else:
                    sleeper = self.slow_down()
                    if sleeper > 0:
                        time.sleep(sleeper)

        except Exception as e:
            print('gemma_4:', e)
            if 'returned -3' in str(e):
                print('gemma_4: llama_decode -3 usually means Metal ran out of memory'
                      ' — reduce n_ctx or close other GPU-heavy apps')
        self.active = False
        self.active_out.send(self.active)
        sys.exit()

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

    def format_and_tokenize_prompt(self, prompt=None):
        llm = self.llm
        if prompt is None:
            prompt = self.prompt
        thinking = self.thinking()
        # close the model's previous turn (the sampled <turn|> stop token is never eval'd)
        prefix = '' if llm.n_tokens == 0 else '<turn|>\n'
        sys_block = ''
        if self.new_system_prompt:
            self.new_system_prompt = False
            think_tag = '<|think|>\n' if thinking else ''
            sys_block = '<|turn>system\n' + think_tag + self.system_prompt.strip() + '<turn|>\n'
        gen_prompt = '<|turn>model\n'
        if not thinking:
            # an empty thought channel makes the model answer directly
            gen_prompt += '<|channel>thought\n<channel|>'
        self.formatted_prompt = (prefix + sys_block + '<|turn>user\n' + prompt
                                 + '<turn|>\n' + gen_prompt)

        add_bos = (llm.n_tokens == 0)
        self.prompt_tokens = llm.tokenizer().encode(self.formatted_prompt, add_bos=add_bos, special=True)

        if self.suppress_prompt_layout:
            self.suppress_prompt_layout = False
        else:
            just_prompt_tokens = self.encode_continuation(prompt)
            for token in just_prompt_tokens:
                self.layout_out.send(['prompt', token, self.decode_token(token)])

    def encode_continuation(self, text):
        tokens = self.llm.tokenizer().encode(text, add_bos=False, special=False)
        if len(tokens) > 0 and tokens[0] == self.bos_token:
            tokens = tokens[1:]
        return tokens

    def update_sampler(self, llm):
        # llama.cpp's dist sampler holds the RNG state: it must persist across
        # tokens (a per-token temporary sampler re-seeds the RNG every step,
        # collapsing sampling to a near-greedy pick). Rebuild only on param change.
        params = (any_to_int(self.top_k()), float(self.top_p()), float(self.min_p()),
                  float(self.temperature()), float(self.repeat_penalty()), int(self.seed))
        if llm._sampler is None or llm._sampler is not self.my_sampler or params != self.sampler_params:
            if llm._sampler is not None:
                llm._sampler.close()
            llm._sampler = llm._init_sampler(
                top_k=params[0],
                top_p=params[1],
                min_p=params[2],
                temp=params[3],
                repeat_penalty=params[4],
                logits_processor=self.logits_processor_list,
            )
            self.my_sampler = llm._sampler
            self.sampler_params = params

    def sample_token(self, tokens, preferred_choice=None):
        llm = self.llm
        llm.eval(tokens)
        sample_idx = llm.n_tokens - 1

        self.update_sampler(llm)
        out_token = llm.sample(idx=sample_idx)
        if preferred_choice is not None and preferred_choice >= 0:
            out_token = preferred_choice

        self.last_generated_token = out_token
        if out_token in self.stop_tokens:
            self.end_of_generation()
            self.last_generated_token = self.double_return_token
            return None, None

        self.new_response = False
        out_string = self.decode_stream_token(out_token)

        if out_string in ['.', '?', '!']:
            self.logits_processor.set_possible_stop()

        if self.logits_processor.logits is not None:
            temp = max(float(self.temperature()), 1e-6)
            scaled = self.logits_processor.logits / temp
            dist = torch.distributions.Categorical(logits=scaled)
            self.entropy = float(dist.entropy())
            self.probability = float(torch.nn.functional.softmax(scaled, dim=-1)[int(out_token)])
            self.unnormed_probability = float(self.unnormalized_probability(self.logits_processor.logits[int(out_token)]))

        return out_string, out_token

    def unnormalized_probability(self, logit):
        return 1 / (1 + torch.exp(self.sigmoid_scaler() * (logit + self.sigmoid_offset())))

    def decode_token(self, token):
        # stateless decode for display lists; partial multibyte tokens show empty
        llm = self.llm
        try:
            return llm.detokenize([int(token)]).decode('utf-8')
        except UnicodeDecodeError:
            return ''

    def decode_stream_token(self, token):
        # sequential decode: hold incomplete utf-8 tails until the next token
        # completes the character, instead of emitting mojibake per token
        llm = self.llm
        buf = self.pending_utf8 + llm.detokenize([int(token)])
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

    def back_step(self, count):
        llm = self.llm
        if llm is None:
            return -1
        n_tokens = llm.n_tokens
        if n_tokens >= count:
            self.last_generated_token = llm._input_ids[n_tokens - count]
            if count == 1:
                preferred_choice = llm._input_ids[n_tokens - count]
            else:
                preferred_choice = llm._input_ids[n_tokens - count + 1]
            self.trim_context_tail(count)
            self.layout_out.send(['step_back', count])
            for i in range(count):
                self.output.send('<backspace>')
            return preferred_choice
        return -1

    def trim_context_tail(self, n_discard):
        # drop the last n_discard tokens from the kv cache and token history;
        # since we only ever trim the tail no position shift is needed
        llm = self.llm
        n_tokens = llm.n_tokens
        n_keep = n_tokens - n_discard
        llm._ctx.kv_cache_seq_rm(-1, n_keep, -1)
        llm.input_ids[n_keep:n_tokens] = 0
        llm.n_tokens = n_keep

    def build_poss_dict(self, chosen_token):
        chosen_index = -1
        if self.scores is not None:
            entries = []
            for entry in self.scores:
                logit = entry[1]
                if logit > -1000:
                    token = int(entry[0])
                    token_word = self.decode_token(token)
                    if token in self.stop_tokens:
                        token_word = '<end>'
                    elif token_word in ['\n', '\r']:
                        token_word = '<return>'
                    elif token_word == '\t':
                        token_word = '<tab>'
                    entries.append([token_word, logit, token])
            entries = truncate_to_sampleable(entries, any_to_int(self.top_k()),
                                             float(self.top_p()), float(self.min_p()))
            # values stay raw logits: llm_layout applies temperature scaling itself,
            # driven by the ['temperature', value] message sent from temp_changed
            self.poss_dict = sorted(entries, key=lambda x: x[1])
            for i, entry in enumerate(self.poss_dict):
                if entry[2] == chosen_token:
                    chosen_index = i
                    break
        return chosen_index

    # ------------------------------------------------------------ controls

    def end_of_generation(self):
        if self.logits_processor is not None:
            self.logits_processor.has_stopped()
        self.new_response = True
        self.active = False
        self.active_out.send(False)
        self.layout_out.send(['add', 0, '\n\n'])
        self.output_end_of_text.send('bang')

    def thinking_changed(self):
        # <|think|> lives in the system turn, so re-emit it with the next prompt
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
            return
        elif len(self.streaming_prompt) == 0:
            self.output.send('<backspace>')
            if len(streaming_prompt_string) > 0:
                if streaming_prompt_string[0] != ' ':
                    self.streaming_prompt += ' '
        self.streaming_prompt += streaming_prompt_string
        self.layout_out.send(['streaming_prompt', [[0, self.streaming_prompt]]])

    def set_target_length(self):
        if self.logits_processor is not None:
            self.logits_processor.set_max_length(max(any_to_int(self.target_length()), 2))

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
        if self.logits_processor is not None:
            self.logits_processor.stop()

    def polite_stop(self):
        if self.logits_processor is not None:
            self.logits_processor.stop_at_next()

    def hard_stop(self):
        self.end_of_generation()

    def show_probs_changed(self):
        self.layout_out.send(['show_probs', self.show_probs()])
        if self.logits_processor is not None:
            self.logits_processor.calc_probs = self.show_probs()

    def randomize_seed(self):
        self.seed = random.randint(0, 2 ** 32 - 1)
        self.seed_input.set(self.seed)

    def set_seed(self):
        self.seed = self.seed_input()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        llm = self.llm
        if llm is not None:
            llm.set_seed(self.seed)
        self.sampler_params = None  # force sampler rebuild with the new seed

    def reset(self):
        self.next_period_counter = 0
        if self.logits_processor is not None:
            self.logits_processor.reset()
        self.do_reset = False
        self.layout_out.send(['reset'])
        self.last_generated_token = None
        self.streaming_prompt = ''
        self.in_thinking = False
        self.thinking_header = False
        self.pending_utf8 = b''
        self.last_key_was_enter = False
        self.suppress_prompt_layout = False
        self.stepping = False
        self.take_step = 0
        llm = self.llm
        if llm is not None:
            llm._ctx.kv_cache_clear()
            llm.input_ids[:] = 0
            llm.n_tokens = 0
        self.new_system_prompt = True
        self.queue.queue.clear()
        self.set_seed()

    def custom_cleanup(self):
        self.active = False
        llm = self.llm
        if llm is not None and llm._sampler is not None and llm._sampler is self.my_sampler:
            llm._sampler.close()
            llm._sampler = None
        if self in Gemma4ChatNode.gemma_nodes:
            Gemma4ChatNode.gemma_nodes.remove(self)


class EosTokenRewardLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id: int, max_length: int):
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f'`eos_token_id` has to be a positive integer, but is {eos_token_id}')
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f'`max_length` has to be a integer bigger than 1, but is {max_length}')

        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.current_start = 0
        self.set_start = False
        self.force_stop = False
        self.possible_stop = False
        self.force_stop_at_next = False
        self.scores = None
        self.collect_scores = False
        self.owner = None
        self.calc_probs = False
        self.entropy = 0.1
        self.probabilities = None
        self.logits = None
        self.suppress_token = -1

    def reset(self):
        self.current_start = 0
        self.set_start = False
        self.force_stop = False
        self.possible_stop = False
        self.force_stop_at_next = False

    def set_owner(self, owner):
        self.owner = owner

    def set_max_length(self, length):
        self.max_length = length

    def set_new_response(self):
        self.set_start = True

    def stop(self):
        self.force_stop = True

    def stop_at_next(self):
        self.force_stop_at_next = True

    def set_possible_stop(self):
        self.possible_stop = True

    def has_stopped(self):
        self.force_stop = False

    def __call__(self, input_ids, scores):
        if self.set_start:
            self.set_start = False
            self.current_start = input_ids.shape[-1]

        if self.suppress_token >= 0:
            scores[self.suppress_token] = -float('inf')

        if self.calc_probs:
            sorted_indices = np.argsort(scores)
            top = sorted_indices[::-1][:40]
            sorted_pairs = []
            for i in top:
                sorted_pairs.append([int(i), float(scores[i])])
            if self.owner is not None:
                self.owner.scores = sorted_pairs

        self.logits = torch.from_numpy(np.array(scores))

        cur_len = input_ids.shape[-1] - self.current_start
        start_eos_ramp = self.max_length * .8
        ratio = (cur_len - start_eos_ramp) / (self.max_length - start_eos_ramp)
        if ratio < 0:
            ratio = 0
        if ratio > 1.0:
            ratio = 1.0

        max_score = np.max(scores)

        if self.force_stop:
            scores[self.eos_token_id] = scores[self.eos_token_id] + max_score * 10
        elif self.possible_stop:
            if self.force_stop_at_next:
                scores[self.eos_token_id] = scores[self.eos_token_id] + max_score * 10
                self.force_stop_at_next = False
            else:
                scores[self.eos_token_id] = scores[self.eos_token_id] + max_score * ratio
            self.possible_stop = False
        if self.collect_scores:
            self.scores = scores.copy()
        return scores
