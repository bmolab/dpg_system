"""
nemotron_nodes.py
Real-time speech-to-text node built on a cache-aware streaming transducer
(NVIDIA Nemotron 3.5 ASR streaming 0.6B) running on MLX via mlx-audio.

Unlike the whisper node, nothing is re-decoded: every token the model emits is
final the instant it appears. The node therefore has no confirmation heuristic;
it only decides where phrases and sentences end.

Outputs
    in_progress          the current phrase, growing token by token (feeds the
                         newest fifo_string register)
    phrase               a phrase, closed by a pause, a sentence end, or a
                         length cap (shifts the fifo)
    sentence_in_progress the sentence so far, re-sent each time a phrase closes
                         (feeds context_tracker 'provisional in')
    sentence             a sentence, closed by sentence-final punctuation, a
                         long pause, or a length cap (feeds context_tracker
                         'text in')

The model emits sentence-final punctuation together with the first word of the
NEXT sentence, so a pause usually closes the phrase before its period arrives.
A bare punctuation token arriving right after such a phrase is attached to the
sentence rather than starting a new phrase.

Created by David Rokeby, 2026.
"""

import os
import queue
import re
from bisect import bisect_right
import threading
import time
import traceback
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from dpg_system.node import Node
from dpg_system.audio_io import AudioSource, RateConverter, input_devices, to_mono
from dpg_system.conversion_utils import *

SAMPLE_RATE = 16000
BLOCK = 1280  # 80 ms at 16 kHz = one encoder frame after 8x subsampling

MODELS = {
    'nemotron-3.5-0.6b': 'mlx-community/nemotron-3.5-asr-streaming-0.6b',
    'nemotron-3.5-0.6b-8bit': 'mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit',
}
# look-ahead label -> right context in encoder frames (80 ms each)
CONTEXTS = {'80 ms': 0, '160 ms': 1, '320 ms': 3, '560 ms': 6, '1120 ms': 13}
LEFT_CONTEXT = 56

SENTENCE_END = ('.', '?', '!')
# punctuation-restoration model used by the semantic sentence splitter (CPU, ~1.1 GB)
SPLIT_MODEL = 'oliverguhr/fullstop-punctuation-multilingual-sonar-base'
BARE_PUNCT = {'.', ',', '?', '!', ';', ':'}


def register_nemotron_nodes():
    Node.app.register_node('nemotron', NemotronNode.factory)


# ─────────────────────────────────────────────────────────────────────────────
# Audio capture: device or external → 16 kHz mono blocks on a queue
# ─────────────────────────────────────────────────────────────────────────────

class NemotronCapture:
    def __init__(self):
        self.blocks: 'queue.Queue[np.ndarray]' = queue.Queue()
        self.gain = 1.0
        self.energy = 0.0
        self.running = False
        self.source: Optional[AudioSource] = None
        self.devices: List[dict] = []
        self.current_device = -1
        self.channels = 1
        self._mic_converter: Optional[RateConverter] = None
        self._external_converter: Optional[RateConverter] = None
        self._external_rate = 0

    def get_device_list(self) -> List[str]:
        self.devices = input_devices()
        return [d['name'] for d in self.devices]

    def init(self, device_index: int) -> bool:
        if not self.devices:
            self.get_device_list()
        if not self.devices:
            return False
        if device_index >= len(self.devices):
            device_index = 0
        device = self.devices[device_index]
        sr = int(device['default_samplerate'])
        self.channels = min(device['channels'], 2)
        self._mic_converter = RateConverter(sr, SAMPLE_RATE)
        try:
            if self.source is not None:
                self.source.stop()
            self.source = AudioSource(channels=self.channels, rate=sr,
                                      chunk=1024, dtype='float32')
            self.source.device_index = device['index']
            self.source.set_callback(self._audio_callback)
            self.current_device = device_index
            return True
        except Exception as e:
            print(f"nemotron: failed to open audio device: {e}")
            return False

    def change_device(self, device_index: int) -> bool:
        if device_index == self.current_device:
            return True
        was_running = self.running
        self.pause()
        ok = self.init(device_index)
        if ok and was_running:
            self.resume()
        return ok

    def resume(self):
        if self.source is not None:
            if self._mic_converter is not None:
                self._mic_converter.reset()
            self.running = self.source.start()

    def pause(self):
        if self.source is not None:
            self.source.stop()
        self.running = False

    def close(self):
        self.pause()
        self.source = None

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        mono = to_mono(indata[:, :self.channels])
        if self._mic_converter is not None:
            mono = self._mic_converter.process(mono)
        self._push(mono)

    def feed_external(self, audio: np.ndarray, sample_rate: int):
        audio = to_mono(audio)
        if sample_rate > 0 and sample_rate != SAMPLE_RATE:
            if self._external_converter is None or self._external_rate != sample_rate:
                self._external_converter = RateConverter(sample_rate, SAMPLE_RATE)
                self._external_rate = sample_rate
            audio = self._external_converter.process(audio)
        self._push(audio)

    def _push(self, audio: np.ndarray):
        if len(audio) == 0:
            return
        audio = np.asarray(audio, dtype=np.float32) * self.gain
        self.energy = float(np.sqrt(np.mean(audio * audio)))
        self.blocks.put(audio)

    def drain(self):
        while True:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                return


# ─────────────────────────────────────────────────────────────────────────────
# Semantic sentence splitter: punctuation restoration over the open text
# ─────────────────────────────────────────────────────────────────────────────

class PunctuationSplitter:
    """Predicts where sentence-final punctuation belongs in unpunctuated text.
    The ASR marks only some sentence ends in continuous speech; this finds the rest."""

    def __init__(self, model_id: str = SPLIT_MODEL):
        self.model_id = model_id
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()
        t = time.time()
        self.pipe = pipeline('token-classification', model=self.model_id,
                             aggregation_strategy='none', device='cpu')
        print(f"nemotron: loaded sentence splitter {self.model_id} in {time.time() - t:.1f}s")

    def marks(self, words: List[str]) -> dict:
        """Per word index, the strongest predicted mark after that word as
        (label, score); words predicted to carry no mark are absent. Labels are the
        model's: '.', '?', ',', ':', '-'."""
        if self.pipe is None or not words:
            return {}
        clean = [re.sub(r"[^\w'\u2019-]", "", w) for w in words]
        starts = []
        pos = 0
        for w in clean:
            starts.append(pos)
            pos += len(w) + 1
        text = ' '.join(clean).lower()
        try:
            out = self.pipe(text)
        except Exception as e:
            print(f"nemotron: splitter failed: {e}")
            return []
        marks = {}
        for tok in out:
            label = tok.get('entity')
            if label in (None, '0', 'O') or tok.get('start') is None:
                continue
            i = bisect_right(starts, tok['start']) - 1
            if 0 <= i < len(words):
                score = float(tok.get('score', 0.0))
                # a sentence end beats a clause mark on the same word; else keep the stronger
                prev = marks.get(i)
                if (prev is None or (label in SENTENCE_END) > (prev[0] in SENTENCE_END)
                        or ((label in SENTENCE_END) == (prev[0] in SENTENCE_END) and score > prev[1])):
                    marks[i] = (label, score)
        return marks

    def sentence_ends(self, words: List[str]) -> List[int]:
        """Indices i such that a sentence ends after words[i]."""
        return sorted(k for k, (label, _) in self.marks(words).items() if label in SENTENCE_END)


# ─────────────────────────────────────────────────────────────────────────────
# Phrase / sentence segmentation over a committed token stream
# ─────────────────────────────────────────────────────────────────────────────

class Segmenter:
    """Turns final tokens into in_progress / phrase / sentence_in_progress /
    sentence events. Text only ever grows between boundaries."""

    def __init__(self):
        self.phrase_silence = 0.8      # s of audio with no token → phrase closes
        self.sentence_silence = 2.0    # s of audio with no token → sentence closes
        self.max_phrase_words = 0      # 0 = off
        self.max_sentence_words = 0    # 0 = off
        # semantic splitting (None = off)
        self.splitter: Optional[PunctuationSplitter] = None
        self.split_min_words = 12      # open text must be at least this long to check
        self.split_guard_words = 3     # never split inside the last N words (no right context yet)
        self.split_check_every = 6     # also check after this many new words without a phrase close
        self.split_max_words = 30      # past this, with no sentence end found, break at the best clause boundary (0 = off)
        self._words_since_check = 0
        self.current = ''
        self.sentence = ''
        self.last_token_audio: Optional[float] = None
        self.last_phrase_by_silence = False
        self.events: List[Tuple[str, str]] = []

    def reset(self):
        self.current = ''
        self.sentence = ''
        self.last_token_audio = None
        self.last_phrase_by_silence = False
        self._words_since_check = 0

    def _emit(self, kind, text):
        if text:
            self.events.append((kind, text))

    def _close_phrase(self, how):
        text = self.current.strip()
        self.current = ''
        if not text:
            return False
        self._emit('phrase', text)
        self.sentence = (self.sentence + ' ' + text).strip()
        self.last_phrase_by_silence = how == 'silence'
        return True

    def _close_sentence(self, how):
        text = self.sentence.strip()
        self.sentence = ''
        self.last_phrase_by_silence = False
        if text:
            self._emit('sentence', text)

    def token(self, text: str, audio_t: float):
        self.last_token_audio = audio_t
        stripped = text.strip()
        # late punctuation belonging to the phrase a pause just closed
        if self.last_phrase_by_silence and not self.current and stripped in BARE_PUNCT:
            self.sentence += stripped
            if stripped in SENTENCE_END:
                self._close_sentence('punct')
            else:
                self._emit('sentence_in_progress', self.sentence)
            return
        self.last_phrase_by_silence = False
        self.current += text
        self._emit('in_progress', self.current.strip())
        if text.startswith(' '):
            self._words_since_check += 1
        if stripped.endswith(SENTENCE_END):
            self._close_phrase('punct')
            self._close_sentence('punct')
        elif self.max_phrase_words > 0 and len(self.current.split()) >= self.max_phrase_words:
            self._close_phrase('length')
            if self.max_sentence_words > 0 and len(self.sentence.split()) >= self.max_sentence_words:
                self._close_sentence('length')
            elif not self._maybe_split():
                self._emit('sentence_in_progress', self.sentence)
        elif self.splitter is not None and self._words_since_check >= self.split_check_every:
            self._maybe_split()

    def tick(self, audio_t: float):
        """Called after every push with the audio time the encoder has actually
        decoded up to (engine.decoded_t), not the time fed - between encoder
        chunks no token can arrive, so counting that time as silence would cut
        a phrase at every chunk boundary at long look-aheads."""
        if self.last_token_audio is None:
            return
        gap = audio_t - self.last_token_audio
        if self.current.strip() and gap >= self.phrase_silence:
            self._close_phrase('silence')
            if self.max_sentence_words > 0 and len(self.sentence.split()) >= self.max_sentence_words:
                self._close_sentence('length')
            elif not self._maybe_split():
                self._emit('sentence_in_progress', self.sentence)
        if self.sentence and not self.current and gap >= self.sentence_silence:
            self._close_sentence('silence')

    def _maybe_split(self) -> bool:
        """Ask the splitter where sentences end in the open text (sentence buffer plus
        current phrase) and close them. Returns True if it emitted anything, in which
        case it has already re-sent sentence_in_progress for what remains."""
        if self.splitter is None:
            return False
        sw = self.sentence.split()
        cw = self.current.split()
        words = sw + cw
        if len(words) < self.split_min_words:
            return False
        self._words_since_check = 0
        limit = len(words) - 1 - self.split_guard_words
        marks = self.splitter.marks(words)
        # where the ASR itself heard a weaker mark (comma, semicolon, colon) trust it:
        # the splitter only fills in the sentence ends the ASR left unmarked
        min_head = max(4, self.split_min_words // 2)
        ends = []
        start = 0
        for k in sorted(marks):
            if (marks[k][0] in SENTENCE_END and 0 <= k <= limit
                    and not words[k].endswith((',', ';', ':'))
                    and k - start + 1 >= min_head):   # no one- or two-word "sentences"
                ends.append(k)
                start = k + 1
        if not ends and self.split_max_words > 0 and len(words) >= self.split_max_words:
            k = self._best_clause_break(words, marks, limit)
            if k is not None:
                ends = [k]
        if not ends:
            return False
        nsw = len(sw)
        into_current = [k for k in ends if k >= nsw]
        if into_current:
            # the last cut falls inside the current phrase: close that much of it.
            # Cut the string at the character, not by re-joining words - the model can
            # emit a bare space token followed by continuation pieces, and re-joining
            # would drop that space and glue the next word on.
            k = into_current[-1]
            spans = [m.span() for m in re.finditer(r'\S+', self.current)]
            cut = spans[k - nsw][1]
            phrase_text = self.current[:cut].strip()
            self._emit('phrase', phrase_text)
            self.sentence = (self.sentence + ' ' + phrase_text).strip()
            self.current = self.current[cut:]
            sw = self.sentence.split()
        start = 0
        for k in ends:
            self._emit('sentence', ' '.join(sw[start:k + 1]))
            start = k + 1
        self.sentence = ' '.join(sw[start:])
        if self.sentence:
            self._emit('sentence_in_progress', self.sentence)
        if into_current and self.current.strip():
            self._emit('in_progress', self.current.strip())
        return True

    def _best_clause_break(self, words, marks, limit) -> Optional[int]:
        """For genuine run-on speech: the most plausible place to break when no
        sentence end can be found. Candidates are clause marks the ASR heard (its own
        commas, semicolons, colons) and clause marks the splitter predicts; the ASR's
        count as certain, the splitter's at their score. Among near-equal candidates
        the later one wins, so the pieces stay as long as they plausibly can."""
        min_head = max(4, self.split_min_words // 2)
        best = None
        for k in range(min_head - 1, limit + 1):
            score = 0.0
            if words[k].endswith((',', ';', ':')):
                score = 1.0
            elif k in marks and marks[k][0] not in SENTENCE_END:
                score = marks[k][1]
            if score <= 0.0:
                continue
            key = (round(score, 1), k)
            if best is None or key > best[0]:
                best = (key, k)
        return None if best is None else best[1]

    def finish(self):
        self._close_phrase('end')
        self._close_sentence('end')

    def take_events(self) -> List[Tuple[str, str]]:
        ev, self.events = self.events, []
        return ev


# ─────────────────────────────────────────────────────────────────────────────
# Streaming engine: mel → cache-aware encoder → prompt → greedy RNNT
# ─────────────────────────────────────────────────────────────────────────────

class NemotronEngine:
    def __init__(self, model_id: str, right_context: int, language: str):
        self.model_id = model_id
        self.right_context = right_context
        self.language = language
        self.model = None
        self.mel = None
        self.enc = None
        self.last_token = None
        self.decoder_hidden = None
        self.audio_t = 0.0      # audio fed so far (s)
        self.decoded_t = 0.0    # audio covered by decoded encoder chunks (s)
        self.frame_sec = 0.08
        self.debug = False
        self._tok = None

    def load(self):
        from mlx_audio.stt import load
        t = time.time()
        self.model = load(self.model_id)
        print(f"nemotron: loaded {self.model_id} in {time.time() - t:.1f}s")
        self.reset_stream()
        # warm the kernels so the first real chunk is not charged for compilation
        self.push(np.zeros(BLOCK * (self.right_context + 1) * 2, np.float32))
        self.reset_stream()

    def reset_stream(self):
        from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
        from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
        from mlx_audio.stt.models.nemotron_asr import tokenizer as tok
        self._tok = tok
        self.mel = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        self.enc = ConformerStreamingState(
            self.model.encoder, att_context_size=[LEFT_CONTEXT, self.right_context])
        self.last_token = self.model.blank_id
        self.decoder_hidden = None
        self.audio_t = 0.0
        self.decoded_t = 0.0
        pc, ec = self.model.preprocessor_config, self.model.encoder_config
        self.frame_sec = ec.subsampling_factor * pc.hop_length / pc.sample_rate

    def push(self, samples: np.ndarray, final=False) -> List[Tuple[str, float]]:
        """Feed 16 kHz mono float32. Returns new (text, audio_time) pieces, specials
        dropped, each stamped at the encoder frame that produced it."""
        import mlx.core as mx
        self.audio_t += len(samples) / SAMPLE_RATE
        pieces = []
        t0 = time.perf_counter()
        mel = self.mel.push(mx.array(samples), final=final)
        chunks = self.enc.push(mel, final=final) if mel.shape[1] > 0 or final else []
        for encoded in chunks:
            self.enc.materialize(encoded)
            prompted = self.model.apply_prompt(encoded, self.language)
            pieces += self._decode(prompted, self.decoded_t)
            self.decoded_t += prompted.shape[1] * self.frame_sec
        if self.debug and chunks:
            print(f"nemotron: {len(chunks)} chunk(s) in {(time.perf_counter() - t0) * 1000:.1f} ms")
        return pieces

    def _decode(self, prompted, chunk_start: float) -> List[Tuple[str, float]]:
        import mlx.core as mx
        m = self.model
        out = []
        chunk_len = prompted.shape[1]
        t = 0
        new_symbols = 0
        while t < chunk_len:
            feature = prompted[:, t:t + 1]
            cur = (mx.array([[self.last_token]], dtype=mx.int32)
                   if self.last_token != m.blank_id else None)
            dec_out, (h, c) = m.decoder(cur, self.decoder_hidden)
            dec_out = dec_out.astype(feature.dtype)
            joint = m.joint(feature, dec_out)
            pred = int(mx.argmax(joint))
            if pred != m.blank_id:
                self.last_token = pred
                self.decoder_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
                if not self._tok.is_special_token(pred, m.vocabulary):
                    out.append((self._tok.piece_to_text(m.vocabulary[pred]),
                                chunk_start + (t + 1) * self.frame_sec))
                new_symbols += 1
                if m.max_symbols is not None and new_symbols >= m.max_symbols:
                    t += 1
                    new_symbols = 0
            else:
                t += 1
                new_symbols = 0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class NemotronNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return NemotronNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        model_name = 'nemotron-3.5-0.6b'
        language = 'en-US'
        context = '160 ms'
        if args:
            for arg in args:
                if arg in MODELS:
                    model_name = arg
                elif arg in CONTEXTS:
                    context = arg
                elif arg == '8bit':
                    model_name = 'nemotron-3.5-0.6b-8bit'
                else:
                    language = arg

        # internal state first: widget callbacks can fire during patch load
        self.capture = NemotronCapture()
        self.engine: Optional[NemotronEngine] = None
        self.segmenter = Segmenter()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.restart_stream = threading.Event()
        self.events: 'queue.Queue[Tuple[str, str]]' = queue.Queue()
        self.using_external_audio = False
        self.last_in_progress = ''
        self.debug = False
        self.record = False
        self.want_split = False
        self.splitter: Optional[PunctuationSplitter] = None
        self._rec_wav = None
        self._rec_log = None

        # ── Inputs ──
        self.on_off_input = self.add_input('on/off', widget_type='checkbox',
                                           default_value=False, triggers_execution=True)
        self.audio_input = self.add_input('audio_in', triggers_execution=True)
        self.sample_rate_in_prop = self.add_input('sample_rate_in', widget_type='drag_int',
                                                  default_value=16000)
        self.model_property = self.add_input('model', widget_type='combo',
                                             default_value=model_name, widget_width=220,
                                             callback=self.model_changed)
        self.model_property.widget.combo_items = list(MODELS.keys())

        try:
            device_names = self.capture.get_device_list()
        except Exception as e:
            print(f"nemotron: no audio hardware available ({e}), use audio_in instead")
            device_names = []
        self.device_property = self.add_input('audio device', widget_type='combo',
                                              default_value=device_names[0] if device_names else 'none',
                                              widget_width=300, callback=self.device_changed)
        self.device_property.widget.combo_items = device_names if device_names else ['none']

        self.language_property = self.add_input('language', widget_type='text_input',
                                                default_value=language, widget_width=100,
                                                callback=self.stream_settings_changed)
        self.context_property = self.add_input('look-ahead', widget_type='combo',
                                               default_value=context, widget_width=120,
                                               callback=self.stream_settings_changed)
        self.context_property.widget.combo_items = list(CONTEXTS.keys())

        # ── Outputs ──
        self.in_progress_output = self.add_output('in_progress')
        self.phrase_output = self.add_output('phrase')
        self.sentence_in_progress_output = self.add_output('sentence_in_progress')
        self.sentence_output = self.add_output('sentence')

        # ── Options ──
        self.gain_option = self.add_option('gain', widget_type='drag_float',
                                           default_value=1.0, min=0.0, max=10.0,
                                           callback=self.options_changed)
        self.phrase_silence_option = self.add_option('phrase_silence', widget_type='drag_float',
                                                     default_value=0.8, min=0.1, max=5.0,
                                                     callback=self.options_changed)
        self.sentence_silence_option = self.add_option('sentence_silence', widget_type='drag_float',
                                                       default_value=2.0, min=0.1, max=10.0,
                                                       callback=self.options_changed)
        self.max_phrase_words_option = self.add_option('max_phrase_words', widget_type='drag_int',
                                                       default_value=0, min=0, max=100,
                                                       callback=self.options_changed)
        self.max_sentence_words_option = self.add_option('max_sentence_words', widget_type='drag_int',
                                                         default_value=0, min=0, max=300,
                                                         callback=self.options_changed)
        # Semantic splitting: a punctuation-restoration model (CPU) finds the sentence
        # ends the ASR leaves unmarked in continuous speech. Loaded only when on.
        self.split_option = self.add_option('semantic_split', widget_type='checkbox',
                                            default_value=False, callback=self.options_changed)
        self.split_min_words_option = self.add_option('split_min_words', widget_type='drag_int',
                                                      default_value=12, min=4, max=100,
                                                      callback=self.options_changed)
        self.split_guard_option = self.add_option('split_guard_words', widget_type='drag_int',
                                                  default_value=3, min=0, max=20,
                                                  callback=self.options_changed)
        self.split_check_every_option = self.add_option('split_check_every', widget_type='drag_int',
                                                        default_value=6, min=1, max=50,
                                                        callback=self.options_changed)
        self.split_max_words_option = self.add_option('split_max_words', widget_type='drag_int',
                                                      default_value=30, min=0, max=200,
                                                      callback=self.options_changed)
        self.debug_option = self.add_option('debug', widget_type='checkbox',
                                            default_value=False, callback=self.options_changed)
        # Writes the 16 kHz audio the engine actually receives, plus a token log with
        # audio-time stamps, to ~/nemotron_debug/. For chasing lost words: if the words
        # are audible in the wav but absent from the log, the model dropped them; if
        # they are faint or missing in the wav, the input side did.
        self.record_option = self.add_option('record_debug', widget_type='checkbox',
                                             default_value=False, callback=self.options_changed)

    # ── settings ──

    def _apply_options(self):
        def val(opt, default):
            v = opt()
            return default if v is None else v
        self.capture.gain = float(val(self.gain_option, 1.0))
        self.segmenter.phrase_silence = float(val(self.phrase_silence_option, 0.8))
        self.segmenter.sentence_silence = float(val(self.sentence_silence_option, 2.0))
        self.segmenter.max_phrase_words = int(val(self.max_phrase_words_option, 0))
        self.segmenter.max_sentence_words = int(val(self.max_sentence_words_option, 0))
        self.debug = bool(val(self.debug_option, False))
        self.record = bool(val(self.record_option, False))
        self.want_split = bool(val(self.split_option, False))
        self.segmenter.split_min_words = int(val(self.split_min_words_option, 12))
        self.segmenter.split_guard_words = int(val(self.split_guard_option, 3))
        self.segmenter.split_check_every = int(val(self.split_check_every_option, 6))
        self.segmenter.split_max_words = int(val(self.split_max_words_option, 30))
        if not self.want_split:
            self.segmenter.splitter = None   # (re)attached on the worker when wanted
        if self.engine is not None:
            self.engine.debug = self.debug

    def options_changed(self):
        self._apply_options()

    def model_changed(self):
        if self.thread is not None and self.thread.is_alive():
            self._stop_processing()
            self._start_processing()

    def device_changed(self):
        device_name = self.device_property()
        for i, d in enumerate(self.capture.devices):
            if d['name'] == device_name:
                self.capture.change_device(i)
                break

    def stream_settings_changed(self):
        # language / look-ahead: rebuild the streaming state on the worker
        if self.engine is not None:
            self.restart_stream.set()

    # ── lifecycle ──

    def _start_processing(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.using_external_audio = (self.audio_input is not None
                                     and len(self.audio_input._parents) > 0)
        if not self.using_external_audio and not self.capture.devices:
            self.using_external_audio = True
            print("nemotron: no audio devices found, using audio_in")
        if not self.using_external_audio:
            device_name = self.device_property()
            device_idx = 0
            for i, d in enumerate(self.capture.devices):
                if d['name'] == device_name:
                    device_idx = i
                    break
            if self.capture.current_device != device_idx or self.capture.source is None:
                if not self.capture.init(device_idx):
                    print("nemotron: mic init failed, falling back to audio_in")
                    self.using_external_audio = True
        self.capture.drain()
        self.segmenter.reset()
        self.last_in_progress = ''
        self.stop_event.clear()
        self.restart_stream.clear()
        self.thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.thread.start()
        self.add_frame_task()

    def _stop_processing(self):
        self.stop_event.set()
        self.capture.pause()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                print("nemotron: processing thread did not exit within 5s")
        self.thread = None

    def _current_settings(self):
        model_id = MODELS.get(self.model_property(), MODELS['nemotron-3.5-0.6b'])
        right = CONTEXTS.get(self.context_property(), 1)
        language = (self.language_property() or 'en-US').strip() or 'en-US'
        return model_id, right, language

    def _processing_thread(self):
        model_id, right, language = self._current_settings()
        try:
            if self.engine is None or self.engine.model_id != model_id:
                self.engine = NemotronEngine(model_id, right, language)
                self.engine.debug = self.debug
                self.engine.load()
            else:
                self.engine.right_context = right
                self.engine.language = language
                self.engine.reset_stream()
        except Exception as e:
            print(f"nemotron: model load failed: {e}")
            traceback.print_exception(e)
            self.engine = None
            return
        self._apply_options()
        if not self.using_external_audio:
            self.capture.resume()
        print("nemotron: listening")

        while not self.stop_event.is_set():
            self._sync_splitter()
            if self.restart_stream.is_set():
                self.restart_stream.clear()
                _, right, language = self._current_settings()
                self.segmenter.finish()
                self._post(self.segmenter.take_events())
                self.engine.right_context = right
                self.engine.language = language
                self.engine.reset_stream()
            try:
                block = self.capture.blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                pieces = self.engine.push(block)
                for piece, at in pieces:
                    self.segmenter.token(piece, at)
                self.segmenter.tick(self.engine.decoded_t)
                events = self.segmenter.take_events()
                self._rec_block(block, pieces, events)
                self._post(events)
            except Exception as e:
                print(f"nemotron: processing error: {e}")
                traceback.print_exception(e)
                time.sleep(0.5)

        # flush what is still open
        try:
            for piece, at in self.engine.push(np.zeros(0, np.float32), final=True):
                self.segmenter.token(piece, at)
        except Exception:
            pass
        self.segmenter.finish()
        self._post(self.segmenter.take_events())
        self._rec_close()

    def _rec_open(self):
        if self._rec_wav is not None:
            return
        try:
            import soundfile as sf
            folder = os.path.expanduser('~/nemotron_debug')
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            self._rec_wav = sf.SoundFile(stem + '.wav', mode='w', samplerate=SAMPLE_RATE,
                                         channels=1, subtype='FLOAT')
            self._rec_log = open(stem + '.log', 'w')
            print(f"nemotron: recording debug audio to {stem}.wav")
        except Exception as e:
            print(f"nemotron: could not open debug recording: {e}")
            self._rec_wav = None
            self._rec_log = None

    def _rec_close(self):
        for f in (self._rec_wav, self._rec_log):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._rec_wav = None
        self._rec_log = None

    def _rec_block(self, block, pieces, events):
        if self.record and self._rec_wav is None:
            self._rec_open()
        elif not self.record and self._rec_wav is not None:
            self._rec_close()
        if self._rec_wav is None:
            return
        try:
            self._rec_wav.write(block)
            for p, at in pieces:
                self._rec_log.write(f"{at:9.3f} tok  {p!r}\n")
            at = self.engine.decoded_t
            for kind, text in events:
                if kind != 'in_progress':
                    self._rec_log.write(f"{at:9.3f} {kind:20s} {text!r}\n")
            if pieces or events:
                self._rec_log.flush()
        except Exception as e:
            print(f"nemotron: debug recording failed: {e}")
            self._rec_close()

    def _sync_splitter(self):
        """Worker thread: load and attach the splitter when the option is on."""
        if self.want_split and self.segmenter.splitter is None:
            try:
                if self.splitter is None:
                    self.splitter = PunctuationSplitter()
                if self.splitter.pipe is None:
                    print("nemotron: loading sentence splitter (first time downloads ~1.1 GB)...")
                    self.splitter.load()
                self.segmenter.splitter = self.splitter
            except Exception as e:
                print(f"nemotron: sentence splitter unavailable: {e}")
                self.want_split = False
        elif not self.want_split and self.segmenter.splitter is not None:
            self.segmenter.splitter = None

    def _post(self, events):
        for ev in events:
            self.events.put(ev)

    # ── node callbacks ──

    def execute(self):
        if self.audio_input.fresh_input:
            audio_data = self.audio_input()
            if self.thread is None:
                return
            try:
                if hasattr(audio_data, 'detach'):
                    audio_np = audio_data.detach().cpu().numpy()
                elif isinstance(audio_data, np.ndarray):
                    audio_np = audio_data
                elif isinstance(audio_data, (list, tuple)):
                    audio_np = np.array(audio_data, dtype=np.float32)
                else:
                    audio_np = None
                if audio_np is not None and audio_np.ndim > 0:
                    self.capture.feed_external(audio_np, int(self.sample_rate_in_prop()))
            except Exception as e:
                if self.debug:
                    print(f"nemotron: audio_in error: {e}")
            return

        if self.on_off_input():
            if self.thread is None or not self.thread.is_alive():
                self._start_processing()
        else:
            self._stop_processing()

    def frame_task(self):
        """Main thread: hand the worker's events to the outputs."""
        while True:
            try:
                kind, text = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == 'in_progress':
                if text != self.last_in_progress:
                    self.last_in_progress = text
                    self.in_progress_output.send(text)
            elif kind == 'phrase':
                self.last_in_progress = ''
                self.phrase_output.send(text)
            elif kind == 'sentence_in_progress':
                self.sentence_in_progress_output.send(text)
            elif kind == 'sentence':
                self.sentence_output.send(text)

    def custom_cleanup(self):
        self._stop_processing()
        self.capture.close()
