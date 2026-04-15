"""
whisper_nodes.py
Real-time speech-to-text node for dpg_system using Whisper.

Ported from David Rokeby's ofxWhisper C++ implementation.
Supports faster-whisper (CTranslate2) and pywhispercpp (ggml-metal) backends.

Created by David Rokeby, 2023. Python port 2026.
"""

import numpy as np
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *

WHISPER_SAMPLE_RATE = 16000

# ─────────────────────────────────────────────────────────────────────────────
# Language map (matching the C++ g_lang_map_multi)
# ─────────────────────────────────────────────────────────────────────────────
LANG_LIST = [
    "auto", "english", "chinese", "german", "spanish", "russian", "korean",
    "french", "japanese", "portuguese", "turkish", "polish", "catalan",
    "dutch", "arabic", "swedish", "italian", "indonesian", "hindi",
    "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay",
    "czech", "romanian", "danish", "hungarian", "tamil", "norwegian",
    "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin",
    "maori", "malayalam", "welsh", "slovak", "telugu", "persian",
    "latvian", "bengali", "serbian", "azerbaijani", "slovenian",
    "kannada", "estonian", "macedonian", "breton", "basque", "icelandic",
    "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian",
    "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer",
    "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian",
    "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish",
    "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen",
    "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar",
    "tibetan", "tagalog", "malagasy", "assamese", "tatar", "hawaiian",
    "lingala", "hausa", "bashkir", "javanese", "sundanese", "cantonese"
]

LANG_CODES = [
    "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he",
    "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv",
    "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
    "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
    "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb",
    "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
    "su", "yue"
]

MODEL_SIZES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v3-turbo", "large-v3",
]

# Noise characters (matching the C++ noise[] table)
NOISE_CHARS = set('([*<)]/>')


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Backend Abstraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenInfo:
    text: str
    probability: float


@dataclass
class SegmentInfo:
    text: str
    tokens: List[TokenInfo]
    start: float  # whisper timestamp (centiseconds)
    end: float
    language: str = ""


class WhisperBackend(ABC):
    """Abstract interface for whisper inference engines."""

    @abstractmethod
    def load_model(self, model_name: str, device: str = "auto",
                   compute_type: str = "auto") -> bool:
        ...

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, language: str = "auto",
                   translate: bool = False) -> Tuple[List[SegmentInfo], str]:
        """Run inference. Returns (segments, detected_language)."""
        ...

    @abstractmethod
    def is_multilingual(self) -> bool:
        ...


class FasterWhisperBackend(WhisperBackend):
    """Backend using faster-whisper (CTranslate2). CPU-only on macOS."""

    def __init__(self):
        self.model = None
        self._is_multilingual = True
        self.debug = False

    def load_model(self, model_name: str, device: str = "auto",
                   compute_type: str = "auto") -> bool:
        try:
            from faster_whisper import WhisperModel
            import platform

            # On macOS: force CPU with int8 (fastest for CTranslate2 without CUDA)
            if platform.system() == "Darwin":
                device = "cpu"
                if compute_type == "auto":
                    compute_type = "int8"

            cpu_threads = 4  # reasonable default for M-series
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            self._is_multilingual = not model_name.endswith('.en')
            print(f"faster-whisper: loaded '{model_name}' on {device} "
                  f"({compute_type}), threads={cpu_threads}")
            return True
        except Exception as e:
            print(f"Failed to load faster-whisper model '{model_name}': {e}")
            traceback.print_exception(e)
            return False

    def transcribe(self, audio_data: np.ndarray, language: str = "auto",
                   translate: bool = False) -> Tuple[List[SegmentInfo], str]:
        if self.model is None:
            return [], ""

        task = "translate" if translate else "transcribe"
        lang = None if language == "auto" else language

        try:
            t0 = time.time()
            segments_gen, info = self.model.transcribe(
                audio_data,
                language=lang,
                task=task,
                beam_size=1,
                best_of=1,
                patience=1.0,
                word_timestamps=False,
                vad_filter=False,          # we do our own VAD
                without_timestamps=True,
                condition_on_previous_text=False,  # faster, avoids hallucination loops
                temperature=0.0,                   # single-pass, no fallback retries
                compression_ratio_threshold=None,   # skip compression checks
                log_prob_threshold=None,            # skip log-prob filtering
                no_speech_threshold=None,           # skip no-speech filtering
            )

            detected_language = info.language if info.language else ""
            segments = []
            for seg in segments_gen:
                tokens = []
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        tokens.append(TokenInfo(text=w.word,
                                                probability=w.probability))
                else:
                    import math
                    avg_prob = math.exp(seg.avg_logprob) if seg.avg_logprob > -10 else 0.0
                    tokens.append(TokenInfo(text=seg.text, probability=avg_prob))

                segments.append(SegmentInfo(
                    text=seg.text,
                    tokens=tokens,
                    start=seg.start * 100,
                    end=seg.end * 100,
                    language=detected_language,
                ))

            if self.debug:
                elapsed = (time.time() - t0) * 1000
                audio_ms = len(audio_data) / WHISPER_SAMPLE_RATE * 1000
                print(f"faster-whisper: {elapsed:.0f}ms inference "
                      f"for {audio_ms:.0f}ms audio "
                      f"({len(segments)} segments)")

            return segments, detected_language

        except Exception as e:
            print(f"Whisper transcription error: {e}")
            traceback.print_exception(e)
            return [], ""

    def is_multilingual(self) -> bool:
        return self._is_multilingual


class WhisperCppBackend(WhisperBackend):
    """Backend using pywhispercpp (ggml with Metal/CoreML GPU support)."""

    def __init__(self):
        self.model = None
        self._is_multilingual = True
        self._ctx = None
        self.debug = False

    def load_model(self, model_name: str, device: str = "auto",
                   compute_type: str = "auto") -> bool:
        try:
            from pywhispercpp.model import Model
            import os

            n_threads = os.cpu_count() or 4

            self.model = Model(
                model_name,
                n_threads=n_threads,
                single_segment=False,
                print_progress=False,
                print_realtime=False,
                print_timestamps=False,
                suppress_blank=True,
                no_context=True,           # don't condition on previous text
                temperature_inc=0.0,       # single-pass, no fallback retries
                redirect_whispercpp_logs_to=None,  # suppress C logs
            )
            self._ctx = self.model._ctx

            sys_info = Model.system_info()
            print(f"whisper.cpp: loaded '{model_name}', "
                  f"threads={n_threads}, system={sys_info}")
            return True
        except Exception as e:
            print(f"Failed to load pywhispercpp model '{model_name}': {e}")
            traceback.print_exception(e)
            return False

    def transcribe(self, audio_data: np.ndarray, language: str = "auto",
                   translate: bool = False) -> Tuple[List[SegmentInfo], str]:
        if self.model is None:
            return [], ""
        try:
            import _pywhispercpp as pw

            t0 = time.time()
            params = {}
            if language != "auto":
                params['language'] = language
            if translate:
                params['translate'] = True

            # Run transcription — pywhispercpp accepts numpy arrays directly
            result = self.model.transcribe(
                audio_data,
                extract_probability=True,
                **params
            )

            detected_language = ""
            segments = []
            # Use the high-level result from model.transcribe() for clean text,
            # then try to enhance with per-token probabilities from C context
            ctx = self.model._ctx
            n_segs = pw.whisper_full_n_segments(ctx)

            for idx, seg_result in enumerate(result):
                if idx >= n_segs:
                    break

                seg_t0 = seg_result.t0
                seg_t1 = seg_result.t1
                seg_text = seg_result.text  # already clean, stripped by pywhispercpp
                seg_prob = float(seg_result.probability) if not np.isnan(seg_result.probability) else 0.5

                # Try per-token probability extraction from C context
                tokens = []
                try:
                    n_tokens = pw.whisper_full_n_tokens(ctx, idx)
                    eot_id = pw.whisper_token_eot(ctx)
                    for j in range(n_tokens):
                        tok_id = pw.whisper_full_get_token_id(ctx, idx, j)
                        if tok_id >= eot_id:
                            continue
                        tok_text_bytes = pw.whisper_full_get_token_text(ctx, idx, j)
                        tok_text = tok_text_bytes.decode('utf-8', errors='replace')
                        if tok_text.startswith('[_') or tok_text.startswith('<|'):
                            continue
                        tok_prob = pw.whisper_full_get_token_p(ctx, idx, j)
                        tokens.append(TokenInfo(text=tok_text, probability=float(tok_prob)))
                except Exception:
                    tokens = []

                # Fallback: use segment text as single token (same as faster-whisper)
                if not tokens:
                    tokens = [TokenInfo(text=seg_text, probability=seg_prob)]

                segments.append(SegmentInfo(
                    text=seg_text,
                    tokens=tokens,
                    start=float(seg_t0),
                    end=float(seg_t1),
                    language=detected_language,
                ))

            if self.debug:
                elapsed = (time.time() - t0) * 1000
                audio_ms = len(audio_data) / WHISPER_SAMPLE_RATE * 1000
                print(f"whisper.cpp: {elapsed:.0f}ms inference "
                      f"for {audio_ms:.0f}ms audio "
                      f"({len(segments)} segments)")

            return segments, detected_language
        except Exception as e:
            print(f"WhisperCpp transcription error: {e}")
            traceback.print_exception(e)
            return [], ""

    def is_multilingual(self) -> bool:
        return self._is_multilingual


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Audio Capture (port of audio_async)
# ─────────────────────────────────────────────────────────────────────────────

class AudioCapture:
    """Circular-buffer audio capture with VAD, ported from audio_async.cpp."""

    def __init__(self, length_ms: int = 30000):
        self.length_ms = length_ms
        self.sample_rate = WHISPER_SAMPLE_RATE
        self.buffer_size = (self.sample_rate * length_ms) // 1000

        # Circular buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.audio_pos = 0
        self.audio_start = 0

        # State
        self.running = False
        self.new_audio = False
        self.pause_cleared_audio = False
        self.audio_ready = False
        self.external_mode = False  # True when using external audio input

        # VAD
        self.voice_threshold = 0.02
        self.silence_period_threshold = 15
        self.silence_count = 0
        self.diff_energy_all = 0.0
        self.diff_energy_last = 0.0
        self.energy = 0.0
        self.gain = 1.0

        # Previous audio chunk (for prepending on speech onset)
        self.last_audio: Optional[np.ndarray] = None

        # Device info
        self.stream = None
        self.devices: List[dict] = []
        self.current_device = 0
        self.device_sample_rate = 16000
        self.downsampling = 1
        self.channels = 1
        self.next_buffer_start_frame = 0

        self.mutex = threading.Lock()

    def get_device_list(self) -> List[str]:
        """Return list of available input device names."""
        try:
            import sounddevice as sd
            all_devices = sd.query_devices()
            self.devices = []
            for i, dev in enumerate(all_devices):
                if dev['max_input_channels'] > 0:
                    self.devices.append({
                        'name': dev['name'],
                        'index': i,
                        'channels': dev['max_input_channels'],
                        'default_samplerate': dev['default_samplerate'],
                    })
            return [d['name'] for d in self.devices]
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            return []

    def init(self, device_index: int = 0) -> bool:
        """Initialize audio capture for the given device."""
        import sounddevice as sd

        if not self.devices:
            self.get_device_list()
        if not self.devices:
            print("No audio input devices found")
            return False

        if device_index >= len(self.devices):
            device_index = 0
        self.current_device = device_index
        device = self.devices[device_index]

        # Choose sample rate: prefer lowest multiple of 16000
        sr = int(device['default_samplerate'])
        for candidate in [16000, 32000, 48000]:
            # sounddevice will resample if needed; just use the default
            pass
        self.device_sample_rate = sr
        self.downsampling = max(1, sr // 16000)
        self.channels = min(device['channels'], 2)

        # Reset buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.audio_pos = 0
        self.audio_start = 0
        self.silence_count = 0
        self.next_buffer_start_frame = 0
        self.last_audio = None

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()

            self.stream = sd.InputStream(
                device=device['index'],
                channels=self.channels,
                samplerate=self.device_sample_rate,
                blocksize=1024 * self.downsampling,
                dtype='float32',
                callback=self._audio_callback,
            )
            self.audio_ready = True
            return True
        except Exception as e:
            print(f"Failed to init audio device: {e}")
            traceback.print_exception(e)
            return False

    def change_device(self, device_index: int) -> bool:
        """Switch to a different audio device."""
        if device_index == self.current_device:
            return True
        was_running = self.running
        self.pause()
        success = self.init(device_index)
        if success and was_running:
            self.resume()
        return success

    def resume(self):
        if self.stream is not None and self.audio_ready:
            self.stream.start()
            self.running = True

    def pause(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.running = False

    def close(self):
        self.pause()
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs on audio thread."""
        if not self.running:
            return

        # Downsample and mix channels to mono at 16kHz
        if self.channels >= 2:
            mono = np.mean(indata[:, :self.channels], axis=1)
        else:
            mono = indata[:, 0]

        # Downsample
        if self.downsampling > 1:
            mono = mono[self.next_buffer_start_frame::self.downsampling]
            leftover = frames - (self.next_buffer_start_frame +
                                 (len(mono)) * self.downsampling)
            if leftover < 0:
                leftover = 0
            self.next_buffer_start_frame = self.downsampling - leftover
            if self.next_buffer_start_frame < 0:
                self.next_buffer_start_frame = 0
        else:
            self.next_buffer_start_frame = 0

        # Apply gain
        audio_new = mono * self.gain
        samples_16k = len(audio_new)

        if samples_16k == 0:
            return

        # VAD: differential energy
        channel_active = self._voice_active(audio_new)

        with self.mutex:
            if channel_active:
                if self.silence_count > self.silence_period_threshold:
                    # Starting new utterance — prepend previous chunk
                    if self.last_audio is not None and len(self.last_audio) > 0:
                        self._insert_audio(self.last_audio)
                self.silence_count = 0
                self._insert_audio(audio_new)
            else:
                if self.silence_count < self.silence_period_threshold:
                    self._insert_audio(audio_new)
                else:
                    self.new_audio = False
                    self.audio_start = self.audio_pos = 0
                    self.pause_cleared_audio = True
                self.silence_count += 1
                self.last_audio = audio_new.copy()

            self.new_audio = True

    def _voice_active(self, audio: np.ndarray) -> bool:
        """VAD via differential energy, matching audio_async::voice_active."""
        n = len(audio)
        if n < 2:
            return False
        diffs = np.abs(np.diff(audio))
        self.diff_energy_all = float(np.sum(diffs))
        half = n // 2
        self.diff_energy_last = float(np.sum(diffs[half:]))
        self.energy = self.diff_energy_all
        return (self.diff_energy_all > self.voice_threshold or
                self.diff_energy_last > self.voice_threshold)

    def _insert_audio(self, audio: np.ndarray):
        """Insert audio into circular buffer, matching InsertAudioIntoBuffer."""
        n = len(audio)
        buf_size = len(self.audio_buffer)

        if n >= buf_size:
            # Just fill the buffer
            start = n - buf_size + 1
            self.audio_buffer[:buf_size - 1] = audio[start:start + buf_size - 1]
            self.audio_pos = buf_size - 1
            self.audio_start = 0
            return

        # Check if we'd overwrite the start
        active = (self.audio_pos - self.audio_start) % buf_size
        available = buf_size - active
        lapped = available <= n

        end_pos = self.audio_pos + n
        if end_pos > buf_size:
            n0 = buf_size - self.audio_pos
            n1 = n - n0
            self.audio_buffer[self.audio_pos:self.audio_pos + n0] = audio[:n0]
            self.audio_buffer[:n1] = audio[n0:n0 + n1]
            self.audio_pos = n1
        else:
            self.audio_buffer[self.audio_pos:self.audio_pos + n] = audio
            self.audio_pos += n

        if lapped:
            self.audio_start = (self.audio_pos + 1) % buf_size

    def feed_external(self, audio: np.ndarray, sample_rate: int = 16000):
        """Feed externally-sourced audio into the buffer (from a node input).
        Handles downsampling to 16kHz, mono conversion, VAD, and gain."""
        if not self.audio_ready and not self.external_mode:
            # Auto-init in external mode (no mic needed)
            self.external_mode = True
            self.audio_ready = True

        # Convert to mono float32 if needed
        if audio.ndim > 1:
            if audio.shape[0] <= audio.shape[-1]:
                # (channels, samples) — take first channel
                audio = audio[0]
            else:
                # (samples, channels) — take first channel
                audio = audio[:, 0]
        audio = audio.flatten().astype(np.float32)

        # Downsample to 16kHz if needed
        if sample_rate != WHISPER_SAMPLE_RATE and sample_rate > 0:
            ratio = sample_rate / WHISPER_SAMPLE_RATE
            if ratio > 1:
                indices = np.arange(0, len(audio), ratio).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

        # Apply gain
        audio = audio * self.gain

        if len(audio) == 0:
            return

        # VAD
        channel_active = self._voice_active(audio)

        with self.mutex:
            if channel_active:
                if self.silence_count > self.silence_period_threshold:
                    if self.last_audio is not None and len(self.last_audio) > 0:
                        self._insert_audio(self.last_audio)
                self.silence_count = 0
                self._insert_audio(audio)
            else:
                if self.silence_count < self.silence_period_threshold:
                    self._insert_audio(audio)
                else:
                    self.new_audio = False
                    self.audio_start = self.audio_pos = 0
                    self.pause_cleared_audio = True
                self.silence_count += 1
                self.last_audio = audio.copy()

            self.new_audio = True

    def fraction_full(self) -> float:
        length = (self.audio_pos - self.audio_start) % len(self.audio_buffer)
        return length / len(self.audio_buffer)

    def get(self, ms: int = 0) -> Optional[np.ndarray]:
        """Get audio from buffer, matching audio_async::get."""
        if not self.new_audio or not self.audio_ready:
            return None
        if not self.running:
            return None

        with self.mutex:
            if ms <= 0:
                ms = self.length_ms

            n_samples = (self.sample_rate * ms) // 1000
            buf_size = len(self.audio_buffer)

            available = (self.audio_pos - self.audio_start) % buf_size
            if n_samples > available:
                n_samples = available
            if n_samples > buf_size:
                n_samples = buf_size
            if n_samples < self.sample_rate:  # less than 1 second
                return None

            result = np.zeros(n_samples, dtype=np.float32)
            s0 = (self.audio_pos - n_samples) % buf_size

            if s0 + n_samples > buf_size:
                n0 = buf_size - s0
                result[:n0] = self.audio_buffer[s0:s0 + n0]
                result[n0:] = self.audio_buffer[:n_samples - n0]
            else:
                result[:] = self.audio_buffer[s0:s0 + n_samples]

            self.audio_start = s0
            self.new_audio = False
            return result

    def set_start(self, whisper_time: float, ref_start: int,
                  samples_to_keep: int):
        """Advance the buffer start, matching audio_async::set_start."""
        with self.mutex:
            buf_size = len(self.audio_buffer)
            new_start = self._whisper_time_to_sample(whisper_time,
                                                     ref_start) - samples_to_keep
            new_start = new_start % buf_size

            pre_size = (self.audio_pos - self.audio_start) % buf_size
            post_size = (self.audio_pos - new_start) % buf_size

            if post_size > pre_size:
                self.audio_start = self.audio_pos
            else:
                self.audio_start = new_start

    def push_start_forward(self, samples: int):
        """Push buffer start forward, matching audio_async::push_start_forward."""
        with self.mutex:
            buf_size = len(self.audio_buffer)
            current_size = (self.audio_pos - self.audio_start) % buf_size
            new_start = (self.audio_start + samples) % buf_size
            new_size = (self.audio_pos - new_start) % buf_size
            if new_size > current_size:
                return
            self.audio_start = new_start

    def _whisper_time_to_sample(self, whisper_time: float,
                                ref_start: int = -1) -> int:
        """Convert whisper centisecond timestamp to buffer sample position."""
        buf_size = len(self.audio_buffer)
        sample = int(whisper_time * self.sample_rate / 100.0)
        if ref_start >= 0:
            return (ref_start + sample) % buf_size
        return (self.audio_start + sample) % buf_size


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Processing Pipeline (port of ofxWhisperMulti)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurrentPhrase:
    segment_string: str = ""
    noise: bool = False
    duration: float = 0.0
    token_count: int = 0
    confidence: float = 0.0
    start: float = 0.0
    end: float = 0.0
    last_processed_sample: int = 0
    energy: float = 0.0

    def reset(self):
        self.segment_string = ""
        self.noise = False
        self.duration = 0.0
        self.token_count = 0
        self.confidence = 0.0
        self.start = 0.0
        self.end = 0.0
        self.last_processed_sample = 0
        self.energy = 0.0


@dataclass
class Phrase:
    segment_string: str = ""
    age: int = 0
    noise: bool = False
    noise_output: bool = False
    rate: float = 0.0
    duration: float = 0.0
    token_count: int = 0
    confidence: float = 0.0
    lifespan: int = 0
    energy: float = 0.0

    def set_from_current(self, current: CurrentPhrase):
        self.segment_string = current.segment_string
        self.noise = current.noise
        self.duration = current.duration
        self.token_count = current.token_count
        self.confidence = current.confidence
        self.energy = current.energy


class WhisperProcessor:
    """
    Core real-time whisper processing pipeline.
    Port of ofxWhisperMulti, preserving all custom logic:
    segment aging, overlap detection, noise filtering, phrase emission.
    """

    def __init__(self, backend: WhisperBackend, audio: AudioCapture):
        self.backend = backend
        self.audio = audio

        # Processing parameters (matching C++ defaults)
        self.chunk_size_ms = 300  # update_period
        self.confirmation_age = 1
        self.minimum_lifespan = 4
        self.minimum_confidence = 0.5
        self.min_trailing_probability = 0.6
        self.length_factor = 50
        self.buffer_overflow_fraction = 0.9
        self.overlap_ms = 200.0
        self.n_samples_keep = int(self.overlap_ms * 1e-3 * WHISPER_SAMPLE_RATE)
        self.overlap_max = 20
        self.debug = False

        # State
        self.is_running = False
        self.processing = False
        self.language = "auto"
        self.translate = False
        self.current_language = ""
        self.n_iter = 0
        self.n_segments = 0
        self.last_processed_sample = 0
        self.past_rate = 0.0

        # Segment tracking
        self.last_segment_strings: List[Phrase] = []
        self.current_segment_strings: List[CurrentPhrase] = []
        self.last_phrase_emitted = ""

        # Output strings (accumulated per process cycle)
        self.emit_string = ""
        self.noise_string = ""
        self.in_progress_string = ""
        self.last_in_progress_string = ""

        # Timing
        self.t_last = time.time()

        # Thread-safe result passing
        self.result_mutex = threading.Lock()
        self.fresh_phrase = False
        self.fresh_progress = False
        self.fresh_noise = False
        self.phrases_to_output = ""
        self.in_progress_output = ""
        self.noise_output = ""

    def set_overlap(self, overlap_ms: float):
        self.overlap_ms = overlap_ms
        self.n_samples_keep = int(overlap_ms * 1e-3 * WHISPER_SAMPLE_RATE)

    def process(self) -> int:
        """
        Main processing loop iteration.
        Returns 0 on success, 1 on skip, error code otherwise.
        Port of ofxWhisperMulti::Process.
        """
        self.emit_string = ""
        self.noise_string = ""
        self.in_progress_string = ""

        # Check for pause (silence cleared audio)
        if self.audio.pause_cleared_audio:
            self._emit_and_clear("PAUSE")

        # Check for buffer overflow
        fullness = self.audio.fraction_full()
        if fullness > self.buffer_overflow_fraction:
            if self.debug:
                print(f"Buffer fullness: {fullness:.2f}")
            self._emit_and_clear("OVERFLOW")
            self.audio.audio_start = self.last_processed_sample

        # Timing gate
        t_now = time.time()
        t_diff_ms = (t_now - self.t_last) * 1000

        if t_diff_ms < self.chunk_size_ms:
            time.sleep(0.01)  # tight poll when near update threshold
            if self.emit_string:
                self._do_callbacks()
            self.processing = False
            return 1

        # Get audio
        audio_data = self.audio.get(self.audio.length_ms)
        if audio_data is None:
            time.sleep(self.chunk_size_ms / 4000.0)  # shorter wait
            self.processing = False
            if self.emit_string:
                self._do_callbacks()
            return 1

        self.t_last = t_now

        if self.debug:
            print(f"\nbuffer size = {len(audio_data) / 16000:.2f}s")

        if len(audio_data) >= 16000:
            # Run inference
            lang_code = self.language
            if lang_code != "auto":
                idx = LANG_LIST.index(lang_code) if lang_code in LANG_LIST else 0
                if 0 < idx < len(LANG_CODES):
                    lang_code = LANG_CODES[idx]

            segments, detected_lang = self.backend.transcribe(
                audio_data, language=lang_code, translate=self.translate
            )
            self.current_language = detected_lang

            if not segments:
                self.processing = False
                return 0

            self.n_segments = len(segments)
            self.current_segment_strings = [CurrentPhrase() for _ in range(self.n_segments)]

            for i, seg in enumerate(segments):
                text = seg.text
                if text and len(text) > 0:
                    self._check_for_overlap(i, text)
                    self._build_segment_string(i, seg)
                    self.last_processed_sample = self.current_segment_strings[i].last_processed_sample
                    self._age_unchanging_strings(i)

                    if (i < len(self.last_segment_strings) and
                            self.last_segment_strings[i].noise and
                            not self.last_segment_strings[i].noise_output):
                        self.last_segment_strings[i].noise_output = True
                        self.last_segment_strings[i].lifespan = 0
                        # Extract noise text (strip noise chars from edges)
                        noise_text = self._strip_noise_chars(text)
                        if len(noise_text) > 1:
                            self._emit_string(noise_text + ' ', "NOISE")

            self._check_for_phrase_endings()
            self.n_iter += 1

        self._do_callbacks()
        return 0

    def _build_segment_string(self, which_segment: int, seg: SegmentInfo):
        """
        Build the segment string with probability trimming.
        Port of ofxWhisperMulti::BuildSegmentString.
        """
        segment_string = ""
        probable_string_length = 0
        mean_probs = 0.0
        prob_count = 0

        for token in seg.tokens:
            token_text = token.text
            prob = token.probability

            # Strip leading " -" pattern
            if len(token_text) >= 2:
                if token_text[0] == ' ' and token_text[1] == '-':
                    if len(token_text) == 2:
                        continue
                    token_text = ' ' + token_text[2:]

            if self._token_is_not_noise(token_text):
                segment_string += token_text
                if len(token_text) > 1:
                    mean_probs += prob
                    prob_count += 1
                if prob > self.min_trailing_probability:
                    probable_string_length = len(segment_string)
            elif segment_string:  # break on noise after content
                break

        # Hallucination detection
        hallucination = False
        if len(segment_string) > 3 and segment_string[1:4] == 'MBC':
            hallucination = True
            if self.debug:
                print("HALLUCINATION!")
            segment_string = "-"

        # Trim to probable length (only for multi-token sequences;
        # with a single segment-level token, trimming would empty the string)
        if (probable_string_length < len(segment_string) and
                len(seg.tokens) > 1):
            # Handle UTF-8 3-byte boundaries
            if (probable_string_length > 2 and
                    probable_string_length < len(segment_string)):
                try:
                    segment_string[:probable_string_length].encode('utf-8')
                except UnicodeEncodeError:
                    probable_string_length = max(0, probable_string_length - 2)
            segment_string = segment_string[:probable_string_length]

        # Calculate confidence
        prob = mean_probs / prob_count if prob_count > 0 else 0.0
        if hallucination:
            prob = 0.0

        duration = seg.end - seg.start
        if seg.start > seg.end:
            duration += 3000  # 30s in centiseconds

        cs = self.current_segment_strings[which_segment]
        cs.reset()
        cs.duration = duration
        cs.token_count = len(seg.tokens)
        cs.confidence = prob
        cs.segment_string = segment_string
        cs.start = seg.start
        cs.end = seg.end
        cs.last_processed_sample = self.audio._whisper_time_to_sample(
            seg.end, self.audio.audio_start
        )

    def _age_unchanging_strings(self, which_segment: int):
        """
        Track segment stability across iterations.
        Port of ofxWhisperMulti::AgeUnchangingStrings.
        """
        text = self.current_segment_strings[which_segment].segment_string

        if which_segment < len(self.last_segment_strings):
            last_text = self.last_segment_strings[which_segment].segment_string

            if text == last_text:
                self.last_segment_strings[which_segment].age += 1
                self.last_segment_strings[which_segment].lifespan += 1
                self.last_segment_strings[which_segment].set_from_current(
                    self.current_segment_strings[which_segment])
            else:
                # Check for minor variations (trailing punctuation, leading spaces)
                fixed = False
                text_stripped = text.strip()
                last_stripped = last_text.strip()

                # Strip trailing punctuation for comparison
                for punct in '.?!':
                    text_stripped = text_stripped.rstrip(punct)
                    last_stripped = last_stripped.rstrip(punct)

                if text_stripped == last_stripped and text_stripped:
                    self.last_segment_strings[which_segment].age += 1
                    self.last_segment_strings[which_segment].lifespan += 1
                    self.last_segment_strings[which_segment].set_from_current(
                        self.current_segment_strings[which_segment])
                    fixed = True

                if not fixed:
                    self.last_segment_strings[which_segment].set_from_current(
                        self.current_segment_strings[which_segment])
                    self.last_segment_strings[which_segment].age = 0
                    is_not_noise = self._segment_is_not_noise(text)
                    if is_not_noise:
                        self.last_segment_strings[which_segment].lifespan += 1
                    else:
                        self.last_segment_strings[which_segment].lifespan = 0
                    self.last_segment_strings[which_segment].noise = not is_not_noise
                    self.last_segment_strings[which_segment].noise_output = False
        else:
            # New segment
            while len(self.last_segment_strings) <= which_segment:
                self.last_segment_strings.append(Phrase())
            self.last_segment_strings[which_segment].set_from_current(
                self.current_segment_strings[which_segment])
            self.last_segment_strings[which_segment].lifespan = 0
            self.last_segment_strings[which_segment].age = 0
            self.last_segment_strings[which_segment].noise = not self._segment_is_not_noise(text)
            self.last_segment_strings[which_segment].noise_output = False

        if self.debug and which_segment < len(self.last_segment_strings):
            p = self.last_segment_strings[which_segment]
            print(f"  seg {which_segment}: '{p.segment_string}' "
                  f"life={p.lifespan} age={p.age} conf={p.confidence:.2f}")

    def _check_for_phrase_endings(self):
        """
        Check confirmed segments for phrase endings.
        Port of ofxWhisperMulti::CheckForPhraseEndings.
        """
        new_start_offset = -1
        emitted = -1
        noise_emitted = -1

        # Dynamic confirmation age
        prev_len = len(self.last_in_progress_string)
        scale = prev_len // self.length_factor
        adjusted_confirmation_age = max(0, self.confirmation_age - scale)

        for i in range(self.n_segments):
            if i >= len(self.last_segment_strings):
                break

            seg = self.last_segment_strings[i]

            if seg.age > adjusted_confirmation_age:
                if seg.noise:
                    if emitted == -1:
                        if i < self.n_segments and i < len(self.current_segment_strings):
                            new_start_offset = self.current_segment_strings[i].end
                        noise_emitted = i
                elif len(seg.segment_string) > 0:
                    # Check for sentence-ending punctuation
                    ender = seg.segment_string[-1]
                    if ender == '"' and len(seg.segment_string) > 1:
                        ender = seg.segment_string[-2]

                    if ender in '.?!':
                        hold_offset = new_start_offset
                        if i < len(self.current_segment_strings):
                            new_start_offset = self.current_segment_strings[i].end
                        if self.debug:
                            print(f"Phrase Ending '{ender}' seg={i} "
                                  f"conf_age={adjusted_confirmation_age}")
                        result = self._emit_phrases(emitted + 1, i, force=True)
                        if result != -1:
                            emitted = i
                        else:
                            new_start_offset = hold_offset
                    else:
                        # Force emit if followed by additional segments
                        if i < self.n_segments - 2:
                            if self.debug:
                                print(f"Age emit seg={i} "
                                      f"conf_age={adjusted_confirmation_age}")
                            hold_offset = new_start_offset
                            if i < len(self.current_segment_strings):
                                new_start_offset = self.current_segment_strings[i].end
                            self._emit_phrases(emitted + 1, i, force=True)
                            emitted = i
                else:
                    # Empty string
                    if i < len(self.current_segment_strings):
                        new_start_offset = self.current_segment_strings[i].end
                    emitted = i
            else:
                break

        # Trim emitted segments from tracking list
        if new_start_offset != -1:
            if emitted != -1:
                del self.last_segment_strings[:emitted + 1]
            elif noise_emitted != -1:
                del self.last_segment_strings[:noise_emitted + 1]
            elif len(self.last_segment_strings) != self.n_segments:
                self.last_segment_strings = self.last_segment_strings[:self.n_segments]

            self.audio.set_start(new_start_offset, self.audio.audio_start,
                                 self.n_samples_keep)
        elif len(self.last_segment_strings) != self.n_segments:
            self.last_segment_strings = self.last_segment_strings[:self.n_segments]

        # Build in-progress string from remaining segments
        for i in range(len(self.last_segment_strings)):
            seg = self.last_segment_strings[i]
            if not seg.noise and seg.confidence > self.minimum_confidence:
                self._emit_string(seg.segment_string, "IN_PROGRESS")

    def _check_for_overlap(self, i: int, text: str):
        """
        Detect overlapping text between last emitted phrase and new segment.
        Port of ofxWhisperMulti::CheckForOverlap.
        """
        if i != 0 or not self.last_phrase_emitted or not text:
            return

        size_to_check = min(self.overlap_max, len(text),
                            len(self.last_phrase_emitted))
        matched = False
        last_size = len(self.last_phrase_emitted)

        for j in range(size_to_check):
            if text[j] != ' ':
                continue
            for k in range(last_size - 1, max(last_size - size_to_check - 1, -1), -1):
                if text[j] == self.last_phrase_emitted[k]:
                    same_count = 1
                    match_str = text[j]
                    l = 1
                    while (j + l < size_to_check and
                           k + l < last_size and not matched):
                        if text[j + l] == self.last_phrase_emitted[k + l]:
                            match_str += text[j + l]
                            same_count += 1
                            l += 1
                        else:
                            if match_str and match_str[-1] in '.?!':
                                if len(match_str) > 2:
                                    if self.debug:
                                        print(f"Matching overlap: {match_str}")
                                    matched = True
                                    self.audio.push_start_forward(
                                        self.n_samples_keep)
                            break
                    if matched:
                        break
            if matched:
                break

        if matched:
            self.last_phrase_emitted = ""

    def _emit_phrases(self, first_phrase: int, last_phrase: int,
                      force: bool = False) -> int:
        """
        Emit confirmed phrases.
        Port of ofxWhisperMulti::EmitPhrases.
        """
        result = self._emit("EMIT", first_phrase, last_phrase, force)
        if last_phrase < len(self.last_segment_strings):
            self.last_phrase_emitted = self.last_segment_strings[last_phrase].segment_string
        if not result:
            return -1
        return last_phrase

    def _emit(self, reason: str, first_segment: int, last_segment: int,
              force: bool = False) -> bool:
        """
        Emit segments that pass confidence/lifespan gates.
        Port of ofxWhisperMulti::Emit.
        """
        emitted = False
        self.last_phrase_emitted = ""

        if not self.last_segment_strings:
            return False

        full_string = ""
        for i in range(first_segment, min(last_segment + 1,
                                           len(self.last_segment_strings))):
            seg = self.last_segment_strings[i]
            if not seg.noise:
                if ((seg.confidence > self.minimum_confidence and
                     seg.lifespan > self.minimum_lifespan) or force):
                    ender = seg.segment_string[-1] if seg.segment_string else ''
                    if ender == '"' and len(seg.segment_string) > 1:
                        ender = seg.segment_string[-2]

                    full_string += seg.segment_string
                    if ender in '.?!':
                        self._emit_string(full_string, reason)
                        emitted = True
                        full_string = ""
                elif not seg.segment_string:
                    emitted = True
            else:
                if self.debug:
                    print(f"Noisy segment: {seg.segment_string}")
                seg.noise = True
                seg.noise_output = False

        if full_string:
            self._emit_string(full_string, reason)
            emitted = True

        return emitted

    def _emit_and_clear(self, reason: str):
        """Port of ofxWhisperMulti::EmitAndClear."""
        self._emit(reason, 0, len(self.last_segment_strings) - 1, force=True)
        self.last_segment_strings.clear()
        self.audio.pause_cleared_audio = False

    def _emit_string(self, text: str, reason: str):
        """Route emitted text to appropriate output string."""
        if not text:
            return
        # Strip trailing 0xeb bytes (Korean artifact)
        text = text.rstrip('\xeb')

        if reason == "IN_PROGRESS":
            self.in_progress_string += text
        elif reason == "NOISE":
            self.noise_string += text
        else:  # EMIT, OVERFLOW, PAUSE, etc.
            if self.debug and reason != "NOISE":
                print(f"──────────────> {reason} {text}")
            self.emit_string += text

    def _do_callbacks(self):
        """Transfer results to thread-safe output buffers."""
        with self.result_mutex:
            if self.in_progress_string or self.in_progress_string != self.last_in_progress_string:
                self.last_in_progress_string = self.in_progress_string
                self.in_progress_output = self.in_progress_string
                self.fresh_progress = True

            if self.emit_string:
                self.phrases_to_output += self.emit_string
                self.fresh_phrase = True

            if self.noise_string:
                self.noise_output += self.noise_string
                self.fresh_noise = True

    def _segment_is_not_noise(self, text: str) -> bool:
        """Port of ofxWhisperMulti::segment_is_not_noise."""
        if len(text) < 2:
            return len(text) > 0 and text[0] not in NOISE_CHARS
        return text[0] not in NOISE_CHARS and text[1] not in NOISE_CHARS

    def _token_is_not_noise(self, token_text: str) -> bool:
        """Port of ofxWhisperMulti::token_is_not_noise."""
        if not token_text:
            return False

        text = token_text
        if text[0] == ' ':
            text = text[1:]
        if not text:
            return True  # just a space

        char0 = text[0] if len(text) > 0 else '\0'
        char1 = text[1] if len(text) > 1 else '\0'

        if char0 not in NOISE_CHARS and char1 not in NOISE_CHARS:
            if not text.startswith('AUD'):
                return True
        return False

    def _strip_noise_chars(self, text: str) -> str:
        """Strip noise characters from edges of text."""
        start = 0
        for i, c in enumerate(text):
            if c == ' ' or c in NOISE_CHARS:
                continue
            start = i
            break

        end = len(text) - 1
        for i in range(len(text) - 1, start, -1):
            if text[i] == ' ' or text[i] in NOISE_CHARS:
                continue
            end = i
            break

        return text[start:end + 1]

    def get_speaking_rate(self) -> float:
        """Port of ofxWhisperMulti::GetSpeakingRate."""
        duration_acc = 0.0
        length_acc = 0
        for seg in self.last_segment_strings:
            duration_acc += seg.duration
            length_acc += len(seg.segment_string)
        if duration_acc > 0:
            rate = length_acc * 4.0 / duration_acc
            self.past_rate = rate
            return rate
        return self.past_rate

    def get_results(self):
        """
        Retrieve pending results (called from main thread).
        Returns (phrases, in_progress, noise, has_new_data).
        """
        with self.result_mutex:
            phrases = ""
            in_progress = ""
            noise = ""
            has_data = False

            if self.fresh_phrase:
                phrases = self.phrases_to_output
                self.phrases_to_output = ""
                self.fresh_phrase = False
                has_data = True

            if self.fresh_progress:
                in_progress = self.in_progress_output
                self.in_progress_output = ""
                self.fresh_progress = False
                has_data = True

            if self.fresh_noise:
                noise = self.noise_output
                self.noise_output = ""
                self.fresh_noise = False
                has_data = True

            return phrases, in_progress, noise, has_data


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: dpg_system Node
# ─────────────────────────────────────────────────────────────────────────────

def register_whisper_nodes():
    Node.app.register_node("whisper", WhisperNode.factory)


class WhisperNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WhisperNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Parse args for model name and language
        model_name = "base.en"
        language = "english"
        translate = False
        backend_name = "whisper.cpp"

        if args:
            for arg in args:
                if arg in MODEL_SIZES:
                    model_name = arg
                elif arg == "translate":
                    translate = True
                elif arg in ("whisper.cpp", "whisper_cpp", "pywhispercpp"):
                    backend_name = "whisper.cpp"
                elif arg in LANG_LIST:
                    language = arg
                elif arg in LANG_CODES:
                    idx = LANG_CODES.index(arg)
                    language = LANG_LIST[idx]

        # ── Inputs ──
        self.on_off_input = self.add_input('on/off', widget_type='checkbox',
                                           default_value=False,
                                           triggers_execution=True)
        self.audio_input = self.add_input('audio_in', triggers_execution=True)
        self.sample_rate_in_prop = self.add_input('sample_rate_in',
                                                   widget_type='drag_int',
                                                   default_value=16000)

        # ── Properties ──
        self.model_property = self.add_input('model', widget_type='combo',
                                             default_value=model_name,
                                             widget_width=200,
                                             callback=self.model_changed)
        self.model_property.widget.combo_items = MODEL_SIZES

        # Audio device — gracefully handle missing audio hardware
        self.audio_capture = AudioCapture()
        try:
            device_names = self.audio_capture.get_device_list()
        except Exception as e:
            print(f"Whisper: no audio hardware available ({e}), use audio_in input instead")
            device_names = []
        self.device_property = self.add_input('audio device', widget_type='combo',
                                              default_value=device_names[0] if device_names else 'none',
                                              widget_width=300,
                                              callback=self.device_changed)
        self.device_property.widget.combo_items = device_names if device_names else ['none']

        # Language
        self.language_property = self.add_input('language', widget_type='combo',
                                                default_value=language,
                                                widget_width=200,
                                                callback=self.language_changed)
        self.language_property.widget.combo_items = LANG_LIST

        # Translate
        self.translate_property = self.add_input('translate', widget_type='checkbox',
                                                 default_value=translate,
                                                 callback=self.translate_changed)

        # Backend
        self.backend_property = self.add_input('backend', widget_type='combo',
                                               default_value=backend_name,
                                               widget_width=200,
                                               callback=self.backend_changed)
        self.backend_property.widget.combo_items = ["faster-whisper", "whisper.cpp"]

        # ── Outputs ──
        self.phrases_output = self.add_output('phrases')
        self.in_progress_output = self.add_output('in_progress')
        self.noises_output = self.add_output('noises')
        self.energy_output = self.add_output('energy')
        self.rate_output = self.add_output('rate')
        self.language_output = self.add_output('language')

        # ── Options (hidden by default) ──
        self.gain_option = self.add_option('gain', widget_type='drag_float',
                                           default_value=1.0, min=0.0, max=10.0,
                                           callback=self.options_changed)
        self.silence_threshold_option = self.add_option('silence_threshold',
                                                        widget_type='drag_float',
                                                        default_value=0.02,
                                                        min=0.0, max=1.0,
                                                        callback=self.options_changed)
        self.silence_period_option = self.add_option('silence_period',
                                                     widget_type='drag_int',
                                                     default_value=30, min=1, max=100,
                                                     callback=self.options_changed)
        self.update_period_option = self.add_option('update_period',
                                                    widget_type='drag_int',
                                                    default_value=150, min=100, max=5000,
                                                    callback=self.options_changed)
        self.confirmation_age_option = self.add_option('confirmation_age',
                                                       widget_type='drag_int',
                                                       default_value=1, min=0, max=10,
                                                       callback=self.options_changed)
        self.minimum_lifespan_option = self.add_option('minimum_lifespan',
                                                       widget_type='drag_int',
                                                       default_value=4, min=0, max=20,
                                                       callback=self.options_changed)
        self.minimum_confidence_option = self.add_option('minimum_confidence',
                                                         widget_type='drag_float',
                                                         default_value=0.5,
                                                         min=0.0, max=1.0,
                                                         callback=self.options_changed)
        self.min_trailing_conf_option = self.add_option('min_trailing_confidence',
                                                        widget_type='drag_float',
                                                        default_value=0.6,
                                                        min=0.0, max=1.0,
                                                        callback=self.options_changed)
        self.length_factor_option = self.add_option('length_factor',
                                                    widget_type='drag_int',
                                                    default_value=150, min=1, max=200,
                                                    callback=self.options_changed)
        self.buffer_overflow_option = self.add_option('buffer_overflow_fraction',
                                                      widget_type='drag_float',
                                                      default_value=0.9,
                                                      min=0.1, max=1.0,
                                                      callback=self.options_changed)
        self.overlap_option = self.add_option('overlap', widget_type='drag_float',
                                              default_value=200.0, min=0.0, max=2000.0,
                                              callback=self.options_changed)
        self.debug_option = self.add_option('debug', widget_type='checkbox',
                                            default_value=False,
                                            callback=self.options_changed)

        # ── Internal state ──
        self.backend: Optional[WhisperBackend] = None
        self.processor: Optional[WhisperProcessor] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.model_loaded = False
        self.last_progress_text = ""
        self.using_external_audio = False

    def _create_backend(self, backend_name: str) -> WhisperBackend:
        if backend_name == "whisper.cpp":
            return WhisperCppBackend()
        return FasterWhisperBackend()

    def _load_model(self):
        """Load the whisper model (may take a few seconds)."""
        backend_name = self.backend_property()
        model_name = self.model_property()

        self.backend = self._create_backend(backend_name)
        success = self.backend.load_model(model_name)
        if success:
            self.model_loaded = True
        else:
            self.model_loaded = False
            print(f"Failed to load whisper model '{model_name}'")
        return success

    def _start_processing(self):
        """Initialize audio and start the processing thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        # Check if external audio is connected
        self.using_external_audio = (self.audio_input is not None and
                                      len(self.audio_input._parents) > 0)

        # Also force external mode if no audio devices are available
        if not self.using_external_audio and not self.audio_capture.devices:
            self.using_external_audio = True
            print("Whisper: no audio devices found, using external audio input mode")

        if self.using_external_audio:
            # External audio mode — no mic needed, just mark capture ready
            self.audio_capture.external_mode = True
            self.audio_capture.audio_ready = True
            self.audio_capture.running = True
            print("Whisper: using external audio input")
        else:
            # Init mic on main thread (safe — it's just sounddevice)
            device_name = self.device_property()
            device_idx = 0
            for i, d in enumerate(self.audio_capture.devices):
                if d['name'] == device_name:
                    device_idx = i
                    break

            if not self.audio_capture.audio_ready:
                try:
                    if not self.audio_capture.init(device_idx):
                        print("Whisper: mic init failed, falling back to external audio mode")
                        self.using_external_audio = True
                        self.audio_capture.external_mode = True
                        self.audio_capture.audio_ready = True
                        self.audio_capture.running = True
                except Exception as e:
                    print(f"Whisper: mic init error ({e}), falling back to external audio mode")
                    self.using_external_audio = True
                    self.audio_capture.external_mode = True
                    self.audio_capture.audio_ready = True
                    self.audio_capture.running = True

        # Start background thread which handles model loading + processing
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._processing_thread,
                                       daemon=True)
        self.thread.start()
        self.add_frame_task()

    def _stop_processing(self):
        """Stop audio and processing thread."""
        if self.processor:
            self.processor.is_running = False
        self.stop_event.set()
        self.audio_capture.pause()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None

    def _processing_thread(self):
        """Background thread: load model, then run processing loop."""
        # --- Model loading (off the main thread) ---
        if not self.model_loaded:
            print("Whisper: loading model on background thread...")
            try:
                if not self._load_model():
                    print("Whisper: model load failed, thread exiting.")
                    return
            except Exception as e:
                print(f"Whisper: model load crashed: {e}")
                traceback.print_exception(e)
                return

        # --- Create processor and start audio ---
        self.processor = WhisperProcessor(self.backend, self.audio_capture)
        self._apply_options()
        self.audio_capture.resume()
        self.processor.is_running = True

        print("Whisper: processing started.")

        # --- Main processing loop ---
        while not self.stop_event.is_set():
            if self.processor and self.processor.is_running:
                try:
                    self.processor.process()
                except Exception as e:
                    print(f"Whisper processing error: {e}")
                    traceback.print_exception(e)
                    time.sleep(0.5)
            else:
                time.sleep(0.1)

    def _apply_options(self):
        """Apply option widget values to processor and audio capture."""
        if self.processor:
            self.processor.chunk_size_ms = self.update_period_option()
            self.processor.confirmation_age = self.confirmation_age_option()
            self.processor.minimum_lifespan = self.minimum_lifespan_option()
            self.processor.minimum_confidence = self.minimum_confidence_option()
            self.processor.min_trailing_probability = self.min_trailing_conf_option()
            self.processor.length_factor = self.length_factor_option()
            self.processor.buffer_overflow_fraction = self.buffer_overflow_option()
            self.processor.set_overlap(self.overlap_option())
            self.processor.debug = self.debug_option()
            # Propagate debug to the backend for timing info
            if self.backend and hasattr(self.backend, 'debug'):
                self.backend.debug = self.processor.debug
        self.audio_capture.gain = self.gain_option()
        self.audio_capture.voice_threshold = self.silence_threshold_option()
        self.audio_capture.silence_period_threshold = self.silence_period_option()

    # ── Callbacks ──

    def execute(self):
        # Check if this was triggered by audio_in
        audio_data = self.audio_input()
        if audio_data is not None and self.processor is not None:
            # Feed external audio into the capture buffer
            try:
                if hasattr(audio_data, 'detach'):
                    # Torch tensor
                    audio_np = audio_data.detach().cpu().numpy()
                elif isinstance(audio_data, np.ndarray):
                    audio_np = audio_data
                elif isinstance(audio_data, (list, tuple)):
                    audio_np = np.array(audio_data, dtype=np.float32)
                else:
                    audio_np = None

                if audio_np is not None:
                    sr = int(self.sample_rate_in_prop())
                    self.audio_capture.feed_external(audio_np, sample_rate=sr)
            except Exception as e:
                if self.processor and self.processor.debug:
                    print(f"Whisper: audio_in error: {e}")
            return

        # On/off toggle
        on = self.on_off_input()
        if on:
            if self.thread is None or not self.thread.is_alive():
                self._start_processing()
        else:
            self._stop_processing()

    def frame_task(self):
        """Called every frame on the main thread — poll for results."""
        if self.processor is None:
            return

        phrases, in_progress, noise, has_data = self.processor.get_results()

        if has_data:
            if phrases:
                self.phrases_output.send(phrases)
            if noise:
                self.noises_output.send(noise)
            if in_progress and in_progress != self.last_progress_text:
                self.in_progress_output.send(in_progress)
                self.last_progress_text = in_progress
                if self.processor.current_language:
                    self.language_output.send(self.processor.current_language)

        # Always output energy and rate if running
        if self.processor.is_running:
            self.energy_output.send(self.audio_capture.energy)
            rate = self.processor.get_speaking_rate()
            self.rate_output.send(rate)

    def model_changed(self):
        """Reload model when model selection changes."""
        was_running = self.processor is not None and self.processor.is_running
        if was_running:
            self._stop_processing()
        self.model_loaded = False
        if was_running:
            self._start_processing()

    def device_changed(self):
        """Switch audio device."""
        device_name = self.device_property()
        for i, d in enumerate(self.audio_capture.devices):
            if d['name'] == device_name:
                self.audio_capture.change_device(i)
                break

    def language_changed(self):
        if self.processor:
            self.processor.language = self.language_property()

    def translate_changed(self):
        if self.processor:
            self.processor.translate = self.translate_property()

    def backend_changed(self):
        """Reload with different backend."""
        was_running = self.processor is not None and self.processor.is_running
        if was_running:
            self._stop_processing()
        self.model_loaded = False
        if was_running:
            self._start_processing()

    def options_changed(self):
        self._apply_options()

    def custom_cleanup(self):
        """Called when node is deleted."""
        self._stop_processing()
        self.audio_capture.close()
