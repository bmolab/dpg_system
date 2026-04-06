"""
speech_analysis_nodes.py
Non-semantic speech analysis nodes for dpg_system.

SpeechPitchNode  — real-time F0 extraction (PYIN / Parselmouth / Kaldi backends)
SpeechProsodyNode — windowed prosody statistics derived from an F0 contour

Created 2026-03-29.
"""

import numpy as np
import time
import traceback

from dpg_system.torch_base_nodes import *

# ─────────────────────────────────────────────────────────────────────────────
# Optional backend availability
# ─────────────────────────────────────────────────────────────────────────────

_librosa_available = False
try:
    import librosa
    _librosa_available = True
except ImportError:
    pass

_parselmouth_available = False
try:
    import parselmouth
    _parselmouth_available = True
except ImportError:
    pass

_torchaudio_available = False
try:
    import torchaudio
    _torchaudio_available = True
except ImportError:
    pass

_scipy_available = False
try:
    from scipy.signal import savgol_filter
    _scipy_available = True
except ImportError:
    pass


def _best_available_backend():
    """Return the name of the best pitch backend that is actually importable."""
    if _librosa_available:
        return 'pyin'
    if _parselmouth_available:
        return 'parselmouth'
    if _torchaudio_available:
        return 'kaldi'
    return 'none'


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight circular buffer for raw audio
# ─────────────────────────────────────────────────────────────────────────────

class AudioRingBuffer:
    """
    Fixed-length circular buffer that accumulates float32 audio chunks
    and returns the most recent N samples as a contiguous numpy array.
    """

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0  # monotonically increasing

    def resize(self, new_max: int):
        if new_max == self.max_samples:
            return
        self.max_samples = new_max
        self.buffer = np.zeros(new_max, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0

    def push(self, chunk: np.ndarray):
        """Append a 1-D chunk, oldest data overwritten when full."""
        n = len(chunk)
        if n == 0:
            return
        if n >= self.max_samples:
            # chunk is larger than the whole buffer — keep last max_samples
            self.buffer[:] = chunk[-self.max_samples:]
            self.write_pos = 0
            self.total_written += n
            return

        end = self.write_pos + n
        if end <= self.max_samples:
            self.buffer[self.write_pos:end] = chunk
        else:
            first = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = chunk[:first]
            self.buffer[:n - first] = chunk[first:]

        self.write_pos = end % self.max_samples
        self.total_written += n

    def available(self) -> int:
        """Number of valid samples in the buffer (up to max_samples)."""
        return min(self.total_written, self.max_samples)

    def get_last_n(self, n: int) -> np.ndarray:
        """Return the most recent *n* samples as a contiguous float32 array."""
        avail = self.available()
        if n > avail:
            n = avail
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        start = (self.write_pos - n) % self.max_samples
        if start + n <= self.max_samples:
            return self.buffer[start:start + n].copy()
        else:
            first = self.max_samples - start
            return np.concatenate([self.buffer[start:], self.buffer[:n - first]])

    def get_all(self) -> np.ndarray:
        """Return all available samples, oldest first."""
        return self.get_last_n(self.available())


# ─────────────────────────────────────────────────────────────────────────────
# F0 Ring Buffer — for accumulating pitch frames
# ─────────────────────────────────────────────────────────────────────────────

class F0RingBuffer:
    """Circular buffer for f0 / voiced_prob frames."""

    def __init__(self, max_frames: int):
        self.max_frames = max_frames
        self.f0 = np.zeros(max_frames, dtype=np.float32)
        self.voiced_prob = np.zeros(max_frames, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0

    def resize(self, new_max: int):
        if new_max == self.max_frames:
            return
        self.max_frames = new_max
        self.f0 = np.zeros(new_max, dtype=np.float32)
        self.voiced_prob = np.zeros(new_max, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0

    def push_frames(self, f0_frames: np.ndarray, vp_frames: np.ndarray):
        n = len(f0_frames)
        if n == 0:
            return
        if n >= self.max_frames:
            self.f0[:] = f0_frames[-self.max_frames:]
            self.voiced_prob[:] = vp_frames[-self.max_frames:]
            self.write_pos = 0
            self.total_written += n
            return
        end = self.write_pos + n
        if end <= self.max_frames:
            self.f0[self.write_pos:end] = f0_frames
            self.voiced_prob[self.write_pos:end] = vp_frames
        else:
            first = self.max_frames - self.write_pos
            self.f0[self.write_pos:] = f0_frames[:first]
            self.f0[:n - first] = f0_frames[first:]
            self.voiced_prob[self.write_pos:] = vp_frames[:first]
            self.voiced_prob[:n - first] = vp_frames[first:]
        self.write_pos = end % self.max_frames
        self.total_written += n

    def available(self) -> int:
        return min(self.total_written, self.max_frames)

    def get_last_n(self, n: int):
        avail = self.available()
        if n > avail:
            n = avail
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        start = (self.write_pos - n) % self.max_frames
        if start + n <= self.max_frames:
            return (self.f0[start:start + n].copy(),
                    self.voiced_prob[start:start + n].copy())
        else:
            first = self.max_frames - start
            f0_out = np.concatenate([self.f0[start:], self.f0[:n - first]])
            vp_out = np.concatenate([self.voiced_prob[start:],
                                      self.voiced_prob[:n - first]])
            return f0_out, vp_out

    def get_all(self):
        return self.get_last_n(self.available())


# ─────────────────────────────────────────────────────────────────────────────
# Node registration
# ─────────────────────────────────────────────────────────────────────────────

def register_speech_analysis_nodes():
    Node.app.register_node('speech_pitch', SpeechPitchNode.factory)
    Node.app.register_node('speech_prosody', SpeechProsodyNode.factory)


# ─────────────────────────────────────────────────────────────────────────────
# SpeechPitchNode
# ─────────────────────────────────────────────────────────────────────────────

class SpeechPitchNode(TorchNode):
    """
    Real-time F0 (pitch) extraction from streaming audio tensors.

    Backends (in fallback order):
      1. PYIN   — librosa.pyin (probabilistic YIN, best for speech)
      2. Praat  — parselmouth  (autocorrelation-based, rich voice quality)
      3. Kaldi  — torchaudio.functional.compute_kaldi_pitch (GPU-capable)

    Usage: t.audio_source → speech_pitch → [f0, voiced_prob, voiced]
    """

    @staticmethod
    def factory(name, data, args=None):
        node = SpeechPitchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        default_backend = _best_available_backend()
        self._available_backends = []
        if _librosa_available:
            self._available_backends.append('pyin')
        if _parselmouth_available:
            self._available_backends.append('parselmouth')
        if _torchaudio_available:
            self._available_backends.append('kaldi')
        if not self._available_backends:
            self._available_backends.append('none')

        # Inputs
        self.input = self.add_input('audio tensor in', triggers_execution=True)

        # Properties
        self.sample_rate_prop = self.add_property('sample_rate', widget_type='drag_int',
                                                   default_value=16000)
        self.buffer_sec_prop = self.add_input('buffer_sec', widget_type='drag_float',
                                               default_value=1.0,
                                               callback=self._buffer_size_changed)
        self.min_freq_prop = self.add_input('min_freq', widget_type='drag_float',
                                             default_value=65.0)
        self.max_freq_prop = self.add_input('max_freq', widget_type='drag_float',
                                             default_value=600.0)
        self.output_mode = self.add_input('output_mode', widget_type='combo',
                                           default_value='summary')
        self.output_mode.widget.combo_items = ['summary', 'time_series']

        self.backend_prop = self.add_input('backend', widget_type='combo',
                                            default_value=default_backend,
                                            callback=self._backend_changed)
        self.backend_prop.widget.combo_items = self._available_backends

        self.analysis_fps_prop = self.add_input('analysis_fps', widget_type='drag_float',
                                                 default_value=20.0)

        # Status label
        self.status_label = self.add_label(f'backend: {default_backend}')

        # Outputs
        self.f0_output = self.add_output('f0')
        self.voiced_prob_output = self.add_output('voiced_prob')
        self.voiced_output = self.add_output('voiced')
        self.f0_raw_output = self.add_output('f0_raw')  # always time-series for prosody

        # Internal state
        sr = 16000
        max_samples = int(1.0 * sr)
        self._ring = AudioRingBuffer(max_samples)
        self._active_backend = default_backend
        self._hop_length = 160  # 10 ms at 16 kHz
        self._last_analysis_time = 0.0

    def _buffer_size_changed(self):
        sr = int(self.sample_rate_prop())
        buf_sec = float(self.buffer_sec_prop())
        new_size = max(int(buf_sec * sr), sr)  # at least 1 second
        self._ring.resize(new_size)

    def _backend_changed(self):
        requested = self.backend_prop()
        if requested in self._available_backends and requested != 'none':
            self._active_backend = requested
            self.status_label.set(f'backend: {requested}')
        else:
            fallback = _best_available_backend()
            self._active_backend = fallback
            self.status_label.set(f'backend: {fallback} (fallback)')

    def execute(self):
        data = self.input_to_tensor()
        if data is None:
            return

        # Convert tensor to mono float32 numpy
        audio_np = data.detach().cpu().numpy().astype(np.float32)
        if audio_np.ndim > 1:
            # Take first channel or average
            if audio_np.shape[0] <= audio_np.shape[-1]:
                audio_np = audio_np[0]  # (channels, samples) → first channel
            else:
                audio_np = audio_np[:, 0]  # (samples, channels) → first channel
        audio_np = audio_np.flatten()

        # Push into ring buffer (sliding window — always accumulate)
        self._ring.push(audio_np)

        # Throttle analysis rate based on user-configurable fps
        now = time.time()
        fps = max(float(self.analysis_fps_prop()), 1.0)
        if now - self._last_analysis_time < 1.0 / fps:
            return
        self._last_analysis_time = now

        # Analyse the full buffer contents (sliding window re-analysis)
        sr = int(self.sample_rate_prop())
        buf_sec = float(self.buffer_sec_prop())
        n_samples = min(int(buf_sec * sr), self._ring.available())
        # Need enough data for at least a few pitch frames (~0.1s)
        if n_samples < max(sr // 10, 2048):
            return

        audio = self._ring.get_last_n(n_samples)
        fmin = float(self.min_freq_prop())
        fmax = float(self.max_freq_prop())

        # Run pitch extraction
        f0, voiced_prob, voiced_flag = self._extract_pitch(audio, sr, fmin, fmax)

        if f0 is None or len(f0) == 0:
            return

        # Always send raw time-series for downstream prosody
        self.f0_raw_output.send(f0)

        # Output based on mode
        mode = self.output_mode()
        if mode == 'summary':
            # Summarize over voiced frames
            voiced_mask = voiced_flag.astype(bool)
            if np.any(voiced_mask):
                mean_f0 = float(np.nanmean(f0[voiced_mask]))
                mean_vp = float(np.nanmean(voiced_prob[voiced_mask]))
            else:
                mean_f0 = 0.0
                mean_vp = 0.0
            any_voiced = bool(np.any(voiced_mask))

            self.f0_output.send(mean_f0)
            self.voiced_prob_output.send(mean_vp)
            self.voiced_output.send(any_voiced)
        else:
            # time_series mode
            self.f0_output.send(f0)
            self.voiced_prob_output.send(voiced_prob)
            self.voiced_output.send(voiced_flag)

    def _extract_pitch(self, audio, sr, fmin, fmax):
        """Dispatch to the active backend. Returns (f0, voiced_prob, voiced_flag)."""
        backend = self._active_backend

        if backend == 'pyin' and _librosa_available:
            return self._pyin(audio, sr, fmin, fmax)
        elif backend == 'parselmouth' and _parselmouth_available:
            return self._parselmouth(audio, sr, fmin, fmax)
        elif backend == 'kaldi' and _torchaudio_available:
            return self._kaldi(audio, sr, fmin, fmax)
        else:
            # Try all backends in priority order
            if _librosa_available:
                return self._pyin(audio, sr, fmin, fmax)
            if _parselmouth_available:
                return self._parselmouth(audio, sr, fmin, fmax)
            if _torchaudio_available:
                return self._kaldi(audio, sr, fmin, fmax)
            return None, None, None

    def _pyin(self, audio, sr, fmin, fmax):
        """PYIN pitch tracking via librosa."""
        try:
            hop = self._hop_length
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio, fmin=fmin, fmax=fmax, sr=sr,
                frame_length=2048, hop_length=hop,
                fill_na=0.0
            )
            # voiced_flag: bool array, voiced_prob: float array
            # f0 has 0.0 where unvoiced (because fill_na=0.0)
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            voiced_prob = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
            voiced_flag = voiced_flag.astype(np.float32)
            return f0, voiced_prob, voiced_flag
        except Exception as e:
            print(f'speech_pitch PYIN error: {e}')
            traceback.print_exc()
            return None, None, None

    def _parselmouth(self, audio, sr, fmin, fmax):
        """Praat pitch tracking via parselmouth."""
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            hop_sec = self._hop_length / sr
            pitch_obj = snd.to_pitch(time_step=hop_sec,
                                      pitch_floor=fmin,
                                      pitch_ceiling=fmax)
            f0 = pitch_obj.selected_array['frequency'].astype(np.float32)

            # Derive voiced probability from harmonicity (HNR)
            # Higher HNR → more periodic → more likely voiced
            n_frames = len(f0)
            voiced_flag = (f0 > 0).astype(np.float32)
            try:
                harmonicity = snd.to_harmonicity(time_step=hop_sec,
                                                  minimum_pitch=fmin)
                hnr_values = harmonicity.values[0]  # shape: (1, n_frames)
                # Resample/align if frame counts differ
                if len(hnr_values) != n_frames:
                    # Use nearest-neighbor resampling
                    indices = np.linspace(0, len(hnr_values) - 1, n_frames).astype(int)
                    hnr_values = hnr_values[indices]
                # Map HNR (typically -200 to +40 dB) to 0–1 probability
                # HNR > 0 dB is reasonably voiced; > 20 dB is very clean
                hnr_values = np.nan_to_num(hnr_values, nan=-200.0)
                voiced_prob = np.clip((hnr_values + 5.0) / 25.0, 0.0, 1.0).astype(np.float32)
            except Exception:
                # Fallback: binary voiced probability from f0
                voiced_prob = voiced_flag.copy()

            return f0, voiced_prob, voiced_flag
        except Exception as e:
            print(f'speech_pitch Parselmouth error: {e}')
            traceback.print_exc()
            return None, None, None

    def _kaldi(self, audio, sr, fmin, fmax):
        """Kaldi pitch tracking via torchaudio."""
        try:
            import torch
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            pitch_feature = torchaudio.functional.compute_kaldi_pitch(
                audio_tensor, sr,
                min_f0=fmin, max_f0=fmax,
                frame_shift=self._hop_length / sr * 1000  # in ms
            )
            # pitch_feature shape: (1, n_frames, 2) — [nccf, pitch]
            nccf = pitch_feature[0, :, 0].numpy().astype(np.float32)
            f0 = pitch_feature[0, :, 1].numpy().astype(np.float32)

            # Use NCCF as voiced probability proxy (higher = more periodic)
            voiced_prob = np.clip(nccf, 0, 1)
            voiced_flag = (voiced_prob > 0.3).astype(np.float32)

            # Zero out f0 where unvoiced
            f0[voiced_flag < 0.5] = 0.0
            return f0, voiced_prob, voiced_flag
        except Exception as e:
            print(f'speech_pitch Kaldi error: {e}')
            traceback.print_exc()
            return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# SpeechProsodyNode
# ─────────────────────────────────────────────────────────────────────────────

class SpeechProsodyNode(Node):
    """
    Compute windowed prosody statistics from an incoming F0 time series.

    Outputs pitch slope, range, variability, mean, and intonation direction.

    Usage: speech_pitch.f0_raw → speech_prosody → [slope, range, variability, ...]
    """

    @staticmethod
    def factory(name, data, args=None):
        node = SpeechProsodyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Inputs
        self.f0_input = self.add_input('f0_in', triggers_execution=True)

        # Properties
        self.hop_time_prop = self.add_input('hop_time_ms', widget_type='drag_float',
                                             default_value=10.0)
        self.window_sec_prop = self.add_input('window_sec', widget_type='drag_float',
                                               default_value=1.0,
                                               callback=self._window_changed)
        self.smoothing_prop = self.add_input('smoothing', widget_type='drag_float',
                                              default_value=0.3)
        self.output_mode = self.add_input('output_mode', widget_type='combo',
                                           default_value='summary')
        self.output_mode.widget.combo_items = ['summary', 'time_series']

        # Outputs
        self.slope_output = self.add_output('pitch_slope')
        self.range_output = self.add_output('pitch_range')
        self.variability_output = self.add_output('pitch_variability')
        self.mean_output = self.add_output('pitch_mean')
        self.intonation_output = self.add_output('intonation')

        # Internal F0 ring buffer (default: ~100 frames = 1 sec at 10ms hop)
        self._f0_ring = F0RingBuffer(200)

    def _window_changed(self):
        hop_ms = float(self.hop_time_prop())
        window_sec = float(self.window_sec_prop())
        frames_needed = int((window_sec * 1000) / max(hop_ms, 1.0))
        # Keep buffer 2x the analysis window for headroom
        self._f0_ring.resize(max(frames_needed * 2, 50))

    def execute(self):
        raw = self.f0_input()
        if raw is None:
            return

        # Accept numpy, torch tensor, list, or scalar
        if hasattr(raw, 'detach'):
            f0_new = raw.detach().cpu().numpy().astype(np.float32).flatten()
        elif isinstance(raw, np.ndarray):
            f0_new = raw.astype(np.float32).flatten()
        elif isinstance(raw, (list, tuple)):
            f0_new = np.array(raw, dtype=np.float32)
        else:
            # single value — wrap
            f0_new = np.array([float(raw)], dtype=np.float32)

        if len(f0_new) == 0:
            return

        # We don't have per-frame voiced_prob from raw f0, so derive from f0 > 0
        vp_new = (f0_new > 0).astype(np.float32)
        self._f0_ring.push_frames(f0_new, vp_new)

        # Determine analysis window
        hop_ms = float(self.hop_time_prop())
        window_sec = float(self.window_sec_prop())
        window_frames = int((window_sec * 1000) / max(hop_ms, 1.0))
        window_frames = min(window_frames, self._f0_ring.available())

        if window_frames < 5:  # need at least a few frames
            return

        f0_win, vp_win = self._f0_ring.get_last_n(window_frames)

        # Mask for voiced frames
        voiced_mask = f0_win > 0

        if not np.any(voiced_mask):
            # All unvoiced
            self.slope_output.send(0.0)
            self.range_output.send(0.0)
            self.variability_output.send(0.0)
            self.mean_output.send(0.0)
            self.intonation_output.send('unvoiced')
            return

        # Work with voiced portion for statistics
        f0_voiced = f0_win.copy()
        f0_voiced[~voiced_mask] = np.nan

        # Smooth the F0 contour
        smoothing = float(self.smoothing_prop())
        f0_smoothed = self._smooth_f0(f0_voiced, smoothing)

        # ── Compute features ──

        # Pitch mean
        pitch_mean = float(np.nanmean(f0_smoothed))

        # Pitch range
        f0_valid = f0_smoothed[~np.isnan(f0_smoothed)]
        if len(f0_valid) > 0:
            pitch_range = float(np.max(f0_valid) - np.min(f0_valid))
        else:
            pitch_range = 0.0

        # Pitch variability (std dev)
        if len(f0_valid) > 1:
            pitch_variability = float(np.nanstd(f0_smoothed))
        else:
            pitch_variability = 0.0

        # Pitch slope (gradient in Hz per frame → convert to Hz/sec)
        hop_sec = hop_ms / 1000.0
        gradient = np.gradient(f0_smoothed)
        # Use nanmean of gradient for summary slope
        gradient_valid = gradient[~np.isnan(gradient)]

        if len(gradient_valid) > 0:
            mean_slope_per_frame = float(np.nanmean(gradient_valid))
            pitch_slope = mean_slope_per_frame / hop_sec  # Hz/second
        else:
            pitch_slope = 0.0

        # Intonation direction
        if abs(pitch_slope) < 5.0:  # less than 5 Hz/sec → flat
            intonation = 'flat'
        elif pitch_slope > 0:
            intonation = 'rising'
        else:
            intonation = 'falling'

        # ── Output ──
        mode = self.output_mode()
        if mode == 'summary':
            self.slope_output.send(pitch_slope)
            self.range_output.send(pitch_range)
            self.variability_output.send(pitch_variability)
            self.mean_output.send(pitch_mean)
            self.intonation_output.send(intonation)
        else:
            # time_series mode: output per-frame arrays where applicable
            slope_ts = np.gradient(f0_smoothed) / hop_sec
            slope_ts = np.nan_to_num(slope_ts, nan=0.0).astype(np.float32)
            self.slope_output.send(slope_ts)
            self.range_output.send(pitch_range)       # scalar — range is window-level
            self.variability_output.send(pitch_variability)  # scalar
            self.mean_output.send(pitch_mean)          # scalar
            self.intonation_output.send(intonation)    # string

    @staticmethod
    def _smooth_f0(f0: np.ndarray, smoothing: float) -> np.ndarray:
        """
        Smooth F0 contour, interpolating over unvoiced (NaN) gaps.
        smoothing: 0.0 = no smoothing, 1.0 = maximum smoothing.
        Uses Savitzky-Golay filter when scipy is available, else simple EMA.
        """
        if smoothing <= 0.0:
            return f0.copy()

        result = f0.copy()
        n = len(result)

        # Interpolate NaN gaps for smoother derivative computation
        nans = np.isnan(result)
        if np.all(nans):
            return result
        if np.any(nans):
            # Linear interpolation over NaN regions
            not_nan = ~nans
            indices = np.arange(n)
            result[nans] = np.interp(indices[nans], indices[not_nan], result[not_nan])

        if _scipy_available and n >= 5:
            # Map smoothing 0–1 to window length 5–n (must be odd)
            max_win = min(n, 51)
            if max_win % 2 == 0:
                max_win -= 1
            win_len = int(5 + smoothing * (max_win - 5))
            if win_len % 2 == 0:
                win_len += 1
            win_len = max(5, min(win_len, n))
            if win_len % 2 == 0:
                win_len -= 1
            if win_len >= 5 and n >= win_len:
                polyorder = min(3, win_len - 1)
                result = savgol_filter(result, win_len, polyorder)
        else:
            # Simple exponential moving average fallback
            alpha = 1.0 - smoothing * 0.9  # smoothing=1 → alpha=0.1
            for i in range(1, n):
                if not np.isnan(result[i]):
                    result[i] = alpha * result[i] + (1 - alpha) * result[i - 1]

        # Restore NaN where originally unvoiced
        result[nans] = np.nan
        return result
