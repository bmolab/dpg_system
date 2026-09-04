"""Shared audio input plumbing: one microphone wrapper, one rate converter.

Before this module, t.audio_source (torchaudio_nodes) and whisper each carried
their own PortAudio input wrapper, their own device enumeration and their own
resampling -- whisper's by integer decimation with no anti-alias filter. Both
now open their stream through `AudioSource` and convert rate through
`RateConverter`. Output streams are not here: the single output stream lives
in sampler.SamplerEngine and the synth graph and player nodes mix into it.

Deliberately numpy + sounddevice only, so a node module that does not need
torch (whisper) can import it without pulling torch in.
"""

import math

import numpy as np

try:
    import sounddevice as sd
    sounddevice_available = True
except ImportError:
    sd = None
    sounddevice_available = False

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    _scipy_available = True
except ImportError:
    _scipy_available = False


def input_devices():
    """Every PortAudio device with at least one input channel.

    Returns a list of dicts: name, index, channels, default_samplerate. The
    list is whatever PortAudio saw at process start; a device plugged in
    later does not appear until the next launch (see sampler.output_devices
    for why rescanning is not attempted).
    """
    if not sounddevice_available:
        return []
    devices = []
    try:
        for index, info in enumerate(sd.query_devices()):
            if info.get('max_input_channels', 0) > 0:
                devices.append({
                    'name': info['name'],
                    'index': index,
                    'channels': int(info['max_input_channels']),
                    'default_samplerate': float(info['default_samplerate']),
                })
    except Exception as error:
        print(f'audio_io: could not list input devices ({error})')
    return devices


def default_input_index(devices=None):
    """PortAudio's default input, or the first input device, or None."""
    if devices is None:
        devices = input_devices()
    if not devices:
        return None
    try:
        index = sd.default.device['input']
        if index is not None and index >= 0 and any(d['index'] == index for d in devices):
            return index
    except Exception:
        pass
    return devices[0]['index']


class AudioSource:
    """One PortAudio input stream, delivering float32 (frames, channels).

    Construct it with the format you want; `start()` opens the stream and
    returns False (rather than raising) if the device refuses. The callback
    you install with `set_callback` runs on PortAudio's thread with
    (indata, frames, time_info, status) -- keep it short, and never let an
    exception escape it or the stream silently dies.

    A machine with no input device gets a source whose `device_index` is
    None; `start()` then fails cleanly, so a node can still be built and a
    patch still loaded.
    """

    def __init__(self, channels=1, rate=16000, chunk=1024, dtype='float32',
                 data_format=None):
        self.samplerate = int(rate)
        self.channels = int(channels)
        self.blocksize = int(chunk)
        # `data_format` survives from the pyaudio days for callers that
        # still pass it; only the dtype string matters now.
        self.dtype = dtype if isinstance(dtype, str) else 'float32'
        self.stream = None
        self.callback_routine = None

        self.devices = input_devices()
        self.sources = {d['index']: d['name'] for d in self.devices}
        self.device_index = default_input_index(self.devices)
        if self.device_index is None:
            print('AudioSource: no input audio device found')

    # -- devices ------------------------------------------------------------

    def get_device_list(self):
        return list(self.sources.values())

    def change_source(self, source_name):
        for index, name in self.sources.items():
            if name == source_name:
                self.device_index = index
                return True
        print(f"AudioSource: source '{source_name}' not found, no change made")
        return False

    def get_device_info(self):
        if self.device_index is None or not sounddevice_available:
            return {}
        return sd.query_devices(self.device_index)

    def get_max_input_channels(self):
        return int(self.get_device_info().get('max_input_channels', 0))

    def get_default_sample_rate(self):
        return int(self.get_device_info().get('default_samplerate', self.samplerate))

    def check_format(self, rate, channels, dtype=None):
        if self.device_index is None or not sounddevice_available:
            return False
        try:
            sd.check_input_settings(device=self.device_index, channels=channels,
                                    samplerate=rate, dtype=dtype or self.dtype)
            return True
        except Exception:
            return False

    # -- stream -------------------------------------------------------------

    def set_callback(self, routine):
        self.callback_routine = routine

    def _internal_callback(self, indata, frames, time_info, status):
        if self.callback_routine is not None:
            self.callback_routine(indata, frames, time_info, status)

    @property
    def active(self):
        return self.stream is not None and self.stream.active

    def start(self):
        if self.active:
            return True
        if self.device_index is None or not sounddevice_available:
            print('AudioSource: no input device to open')
            return False
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                device=self.device_index,
                dtype=self.dtype,
                blocksize=self.blocksize,
                callback=self._internal_callback,
            )
            self.stream.start()
            return True
        except Exception as error:
            print(f'AudioSource: error starting stream ({error})')
            self.stream = None
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        return True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class RateConverter:
    """Stateful sample-rate conversion for a mono stream fed in blocks.

    Going down in rate, a fourth-order Butterworth at 0.45 of the target
    rate removes what would otherwise fold back as aliasing; its state is
    carried between calls, so block boundaries are seamless. Both directions
    then read the input at a fractional stride with linear interpolation,
    with the fractional phase and the last input sample carried over, so the
    output is continuous no matter how the input is chunked.

    Linear interpolation is not a windowed-sinc, but for the traffic here --
    speech into whisper at 16 kHz, mic and file audio into analysis nodes --
    it is a large step up from dropping samples, and it costs nothing.
    """

    def __init__(self, source_rate, target_rate):
        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        self.ratio = self.source_rate / self.target_rate
        self.identity = abs(self.ratio - 1.0) < 1.0e-9
        self._phase = 0.0          # fractional read position past the carried sample
        self._carry = np.zeros(1, dtype=np.float32)
        self._have_carry = False
        self._sos = None
        self._zi = None
        if not self.identity and self.ratio > 1.0 and _scipy_available:
            cutoff = 0.45 * self.target_rate / (0.5 * self.source_rate)
            self._sos = butter(4, min(0.99, cutoff), output='sos')
            self._zi = sosfilt_zi(self._sos)

    def process(self, x):
        """Convert one block. Returns float32; may be empty for tiny blocks."""
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size == 0:
            return x
        if self.identity:
            return x
        if self._sos is not None:
            x, self._zi = sosfilt(self._sos, x, zi=self._zi)
            x = x.astype(np.float32, copy=False)

        # Input line for this call: last sample of the previous call, then
        # this block. Read positions are measured from that carried sample.
        if self._have_carry:
            line = np.concatenate((self._carry, x))
        else:
            line = x
            self._phase = 0.0
        last = line.size - 1                         # highest index we may interpolate to
        first = self._phase
        if first > last:
            # Block too short to reach the next output sample; keep waiting.
            self._phase = first - last
            self._carry[0] = line[-1]
            self._have_carry = True
            return np.zeros(0, dtype=np.float32)
        count = int(math.floor((last - first) / self.ratio)) + 1
        positions = first + np.arange(count, dtype=np.float64) * self.ratio
        idx = np.floor(positions).astype(np.int64)
        frac = (positions - idx).astype(np.float32)
        idx_next = np.minimum(idx + 1, last)
        out = line[idx] * (1.0 - frac) + line[idx_next] * frac
        # Where the next output would land, relative to the new carried sample.
        self._phase = (first + count * self.ratio) - last
        self._carry[0] = line[-1]
        self._have_carry = True
        return out.astype(np.float32, copy=False)

    def reset(self):
        self._phase = 0.0
        self._have_carry = False
        if self._sos is not None:
            self._zi = sosfilt_zi(self._sos)


def to_mono(block):
    """(frames, channels), (channels, frames) or 1-D -> 1-D float32 mean.

    The smaller dimension is taken as channels, which is right for anything
    up to a few dozen channels of more than a few dozen samples.
    """
    block = np.asarray(block)
    if block.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.ndim > 2:
        block = block.reshape(block.shape[0], -1)
    if block.shape[0] <= block.shape[1]:
        block = block.T
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)
