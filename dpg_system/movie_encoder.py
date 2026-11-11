"""Background movie encoder (PyAV) used by the movie_writer node.

    enc = MovieEncoder('/path/take.mov', codec='prores', fps=29.97)
    enc.write(frame)          # (H, W, 3) uint8 RGB; queued, encoded on a thread
    enc.close()               # flush and finish the file (returns at once)

Encoding runs on its own thread behind a short queue. write() waits when the
queue is full unless drop_when_behind is set, in which case the frame is
counted in .dropped and discarded. The first frame fixes the movie size; later
frames of another size are scaled to it, and an odd width or height is padded
to even (the 4:2:x formats need it).

Codecs ('prores', 'prores hq', 'h264', 'hevc') use the Apple hardware encoder
when present and fall back to software encoders otherwise. Output is tagged
BT.709, limited range, and converted with that matrix -- without the explicit
matrix the hardware ProRes files decode with a colour cast (swscale defaults to
BT.601 while the encoder tags 709). 10-bit formats are fed 16-bit RGB, which
keeps swscale's rounding offset under about 1 grey level.
"""
import os
import queue
import threading
import time
from fractions import Fraction

import numpy as np

try:
    import av
    from av.video.reformatter import Colorspace
    AV_AVAILABLE = True
except Exception:  # pragma: no cover
    av = None
    AV_AVAILABLE = False

CODECS = ['prores', 'prores hq', 'h264', 'hevc']

# (encoder name, pixel format, options, is_hardware) in order of preference
_CANDIDATES = {
    'prores': [('prores_videotoolbox', 'p010le', {'profile': 'standard'}, True),
               ('prores_ks', 'yuv422p10le', {'profile': '2', 'vendor': 'apl0'}, False)],
    'prores hq': [('prores_videotoolbox', 'p010le', {'profile': 'hq'}, True),
                  ('prores_ks', 'yuv422p10le', {'profile': '3', 'vendor': 'apl0'}, False)],
    'h264': [('h264_videotoolbox', 'nv12', {}, True),
             ('libx264', 'yuv420p', {'crf': '16', 'preset': 'medium'}, False)],
    'hevc': [('hevc_videotoolbox', 'nv12', {}, True),
             ('libx265', 'yuv420p', {'crf': '18', 'preset': 'medium'}, False)],
}


def frame_rate(fps):
    """Exact rate: the NTSC rates (23.976, 29.97, 59.94 ...) map to N*1000/1001,
    as ffmpeg's -r 29.97 does, rather than 2997/100."""
    fps = float(fps)
    for base in (24, 30, 48, 60, 120):
        if abs(fps - base * 1000 / 1001) < 0.005:
            return Fraction(base * 1000, 1001)
    return Fraction(fps).limit_denominator(1000)


def unique_path(path):
    """path, or path with _2, _3 ... before the extension if it exists."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists('%s_%d%s' % (base, n, ext)):
        n += 1
    return '%s_%d%s' % (base, n, ext)


def resolve_movie_path(path, prefix='movie'):
    """A folder (or empty path) gets a timestamped file name inside it; a
    path without an extension gets .mov. Never returns an existing file."""
    path = os.path.expanduser(path or '')
    if path == '' or os.path.isdir(path) or path.endswith(os.sep):
        folder = path or os.getcwd()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, prefix + '_' + time.strftime('%Y-%m-%d_%Hh%Mm%S') + '.mov')
    elif os.path.splitext(path)[1] == '':
        path += '.mov'
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return unique_path(path)


def to_rgb_uint8(frame):
    """numpy / torch, HxW, HxWx1/3/4, uint8 or float 0..1 -> HxWx3 uint8."""
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('movie_writer: frame must be HxW, HxWx3 or HxWx4, got shape %s' % (a.shape,))
    return np.ascontiguousarray(a)


class MovieEncoder:
    def __init__(self, path, codec='prores', fps=29.97, bitrate_mbps=40.0,
                 drop_when_behind=False, queue_frames=6):
        if not AV_AVAILABLE:
            raise RuntimeError('movie_writer needs PyAV (pip install av)')
        if codec not in _CANDIDATES:
            raise ValueError('unknown codec ' + str(codec))
        self.path = path
        self.codec = codec
        self.fps = frame_rate(fps)
        self.bitrate = int(bitrate_mbps * 1e6)
        self.drop_when_behind = drop_when_behind
        self.size = None              # (w, h) of the movie
        self.encoder_name = None
        self.written = 0
        self.dropped = 0
        self.error = None
        self.finished = False
        self._q = queue.Queue(maxsize=queue_frames)
        self._closing = False
        self._thread = threading.Thread(target=self._run, name='movie_writer', daemon=True)
        self._thread.start()

    # ---- producer side ---------------------------------------------------
    def write(self, frame):
        if self._closing or self.error is not None:
            return False
        frame = to_rgb_uint8(frame)
        if self.drop_when_behind:
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                self.dropped += 1
                return False
            return True
        while True:
            try:
                self._q.put(frame, timeout=0.5)
                return True
            except queue.Full:
                if self.error is not None or not self._thread.is_alive():
                    return False

    def close(self):
        """Finish the file in the background; .finished turns True when done."""
        if not self._closing:
            self._closing = True
            self._q.put(None)

    def wait(self, timeout=None):
        self._thread.join(timeout)
        return self.finished

    @property
    def queued(self):
        return self._q.qsize()

    # ---- encoder thread --------------------------------------------------
    def _open(self, w, h):
        last_error = None
        for name, pix, opts, hw in _CANDIDATES[self.codec]:
            container = None
            try:
                container = av.open(self.path, 'w')
                stream = container.add_stream(name, rate=self.fps)
                stream.width, stream.height, stream.pix_fmt = w, h, pix
                stream.options = dict(opts)
                cc = stream.codec_context
                cc.colorspace = 1          # BT.709
                cc.color_primaries = 1
                cc.color_trc = 1
                cc.color_range = 1         # limited (tv)
                if self.codec in ('h264', 'hevc') and hw:
                    stream.bit_rate = self.bitrate
                if name == 'hevc_videotoolbox' or name == 'libx265':
                    cc.codec_tag = 'hvc1'  # QuickTime plays hvc1, not hev1
                # encode a probe frame now so a missing hardware encoder fails here
                stream.codec_context.open()
                self._container, self._stream, self._pix = container, stream, pix
                self._ten_bit = '10' in pix
                self.encoder_name = name
                return
            except Exception as e:
                last_error = e
                try:
                    if container is not None:
                        container.close()
                except Exception:
                    pass
                try:
                    os.remove(self.path)
                except OSError:
                    pass
        raise RuntimeError('could not open a %s encoder: %s' % (self.codec, last_error))

    def _encode(self, rgb):
        h, w = rgb.shape[:2]
        if self.size is None:
            ew, eh = w + (w & 1), h + (h & 1)
            self.size = (ew, eh)
            self._open(ew, eh)
        ew, eh = self.size
        if (w, h) != (ew, eh):
            if w + (w & 1) == ew and h + (h & 1) == eh:
                rgb = np.pad(rgb, ((0, eh - h), (0, ew - w), (0, 0)), mode='edge')
            else:
                import cv2
                rgb = cv2.resize(rgb, (ew, eh), interpolation=cv2.INTER_AREA if w > ew else cv2.INTER_LINEAR)
        if self._ten_bit:
            vf = av.VideoFrame.from_ndarray(rgb.astype(np.uint16) * 257, format='rgb48le')
        else:
            vf = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        vf = vf.reformat(format=self._pix, dst_colorspace=Colorspace.ITU709)
        vf.pts = self.written
        vf.time_base = 1 / self.fps
        for packet in self._stream.encode(vf):
            self._container.mux(packet)
        self.written += 1

    def _run(self):
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                if self.error is None:
                    self._encode(item)
        except Exception as e:
            self.error = str(e)
            # keep draining so a waiting writer is released
            while not self._closing:
                try:
                    if self._q.get(timeout=0.5) is None:
                        break
                except queue.Empty:
                    pass
        finally:
            try:
                if self.size is not None and getattr(self, '_stream', None) is not None:
                    for packet in self._stream.encode():
                        self._container.mux(packet)
                    self._container.close()
            except Exception as e:
                if self.error is None:
                    self.error = str(e)
            self.finished = True
