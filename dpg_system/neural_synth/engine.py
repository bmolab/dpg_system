"""Neural synth engine: the June 2018 NeuralSynth octave loop as a resumable
generator.

    engine = SynthEngine()
    engine.set_target(['mixed5b_5x5_pre_relu'], [17])
    for info in engine.run_image():      # one yield per gradient step
        frame = engine.render_frame()    # (H, W, 3) uint8, whenever wanted

The image starts as uniform noise generated at full size and sampled down to
the smallest octave. Each octave upsamples the image by octave_factor and runs
a falling number of gradient-ascent steps on one feature channel of one
network layer. The gradient is randomly shifted, split into frequency bands,
each band scaled to unit spread and reweighted (GradientShaper), then added
with a step that shrinks every octave.

Dual path (2018 dual_path=True): the shaped gradient drives the image fed
back into the network, while a second shaper builds the displayed image from
the same raw gradient with its own saturation, band weights and step.

Targets: set_target(layers, features) stores a pending target. Lists give one
entry per octave (the last entry holds for the remaining octaves). When the
pending target takes effect is settings.apply: 'next image', 'next octave' or
'next step'. A new image always takes the pending target.

Settings are read live: image size, octave count, iterations and the noise are
fixed when an image starts; shaper settings are re-read at each octave; the
remaining settings (roll, tanh gain, objective, model, device) at each step.

A feature can also be an atlas Direction (neural_synth.atlas): a weighting of
every channel of a concatenated Inception block. Its objective is Lucid's
activation-atlas one, over every position of the picture: the activation's
dot product with the direction, times its cosine similarity to it raised to
direction_cossim (clamped at 0.1), so the picture is pushed to match the
direction rather than just to be loud in it.
"""
import math
import os
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import load_model, DEFAULT_EXIT
from .atlas import Direction
from .shaping import (BandSplitter, GradientShaper, shift_zero, shift_wrap,
                      display_normalize, octave_sizes)

# cuBLAS only repeats its results with a fixed workspace, and reads this when
# it first starts, so it has to be set before any CUDA work in the process
# (no effect, and no cost, without CUDA)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def default_settings():
    """June 2018 (neural_synth_inception5h_June3.py) defaults."""
    return SimpleNamespace(
        model='inception5h',
        device=best_device(),
        width=1920, height=1080,
        octaves=13,
        octave_factor=math.sqrt(2.0),
        iter_base=200,
        iter_factor=1.1,
        seed=78190,                   # image noise seed
        seed_mode='fixed',            # 'fixed' or 'random' per image
        deterministic=False,          # cuda only: repeatable (slower) arithmetic
        noise_amplitude=1.0,
        noise_spectrum=0.0,           # 0 white (2018), 1 pink, 2 brown: power falls as 1/f^spectrum
        noise_colour=1.0,             # 1 independent per colour (2018), 0 the same in all three
        noise_tint=(1.0, 1.0, 1.0),   # per-colour multiplier of the noise
        noise_base=(0.0, 0.0, 0.0),   # per-colour level added (network pixel scale, 0..255)
        start_octave=0,               # begin the image at this octave (smaller ones skipped)
        # init image (engine.set_init): the start is the source shrunk to the
        # start octave's size, mixed toward the noise by init_noise x mask
        init_noise=0.0,               # 0 the source alone, 1 the noise alone
        init_blur=0.0,                # extra gaussian blur, pixels at the start octave's size
        init_level=1.0,               # source multiplier (after centring)
        init_centre=True,             # subtract the source's mean brightness
        objective='sum',              # 'sum' / 'square' of one channel, 'all' = square of whole layer
        direction_cossim=1.0,         # atlas directions: power of the cosine-similarity factor
        lap_levels=4,
        band_padding='zero',          # 2018 alt_pad = zero; 'replicate'
        roll=0.01,                    # max random shift, fraction of image height
        roll_mode='zero',             # 'zero' fill (2018 roll_zero) or 'wrap'
        pre_tanh_gain=0.0,
        lrn=(11, 1e-4, 0.5, 2.0),
        desaturate='row mean',
        # feedback path (drives the image)
        sat_elbow=0.7, octave_scaler=1.2, base_sat=1.0,
        lap_weights_start=(.6, .7, .8, .9, 1.0),
        lap_weights_end=(.8, .85, .9, .95, 1.0),
        lerp_lap_weights=True,
        # output path (dual path only)
        dual_path=False,
        out_sat_elbow=0.7, out_octave_scaler=1.2, out_base_sat=1.0,
        out_lap_weights_start=(.6, .7, .8, .9, 1.0),
        out_lap_weights_end=(.8, .85, .9, .95, 1.0),
        out_lerp_lap_weights=True,
        # display
        norm_scale=4, norm_sigma=0.0, norm_gamma=1.0, norm_min_range=64.0,
        norm_saturation=0.5,          # display only: 1 = as 2018, 0 = grey
        octave_fade=12,               # display only: steps over which an octave change is eased in (0 = off)
        apply='next image',
    )


def best_device():
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


@dataclass
class Target:
    layers: List[str]
    features: list                # ints (channels) or atlas Directions

    def at(self, octave):
        layer = self.layers[min(octave, len(self.layers) - 1)]
        feature = self.features[min(octave, len(self.features) - 1)]
        return layer, feature


@dataclass
class StepInfo:
    image: int
    octave: int
    octaves: int
    step: int                 # step within the octave
    steps: int                # steps in this octave
    total_step: int           # step within the image
    total_steps: int
    layer: str
    feature: object           # channel number, or an atlas Direction
    size: tuple
    octave_end: bool = False
    image_end: bool = False
    seed: int = 0
    used: Optional[list] = None   # at image_end: (layer, feature) actually grown at each octave


class SynthEngine:
    def __init__(self, settings=None):
        self.s = settings if settings is not None else default_settings()
        self.model = None
        self.model_key = None
        self.device = None
        self.splitter = None
        self.target = Target([DEFAULT_EXIT[self.s.model]], [0])
        self.pending: Optional[Target] = None
        self.image_index = 0
        self.x = None              # feedback image (NCHW, float32)
        self.out = None            # output image (dual path)
        self.rng = np.random.RandomState(123457890)
        self.info: Optional[StepInfo] = None
        self._fade = None          # display easing across the current octave change
        self.lock = threading.Lock()   # guards x/out against render during a step
        self.init_image = None     # (1, 3, H, W) float32 CPU, network pixel scale, or None
        self.init_mask = None      # (1, 1, H, W) float32 CPU 0..1, or None
        self.seed_used = None

    # ---- start image ---------------------------------------------------
    @staticmethod
    def _as_nchw(data, channels):
        """An image as (1, C, H, W) float32 CPU: numpy or torch, (H, W),
        (H, W, C) or (C, H, W) / (1, C, H, W) torch; uint8 stays 0..255, floats
        are taken as they are."""
        if data is None:
            return None
        t = data.detach().float().cpu() if isinstance(data, torch.Tensor) else \
            torch.from_numpy(np.asarray(data, dtype=np.float32).copy())
        if t.dim() == 2:
            t = t[None, None]
        elif t.dim() == 3:
            if isinstance(data, torch.Tensor) and t.shape[0] in (1, 3, 4) and t.shape[2] not in (1, 3, 4):
                t = t[None]
            else:
                t = t.permute(2, 0, 1)[None]
        elif t.dim() != 4:
            raise ValueError('not an image: shape %s' % (tuple(t.shape),))
        c = t.shape[1]
        if c == 4:
            t = t[:, :3]
        if channels == 3 and t.shape[1] == 1:
            t = t.expand(1, 3, -1, -1)
        if channels == 1 and t.shape[1] != 1:
            t = t.mean(1, keepdim=True)
        return t.contiguous()

    def set_init(self, image):
        """Source image for the start of each picture (None to go back to
        noise), on the 0..255 scale; an image whose values all lie within 0..1
        is taken as 0..1 and scaled up. Takes effect with the next picture."""
        img = self._as_nchw(image, 3)
        if img is not None and float(img.max()) <= 1.0 and float(img.min()) >= 0.0:
            img = img * 255.0
        self.init_image = img

    def set_mask(self, mask):
        """Where the noise goes (1) and where the source stays (0): 0..1, or
        0..255 if any value is above 1. None: everywhere."""
        m = self._as_nchw(mask, 1)
        if m is not None and float(m.max()) > 1.0:
            m = m / 255.0
        self.init_mask = None if m is None else m.clamp(0.0, 1.0)

    # ---- targets -------------------------------------------------------
    def set_target(self, layers, features):
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(features, (int, np.integer, Direction)):
            features = [features]
        features = [f if isinstance(f, Direction) else int(f) for f in features]
        layers = list(layers)
        for i, f in enumerate(features):
            if isinstance(f, Direction):
                # a direction belongs to its own layer
                while len(layers) <= i:
                    layers.append(layers[-1])
                layers[i] = f.layer
        self.pending = Target(layers, features)

    def _take_pending(self):
        p = self.pending
        if p is not None:
            self.pending = None
            self.target = p
            return True
        return False

    def exits(self):
        self._ensure_model()
        return list(self.model.exits)

    def feature_count(self, layer):
        self._ensure_model()
        return self.model.feature_count(layer)

    def _resolve(self, octave):
        layer, feature = self.target.at(octave)
        if isinstance(feature, Direction):
            if (feature.layer in self.model._counts
                    and self.model.feature_count(feature.layer) == feature.vector.shape[0]):
                return feature.layer, feature
            feature, layer = 0, DEFAULT_EXIT[self.model.name]   # not this model's
        if layer not in self.model._counts:
            layer = DEFAULT_EXIT[self.model.name]
        n = self.model.feature_count(layer)
        return layer, int(feature) % n

    # ---- model / device ------------------------------------------------
    def _ensure_model(self):
        key = (self.s.model, self.s.device)
        if key != self.model_key:
            device = torch.device(self.s.device)
            self.model = load_model(self.s.model, device)
            if self.device is None or device != self.device:
                self.splitter = BandSplitter(device)
                if self.x is not None:
                    self.x = self.x.to(device)
                if self.out is not None:
                    self.out = self.out.to(device)
            self.device = device
            self.model_key = key
        self.model.set_lrn(*self.s.lrn)
        self._apply_determinism()

    def _apply_determinism(self):
        """On CUDA, 'deterministic' trades speed for repeatable arithmetic:
        cuDNN's kernel search off, and PyTorch's deterministic algorithms on
        (warn_only: an operation with no repeatable version prints a warning
        naming itself rather than stopping the synth). These switches are
        process-wide, so they are only touched when this engine's wish
        changes. MPS and CPU repeat without them."""
        if self.device is None or self.device.type != 'cuda':
            return
        want = bool(self.s.deterministic)
        if getattr(self, '_det_applied', None) == want:
            return
        self._det_applied = want
        torch.backends.cudnn.deterministic = want
        # sizes are fixed within an octave: otherwise let cuDNN pick its
        # fastest kernels once per size
        torch.backends.cudnn.benchmark = not want
        torch.use_deterministic_algorithms(want, warn_only=True)

    # ---- one image -----------------------------------------------------
    def _base_noise(self, sizes, start=0):
        """The picture's first state, at the size of octave `start`.

        2018: uniform noise 0..1 per pixel and colour, made at full size and
        sampled down. The spectrum, colour, tint and base settings reshape it
        (at their defaults it is exactly the 2018 noise); an init image
        replaces it, with the noise mixed in by init_noise x mask."""
        s = self.s
        seed = s.seed if s.seed_mode == 'fixed' else int(np.random.randint(0, 2 ** 31 - 1))
        rs = np.random.RandomState(seed)
        # the per-step jitter restarts from the image's seed too, so a fixed
        # seed repeats the whole picture exactly, not just its starting noise
        self.rng = np.random.RandomState([seed, 1])
        self.seed_used = seed
        noise = rs.uniform(size=(1, s.height, s.width, 3))
        t = torch.from_numpy(np.ascontiguousarray(noise.transpose(0, 3, 1, 2)).astype(np.float32))
        t = self._shape_noise(t)
        t = t * s.noise_amplitude
        tint, base = tuple(s.noise_tint), tuple(s.noise_base)
        if tint != (1.0, 1.0, 1.0):
            t = t * torch.tensor(tint, dtype=torch.float32).view(1, 3, 1, 1)
        if base != (0.0, 0.0, 0.0):
            t = t + torch.tensor(base, dtype=torch.float32).view(1, 3, 1, 1)
        t = t.to(self.device)
        size = sizes[start]
        x = F.interpolate(t, size=size, mode='bilinear', align_corners=True)
        src = self.init_image
        if src is not None:
            x = self._mix_init(src, x, size)
        return x

    def _shape_noise(self, t):
        """Spectrum and colour of the uniform noise, keeping its mean (0.5)
        and spread (that of uniform 0..1), so amplitude means the same."""
        s = self.s
        spectrum, colour = float(s.noise_spectrum), float(s.noise_colour)
        if spectrum == 0.0 and colour == 1.0:
            return t                                  # the 2018 noise, untouched
        t = t - 0.5
        if colour != 1.0:
            grey = t.mean(1, keepdim=True)
            # the mean of three has a third of the variance: bring it back,
            # so colour 0 is grey noise as strong as the coloured one
            t = grey * math.sqrt(3.0) + (t - grey) * colour
        if spectrum != 0.0:
            h, w = t.shape[2], t.shape[3]
            fy = torch.fft.fftfreq(h).view(-1, 1)
            fx = torch.fft.rfftfreq(w).view(1, -1)
            f = torch.sqrt(fy * fy + fx * fx)
            f[0, 0] = 1.0
            gain = f.pow(-spectrum / 2.0)            # power ~ 1/f^spectrum
            gain[0, 0] = 0.0
            t = torch.fft.irfft2(torch.fft.rfft2(t) * gain, s=(h, w))
        sd = t.std().clamp_min(1e-12)
        return t * (1.0 / math.sqrt(12.0) / sd) + 0.5

    def _mix_init(self, src, noise, size):
        s = self.s
        img = F.interpolate(src, size=size, mode='bilinear', align_corners=False, antialias=True)
        sigma = float(s.init_blur)
        if sigma > 0:
            r = max(1, int(math.ceil(3 * sigma)))
            k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
            k = (k / k.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
            img = F.conv2d(F.pad(img, (r, r, 0, 0), mode='replicate'), k, groups=3)
            img = F.conv2d(F.pad(img, (0, 0, r, r), mode='replicate'), k.transpose(2, 3), groups=3)
        if s.init_centre:
            img = img - img.mean()
        img = (img * float(s.init_level)).to(noise.device)
        amount = float(s.init_noise)
        if amount == 0.0:
            return img
        m = self.init_mask
        if m is not None:
            m = F.interpolate(m, size=size, mode='bilinear', align_corners=False, antialias=True)
            amount = (m * amount).to(noise.device)
        return img + (noise - img) * amount

    def _shapers(self, octave):
        s = self.s
        fb = GradientShaper(s.octaves, s.sat_elbow, s.octave_scaler, s.lap_weights_start,
                            s.lap_weights_end, s.lerp_lap_weights, s.base_sat, s.desaturate)
        fb.set_current_octave(octave)
        if not s.dual_path:
            return fb, None
        out = GradientShaper(s.octaves, s.out_sat_elbow, s.out_octave_scaler, s.out_lap_weights_start,
                             s.out_lap_weights_end, s.out_lerp_lap_weights, s.out_base_sat, s.desaturate)
        out.set_current_octave(octave)
        return fb, out

    def _gradient(self, x, layer, feature):
        s = self.s
        h = x.shape[2]
        max_roll = int(s.roll * h)
        sx = sy = 0
        if max_roll > 0:
            sx, sy = (int(v) for v in self.rng.randint(-max_roll, max_roll, size=2))
        shift = shift_zero if s.roll_mode == 'zero' else shift_wrap
        local = shift(x, sy, sx).detach().requires_grad_(True)

        # float32 only: the network takes raw 0..255-scale pixels, activations
        # reach the thousands and LRN squares them -- float16 overflows and
        # bfloat16 goes NaN at 1920 x 1080
        if isinstance(feature, Direction):
            loss = self._direction_loss(self.model(local, layer, None), feature)
        else:
            ch = None if s.objective == 'all' else feature
            f = self.model(local, layer, ch)
            loss = f.sum() if s.objective == 'sum' else 0.5 * (f * f).sum()
        g, = torch.autograd.grad(loss, local)
        g = shift(g, -sy, -sx)
        if s.pre_tanh_gain != 0:
            g = torch.tanh(g * (s.pre_tanh_gain / g.std().clamp_min(1e-10)))
        return g

    def _direction_loss(self, f, direction):
        key = id(direction), f.device
        cached = getattr(self, '_dir_cache', None)
        if cached is None or cached[0] != key:
            d = torch.from_numpy(direction.vector).to(f.device).view(1, -1, 1, 1)
            self._dir_cache = cached = (key, d, direction)   # holds the Direction so id() stays unique
        d = cached[1]
        dot = (f * d).sum(1)
        p = float(self.s.direction_cossim)
        if p == 0:
            return dot.sum()
        cos = dot / (f.norm(dim=1) + 1e-4)
        return (dot * cos.clamp_min(0.1) ** p).sum()

    def run_image(self):
        """Generator: one StepInfo per gradient step of a whole image."""
        s = self.s
        self._take_pending()
        self._ensure_model()
        self.image_index += 1
        octaves = int(s.octaves)
        sizes = octave_sizes(int(s.height), int(s.width), octaves, s.octave_factor)
        iters = [int(s.iter_base)]
        for _ in range(1, octaves):
            iters.append(int(iters[-1] / s.iter_factor))
        start = min(max(0, int(s.start_octave)), octaves - 1)
        total = sum(iters[start:])
        used = [None] * octaves

        self._fade = None
        with torch.no_grad():
            base = self._base_noise(sizes, start)
        with self.lock:
            self.x = base
            self.out = base.clone() if s.dual_path else None
        done = 0
        for octave in range(start, octaves):
            if s.apply in ('next octave', 'next step'):
                self._take_pending()
            if octave > start:
                with self.lock, torch.no_grad():
                    before = self._display_source()
                    self.x = F.interpolate(self.x, size=sizes[octave], mode='bilinear', align_corners=True)
                    if self.out is not None:
                        self.out = F.interpolate(self.out, size=sizes[octave], mode='bilinear', align_corners=True)
                    # the step tensors are replaced, never modified in place,
                    # so holding references is enough
                    self._fade = {'octave': octave, 'before': before,
                                  'after': self._display_source(), 'offset': None}
            fb, outs = self._shapers(octave)
            for i in range(iters[octave]):
                if s.apply == 'next step':
                    self._take_pending()
                self._ensure_model()
                layer, feature = self._resolve(octave)
                used[octave] = (layer, feature)
                g = self._gradient(self.x, layer, feature)
                with torch.no_grad():
                    bands = self.splitter.split(g, int(s.lap_levels), s.band_padding)
                    gf = fb.shape(bands)
                    go = outs.shape(bands) if (outs is not None and self.out is not None) else None
                    with self.lock:
                        self.x = self.x + gf * fb.current_step
                        if go is not None:
                            self.out = self.out + go * outs.current_step
                done += 1
                self.info = StepInfo(self.image_index, octave, octaves, i, iters[octave], done, total,
                                     layer, feature, tuple(sizes[octave]),
                                     octave_end=(i == iters[octave] - 1),
                                     image_end=(octave == octaves - 1 and i == iters[octave] - 1),
                                     seed=self.seed_used)
                if self.info.image_end:
                    # octaves skipped by start_octave report the target's own
                    # entries, so the list goes back as the same schedule
                    self.info.used = [u if u is not None else self._resolve(o) for o, u in enumerate(used)]
                yield self.info

    def advance_feature(self, delta=1):
        """For 'next feature' at the end of an image: shift every scheduled
        feature by delta (wrapping in each layer's range)."""
        self._ensure_model()
        t = self.target
        feats = []
        for o in range(len(t.features)):
            layer = t.layers[min(o, len(t.layers) - 1)]
            n = self.model.feature_count(layer) if layer in self.model._counts else 1
            f = t.features[o]
            feats.append(f if isinstance(f, Direction) else (f + delta) % n)
        self.target = Target(list(t.layers), feats)

    # ---- display -------------------------------------------------------
    def _display_source(self):
        return self.out if (self.s.dual_path and self.out is not None) else self.x

    def _render_tensor(self, img, width, height):
        """Display picture as NCHW 0..1 at the output size.

        Scaled to the output size FIRST, then normalised. Normalising at the
        working size (as the 2018 per-octave display did) computes the
        contrast blocks on a different pixel grid every octave, and in a
        per-step stream that showed as a jump at every octave change.
        align_corners=True matches the synth's octave enlargement, so content
        does not shift in the frame when the octave changes."""
        s = self.s
        if img.shape[2] != height or img.shape[3] != width:
            img = F.interpolate(img, size=(height, width), mode='bilinear', align_corners=True)
        return display_normalize(img, int(s.norm_scale), float(s.norm_sigma), float(s.norm_gamma),
                                 float(s.norm_min_range), float(s.norm_saturation))

    def render_frame(self, width=None, height=None):
        """Current image, local-contrast normalised and scaled to the output
        size: (H, W, 3) uint8 numpy.

        Octave fade (display only): enlarging the image at an octave change
        resamples it, which softens the finest detail at once. The difference
        between the picture just before and just after the enlargement is
        added back to the frames and faded out over octave_fade steps, so the
        first frame after the change matches the last one before it and the
        change is spread out while the growth carries on. The synthesis, and
        every frame once the fade is over, are untouched."""
        s = self.s
        with self.lock:
            img = self._display_source()
            fade = self._fade
            info = self.info
        if img is None:
            return None
        width = int(width or s.width)
        height = int(height or s.height)
        with torch.no_grad():
            n = self._render_tensor(img, width, height)
            steps = int(s.octave_fade)
            if fade is not None and info is not None and info.octave == fade['octave'] and steps > 0:
                weight = 1.0 - (info.step + 1) / steps
                if weight > 0:
                    off = fade['offset']
                    if off is None or off.shape[2] != height or off.shape[3] != width:
                        off = (self._render_tensor(fade['before'], width, height)
                               - self._render_tensor(fade['after'], width, height))
                        fade['offset'] = off
                    n = n + off * weight
                else:
                    fade['before'] = fade['after'] = fade['offset'] = None
            u8 = (n[0] * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        return u8.cpu().numpy()
