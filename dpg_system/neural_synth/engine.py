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
"""
import math
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import load_model, DEFAULT_EXIT
from .shaping import (BandSplitter, GradientShaper, shift_zero, shift_wrap,
                      display_normalize, octave_sizes)


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
        noise_amplitude=1.0,
        objective='sum',              # 'sum' / 'square' of one channel, 'all' = square of whole layer
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
    features: List[int]

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
    feature: int
    size: tuple
    octave_end: bool = False
    image_end: bool = False


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

    # ---- targets -------------------------------------------------------
    def set_target(self, layers, features):
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(features, (int, np.integer)):
            features = [int(features)]
        self.pending = Target(list(layers), [int(f) for f in features])

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
        if layer not in self.model._counts:
            layer = DEFAULT_EXIT[self.model.name]
        n = self.model.feature_count(layer)
        return layer, int(feature) % n

    # ---- model / device ------------------------------------------------
    def _ensure_model(self):
        key = (self.s.model, self.s.device)
        if key != self.model_key:
            device = torch.device(self.s.device)
            if device.type == 'cuda':
                # sizes are fixed within an octave: let cuDNN pick kernels once per size
                torch.backends.cudnn.benchmark = True
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

    # ---- one image -----------------------------------------------------
    def _base_noise(self, sizes):
        s = self.s
        seed = s.seed if s.seed_mode == 'fixed' else int(np.random.randint(0, 2 ** 31 - 1))
        rs = np.random.RandomState(seed)
        noise = rs.uniform(size=(1, s.height, s.width, 3)) * s.noise_amplitude
        t = torch.from_numpy(np.ascontiguousarray(noise.transpose(0, 3, 1, 2)).astype(np.float32))
        t = t.to(self.device)
        return F.interpolate(t, size=sizes[0], mode='bilinear', align_corners=True)

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
        ch = None if s.objective == 'all' else feature
        f = self.model(local, layer, ch)
        loss = f.sum() if s.objective == 'sum' else 0.5 * (f * f).sum()
        g, = torch.autograd.grad(loss, local)
        g = shift(g, -sy, -sx)
        if s.pre_tanh_gain != 0:
            g = torch.tanh(g * (s.pre_tanh_gain / g.std().clamp_min(1e-10)))
        return g

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
        total = sum(iters)

        self._fade = None
        with torch.no_grad():
            base = self._base_noise(sizes)
        with self.lock:
            self.x = base
            self.out = base.clone() if s.dual_path else None
        done = 0
        for octave in range(octaves):
            if s.apply in ('next octave', 'next step'):
                self._take_pending()
            if octave > 0:
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
                                     image_end=(octave == octaves - 1 and i == iters[octave] - 1))
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
            feats.append((t.features[o] + delta) % n)
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
                                 float(s.norm_min_range))

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
