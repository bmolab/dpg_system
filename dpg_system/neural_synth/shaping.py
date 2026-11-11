"""Gradient shaping and display normalization for the neural synth, ported
from the June 2018 NeuralSynth code (Laplacian_Normalization.py,
GradientShaper.py, adaptive_normalization.py, roll_tensor.py).

Everything runs on the tensor's device with no host synchronisation, except
display_normalize's optional grid smoothing.

Band split: the gradient is split into frequency bands by repeated
full-resolution Gaussian blurs of 5, 9, 17 and 33 taps (no downsampling).
The kernels are the 2018 ones exactly: a ±2-sigma span cut into n taps, not
renormalised, so each blur keeps 93-99% of the DC. Each band is scaled to unit
standard deviation and the bands are recombined with per-band weights.

The 2018 code expressed the blur as a 3x3 channel-identity ("eye") kernel;
here it is a depthwise convolution (groups=3), which is the same arithmetic
with a third of the multiplies.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

BAND_TAPS = (5, 9, 17, 33)


def _gaussian_taps(n, span=2.0):
    """2018 gaussian_kernel(n, sigma=2.0): differences of the normal CDF over
    n equal intervals covering ±(span + interval/2)."""
    interval = (2 * span + 1.0) / n
    x = np.linspace(-span - interval / 2.0, span + interval / 2.0, n + 1)
    cdf = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x])
    return np.diff(cdf).astype(np.float32)


class BandSplitter:
    """Holds the blur kernels on one device."""

    def __init__(self, device, dtype=torch.float32):
        self.device = device
        self.kernels = []
        for n in BAND_TAPS:
            k = torch.from_numpy(_gaussian_taps(n)).to(device=device, dtype=dtype)
            kh = k.view(1, 1, 1, n).repeat(3, 1, 1, 1).contiguous()
            kv = k.view(1, 1, n, 1).repeat(3, 1, 1, 1).contiguous()
            self.kernels.append((n, kh, kv))

    def _blur(self, x, level, pad_mode):
        n, kh, kv = self.kernels[level]
        r = n // 2
        if pad_mode == 'zero':
            x = F.conv2d(x, kh, padding=(0, r), groups=3)
            return F.conv2d(x, kv, padding=(r, 0), groups=3)
        x = F.pad(x, (r, r, r, r), mode='replicate')
        x = F.conv2d(x, kh, groups=3)
        return F.conv2d(x, kv, groups=3)

    def split(self, img, levels=4, pad_mode='zero'):
        """[hi, hi2, ..., lo] with the 2018 level-count rule: a level is
        added while half the smaller image side stays >= 1. Returns [img]
        when no split is possible (the 2018 code returned img unshaped)."""
        min_shape = min(img.shape[2], img.shape[3])
        if min_shape < 1 or levels < 1:
            return [img]
        bands = []
        cur = img
        for level in range(min(levels, len(BAND_TAPS))):
            lo = self._blur(cur, level, pad_mode)
            bands.append(cur - lo)
            cur = lo
            min_shape //= 2
            if min_shape < 1:
                break
        bands.append(cur)
        return bands


def normalize_std(x, eps=1e-10):
    return x / torch.std(x).clamp_min(eps)


class GradientShaper:
    """One gradient path's shaping schedule (2018 GradientShaper).

    Per octave: saturation falls from 1 toward (1 - sat_elbow) as the image
    grows; the step shrinks by octave_scaler per octave; band weights lerp
    from lap_weights_start to lap_weights_end over the octaves (if
    lerp_lap_weights), then are scaled to average 1.

    Weight index 4 goes to the finest band, index 0 to the coarsest
    remainder, as in 2018 (bands[i] gets weights[4 - i]).

    desaturate: 'row mean' is what the June 2018 code actually did -- it
    averaged the gradient over dim 3 of an NCHW tensor, i.e. along each image
    row, not across colour. 'luma' averages the colour channels instead.
    """

    def __init__(self, octaves=13, sat_elbow=0.7, octave_scaler=1.2,
                 lap_weights_start=(.6, .7, .8, .9, 1.0),
                 lap_weights_end=(.8, .85, .9, .95, 1.0),
                 lerp_lap_weights=True, base_sat=1.0, desaturate='row mean'):
        self.octaves = octaves
        self.sat_elbow = sat_elbow
        self.octave_scaler = octave_scaler
        self.lap_weights_start = tuple(lap_weights_start)
        self.lap_weights_end = tuple(lap_weights_end) if lap_weights_end is not None else None
        self.lerp_lap_weights = lerp_lap_weights
        self.base_sat = base_sat
        self.desaturate = desaturate
        self.current_sat = 1.0
        self.current_weights = (1.0,) * 5
        self.current_step = 1.0

    def set_current_octave(self, octave):
        self.current_sat = self.octave_saturation(octave) * self.base_sat
        self.current_weights = self.octave_lap_weights(octave)
        self.current_step = 1.0 / math.pow(self.octave_scaler, octave)

    def octave_saturation(self, octave):
        sat = (self.octaves - octave) / self.octaves
        sat = min(sat, self.sat_elbow)
        return sat + (1.0 - self.sat_elbow)

    def octave_lap_weights(self, octave):
        start = np.array(self.lap_weights_start, dtype=np.float64)
        if self.lerp_lap_weights:
            end = start[::-1] if self.lap_weights_end is None else np.array(self.lap_weights_end, dtype=np.float64)
            pos = octave / max(self.octaves - 1, 1)
            w = pos * end + (1 - pos) * start
        else:
            w = start.copy()
        mean = w.sum() / 5
        if mean != 0:
            w = w / mean
        return tuple(float(v) for v in w)

    def shape(self, bands):
        g = None
        for i, b in enumerate(bands):
            term = normalize_std(b) * self.current_weights[4 - i]
            g = term if g is None else g + term
        if self.desaturate == 'luma':
            grey = g.mean(dim=1, keepdim=True)
        else:
            grey = g.mean(dim=3, keepdim=True)
        return torch.lerp(grey.expand_as(g), g, self.current_sat)


def shift_zero(x, dy, dx):
    """2018 roll_tensor_zero_outside(x, dy, dx): content moves by (-dy, -dx),
    zeros fill the uncovered edge. dy, dx are python ints."""
    if dy == 0 and dx == 0:
        return x
    h, w = x.shape[2], x.shape[3]
    # pad then crop: positive d moves content toward index 0
    x = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)))
    y0 = max(dy, 0)
    x0 = max(dx, 0)
    return x[:, :, y0:y0 + h, x0:x0 + w]


def shift_wrap(x, dy, dx):
    """2018 roll_tensor: circular version of shift_zero."""
    if dy == 0 and dx == 0:
        return x
    return torch.roll(x, shifts=(-dy, -dx), dims=(2, 3))


def _hist_quantile(v, q, bins):
    """np.histogram-bin-edge quantile as 2018 find_quantile_2 computed it:
    the left edge of the first bin whose running count exceeds q * N."""
    flat = v.flatten()
    n = flat.numel()
    vmin, vmax = flat.min(), flat.max()
    k = min(int(math.floor(q * n)) + 1, n)
    val = torch.kthvalue(flat, k).values
    span = (vmax - vmin).clamp_min(1e-30)
    idx = torch.floor((val - vmin) / span * bins).clamp(0, bins - 1)
    return vmin + idx * (vmax - vmin) / bins


def display_normalize(img, scale=4, smoothing=0.0, gamma=1.0, min_range=64.0, saturation=1.0):
    """2018 adaptive_normalize_B_W_H_C on an NCHW tensor; returns 0..1.

    Local contrast stretch: per-pixel colour min/max are reduced to a coarse
    scale x scale grid (min blended 3:1 with the grey min), the grids are
    bilinearly upsampled, and each pixel is stretched between them. The range
    is never stretched beyond 1/min_range gain (2018: 64, in the image's raw
    units, so the first steps of a fresh image look dim).

    saturation (not in 2018) blends the result toward its Rec. 709 luma
    (1 = unchanged, 0 = grey). The stretch pushes each block's weakest
    colour toward black, which by itself makes colours purer."""
    _, _, h, w = img.shape
    sy, sx = h // scale, w // scale
    for _ in range(3):
        if sy < 4 and scale > 1:
            scale //= 2
            sy, sx = h // scale, w // scale
    scale = max(scale, 1)
    sy, sx = max(sy, 1), max(sx, 1)

    grey = img.mean(dim=1, keepdim=True)
    mn = img.amin(dim=1, keepdim=True)
    mx = img.amax(dim=1, keepdim=True)

    gh, gw = -(-h // sy), -(-w // sx)
    ph, pw = gh * sy - h, gw * sx - w

    def block(t, fn, cval):
        # reshape-and-reduce: a (sy, sx) max pool is very slow on MPS when
        # the window is a quarter of the image
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), value=cval)
        t = t.view(t.shape[0], 1, gh, sy, gw, sx)
        return t.amin(dim=(3, 5)) if fn == 'min' else t.amax(dim=(3, 5))

    rmin = block(mn, 'min', 0.0)
    rmax = block(mx, 'max', 255.0)
    gmin = block(grey, 'min', 0.0)
    rmin = (rmin * 3 + gmin) / 4
    # the partial edge blocks are replaced by their inner neighbours
    if gh > scale:
        rmin[..., scale:, :scale] = rmin[..., scale - 1:scale, :scale]
        rmax[..., scale:, :scale] = rmax[..., scale - 1:scale, :scale]
    if gw > scale:
        rmin[..., :, scale:] = rmin[..., :, scale - 1:scale]
        rmax[..., :, scale:] = rmax[..., :, scale - 1:scale]

    bins = sx * sy
    gpmin = _hist_quantile(rmin, 0.01, bins)
    gpmax = _hist_quantile(rmax, 0.99, bins)

    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        rmin = torch.from_numpy(gaussian_filter(rmin[0, 0].float().cpu().numpy(), smoothing))[None, None].to(img)
        rmax = torch.from_numpy(gaussian_filter(rmax[0, 0].float().cpu().numpy(), smoothing))[None, None].to(img)

    lo = F.interpolate(rmin, size=(h, w), mode='bilinear', align_corners=False)
    hi = F.interpolate(rmax, size=(h, w), mode='bilinear', align_corners=False)
    lo = lo + (gpmin - lo.amin())
    hi = hi + (gpmax - hi.amax()) * 0.9
    out = ((img - lo) / (hi - lo).clamp(min_range, 1e6)).clamp(0.0, 1.0)
    if gamma != 1.0:
        out = out.pow(gamma)
    if saturation != 1.0:
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=out.device, dtype=out.dtype).view(1, 3, 1, 1)
        luma = (out * w).sum(1, keepdim=True)
        out = (luma + (out - luma) * saturation).clamp(0.0, 1.0)
    return out


def octave_sizes(height, width, octaves, factor=math.sqrt(2.0)):
    """2018 get_image_sizes: smallest first, largest = (height, width)."""
    sizes = []
    for o in range(octaves):
        d = factor ** o
        sizes.append((int(np.round(np.float32(height) / d)), int(np.round(np.float32(width) / d))))
    return sizes[::-1]
