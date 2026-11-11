"""Feature-visualization networks for the neural synth.

Each model exposes an ordered list of named exits and

    forward(x, exit_name, feature=None)

which runs the network only as far as that exit. When `feature` is given the
result is just that one channel, (B, 1, H, W), and the work is trimmed to
match: the exit convolution is computed for that single output channel, and at
an Inception concat exit only the branch that holds the channel runs. Every
objective the synth uses (sum or square of one channel) has a gradient that
depends only on that channel, so the trimming is exact, not an approximation.

Inception5h is the 2018 NeuralSynth port of Google's TensorFlow GoogLeNet
(the "deepdream" network) in its June 2018 form: TF-style SAME padding on
conv2d0 and the four stride-2 max pools, symmetric padding elsewhere, and
adjustable local response normalization (the 2018 synth ran it at size 11).
An exit whose name ends in _pre_relu is the convolution before its ReLU (the
2018 layer spec ['mixed5b', '5x5', 'conv'] is mixed5b_5x5_pre_relu).

VGG19 uses torchvision's ImageNet weights. The 2018 VGG synth's exit index 32
is conv5_3_pre_relu.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
INCEPTION5H_WEIGHTS = os.path.join(HERE, 'inception5h.pt')
INCEPTION5H_HEAD = os.path.join(HERE, 'inception5h_head.pt')

# (name, in_channels, (1x1, 3x3 bottleneck, 3x3, 5x5 bottleneck, 5x5, pool_reduce))
_MIXED = [
    ('mixed3a', 192, (64, 96, 128, 16, 32, 32)),
    ('mixed3b', 256, (128, 128, 192, 32, 96, 64)),
    ('mixed4a', 480, (192, 96, 204, 16, 48, 64)),
    ('mixed4b', 508, (160, 112, 224, 24, 64, 64)),
    ('mixed4c', 512, (128, 128, 256, 24, 64, 64)),
    ('mixed4d', 512, (112, 144, 288, 32, 64, 64)),
    ('mixed4e', 528, (256, 160, 320, 32, 128, 128)),
    ('mixed5a', 832, (256, 160, 320, 48, 128, 128)),
    ('mixed5b', 832, (384, 192, 384, 48, 128, 128)),
]
# a stride-2 SAME max pool follows these blocks
_POOL_AFTER = {'mixed3b': 'maxpool4', 'mixed4e': 'maxpool10'}


def inception5h_exits():
    """(ordered exit names, {exit: channel count}) -- no weights needed."""
    ex, counts = [], {}

    def conv(name, n):
        ex.extend([name + '_pre_relu', name])
        counts[name + '_pre_relu'] = counts[name] = n

    def plain(name, n):
        ex.append(name)
        counts[name] = n

    conv('conv2d0', 64)
    plain('maxpool0', 64)
    plain('localresponsenorm0', 64)
    conv('conv2d1', 64)
    conv('conv2d2', 192)
    plain('localresponsenorm1', 192)
    plain('maxpool1', 192)
    for m, cin, f in _MIXED:
        conv(m + '_1x1', f[0])
        conv(m + '_3x3_bottleneck', f[1])
        conv(m + '_3x3', f[2])
        conv(m + '_5x5_bottleneck', f[3])
        conv(m + '_5x5', f[4])
        plain(m + '_pool', cin)
        conv(m + '_pool_reduce', f[5])
        plain(m, f[0] + f[2] + f[4] + f[5])
        if m in _POOL_AFTER:
            plain(_POOL_AFTER[m], counts[m])
    plain('avgpool0', counts['mixed5b'])
    return ex, counts


def vgg19_exits():
    """(ordered exit names, {exit: channel count},
    {exit: (torchvision features index, 'conv' | 'relu' | 'pool')})."""
    ex, counts, plan = [], {}, {}
    block, conv_i, idx, ch = 1, 1, 0, 3
    for v in _VGG19_CFG:
        if v == 'M':
            name = 'pool%d' % block
            ex.append(name); counts[name] = ch; plan[name] = (idx, 'pool')
            idx += 1
            block += 1
            conv_i = 1
        else:
            base = '%d_%d' % (block, conv_i)
            for name, kind in (('conv' + base + '_pre_relu', 'conv'), ('relu' + base, 'relu')):
                ex.append(name); counts[name] = v; plan[name] = (idx, kind)
            idx += 2
            conv_i += 1
            ch = v
    return ex, counts, plan


def model_exits(name):
    """Ordered exit names and channel counts for a model, without loading it."""
    if name == 'inception5h':
        return inception5h_exits()
    if name == 'vgg19':
        return vgg19_exits()[:2]
    raise KeyError(name)


def local_response_norm(x, size, alpha, beta, k):
    """Same result as torch.nn.functional.local_response_norm, with the
    channel window summed by a box convolution instead of avg_pool3d.

    torch's version is wrong on MPS (torch 2.9) for large inputs: for a
    64 x 270 x 480 tensor -- the first LRN when the image is 1920 x 1080 --
    it is off by ~10 on outputs of ~70, while 192 x 135 x 240 is correct.
    This version matches float64 to 2e-7 relative (as CPU torch does) on
    both devices and is ~3x faster on MPS. (A running sum over channels is
    faster still but loses ~1e-5 relative to cancellation.)"""
    b, c, h, w = x.shape
    sq = (x * x).reshape(b, 1, c, h * w)
    sq = F.pad(sq, (0, 0, size // 2, (size - 1) // 2))
    box = torch.ones(1, 1, size, 1, device=x.device, dtype=x.dtype)
    window = F.conv2d(sq, box).reshape(b, c, h, w)
    return x / (k + alpha * (window / size)).pow(beta)


def _same_pad(x, k, s):
    """TF 'SAME' padding, as the 2018 PaddedConv2d / PaddedMaxPool did it
    (zero fill; the pools follow ReLUs, so zero acts like -inf)."""
    h, w = x.shape[2], x.shape[3]
    ph = max(k - (s if h % s == 0 else h % s), 0)
    pw = max(k - (s if w % s == 0 else w % s), 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


class Inception5h(nn.Module):
    name = 'inception5h'

    def __init__(self, weights_path=INCEPTION5H_WEIGHTS):
        super().__init__()
        self.convs = nn.ModuleDict()
        self._add('conv2d0', 3, 64, 7)
        self._add('conv2d1', 64, 64, 1)
        self._add('conv2d2', 64, 192, 3)
        for m, cin, f in _MIXED:
            self._add(m + '_1x1', cin, f[0], 1)
            self._add(m + '_3x3_bottleneck', cin, f[1], 1)
            self._add(m + '_3x3', f[1], f[2], 3)
            self._add(m + '_5x5_bottleneck', cin, f[3], 1)
            self._add(m + '_5x5', f[3], f[4], 5)
            self._add(m + '_pool_reduce', cin, f[5], 1)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                'inception5h weights not found at ' + weights_path +
                ' -- run python -m dpg_system.neural_synth.convert_inception5h')
        state = torch.load(weights_path, map_location='cpu')
        self.convs.load_state_dict(state)
        for p in self.parameters():
            p.requires_grad_(False)
        self.lrn = (5, 1e-4, 0.5, 2.0)
        self._build_exits()

    def _add(self, name, cin, cout, k):
        self.convs[name] = nn.Conv2d(cin, cout, k)

    def set_lrn(self, size=5, alpha=1e-4, beta=0.5, k=2.0):
        self.lrn = (int(size), float(alpha), float(beta), float(k))

    # ---- exits ---------------------------------------------------------
    def _build_exits(self):
        self.exits, self._counts = inception5h_exits()

    def feature_count(self, exit_name):
        return self._counts[exit_name]

    # ---- computation ---------------------------------------------------
    def _conv(self, x, name, relu=True, ch=None, same_stride=None):
        c = self.convs[name]
        w, b = c.weight, c.bias
        if ch is not None:
            w, b = w[ch:ch + 1], b[ch:ch + 1]
        k = c.kernel_size[0]
        if same_stride is not None:
            x = F.conv2d(_same_pad(x, k, same_stride), w, b, stride=same_stride)
        else:
            x = F.conv2d(x, w, b, padding=k // 2)
        return F.relu(x) if relu else x

    def _lrn(self, x):
        size, alpha, beta, k = self.lrn
        return local_response_norm(x, size, alpha, beta, k)

    def _mixed_exit(self, x, m, part, ch):
        """One Inception block, stopping at `part` ('' = concat)."""
        if part == '':
            sizes = self._branch_sizes(m)
            if ch is None:
                return torch.cat([self._mixed_exit(x, m, p, None) for p in
                                  ('1x1', '3x3', '5x5', 'pool_reduce')], 1)
            for p, n in sizes:
                if ch < n:
                    return self._mixed_exit(x, m, p, ch)
                ch -= n
            raise IndexError('feature out of range for ' + m)
        pre = part.endswith('_pre_relu')
        base = part[:-len('_pre_relu')] if pre else part
        if base == '1x1':
            return self._conv(x, m + '_1x1', not pre, ch)
        if base in ('3x3_bottleneck', '5x5_bottleneck'):
            return self._conv(x, m + '_' + base, not pre, ch)
        if base in ('3x3', '5x5'):
            y = self._conv(x, m + '_' + base + '_bottleneck')
            return self._conv(y, m + '_' + base, not pre, ch)
        if base == 'pool':
            if ch is not None:
                x = x[:, ch:ch + 1]
            return F.max_pool2d(x, 3, 1, 1)
        if base == 'pool_reduce':
            y = F.max_pool2d(x, 3, 1, 1)
            return self._conv(y, m + '_pool_reduce', not pre, ch)
        raise KeyError(part)

    def _branch_sizes(self, m):
        f = dict((n, ff) for n, _, ff in _MIXED)[m]
        return [('1x1', f[0]), ('3x3', f[2]), ('5x5', f[4]), ('pool_reduce', f[5])]

    def forward(self, x, exit_name, feature=None):
        ch = feature
        if exit_name in ('conv2d0_pre_relu', 'conv2d0'):
            return self._conv(x, 'conv2d0', exit_name == 'conv2d0', ch, same_stride=2)
        x = self._conv(x, 'conv2d0', same_stride=2)
        if exit_name == 'maxpool0':
            if ch is not None:
                x = x[:, ch:ch + 1]
            return F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        x = F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        x = self._lrn(x)
        if exit_name == 'localresponsenorm0':
            return x if ch is None else x[:, ch:ch + 1]
        if exit_name in ('conv2d1_pre_relu', 'conv2d1'):
            return self._conv(x, 'conv2d1', exit_name == 'conv2d1', ch)
        x = self._conv(x, 'conv2d1')
        if exit_name in ('conv2d2_pre_relu', 'conv2d2'):
            return self._conv(x, 'conv2d2', exit_name == 'conv2d2', ch)
        x = self._conv(x, 'conv2d2')
        x = self._lrn(x)
        if exit_name == 'localresponsenorm1':
            return x if ch is None else x[:, ch:ch + 1]
        if exit_name == 'maxpool1' and ch is not None:
            x = x[:, ch:ch + 1]
        x = F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        if exit_name == 'maxpool1':
            return x
        for m, _, _ in _MIXED:
            if exit_name.startswith(m) and not exit_name == _POOL_AFTER.get(m):
                return self._mixed_exit(x, m, exit_name[len(m) + 1:], ch)
            pool = _POOL_AFTER.get(m)
            if pool == exit_name or (exit_name == 'avgpool0' and m == 'mixed5b'):
                y = self._mixed_exit(x, m, '', ch)
                if exit_name == 'avgpool0':
                    return F.avg_pool2d(y, 7, padding=3)
                return F.max_pool2d(_same_pad(y, 3, 2), 3, 2)
            x = self._mixed_exit(x, m, '', None)
            if pool is not None:
                x = F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        raise KeyError('unknown exit ' + exit_name)


    # ---- classifier (activation atlas labels) ---------------------------
    def classify_from(self, a, layer):
        """Class logits (B, 1008) for activations `a` (B, C, H, W) of the
        concatenated block `layer` (e.g. 'mixed4c'), run through the rest of
        the network and the final classifier. Index i is line i of the
        InceptionV1 label list (0 is 'dummy'; 1001..1007 are unused).

        Fed a single averaged activation as a 1 x 1 map, this is how the
        activation atlas labels a cell: the convolutions see only their
        centre taps and the pools pass the value through."""
        head = getattr(self, '_head', None)
        if head is None:
            if not os.path.exists(INCEPTION5H_HEAD):
                raise FileNotFoundError(
                    'classifier weights not found at ' + INCEPTION5H_HEAD +
                    ' -- run python -m dpg_system.neural_synth.convert_inception5h')
            st = torch.load(INCEPTION5H_HEAD, map_location='cpu')
            head = self._head = (st['weight'].to(a.device), st['bias'].to(a.device))
        names = [m for m, _, _ in _MIXED]
        if layer not in names:
            raise KeyError('classify_from needs a mixed block, not ' + layer)
        i = names.index(layer)
        x = a
        if layer in _POOL_AFTER:
            x = F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        for m in names[i + 1:]:
            x = self._mixed_exit(x, m, '', None)
            if m in _POOL_AFTER:
                x = F.max_pool2d(_same_pad(x, 3, 2), 3, 2)
        x = x.mean((2, 3))
        return x @ head[0].t() + head[1]


_VGG19_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    name = 'vgg19'

    def __init__(self):
        super().__init__()
        import torchvision
        net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.features = net.features
        del net
        for p in self.parameters():
            p.requires_grad_(False)
        self.exits, self._counts, self._plan = vgg19_exits()

    def set_lrn(self, *args, **kwargs):
        pass

    def feature_count(self, exit_name):
        return self._counts[exit_name]

    def forward(self, x, exit_name, feature=None):
        last, kind = self._plan[exit_name]
        layers = self.features
        stop = last if kind in ('conv', 'relu', 'pool') else last + 1
        for i in range(stop):
            m = layers[i]
            x = F.relu(x) if isinstance(m, nn.ReLU) else m(x)
        m = layers[last]
        if kind == 'pool':
            if feature is not None:
                x = x[:, feature:feature + 1]
            return m(x)
        w, b = m.weight, m.bias
        if feature is not None:
            w, b = w[feature:feature + 1], b[feature:feature + 1]
        x = F.conv2d(x, w, b, padding=1)
        return F.relu(x) if kind == 'relu' else x


MODEL_NAMES = ['inception5h', 'vgg19']
DEFAULT_EXIT = {'inception5h': 'mixed5b_5x5_pre_relu', 'vgg19': 'conv5_3_pre_relu'}

_cache = {}


def load_model(name, device):
    """Shared, frozen, eval-mode model per (name, device)."""
    key = (name, str(device))
    m = _cache.get(key)
    if m is None:
        if name == 'inception5h':
            m = Inception5h()
        elif name == 'vgg19':
            m = VGG19()
        else:
            raise KeyError(name)
        m = m.to(device).eval()
        _cache[key] = m
    return m
