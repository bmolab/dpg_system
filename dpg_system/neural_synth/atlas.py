"""Activation atlases for Inception5h: directions to grow, found by mapping
what the network does with real photographs.

After Carter, Armstrong, Schubert, Johnson and Olah, "Activation Atlas"
(Distill, 2019), https://distill.pub/2019/activation-atlas/ -- made for the
same network (Lucid's InceptionV1 is the DeepDream GoogLeNet, Inception5h).

A layer's activation at one spot of one image is a vector with one number per
channel. Lucid published 100,000 such vectors per layer, sampled at random
spots of ImageNet photographs. The builder lays them out on a 2D map with
UMAP (similar vectors land near each other), cuts the map into a grid, and
averages the vectors in each cell. Each cell's average is a direction in the
layer -- a mix of channels -- that neural_synth can grow, and each cell is
labelled by feeding its average through the rest of the network to the
classifier. Neighbouring cells hold related things, so walking the grid
moves smoothly through what the layer knows.

Building (once per layer; downloads ~10 pages of 10,000 vectors, 100-400 MB,
then ~2-5 minutes of UMAP; needs umap-learn):

    python -m dpg_system.neural_synth.atlas mixed4c [mixed4d ...]
    python -m dpg_system.neural_synth.atlas all

writes neural_synth/atlas/<layer>.npz. The downloaded pages are cached in
~/.cache/dpg_system/neural_synth_activations and deleted after the build
(--keep keeps them).

Our port reproduces Lucid's activations (same channel order; the same scale
when the local response normalization is TensorFlow's own, alpha 1e-4 per
channel = lrn (11, 1.1e-3, 0.5, 2) here). The 2018 synth settings normalize
less, which changes the scale of the activations but not what the channels
are, so the directions still apply.
"""
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_DIR = os.path.join(HERE, 'atlas')
CACHE_DIR = os.path.expanduser('~/.cache/dpg_system/neural_synth_activations')
ACT_URL = ('https://storage.googleapis.com/modelzoo/aligned-activations/InceptionV1/'
           '{layer}-{page:05d}-of-01000.npy')
LABELS_URL = 'https://storage.googleapis.com/distill-2019-activation-atlas/build/labels--model-InceptionV1.txt'
LAYERS = ['mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d',
          'mixed4e', 'mixed5a', 'mixed5b']
GRIDS = (10, 20, 40)
PAGES = 10                       # 10 x 10,000 = all 100,000 published samples
N_CLASSES = 1001                 # label lines; logits 1001..1007 are padding


def atlas_path(layer):
    return os.path.join(ATLAS_DIR, layer + '.npz')


def available_layers():
    return [l for l in LAYERS if os.path.exists(atlas_path(l))]


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Direction:
    """A target for the synth: grow `vector` (one weight per channel of the
    concatenated block `layer`) instead of a single channel."""
    layer: str
    vector: np.ndarray           # (C,) float32, unit length
    name: str = 'direction'

    def __str__(self):
        return self.name


class Atlas:
    """One built layer: cells at grid sizes 10, 20 and 40."""

    def __init__(self, layer):
        path = atlas_path(layer)
        if not os.path.exists(path):
            raise FileNotFoundError(
                'no atlas for ' + layer + ' -- build it with: '
                'python -m dpg_system.neural_synth.atlas ' + layer)
        z = np.load(path, allow_pickle=False)
        self.layer = layer
        self.labels = [str(l) for l in z['labels']]
        self.whitening = z['whitening'].astype(np.float32)
        self.grids = [int(g) for g in z['grids']]
        self.samples = int(z['samples'])
        self.cells = {}
        for g in self.grids:
            self.cells[g] = dict(
                xy=z['xy_%d' % g].astype(np.int32),           # (K, 2) column, row
                count=z['count_%d' % g].astype(np.int64),
                mean=z['mean_%d' % g].astype(np.float32),     # (K, C)
                probs=z['probs_%d' % g].astype(np.float32),   # (K, 1001)
            )
            c = self.cells[g]
            c['index'] = {(int(x), int(y)): k for k, (x, y) in enumerate(c['xy'])}

    def nearest(self, grid, x, y):
        """Index of the occupied cell nearest (x, y)."""
        c = self.cells[grid]
        k = c['index'].get((int(x), int(y)))
        if k is not None:
            return k
        d = ((c['xy'] - np.array([x, y])) ** 2).sum(1)
        return int(np.argmin(d))

    def top_classes(self, grid, k, n=3):
        p = self.cells[grid]['probs'][k]
        order = np.argsort(-p)[:n]
        return [(self.labels[i], float(p[i])) for i in order]

    def find(self, grid, text, limit=20):
        """Cells ranked by how strongly they point to classes whose names
        contain `text` (whole words, any case; any part of a name if no
        whole word matches), best first, at most `limit`. Returns (cell
        indices, matched class names)."""
        words = text.lower().split()
        if not words:
            return [], []
        def tokens(label):
            return re.findall(r"[a-z0-9']+", label.lower())
        hits = [i for i, l in enumerate(self.labels[:N_CLASSES])
                if i > 0 and all(w in tokens(l) or w + 's' in tokens(l) or w.rstrip('s') in tokens(l)
                                 for w in words)]
        if not hits:                       # nothing by whole word: any part of a name
            hits = [i for i, l in enumerate(self.labels[:N_CLASSES])
                    if i > 0 and all(w in l.lower() for w in words)]
        if not hits:
            return [], []
        score = self.cells[grid]['probs'][:, hits].sum(1)
        # cells that point to the class at least a quarter as strongly as the best one
        floor = 0.25 * float(score.max())
        order = [int(k) for k in np.argsort(-score) if score[k] >= floor and score[k] > 0]
        return order[:limit], [self.labels[i] for i in hits]

    def neighbours(self, grid, k):
        c = self.cells[grid]
        x, y = c['xy'][k]
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    j = c['index'].get((int(x + dx), int(y + dy)))
                    if j is not None:
                        out.append(j)
        return out

    def direction(self, grid, k, whiten=True):
        c = self.cells[grid]
        v = c['mean'][k].astype(np.float64)
        if whiten:
            v = self.whitening.astype(np.float64) @ v
        v = v / max(np.linalg.norm(v), 1e-12)
        x, y = c['xy'][k]
        top = self.top_classes(grid, k, 1)[0][0]
        name = '%s %d:%d,%d %s' % (self.layer, grid, x, y, top)
        return Direction(self.layer, v.astype(np.float32), name)


_atlases = {}


def load_atlas(layer):
    a = _atlases.get(layer)
    if a is None:
        a = _atlases[layer] = Atlas(layer)
    return a


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _download(url, path):
    if os.path.exists(path):
        return path
    tmp = path + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return path


def fetch_activations(layer, pages=PAGES, log=print):
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = []
    for p in range(pages):
        path = os.path.join(CACHE_DIR, '%s-%05d.npy' % (layer, p))
        if not os.path.exists(path):
            log('  downloading %s page %d/%d' % (layer, p + 1, pages))
        paths.append(_download(ACT_URL.format(layer=layer, page=p), path))
    return paths


def fetch_labels():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _download(LABELS_URL, os.path.join(CACHE_DIR, 'labels-InceptionV1.txt'))
    labels = open(path).read().splitlines()
    assert len(labels) == N_CLASSES and labels[0] == 'dummy', 'unexpected label list'
    return labels


def normalize_layout(xy, lo=1, hi=99, margin=0.1):
    """Lucid's normalize_layout: clip outliers at the 1st/99th percentile
    (plus a 10% margin) and scale into 0..1."""
    mins = np.percentile(xy, lo, axis=0)
    maxs = np.percentile(xy, hi, axis=0)
    span = maxs - mins
    mins, maxs = mins - margin * span, maxs + margin * span
    xy = np.clip(xy, mins, maxs)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    return xy


def _umap_fit(acts):
    import warnings
    import umap
    warnings.filterwarnings('ignore', category=FutureWarning)
    reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.05, metric='cosine',
                        compatibility_layout=True)
    return reducer.fit_transform(acts)


def _umap_layout(acts):
    """UMAP in a fresh process: pynndescent's numba code segfaults
    (macOS, torch 2.9 / numba 0.67) once torch is loaded in the same process."""
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(1) as pool:
        return pool.apply(_umap_fit, (acts,))


def build(layer, pages=PAGES, keep=False, device='cpu', log=print):
    import torch
    from .models import Inception5h

    t0 = time.time()
    labels = fetch_labels()
    paths = fetch_activations(layer, pages, log)
    acts = np.concatenate([np.load(p) for p in paths]).astype(np.float32)
    acts = acts[np.isfinite(acts).all(1) & (np.abs(acts).sum(1) > 0)]
    n, ch = acts.shape
    log('  %s: %d samples x %d channels' % (layer, n, ch))

    # whitening (Lucid's activation atlas: inverse of the uncentred
    # correlation; a small ridge for channels that are almost never active)
    correl = (acts.astype(np.float64).T @ acts) / n
    ridge = 1e-4 * np.trace(correl) / ch
    whitening = np.linalg.inv(correl + ridge * np.eye(ch)).astype(np.float32)

    log('  UMAP (cosine, 50 neighbours, min dist 0.05) ...')
    xy = normalize_layout(_umap_layout(acts))

    model = Inception5h().eval().to(device)
    out = dict(labels=np.array(labels), whitening=whitening, grids=np.array(GRIDS),
               samples=np.array(n), layout=xy.astype(np.float16))
    for g in GRIDS:
        ij = np.minimum((xy * g).astype(np.int64), g - 1)
        key = ij[:, 0] * g + ij[:, 1]
        count = np.bincount(key, minlength=g * g)
        sums = np.zeros((g * g, ch), np.float64)
        np.add.at(sums, key, acts)
        min_count = max(3, int(0.02 * n / (g * g)))
        keep_cells = np.nonzero(count >= min_count)[0]
        mean = (sums[keep_cells] / count[keep_cells, None]).astype(np.float32)
        with torch.no_grad():
            a = torch.from_numpy(mean)[:, :, None, None].to(device)
            logits = model.classify_from(a, layer)[:, :N_CLASSES].cpu()
            logits[:, 0] = -1e9
            probs = logits.softmax(1).numpy()
        out['xy_%d' % g] = np.stack([keep_cells // g, keep_cells % g], 1).astype(np.int16)
        out['count_%d' % g] = count[keep_cells].astype(np.int32)
        out['mean_%d' % g] = mean.astype(np.float16)
        out['probs_%d' % g] = probs.astype(np.float16)
        log('  grid %d: %d cells (min %d samples)' % (g, len(keep_cells), min_count))

    os.makedirs(ATLAS_DIR, exist_ok=True)
    path = atlas_path(layer)
    np.savez_compressed(path, **out)
    _atlases.pop(layer, None)
    if not keep:
        for p in paths:
            os.remove(p)
    log('  wrote %s (%.1f MB) in %.0f s' % (path, os.path.getsize(path) / 1e6, time.time() - t0))
    return path


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    keep = '--keep' in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    layers = LAYERS if args == ['all'] else args
    for l in layers:
        if l not in LAYERS:
            sys.exit('unknown layer %s (choose from %s)' % (l, ' '.join(LAYERS)))
    for l in layers:
        print('building', l)
        build(l, keep=keep)
