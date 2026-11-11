"""Convert the Inception5h (GoogLeNet) weights exported from TensorFlow as
loose .npy files into one torch state dict, inception5h.pt, beside this file.

The .npy export lives with the 2018 NeuralSynth code:
    ~/Dev/NeuralSynth_PT_VGG/weights/<layer>_w.npy, <layer>_b.npy

Conv weights are TF layout (height, width, in, out) and are transposed to
torch layout (out, in, height, width). Only the convolutions the synth uses
are kept (conv2d0..2, mixed3a..mixed5b); the auxiliary heads and the
classifier are dropped.

    python -m dpg_system.neural_synth.convert_inception5h [weights_dir]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.expanduser('~/Dev/NeuralSynth_PT_VGG/weights')
OUT_PATH = os.path.join(HERE, 'inception5h.pt')

MIXED = ['mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d',
         'mixed4e', 'mixed5a', 'mixed5b']
BRANCHES = ['1x1', '3x3_bottleneck', '3x3', '5x5_bottleneck', '5x5', 'pool_reduce']


def conv_names():
    names = ['conv2d0', 'conv2d1', 'conv2d2']
    for m in MIXED:
        names += [m + '_' + b for b in BRANCHES]
    return names


def convert(src=DEFAULT_SRC, out=OUT_PATH):
    state = {}
    for name in conv_names():
        w = np.load(os.path.join(src, name + '_w.npy'))
        b = np.load(os.path.join(src, name + '_b.npy'))
        state[name + '.weight'] = torch.from_numpy(np.ascontiguousarray(w.transpose(3, 2, 0, 1)))
        state[name + '.bias'] = torch.from_numpy(b)
    torch.save(state, out)
    return out


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    print('wrote', convert(src))
