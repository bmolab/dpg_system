"""Does neural_synth grow exactly the same picture twice?

Grows the same target several times with a fixed seed and compares each
picture with the first: the largest pixel difference (as a fraction of the
picture's range) and the first step at which they part.

    python -m dpg_system.neural_synth.determinism_test
    python -m dpg_system.neural_synth.determinism_test --device cuda --deterministic
    python -m dpg_system.neural_synth.determinism_test --device cuda --size 1920 1080 --octaves 13

Run it with and without --deterministic on an NVIDIA machine to see whether
the option is needed there, and what it costs in time.
"""
import argparse
import time

import torch

from .engine import SynthEngine, best_device


def grow(engine):
    """One whole picture: the final image and a fingerprint of every step."""
    prints = []
    t0 = time.perf_counter()
    for _ in engine.run_image():
        x = engine.x
        prints.append(torch.stack([x.sum(), x.abs().sum(), (x * x).sum()]).cpu())
    final = engine.x.detach().cpu().clone()
    return final, torch.stack(prints), time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--device', default=best_device())
    p.add_argument('--deterministic', action='store_true',
                   help="turn on the 'deterministic' option (cuda only)")
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--size', type=int, nargs=2, default=(640, 360), metavar=('W', 'H'))
    p.add_argument('--octaves', type=int, default=9)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--layer', default='mixed4d_3x3_bottleneck_pre_relu')
    p.add_argument('--feature', type=int, default=139)
    p.add_argument('--seed', type=int, default=78190)
    a = p.parse_args()

    e = SynthEngine()
    s = e.s
    s.device = a.device
    s.deterministic = a.deterministic
    s.width, s.height = a.size
    s.octaves, s.iter_base = a.octaves, a.iterations
    s.seed, s.seed_mode = a.seed, 'fixed'
    e.set_target([a.layer], [a.feature])

    print('device %s, deterministic %s, %dx%d, %d octaves, %d iterations, %s #%d, seed %d, roll %g'
          % (a.device, a.deterministic, a.size[0], a.size[1], a.octaves, a.iterations,
             a.layer, a.feature, a.seed, s.roll))
    if a.device.startswith('cuda'):
        print('torch %s, CUDA %s, cuDNN %s, %s' % (torch.__version__, torch.version.cuda,
              torch.backends.cudnn.version(), torch.cuda.get_device_name()))
    else:
        print('torch %s' % torch.__version__)

    first = None
    all_same = True
    for run in range(a.runs):
        final, prints, secs = grow(e)
        if first is None:
            first = (final, prints)
            print('run 1: %.1f s, %d steps (the reference)' % (secs, prints.shape[0]))
            continue
        diff = float((final - first[0]).abs().max() / first[0].abs().max().clamp_min(1e-12))
        parted = (prints != first[1]).any(1).nonzero()
        where = 'never' if len(parted) == 0 else 'at step %d' % (int(parted[0]) + 1)
        same = diff == 0 and len(parted) == 0
        all_same &= same
        print('run %d: %.1f s, largest difference %.3g of the range, parts from run 1 %s%s'
              % (run + 1, secs, diff, where, '  (identical)' if same else ''))
    print('identical every time' if all_same else 'NOT identical')


if __name__ == '__main__':
    main()
