"""Write the small example data files that help patches load.

    python3 make_help_data.py

Writes into dpg_system/help/data/. The files are tiny and committed, so a help
patch can load them by a path relative to the repo root (the app's working
directory), e.g. dpg_system/help/data/np_load_example.npz.
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def np_load_example():
    # two named 2-D arrays, values 0..1, that read at a glance as pictures
    y, x = np.mgrid[0:16, 0:16].astype(np.float32)
    r = np.hypot(x - 7.5, y - 7.5)
    rings = (0.5 + 0.5 * np.cos(r * 1.2)).astype(np.float32)          # 16 x 16 target
    yy, xx = np.mgrid[0:8, 0:12]
    checker = ((xx // 2 + yy // 2) % 2).astype(np.float32)             # 8 x 12 checker of 2x2 squares
    path = os.path.join(DATA, 'np_load_example.npz')
    np.savez(path, rings=rings, checker=checker)
    return path


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    for p in [np_load_example()]:
        print(p, os.path.getsize(p), 'bytes')
