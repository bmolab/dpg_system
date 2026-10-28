"""Write help/data/elbow_hinge_example.npz: a synthetic 600-frame Shadow take for the
elbow_hinge help page.

    python3 dpg_system/help/tools/make_elbow_hinge_example.py

Both elbows bend forward and straighten on a fixed hinge (0 to 100 degrees). Three
versions: 'clean', where nothing else moves, so each forearm stays exactly on its
circle; 'quats', where each upper-arm SENSOR misreads its roll about the bone by up to
6 degrees, independently of the bend, while the forearm sensor reads true. The local
quaternions are relative, so the error shows in the elbow's too (it takes the opposite
roll): the frame-varying roll error that pushes the forearm off the circle. And
'turned', the clean take as it reads after a T-pose held with the elbows turned (the
inside of the elbow up toward the ceiling): 30 degrees left, 15 right - the constant
roll that elbow_turn measures. Every other joint stays at rest.

For hang_symmetry: 'hanging', both arms lowered as mirror images, splayed out 10 to 20
degrees and swaying together in and out; and 'shifted', the same with both arms turned
15 degrees toward the dancer's right - the asymmetric hang hang_symmetry measures.
"""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'smpl_utilities'))
from diag_magnetometer_deviation import load_skeleton

DATA_DIR = HERE.parent / 'data'


def axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None] * axis], -1)


def qmul(a, b):
    return np.stack([
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]], -1)


def main():
    parent, order, bax = load_skeleton(HERE.parent.parent / 'definition.xml')
    F = 600
    t = np.arange(F) / 60.0
    clean = np.tile([1.0, 0, 0, 0], (F, 37, 1))
    bend = np.radians(50.0) * (1 - np.cos(2 * np.pi * 0.25 * t))          # 0 to 100 degrees, every 4 s
    for elbow, axis in ((9, [0, -1.0, 0]), (23, [0, 1.0, 0])):       # both forearms swing forward
        clean[:, elbow] = axis_angle(axis, bend)
    wobbly = clean.copy()
    for shoulder, phase in ((5, 0.0), (19, 1.3)):
        roll = np.radians(6.0) * np.sin(2 * np.pi * 0.37 * t + phase)
        error = axis_angle(bax[shoulder], roll)
        wobbly[:, shoulder] = error
        elbow = shoulder + 4                                             # 5 -> 9, 19 -> 23
        conj = error * np.array([1, -1, -1, -1.0])
        w = conj
        b = clean[:, elbow]
        wobbly[:, elbow] = np.stack([                                    # conj(error) * bend
            w[:, 0] * b[:, 0] - w[:, 1] * b[:, 1] - w[:, 2] * b[:, 2] - w[:, 3] * b[:, 3],
            w[:, 0] * b[:, 1] + w[:, 1] * b[:, 0] + w[:, 2] * b[:, 3] - w[:, 3] * b[:, 2],
            w[:, 0] * b[:, 2] - w[:, 1] * b[:, 3] + w[:, 2] * b[:, 0] + w[:, 3] * b[:, 1],
            w[:, 0] * b[:, 3] + w[:, 1] * b[:, 2] - w[:, 2] * b[:, 1] + w[:, 3] * b[:, 0]], -1)
    turned = clean.copy()
    for shoulder, degrees in ((5, 30.0), (19, 15.0)):
        # the upper-arm sensor's zero is turned about its local +X; the forearm reads true
        t = axis_angle([1.0, 0, 0], np.full(F, np.radians(degrees)))
        turned[:, shoulder] = qmul(clean[:, shoulder], t)
        turned[:, shoulder + 4] = qmul(t * np.array([1, -1, -1, -1.0]), clean[:, shoulder + 4])
    t = np.arange(F) / 60.0
    splay = np.radians(15.0 + 5.0 * np.sin(2 * np.pi * 0.2 * t))
    hanging, shifted = (np.tile([1.0, 0, 0, 0], (F, 37, 1)) for _ in range(2))
    for shoulder, sign in ((5, -1.0), (19, 1.0)):
        # lowered from the T-pose about the forward axis (+Z), leaving the splay outward
        down = sign * (np.pi / 2 - splay)
        hanging[:, shoulder] = axis_angle([0, 0, 1.0], down)
        shifted[:, shoulder] = axis_angle([0, 0, 1.0], down - np.radians(15.0))   # toward the dancer's right
    path = DATA_DIR / 'elbow_hinge_example.npz'
    np.savez_compressed(path, quats=wobbly, clean=clean, turned=turned, hanging=hanging, shifted=shifted)
    print(path)


if __name__ == '__main__':
    main()
