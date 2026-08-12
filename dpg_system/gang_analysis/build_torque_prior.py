"""Build the corpus prior that whitened surprise is measured against.

Surprise asks how far a torque configuration sits from what bodies usually do.
That needs a prior: the mean and covariance of the 66-channel torque field over
real movement. Both are computed here under the SEG regime (clean_segments from
every file, see CHARACTERIZATION_RESULTS.md section 8).

Two passes, because the distribution of the distance can only be measured once
the distance is defined:

  1. mean + covariance -> eigenbasis -> whitening matrix
  2. the corpus distribution of the whitened distance itself, so a live value
     can be reported as "more unusual than X% of recorded movement" rather than
     as a bare number with no scale.

Mahalanobis distance is invariant under per-channel scaling, so it makes no
difference whether torque is normalized by max_torque first. It is not
computed here.

Usage:
    python build_torque_prior.py --out torque_prior.npz [--stream total]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import dpg_system.gang_core as gc

ROOT = '/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H'
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}
AXES = ('x', 'y', 'z')
CHANNELS = [f'{NAMES[j]}.{AXES[a]}' for j in range(22) for a in range(3)]
FILE_KEY = {'total': 'torque', 'gravity': 'torques_grav_vec',
            'dynamic': 'torques_dyn_vec'}
TRIM_HEAD = 3

# Distance histogram: log-spaced, since d is heavy tailed.
D_LO, D_HI, D_STEP = -2.0, 3.0, 0.01
D_EDGES = np.arange(D_LO, D_HI + D_STEP, D_STEP)
N_D = len(D_EDGES) - 1


def seg_mask(rel, T, index):
    info = index.get(rel)
    if info is None:
        return None
    m = np.zeros(T, bool)
    for a, b in info.get('keep', []):
        m[max(0, min(T - 1, a)):max(0, min(T - 1, b)) + 1] = True
    for a, b in info.get('drop', []):
        m[max(0, min(T - 1, a)):max(0, min(T - 1, b)) + 1] = False
    m[:TRIM_HEAD] = False
    return m


def _init(idx_path, stream, prior_path=None):
    global INDEX, STREAM, PRIOR
    INDEX = json.load(open(idx_path))
    STREAM = stream
    PRIOR = np.load(prior_path) if prior_path else None


def pass1(path):
    """Cross products for mean and covariance."""
    rel = os.path.relpath(path, ROOT)
    try:
        d = np.load(path, allow_pickle=True)
        a = np.asarray(d[FILE_KEY[STREAM]], np.float64)
    except Exception:
        return None
    m = seg_mask(rel, a.shape[0], INDEX)
    if m is None or m.sum() < 30:
        return None
    X = a[m].reshape(int(m.sum()), -1)
    return X.shape[0], X.sum(axis=0), X.T @ X


def pass2(path):
    """Histogram of the whitened distance, using the prior from pass 1."""
    rel = os.path.relpath(path, ROOT)
    try:
        d = np.load(path, allow_pickle=True)
        a = np.asarray(d[FILE_KEY[STREAM]], np.float64)
    except Exception:
        return None
    m = seg_mask(rel, a.shape[0], INDEX)
    if m is None or m.sum() < 30:
        return None
    X = a[m].reshape(int(m.sum()), -1)
    dist = whiten_distance(X, PRIOR)
    dist = dist[dist > 0]
    if dist.size == 0:
        return None
    lg = np.log10(dist)
    h, _ = np.histogram(lg, bins=D_EDGES)
    return (h.astype(np.int64), int((lg < D_LO).sum()), int((lg >= D_HI).sum()),
            int(dist.size))


def whiten_distance(X, prior):
    """Mahalanobis distance of each row of X (frames, 66) under the prior."""
    live = prior['live']
    W = prior['whiten']          # (k, n_live)
    mu = prior['mean'][live]
    Z = (X[:, live] - mu) @ W.T
    return np.sqrt((Z * Z).sum(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='torque_prior.npz')
    ap.add_argument('--index', default='noise_index.json')
    ap.add_argument('--stream', default='total', choices=tuple(FILE_KEY))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ridge', type=float, default=1e-3,
                    help='eigenvalues floored at ridge*lambda_max; caps the '
                         'condition number and stops whitening from amplifying '
                         'directions that are pure noise (default 1e-3)')
    args = ap.parse_args()

    idx_path = os.path.abspath(args.index)
    files = []
    for dp, _d, fns in os.walk(ROOT):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    print(f'{len(files)} files, stream={args.stream}, ridge={args.ridge}')

    # ---- pass 1: mean + covariance ---------------------------------------
    t0 = time.perf_counter()
    n = 0
    s = np.zeros(66)
    xtx = np.zeros((66, 66))
    with Pool(args.workers, initializer=_init,
              initargs=(idx_path, args.stream)) as pool:
        for r in pool.imap_unordered(pass1, files, chunksize=8):
            if r is None:
                continue
            n += r[0]
            s += r[1]
            xtx += r[2]
    print(f'pass 1: {n:,} frames, {time.perf_counter() - t0:.0f}s')

    mean = s / n
    cov = xtx / n - np.outer(mean, mean)
    var = np.diag(cov)
    live = var > 1e-12
    k_live = int(live.sum())
    print(f'live channels: {k_live} of 66  '
          f'(dead: {", ".join(np.array(CHANNELS)[~live][:12])})')

    C = cov[np.ix_(np.where(live)[0], np.where(live)[0])]
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]

    # Regularize: whitening divides by sqrt(lambda), so the smallest
    # eigenvalues dominate the result. Those directions are also where mocap
    # noise lives, so an unfloored whitening measures noise, not surprise.
    floor = args.ridge * lam[0]
    n_floored = int((lam < floor).sum())
    lam_reg = np.maximum(lam, floor)
    print(f'eigenvalues: max={lam[0]:.4g} min={lam[-1]:.4g} '
          f'condition={lam[0] / max(lam[-1], 1e-30):.3g}')
    print(f'floored {n_floored}/{k_live} eigenvalues at {floor:.4g} '
          f'-> condition capped at {1 / args.ridge:.0f}')

    whiten = (V / np.sqrt(lam_reg)).T          # (k, k): z = W (x - mu)

    prior = {
        'stream': args.stream, 'channels': np.array(CHANNELS),
        'live': live, 'mean': mean, 'cov': cov,
        'eigenvalues': lam, 'eigenvalues_reg': lam_reg, 'eigenvectors': V,
        'whiten': whiten, 'ridge': args.ridge, 'n_frames': n,
        'n_live': k_live,
    }
    np.savez_compressed(args.out, **prior)

    # ---- pass 2: distribution of the distance ----------------------------
    t0 = time.perf_counter()
    P = np.load(args.out)
    h = np.zeros(N_D, np.int64)
    n_under = n_over = n_tot = 0
    with Pool(args.workers, initializer=_init,
              initargs=(idx_path, args.stream, os.path.abspath(args.out))) as pool:
        for r in pool.imap_unordered(pass2, files, chunksize=8):
            if r is None:
                continue
            h += r[0]
            n_under += r[1]
            n_over += r[2]
            n_tot += r[3]
    print(f'pass 2: {n_tot:,} frames, {time.perf_counter() - t0:.0f}s')

    centers = 10 ** ((D_EDGES[:-1] + D_EDGES[1:]) / 2)
    counts = np.concatenate([[n_under], h, [n_over]])
    cent = np.concatenate([[10 ** D_LO], centers, [10 ** D_HI]])
    cum = np.cumsum(counts) / counts.sum()
    qs = {q: float(cent[min(int(np.searchsorted(cum, q / 100)), len(cent) - 1)])
          for q in (1, 10, 25, 50, 75, 90, 99, 99.9)}

    np.savez_compressed(args.out, **prior, d_hist=h, d_under=n_under,
                        d_over=n_over, d_edges=D_EDGES, d_n=n_tot)

    print('\ncorpus distribution of whitened distance d:')
    for q, v in qs.items():
        print(f'   p{q:<5} = {v:8.3f}')
    print(f'\n  sqrt(n_live) = {np.sqrt(k_live):.2f}  '
          f'(the chi expectation if the field were Gaussian; the median '
          f'sitting well below it means the field is heavy tailed, not normal)')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
