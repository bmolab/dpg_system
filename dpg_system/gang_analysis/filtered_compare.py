"""Recompute the headline statistics under three quality regimes.

  ALL       every frame (what the first characterization used)
  KEEP      clean + moderate files, with unusable segments and excision zones
            dropped frame-by-frame
  CLEAN     clean files only, same frame-level excision

The comparison is the point. Two opposite errors are possible and only the
spread between regimes distinguishes them:

  - if ALL is inflated relative to KEEP/CLEAN, the tails were glitch-driven and
    the original conditioning numbers overstate real dynamic range;
  - if CLEAN is *lower* than KEEP, filtering removed genuine high-effort motion
    -- the noise detector is known to false-positive on dynamic movement, so
    over-filtering understates the range.
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

ROOT = '/Users/drokeby/dpg_system/AMASS_Dynamic'
SMPLH = os.path.join(ROOT, 'SMPL_H')
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}
TRIM_HEAD = 3
REGIMES = ('ALL', 'KEEP', 'SEG', 'SEGP', 'CLEAN')

LOG_LO, LOG_HI, LOG_STEP = -4.0, 4.5, 0.05
LOG_EDGES = np.arange(LOG_LO, LOG_HI + LOG_STEP, LOG_STEP)
N_LOG = len(LOG_EDGES) - 1


def log_hist(v):
    v = np.abs(np.asarray(v, np.float64).ravel())
    nz = int((v == 0).sum())
    v = v[v > 0]
    if v.size == 0:
        return [np.zeros(N_LOG, np.int64), nz, 0, 0]
    lg = np.log10(v)
    return [np.histogram(lg, bins=LOG_EDGES)[0].astype(np.int64), nz,
            int((lg < LOG_LO).sum()), int((lg >= LOG_HI).sum())]


def pct(h, qs):
    counts = np.concatenate([[h[1] + h[2]], h[0], [h[3]]])
    centers = np.concatenate([[0.0],
                              10 ** ((LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2),
                              [10 ** LOG_HI]])
    t = counts.sum()
    if t == 0:
        return {q: float('nan') for q in qs}
    cum = np.cumsum(counts) / t
    return {q: float(centers[min(int(np.searchsorted(cum, q / 100)),
                                 len(centers) - 1)]) for q in qs}


def build():
    specs, labels, wsq = [], [], []
    for p in gc.preset_names():
        for s in (gc.sides_for(p) or ['none']):
            labels.append(f'{p}|{s}')
            specs.append(gc.spec_from_preset(p, side=s, stream='total',
                                             normalize=False))
            wsq.append(sum(w * w for _j, _a, w in specs[-1].terms))
    prog_raw = gc.compile_specs(specs)
    prog_norm = gc.compile_specs(
        [gc.spec_from_preset(p, side=s, stream='total', normalize=True)
         for p in gc.preset_names() for s in (gc.sides_for(p) or ['none'])])
    return prog_raw, prog_norm, labels, np.array(wsq)


def _init(idx_path):
    # macOS multiprocessing uses spawn, so workers re-import this module and
    # see no globals set inside main(). The index path must be passed in.
    global PR, PN, LABELS, WSQ, INDEX
    PR, PN, LABELS, WSQ = build()
    INDEX = json.load(open(idx_path))


def pad(a, T):
    o = np.zeros((T, 24, 3), np.float32)
    o[:, :22] = a
    return o


def one(path):
    rel = os.path.relpath(path, SMPLH)
    info = INDEX.get(rel)
    try:
        d = np.load(path, allow_pickle=True)
        tq = np.asarray(d['torque'], np.float64)
        w = np.asarray(d['angular_velocity'], np.float64)
    except Exception:
        return None
    T0 = tq.shape[0]
    if T0 < 20 or tq.shape != w.shape:
        return None

    valid = np.ones(T0, bool)
    valid[:TRIM_HEAD] = False
    cls = info['cls'] if info else 'unrated'
    if info:
        for a, b in info['drop']:
            a = max(0, min(T0 - 1, a))
            b = max(0, min(T0 - 1, b))
            valid[a:b + 1] = False

    masks = {'ALL': np.ones(T0, bool)}
    masks['ALL'][:TRIM_HEAD] = False
    masks['KEEP'] = valid if cls in ('clean', 'moderate') else np.zeros(T0, bool)
    masks['CLEAN'] = valid if cls == 'clean' else np.zeros(T0, bool)

    # SEG: the noise work's own clean_segments, applied uniformly to EVERY file
    # regardless of its file-level classification -- so the clean stretches of
    # problematic files are included rather than discarded with the file.
    seg = np.zeros(T0, bool)
    if info:
        for a, b in info.get('keep', []):
            a = max(0, min(T0 - 1, a))
            b = max(0, min(T0 - 1, b))
            seg[a:b + 1] = True
    seg &= valid
    masks['SEG'] = seg
    # SEGP: only the part of SEG contributed by problematic files -- isolates
    # what the whole-file filter was throwing away.
    masks['SEGP'] = seg if cls == 'problematic' else np.zeros(T0, bool)

    out = {}
    for reg, m in masks.items():
        if m.sum() < 10:
            continue
        t, wv = tq[m], w[m]
        P = (t * wv).sum(axis=-1)
        Ptot = P.sum(axis=1)
        stacked_t = gc.stack_streams(
            {'total': pad(t, t.shape[0]), 'gravity': np.zeros((t.shape[0], 24, 3), np.float32),
             'dynamic': np.zeros((t.shape[0], 24, 3), np.float32),
             'passive': np.zeros((t.shape[0], 24, 3), np.float32)})
        stacked_w = gc.stack_streams(
            {'total': pad(wv, wv.shape[0]), 'gravity': np.zeros((wv.shape[0], 24, 3), np.float32),
             'dynamic': np.zeros((wv.shape[0], 24, 3), np.float32),
             'passive': np.zeros((wv.shape[0], 24, 3), np.float32)})
        nt_raw, _a, _b = PR.evaluate(stacked_t)
        nw_raw, _a, _b = PR.evaluate(stacked_w)
        nt_norm, _a, _b = PN.evaluate(stacked_t)
        gp = nt_raw * nw_raw / np.maximum(WSQ, 1e-12)

        out[reg] = {
            'n': int(m.sum()),
            'torque': log_hist(np.linalg.norm(t, axis=-1)),
            'omega': log_hist(np.linalg.norm(wv, axis=-1)),
            'power': log_hist(P),
            'body': log_hist(Ptot),
            'gang_net': [log_hist(nt_norm[:, i]) for i in range(nt_norm.shape[1])],
            'gang_pow': [log_hist(gp[:, i]) for i in range(gp.shape[1])],
        }
    return out


def blank(n_g):
    z = lambda: [np.zeros(N_LOG, np.int64), 0, 0, 0]
    return {r: {'n': 0, 'torque': z(), 'omega': z(), 'power': z(), 'body': z(),
                'gang_net': [z() for _ in range(n_g)],
                'gang_pow': [z() for _ in range(n_g)]} for r in REGIMES}


def acc_add(dst, src):
    for k in range(4):
        dst[k] = dst[k] + src[k]


def add(A, r):
    for reg, v in r.items():
        a = A[reg]
        a['n'] += v['n']
        for key in ('torque', 'omega', 'power', 'body'):
            acc_add(a[key], v[key])
        for i in range(len(v['gang_net'])):
            acc_add(a['gang_net'][i], v['gang_net'][i])
            acc_add(a['gang_pow'][i], v['gang_pow'][i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='noise_index.json')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    idx_path = os.path.abspath(args.index)

    files = []
    for dp, _d, fns in os.walk(SMPLH):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    if args.limit:
        files = [files[i] for i in sorted(set(
            np.linspace(0, len(files) - 1, args.limit).astype(int)))]

    _init(idx_path)
    A = blank(len(LABELS))
    print(f'{len(files)} files, {args.workers} workers')
    t0 = time.perf_counter()
    done = 0
    with Pool(args.workers, initializer=_init, initargs=(idx_path,)) as pool:
        for r in pool.imap_unordered(one, files, chunksize=8):
            done += 1
            if r:
                add(A, r)
            if done % 3000 == 0:
                print(f'  {done}/{len(files)} {time.perf_counter() - t0:.0f}s',
                      flush=True)
    print(f'done {done} files, {time.perf_counter() - t0:.0f}s\n')

    Q = (50, 90, 99, 99.9)
    print('=' * 88)
    print('FRAMES RETAINED')
    print('=' * 88)
    base = A['ALL']['n']
    for r in REGIMES:
        print(f'  {r:6s} {A[r]["n"]:12,d} frames   {A[r]["n"] / base:6.1%} of ALL')

    print('\n' + '=' * 88)
    print('HEADLINE QUANTITIES BY REGIME  (pooled per-joint magnitudes)')
    print('=' * 88)
    print(f"{'quantity':10s}{'regime':8s}{'p50':>10s}{'p90':>10s}{'p99':>10s}"
          f"{'p99.9':>11s}{'crest':>8s}")
    for key, unit in (('torque', 'N.m'), ('omega', 'rad/s'), ('power', 'W')):
        for r in REGIMES:
            p = pct(A[r][key], Q)
            print(f"{key + ' (' + unit + ')' if r == 'ALL' else '':10s}{r:8s}"
                  f"{p[50]:10.4f}{p[90]:10.4f}{p[99]:10.4f}{p[99.9]:11.4f}"
                  f"{p[99] / max(p[50], 1e-9):8.1f}")
        print()

    print('WHOLE-BODY POWER (W)')
    for r in REGIMES:
        p = pct(A[r]['body'], Q)
        print(f'  {r:6s} p50={p[50]:8.1f}  p90={p[90]:8.1f}  p99={p[99]:8.1f}  '
              f'p99.9={p[99.9]:9.1f}  crest={p[99] / max(p[50], 1e-9):6.1f}')

    print('\n' + '=' * 88)
    print('GANG NET TORQUE (normalized) -- p99 and crest by regime')
    print('=' * 88)
    print(f"{'gang':30s}" + ''.join(f'{r + " p99":>13s}' for r in REGIMES)
          + ''.join(f'{r + " crest":>13s}' for r in REGIMES))
    rowsort = []
    for i, lab in enumerate(LABELS):
        ps = {r: pct(A[r]['gang_net'][i], Q) for r in REGIMES}
        rowsort.append((ps['ALL'][99], lab, ps))
    for _k, lab, ps in sorted(rowsort, reverse=True):
        print(f'{lab:30s}'
              + ''.join(f'{ps[r][99]:13.4f}' for r in REGIMES)
              + ''.join(f'{ps[r][99] / max(ps[r][50], 1e-9):13.1f}' for r in REGIMES))

    print('\n' + '=' * 88)
    print('GANG POWER (W) -- p99 by regime')
    print('=' * 88)
    print(f"{'gang':30s}" + ''.join(f'{r:>13s}' for r in REGIMES) + f"{'ALL/CLEAN':>12s}")
    rowsort = []
    for i, lab in enumerate(LABELS):
        ps = {r: pct(A[r]['gang_pow'][i], Q) for r in REGIMES}
        rowsort.append((ps['ALL'][99], lab, ps))
    for _k, lab, ps in sorted(rowsort, reverse=True):
        ratio = ps['ALL'][99] / max(ps['CLEAN'][99], 1e-9)
        print(f'{lab:30s}' + ''.join(f'{ps[r][99]:13.2f}' for r in REGIMES)
              + f'{ratio:12.2f}')


if __name__ == '__main__':
    main()
