"""Characterize the AMASS_Dynamic torque field.

Descriptive, not prescriptive. Two analyses:

  A. CONDITIONING -- what range and time-scale does each gang signal actually
     occupy? Any mapping (direct, inverted, tangential) needs this; none of it
     assumes a mapping.

  B. STRUCTURE -- the empirical co-variation of the 66 torque channels, and its
     eigenvectors. The principal components ARE data-derived gangs. The 15
     hand-written presets are then one hypothesis to test against the data
     rather than the thing assumed.

Aggregation is exact, not sampled: log-magnitude histograms and cross-product
accumulators both combine across files without approximation, so the corpus
result is identical to what a single pass over all frames would give.

Usage:
    python characterize.py --out DIR [--workers 8] [--limit N] [--subset NAME]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import dpg_system.gang_core as gc

ROOT = '/Users/drokeby/dpg_system/AMASS_Dynamic'
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}
AXES = ('x', 'y', 'z')

# Streams we characterize. 'passive' is excluded: 90.9% of its samples are
# zero and only 4 of 42 gangs ever see it non-zero, so its statistics are
# dominated by the zeros and tell us nothing about mapping range.
STREAMS = ('total', 'gravity', 'dynamic')
FILE_KEY = {'total': 'torque', 'gravity': 'torques_grav_vec',
            'dynamic': 'torques_dyn_vec', 'passive': 'torques_passive_vec'}

# Log-magnitude histogram: 10^-6 .. 10^1 in 0.05-dex bins. Magnitudes below
# the floor land in an underflow counter, exact zeros in their own counter,
# so nothing is silently dropped and percentiles stay exact to bin width.
LOG_LO, LOG_HI, LOG_STEP = -6.0, 1.0, 0.05
LOG_EDGES = np.arange(LOG_LO, LOG_HI + LOG_STEP, LOG_STEP)
N_LOG = len(LOG_EDGES) - 1

# Coherence is bounded [0,1]; linear bins are the natural choice.
N_COH = 200
COH_EDGES = np.linspace(0.0, 1.0, N_COH + 1)

# Frames to trim from the head of every file: derivative start-up. Frame 0 of
# angular_velocity is identically zero by construction, and the first couple of
# frames of any differentiated quantity are unreliable.
TRIM_HEAD = 3


def build_programs():
    """One compiled bank per stream, plus the gang label list."""
    specs, labels = [], []
    for preset in gc.preset_names():
        for side in (gc.sides_for(preset) or ['none']):
            labels.append(f'{preset}|{side}')
            specs.append((preset, side))
    programs = {}
    for stream in STREAMS:
        programs[stream] = gc.compile_specs(
            [gc.spec_from_preset(p, side=s, stream=stream) for p, s in specs])
    return programs, labels


def channel_labels():
    return [f'{NAMES[j]}.{AXES[a]}' for j in range(22) for a in range(3)]


def pad24(arr, T):
    out = np.zeros((T, 24, 3), np.float32)
    out[:, :22] = arr
    return out


def log_hist(values):
    """Histogram of log10|x|, with zero and underflow counted separately."""
    v = np.abs(np.asarray(values, np.float64).ravel())
    n_zero = int((v == 0).sum())
    v = v[v > 0]
    if v.size == 0:
        return np.zeros(N_LOG, np.int64), n_zero, 0, 0
    lg = np.log10(v)
    n_under = int((lg < LOG_LO).sum())
    n_over = int((lg >= LOG_HI).sum())
    h, _ = np.histogram(lg, bins=LOG_EDGES)
    return h.astype(np.int64), n_zero, n_under, n_over


def percentile_from_log_hist(h, n_zero, n_under, n_over, qs):
    """Percentiles of |x| recovered from the log histogram."""
    counts = np.concatenate([[n_zero + n_under], h, [n_over]])
    centers = np.concatenate([[0.0],
                              10 ** ((LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2),
                              [10 ** LOG_HI]])
    total = counts.sum()
    if total == 0:
        return {q: float('nan') for q in qs}
    cum = np.cumsum(counts) / total
    return {q: float(centers[int(np.searchsorted(cum, q / 100.0))])
            for q in qs}


def process_file(path):
    """All per-file accumulators. Returns a dict, or None if unusable."""
    try:
        d = np.load(path, allow_pickle=True)
        net_t = np.asarray(d['torque'], np.float32)
    except Exception:
        return None
    T = net_t.shape[0]
    if T <= TRIM_HEAD + 10:
        return None

    fps = 60.0
    for k in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if k in d:
            fps = float(d[k])
            break
    dt = 1.0 / fps

    rel = os.path.relpath(path, os.path.join(ROOT, 'SMPL_H'))
    subset = rel.split(os.sep)[0]

    # SEG regime: the noise work's own clean_segments, applied uniformly to
    # every file regardless of its file-level classification, minus the
    # surgery/excision drop zones. See CHARACTERIZATION_RESULTS.md section 8.
    sl = np.zeros(T, bool)
    if REGIME == 'ALL':
        sl[TRIM_HEAD:] = True
    else:
        info = INDEX.get(rel)
        if info is None:
            return None
        for a0, b0 in info.get('keep', []):
            sl[max(0, min(T - 1, a0)):max(0, min(T - 1, b0)) + 1] = True
        for a0, b0 in info.get('drop', []):
            sl[max(0, min(T - 1, a0)):max(0, min(T - 1, b0)) + 1] = False
        sl[:TRIM_HEAD] = False
    n_keep = int(sl.sum())
    if n_keep < 30:
        return None

    out = {'path': rel, 'subset': subset, 'frames': n_keep, 'fps': fps,
           'chan': {}, 'gang': {}, 'file_gang': {}}

    programs, _labels = PROGRAMS, LABELS

    # ---- channel-level: cross products for correlation, per stream --------
    raw = {}
    for stream in STREAMS:
        key = FILE_KEY[stream]
        if key not in d:
            continue
        a = np.asarray(d[key], np.float64)[sl]           # (T,22,3)
        raw[stream] = a
        X = a.reshape(a.shape[0], -1)                     # (T,66)
        out['chan'][stream] = {
            'n': X.shape[0],
            'sum': X.sum(axis=0),
            'xtx': X.T @ X,
        }

    if 'angular_velocity' in d:
        w = np.asarray(d['angular_velocity'], np.float64)[sl]
        Xw = w.reshape(w.shape[0], -1)
        out['chan']['angvel'] = {'n': Xw.shape[0], 'sum': Xw.sum(axis=0),
                                 'xtx': Xw.T @ Xw}
        # mechanical power: tau . omega per joint, summed -- the honest scalar
        # for "rate of work", which does not exist without omega
        if 'total' in raw:
            pw = (raw['total'] * w).sum(axis=-1)          # (T,22)
            out['power'] = {'p50': float(np.percentile(np.abs(pw), 50)),
                            'p90': float(np.percentile(np.abs(pw), 90)),
                            'p99': float(np.percentile(np.abs(pw), 99)),
                            'total_p99': float(np.percentile(
                                np.abs(pw).sum(axis=1), 99))}

    # ---- gang-level -------------------------------------------------------
    Tn = None
    bundle_cache = {}
    for stream in STREAMS:
        if stream not in raw:
            continue
        Tn = raw[stream].shape[0]
        bundle = {s: np.zeros((Tn, 24, 3), np.float32) for s in
                  ('total', 'gravity', 'dynamic', 'passive')}
        bundle[stream] = pad24(raw[stream], Tn)
        bundle_cache[stream] = gc.stack_streams(bundle)

    for stream, stacked in bundle_cache.items():
        net, tot, coh = programs[stream].evaluate(stacked)
        g = {}
        for i in range(net.shape[1]):
            hn, zn, un, on = log_hist(net[:, i])
            ht, zt, ut, ot = log_hist(tot[:, i])
            # coherence only where the gang is actually doing something --
            # |net|/total is 0/0 in stillness and would otherwise pile mass
            # at zero and make a live gang look incoherent
            active = tot[:, i] > 1e-6
            hc, _ = np.histogram(coh[active, i], bins=COH_EDGES)
            g[i] = {'net': (hn, zn, un, on), 'tot': (ht, zt, ut, ot),
                    'coh': hc.astype(np.int64), 'n_active': int(active.sum()),
                    'n': int(net.shape[0])}
        out['gang'][stream] = g

        # per-file summaries, kept separable so subsets can be sliced later
        with np.errstate(invalid='ignore'):
            d_net = np.diff(net, axis=0) / dt
        fg = {
            'net_p50': np.percentile(np.abs(net), 50, axis=0),
            'net_p90': np.percentile(np.abs(net), 90, axis=0),
            'net_p99': np.percentile(np.abs(net), 99, axis=0),
            'tot_p99': np.percentile(tot, 99, axis=0),
            'coh_med': np.array([np.median(coh[tot[:, i] > 1e-6, i])
                                 if (tot[:, i] > 1e-6).any() else np.nan
                                 for i in range(net.shape[1])]),
            # time-scale: median rate of change relative to the signal's own
            # spread, in 1/s -- how fast this gang moves through its range
            'rate_hz': np.percentile(np.abs(d_net), 50, axis=0)
                       / (np.percentile(np.abs(net), 90, axis=0) + 1e-12),
        }
        out['file_gang'][stream] = fg

    return out


def _init(idx_path=None, regime='ALL'):
    # spawn start method on macOS: workers re-import, so state must be passed
    global PROGRAMS, LABELS, INDEX, REGIME
    PROGRAMS, LABELS = build_programs()
    REGIME = regime
    INDEX = json.load(open(idx_path)) if idx_path else {}


def merge(dst, src):
    """Accumulate one file's result into the running totals."""
    sub = src['subset']
    dst['files'].append({'path': src['path'], 'subset': sub,
                         'frames': src['frames'], 'fps': src['fps']})
    for stream, c in src['chan'].items():
        for scope in ('__all__', sub):
            a = dst['chan'][scope][stream]
            a['n'] += c['n']
            a['sum'] += c['sum']
            a['xtx'] += c['xtx']
    for stream, g in src['gang'].items():
        for i, rec in g.items():
            a = dst['gang'][stream][i]
            for k in ('net', 'tot'):
                a[k][0] += rec[k][0]
                a[k][1] += rec[k][1]
                a[k][2] += rec[k][2]
                a[k][3] += rec[k][3]
            a['coh'] += rec['coh']
            a['n_active'] += rec['n_active']
            a['n'] += rec['n']
    for stream, fg in src['file_gang'].items():
        dst['file_gang'][stream].append(fg)
        dst['file_gang_meta'][stream].append((src['path'], sub, src['frames']))
    if 'power' in src:
        dst['power'].append(src['power'])


def blank_accumulators(subsets):
    chan = {}
    for scope in ['__all__'] + list(subsets):
        chan[scope] = {s: {'n': 0, 'sum': np.zeros(66), 'xtx': np.zeros((66, 66))}
                       for s in list(STREAMS) + ['angvel']}
    gang = {s: {i: {'net': [np.zeros(N_LOG, np.int64), 0, 0, 0],
                    'tot': [np.zeros(N_LOG, np.int64), 0, 0, 0],
                    'coh': np.zeros(N_COH, np.int64),
                    'n_active': 0, 'n': 0}
                for i in range(42)} for s in STREAMS}
    return {'files': [], 'chan': chan, 'gang': gang,
            'file_gang': {s: [] for s in STREAMS},
            'file_gang_meta': {s: [] for s in STREAMS},
            'power': []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--subset', default=None)
    ap.add_argument('--index', default=None)
    ap.add_argument('--regime', default='ALL', choices=('ALL', 'SEG'))
    args = ap.parse_args()

    files = []
    for dp, _d, fns in os.walk(ROOT):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    if args.subset:
        files = [f for f in files if f'/SMPL_H/{args.subset}/' in f]
    if args.limit:
        idx = np.linspace(0, len(files) - 1, args.limit).astype(int)
        files = [files[i] for i in sorted(set(idx))]

    subsets = sorted({os.path.relpath(f, os.path.join(ROOT, 'SMPL_H')).split(os.sep)[0]
                      for f in files})
    print(f'{len(files)} files, {len(subsets)} subsets, {args.workers} workers')

    os.makedirs(args.out, exist_ok=True)
    acc = blank_accumulators(subsets)
    idx_path = os.path.abspath(args.index) if args.index else None
    _init(idx_path, args.regime)
    print(f'regime = {args.regime}')

    t0 = time.perf_counter()
    done = n_bad = 0
    with Pool(args.workers, initializer=_init,
              initargs=(idx_path, args.regime)) as pool:
        for res in pool.imap_unordered(process_file, files, chunksize=8):
            done += 1
            if res is None:
                n_bad += 1
            else:
                merge(acc, res)
            if done % 500 == 0:
                el = time.perf_counter() - t0
                print(f'  {done}/{len(files)}  {el:6.1f}s  '
                      f'eta {el / done * (len(files) - done):6.1f}s  '
                      f'skipped={n_bad}', flush=True)

    el = time.perf_counter() - t0
    total_frames = sum(f['frames'] for f in acc['files'])
    print(f'done: {done} files ({n_bad} skipped), {total_frames} frames, {el:.1f}s')

    save(args.out, acc, subsets, total_frames)


def save(out, acc, subsets, total_frames):
    _programs, labels = build_programs()
    chans = channel_labels()

    # ---- A. conditioning table -------------------------------------------
    rows = []
    for stream in STREAMS:
        for i, lab in enumerate(labels):
            g = acc['gang'][stream][i]
            pn = percentile_from_log_hist(*g['net'], qs=(50, 90, 99, 99.9))
            pt = percentile_from_log_hist(*g['tot'], qs=(50, 90, 99, 99.9))
            ctot = g['coh'].sum()
            if ctot:
                ccum = np.cumsum(g['coh']) / ctot
                cc = {q: float(COH_EDGES[:-1][int(np.searchsorted(ccum, q / 100))]
                               if np.searchsorted(ccum, q / 100) < N_COH else 1.0)
                      for q in (10, 50, 90)}
            else:
                cc = {10: float('nan'), 50: float('nan'), 90: float('nan')}
            rows.append({
                'stream': stream, 'gang': lab,
                'net_p50': pn[50], 'net_p90': pn[90], 'net_p99': pn[99],
                'net_p999': pn[99.9],
                'tot_p50': pt[50], 'tot_p90': pt[90], 'tot_p99': pt[99],
                'coh_p10': cc[10], 'coh_p50': cc[50], 'coh_p90': cc[90],
                'active_frac': g['n_active'] / max(g['n'], 1),
                'dead_frac': g['net'][1] / max(g['n'], 1),
            })
    with open(os.path.join(out, 'conditioning.json'), 'w') as fh:
        json.dump({'total_frames': total_frames, 'rows': rows}, fh, indent=1)

    # ---- B. correlation structure ----------------------------------------
    corr = {}
    for scope, per_stream in acc['chan'].items():
        for stream, a in per_stream.items():
            if a['n'] < 100:
                continue
            n, s, xtx = a['n'], a['sum'], a['xtx']
            cov = xtx / n - np.outer(s / n, s / n)
            sd = np.sqrt(np.clip(np.diag(cov), 0, None))
            live = sd > 1e-12
            C = np.full((66, 66), np.nan)
            idx = np.where(live)[0]
            sub = cov[np.ix_(idx, idx)] / np.outer(sd[idx], sd[idx])
            C[np.ix_(idx, idx)] = sub
            corr[f'{scope}|{stream}'] = C.astype(np.float32)

    np.savez_compressed(
        os.path.join(out, 'correlation.npz'),
        channels=np.array(chans),
        **{k.replace('|', '__'): v for k, v in corr.items()})

    # ---- per-file records, kept separable --------------------------------
    meta = acc['file_gang_meta']
    perfile = {}
    for stream in STREAMS:
        if not acc['file_gang'][stream]:
            continue
        for field in ('net_p50', 'net_p90', 'net_p99', 'tot_p99',
                      'coh_med', 'rate_hz'):
            perfile[f'{stream}__{field}'] = np.array(
                [fg[field] for fg in acc['file_gang'][stream]], np.float32)
        perfile[f'{stream}__paths'] = np.array([m[0] for m in meta[stream]])
        perfile[f'{stream}__subset'] = np.array([m[1] for m in meta[stream]])
        perfile[f'{stream}__frames'] = np.array([m[2] for m in meta[stream]])
    np.savez_compressed(os.path.join(out, 'per_file.npz'),
                        gangs=np.array(labels), **perfile)

    if acc['power']:
        with open(os.path.join(out, 'power.json'), 'w') as fh:
            json.dump({k: float(np.median([p[k] for p in acc['power']]))
                       for k in acc['power'][0]}, fh, indent=1)

    print(f'wrote conditioning.json, correlation.npz, per_file.npz to {out}')


if __name__ == '__main__':
    main()
