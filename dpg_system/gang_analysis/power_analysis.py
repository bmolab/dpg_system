"""Mechanical power analysis over AMASS_Dynamic.

Power is the one quantity that needs both halves of the archive: tau . omega.
It matters here because every physical model in the synth is driven by a
(force, velocity) PAIR -- bow~ velocity+force, motor~ speed+load, rub~
velocity+force -- and torque alone is only the force half of each.

Three questions:

  1. Is power better CONDITIONED than torque? Torque's median crest factor is
     11x, which is what makes linear mapping fail. Power is a product of two
     skewed signals, so it could easily be worse -- that has to be measured,
     not assumed.

  2. What does the SIGN of power buy? tau.omega > 0 is torque doing work along
     the motion (generating, concentric); < 0 is absorbing it (braking,
     eccentric). Unlike |torque| this is a physically grounded signed
     distinction, and generating-vs-absorbing is a real perceptual difference
     in how effort reads.

  3. Gang power. For a gang whose weights define a generalized coordinate q
     with theta_j = w_j q, the generalized force is Q = sum w_j tau_j and the
     generalized velocity is qdot = (sum w_j omega_j)/||w||^2. Power along that
     coordinate is Q * qdot. Both sums are linear, so the existing compiled
     bank computes them -- run it twice, once on torque and once on omega.

Torque is left UNNORMALIZED here so power comes out in watts and can be
sanity-checked against physiology.
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
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}
TRIM_HEAD = 3

LOG_LO, LOG_HI, LOG_STEP = -4.0, 4.5, 0.05     # watts, 1e-4 .. ~30 kW
LOG_EDGES = np.arange(LOG_LO, LOG_HI + LOG_STEP, LOG_STEP)
N_LOG = len(LOG_EDGES) - 1


def log_hist(v):
    v = np.abs(np.asarray(v, np.float64).ravel())
    n_zero = int((v == 0).sum())
    v = v[v > 0]
    if v.size == 0:
        return np.zeros(N_LOG, np.int64), n_zero, 0, 0
    lg = np.log10(v)
    return (np.histogram(lg, bins=LOG_EDGES)[0].astype(np.int64), n_zero,
            int((lg < LOG_LO).sum()), int((lg >= LOG_HI).sum()))


def pct_from_hist(h, n_zero, n_under, n_over, qs):
    counts = np.concatenate([[n_zero + n_under], h, [n_over]])
    centers = np.concatenate([[0.0],
                              10 ** ((LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2),
                              [10 ** LOG_HI]])
    tot = counts.sum()
    if tot == 0:
        return {q: float('nan') for q in qs}
    cum = np.cumsum(counts) / tot
    return {q: float(centers[min(int(np.searchsorted(cum, q / 100.0)),
                                 len(centers) - 1)]) for q in qs}


def build():
    """Unnormalized banks for torque and omega, sharing one weight set."""
    specs, labels, wsq = [], [], []
    for preset in gc.preset_names():
        for side in (gc.sides_for(preset) or ['none']):
            spec = gc.spec_from_preset(preset, side=side, stream='total',
                                       normalize=False)
            labels.append(f'{preset}|{side}')
            wsq.append(sum(w * w for _j, _a, w in spec.terms))
            specs.append(spec)
    return gc.compile_specs(specs), labels, np.array(wsq)


def _init():
    global PROG, LABELS, WSQ
    PROG, LABELS, WSQ = build()


def pad(a, T):
    o = np.zeros((T, 24, 3), np.float32)
    o[:, :22] = a
    return o


def one(path):
    try:
        d = np.load(path, allow_pickle=True)
        tq = np.asarray(d['torque'], np.float64)[TRIM_HEAD:]
        w = np.asarray(d['angular_velocity'], np.float64)[TRIM_HEAD:]
    except Exception:
        return None
    if tq.shape[0] < 10 or tq.shape != w.shape:
        return None
    T = tq.shape[0]
    subset = os.path.relpath(path, os.path.join(ROOT, 'SMPL_H')).split(os.sep)[0]

    P = (tq * w).sum(axis=-1)                    # (T,22) watts, signed
    Ptot = P.sum(axis=1)                         # (T,)  whole body

    r = {'subset': subset, 'n': T,
         'joint': {}, 'gang': {}, 'body': {}}

    for j in range(22):
        pj = P[:, j]
        r['joint'][j] = {'hist': log_hist(pj), 'n_pos': int((pj > 0).sum()),
                         'n_neg': int((pj < 0).sum())}
    r['body'] = {'hist': log_hist(Ptot), 'n_pos': int((Ptot > 0).sum()),
                 'n_neg': int((Ptot < 0).sum()),
                 'gen_hist': log_hist(np.clip(Ptot, 0, None)),
                 'abs_hist': log_hist(np.clip(Ptot, None, 0))}

    # comparison of conditioning: |power| vs |torque| vs |omega|, per joint
    r['cmp'] = {
        'power': log_hist(P),
        'torque': log_hist(np.linalg.norm(tq, axis=-1)),
        'omega': log_hist(np.linalg.norm(w, axis=-1)),
    }

    # -- does the generating/absorbing sign survive integration? ----------
    # Instantaneous sign is ~50/50 for any oscillation, so it says little.
    # What matters is the timescale at which a net direction emerges:
    #   rectification(win) = |mean P over win| / mean |P| over win, in [0,1].
    # 1 = the window is entirely generating or entirely absorbing; 0 = it
    # cancels exactly. Also the mean run length of constant sign.
    fps = 60.0
    for k in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if k in d:
            fps = float(d[k])
            break
    sgn = np.sign(Ptot)
    flips = int((np.diff(sgn[sgn != 0]) != 0).sum())
    r['runlen_ms'] = (len(sgn) / max(flips, 1)) * (1000.0 / fps)
    r['rect'] = {}
    for ms in (50, 200, 1000):
        win = max(2, int(round(ms * fps / 1000.0)))
        if T < win * 2:
            continue
        n_w = T // win
        blk = Ptot[:n_w * win].reshape(n_w, win)
        num = np.abs(blk.mean(axis=1))
        den = np.abs(blk).mean(axis=1)
        good = den > 1e-9
        if good.any():
            r['rect'][ms] = (float(num[good].sum()), float(den[good].sum()),
                             int(good.sum()))

    # ---- gang power ------------------------------------------------------
    bt = {s: np.zeros((T, 24, 3), np.float32) for s in
          ('total', 'gravity', 'dynamic', 'passive')}
    bt['total'] = pad(tq, T)
    net_tau, _tt, _cc = PROG.evaluate(gc.stack_streams(bt))

    bw = {s: np.zeros((T, 24, 3), np.float32) for s in
          ('total', 'gravity', 'dynamic', 'passive')}
    bw['total'] = pad(w, T)
    net_w, _tt2, _cc2 = PROG.evaluate(gc.stack_streams(bw))

    gp = net_tau * net_w / np.maximum(WSQ, 1e-12)      # (T, gangs) watts
    for i in range(gp.shape[1]):
        col = gp[:, i]
        r['gang'][i] = {'hist': log_hist(col), 'n_pos': int((col > 0).sum()),
                        'n_neg': int((col < 0).sum())}
    return r


def blank(n_gangs):
    z = lambda: [np.zeros(N_LOG, np.int64), 0, 0, 0]
    return {
        'n': 0,
        'joint': {j: {'h': z(), 'pos': 0, 'neg': 0} for j in range(22)},
        'gang': {i: {'h': z(), 'pos': 0, 'neg': 0} for i in range(n_gangs)},
        'body': {'h': z(), 'gen': z(), 'abs': z(), 'pos': 0, 'neg': 0},
        'cmp': {k: z() for k in ('power', 'torque', 'omega')},
        'rect': {ms: [0.0, 0.0, 0] for ms in (50, 200, 1000)},
        'runlen': [],
    }


def add(dst, src):
    for k in range(4):
        pass
    for j, v in src['joint'].items():
        t = dst['joint'][j]
        for i in range(4):
            t['h'][i] = t['h'][i] + v['hist'][i]
        t['pos'] += v['n_pos']
        t['neg'] += v['n_neg']
    for i, v in src['gang'].items():
        t = dst['gang'][i]
        for k in range(4):
            t['h'][k] = t['h'][k] + v['hist'][k]
        t['pos'] += v['n_pos']
        t['neg'] += v['n_neg']
    b = dst['body']
    for k in range(4):
        b['h'][k] = b['h'][k] + src['body']['hist'][k]
        b['gen'][k] = b['gen'][k] + src['body']['gen_hist'][k]
        b['abs'][k] = b['abs'][k] + src['body']['abs_hist'][k]
    b['pos'] += src['body']['n_pos']
    b['neg'] += src['body']['n_neg']
    for ms, v in src.get('rect', {}).items():
        t = dst['rect'][ms]
        t[0] += v[0]; t[1] += v[1]; t[2] += v[2]
    if 'runlen_ms' in src:
        dst['runlen'].append(src['runlen_ms'])
    for key, h in src['cmp'].items():
        for k in range(4):
            dst['cmp'][key][k] = dst['cmp'][key][k] + h[k]
    dst['n'] += src['n']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    files = []
    for dp, _d, fns in os.walk(ROOT):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    if args.limit:
        files = [files[i] for i in
                 sorted(set(np.linspace(0, len(files) - 1, args.limit).astype(int)))]

    _init()
    acc = blank(len(LABELS))
    print(f'{len(files)} files, {args.workers} workers')
    t0 = time.perf_counter()
    done = bad = 0
    with Pool(args.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(one, files, chunksize=8):
            done += 1
            if r is None:
                bad += 1
            else:
                add(acc, r)
            if done % 2000 == 0:
                el = time.perf_counter() - t0
                print(f'  {done}/{len(files)} {el:.0f}s', flush=True)
    print(f'done {done} files ({bad} skipped), {acc["n"]:,} frames, '
          f'{time.perf_counter() - t0:.0f}s\n')

    Q = (50, 90, 99, 99.9)

    print('=' * 92)
    print('1. IS POWER BETTER CONDITIONED THAN TORQUE?')
    print('=' * 92)
    print(f"{'quantity':12s}{'p50':>12s}{'p90':>12s}{'p99':>12s}{'p99.9':>12s}"
          f"{'crest p99/p50':>15s}")
    for key, unit in (('torque', 'N.m'), ('omega', 'rad/s'), ('power', 'W')):
        p = pct_from_hist(*acc['cmp'][key], qs=Q)
        print(f"{key + ' (' + unit + ')':12s}{p[50]:12.4f}{p[90]:12.4f}"
              f"{p[99]:12.4f}{p[99.9]:12.4f}{p[99] / max(p[50], 1e-9):15.1f}")
    print('\n  (per-joint magnitudes, pooled over all 22 joints and 20.4M frames)')

    print('\n' + '=' * 92)
    print('2. WHOLE-BODY POWER, AND THE GENERATING / ABSORBING SPLIT')
    print('=' * 92)
    pb = pct_from_hist(*acc['body']['h'], qs=Q)
    pg = pct_from_hist(*acc['body']['gen'], qs=Q)
    pa = pct_from_hist(*acc['body']['abs'], qs=Q)
    tot = acc['body']['pos'] + acc['body']['neg']
    print(f"  |total power|   p50={pb[50]:8.1f} W  p90={pb[90]:8.1f}  "
          f"p99={pb[99]:8.1f}  p99.9={pb[99.9]:8.1f}")
    print(f"  generating      p99={pg[99]:8.1f} W        "
          f"absorbing p99={pa[99]:8.1f} W")
    print(f"  time generating (tau.omega>0): {acc['body']['pos'] / max(tot, 1):.1%}   "
          f"absorbing: {acc['body']['neg'] / max(tot, 1):.1%}")
    print(f"  crest (p99/p50) = {pb[99] / max(pb[50], 1e-9):.1f}")

    print(f"\n  mean run length of constant sign: "
          f"{np.median(acc['runlen']):.0f} ms (median over files)")
    print('  rectification |mean P| / mean |P| over a window '
          '(1 = window is all one direction):')
    for ms in (50, 200, 1000):
        num, den, n = acc['rect'][ms]
        if n:
            print(f'     {ms:5d} ms : {num / max(den, 1e-9):.3f}   ({n:,} windows)')

    print('\n' + '=' * 92)
    print('3. PER-JOINT POWER  (watts, and how much of the time each absorbs)')
    print('=' * 92)
    print(f"{'joint':16s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'p99.9':>10s}"
          f"{'crest':>7s}{'absorb%':>9s}")
    jr = []
    for j in range(22):
        p = pct_from_hist(*acc['joint'][j]['h'], qs=Q)
        n = acc['joint'][j]['pos'] + acc['joint'][j]['neg']
        jr.append((p[99], NAMES[j], p, acc['joint'][j]['neg'] / max(n, 1)))
    for _k, nm, p, ab in sorted(jr, reverse=True):
        print(f"{nm:16s}{p[50]:9.3f}{p[90]:9.3f}{p[99]:9.2f}{p[99.9]:10.2f}"
              f"{p[99] / max(p[50], 1e-9):7.1f}{ab:9.1%}")

    print('\n' + '=' * 92)
    print('4. GANG POWER   Q*qdot along each gang\'s generalized coordinate')
    print('=' * 92)
    print(f"{'gang':32s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'crest':>7s}{'absorb%':>9s}")
    gr = []
    for i, lab in enumerate(LABELS):
        p = pct_from_hist(*acc['gang'][i]['h'], qs=Q)
        n = acc['gang'][i]['pos'] + acc['gang'][i]['neg']
        gr.append((p[99], lab, p, acc['gang'][i]['neg'] / max(n, 1)))
    for _k, lab, p, ab in sorted(gr, reverse=True):
        print(f"{lab:32s}{p[50]:9.3f}{p[90]:9.3f}{p[99]:9.2f}"
              f"{p[99] / max(p[50], 1e-9):7.1f}{ab:9.1%}")

    if args.out:
        json.dump({'frames': acc['n']}, open(args.out, 'w'))


if __name__ == '__main__':
    main()
