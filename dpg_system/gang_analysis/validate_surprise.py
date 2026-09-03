"""Does whitened surprise measure what it claims to?

Four questions, each with a way of being answered no:

  1. CALIBRATION -- on held-out corpus frames, does percentile() come out
     uniform? If not, the prior does not describe the data it was built from.

  2. NOISE SENSITIVITY -- the stated caution is that whitening amplifies the
     low-variance directions where mocap noise also lives. Comparing frames the
     noise work called clean against frames it called problematic measures
     exactly how much of the signal is noise rather than movement.

  3. DISCRIMINATION -- does it separate dance from walking? If DanceDB and
     KIT score the same, it is not detecting unusual movement.

  4. DECOMPOSITION -- how much surprise lies along the declared gang span, and
     how much is orthogonal to all 42 of them? The orthogonal part is movement
     unusual in a way the current vocabulary cannot express.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpg_system.gang_core as gc
from surprise_core import TorquePrior, gang_weight_vectors

ROOT = '/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H'
TRIM = 3
FILE_KEY = {'total': 'torque', 'gravity': 'torques_grav_vec',
            'dynamic': 'torques_dyn_vec'}


def masks(rel, T, index):
    info = index.get(rel)
    if info is None:
        return None, None
    keep = np.zeros(T, bool)
    for a, b in info.get('keep', []):
        keep[max(0, min(T - 1, a)):max(0, min(T - 1, b)) + 1] = True
    drop = np.zeros(T, bool)
    for a, b in info.get('drop', []):
        drop[max(0, min(T - 1, a)):max(0, min(T - 1, b)) + 1] = True
    keep[:TRIM] = False
    drop[:TRIM] = False
    return keep, drop, info['cls']


def main():
    prior = TorquePrior(sys.argv[1] if len(sys.argv) > 1 else 'torque_prior.npz')
    index = json.load(open('noise_index.json'))
    print(f'prior: stream={prior.stream}  {prior.n_live} live channels  '
          f'{prior.n_frames:,} frames  ridge={prior.ridge}')

    key = FILE_KEY[prior.stream]
    print(f'reading archive stream "{key}"')
    labels, W = gang_weight_vectors(gc, prior.stream)
    basis = prior.gang_basis(W)
    print(f'gang span: {len(labels)} gangs -> rank {basis.shape[1]} '
          f'of {prior.n_live} whitened dimensions '
          f'({basis.shape[1] / prior.n_live:.0%} of the space)\n')

    files = []
    for dp, _d, fns in os.walk(ROOT):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    rng = np.random.default_rng(0)
    sample = [files[i] for i in
              sorted(rng.choice(len(files), size=min(600, len(files)),
                                replace=False))]

    by_cls = defaultdict(list)
    by_sub = defaultdict(list)
    pct_all = []
    dec = defaultdict(list)
    for p in sample:
        rel = os.path.relpath(p, ROOT)
        try:
            d = np.load(p, allow_pickle=True)
            X = np.asarray(d[key], np.float64).reshape(
                d[key].shape[0], -1)
        except Exception:
            continue
        mk = masks(rel, X.shape[0], index)
        if mk[0] is None:
            continue
        keep, drop, cls = mk
        sub = rel.split(os.sep)[0]

        if keep.sum() >= 30:
            dd = prior.surprise(X[keep])
            by_cls[cls].append(np.percentile(dd, [50, 90, 99]))
            by_sub[sub].append(np.percentile(dd, [50, 90, 99]))
            pct_all.append(prior.percentile(dd))
            dt, dg, df = prior.decompose(X[keep], basis)
            dec['tot'].append(np.median(dt))
            dec['gang'].append(np.median(dg))
            dec['free'].append(np.median(df))
            dec['frac_free'].append(np.median(df ** 2 / np.maximum(dt ** 2, 1e-12)))
        # frames the noise work excised, i.e. known-bad
        if drop.sum() >= 30:
            dd = prior.surprise(X[drop])
            by_cls['<excised>'].append(np.percentile(dd, [50, 90, 99]))

    # ---- 1. calibration ------------------------------------------------
    print('=' * 78)
    print('1. CALIBRATION  -- percentile() on corpus frames should be uniform')
    print('=' * 78)
    allp = np.concatenate(pct_all)
    print('   decile occupancy (0.100 each if perfectly uniform):')
    hist, _ = np.histogram(allp, bins=np.linspace(0, 1, 11))
    print('   ' + '  '.join(f'{h / len(allp):.3f}' for h in hist))
    print(f'   mean={allp.mean():.3f} (expect 0.500)   '
          f'median={np.median(allp):.3f} (expect 0.500)')

    # ---- 2. noise sensitivity -----------------------------------------
    print('\n' + '=' * 78)
    print('2. NOISE SENSITIVITY  -- clean vs problematic vs excised frames')
    print('=' * 78)
    print(f'   {"population":16s}{"files":>7s}{"d p50":>9s}{"d p90":>9s}{"d p99":>9s}')
    base = None
    for cls in ('clean', 'moderate', 'problematic', '<excised>'):
        if cls not in by_cls:
            continue
        a = np.array(by_cls[cls])
        med = np.median(a, axis=0)
        if cls == 'clean':
            base = med
        rel = f'   ({med[0] / base[0]:.2f}x clean)' if base is not None else ''
        print(f'   {cls:16s}{len(a):7d}{med[0]:9.2f}{med[1]:9.2f}{med[2]:9.2f}{rel}')
    print('   (excised = frames the noise work cut. If these do not score far')
    print('    above clean, surprise is not merely detecting mocap noise.)')

    # ---- 3. discrimination --------------------------------------------
    print('\n' + '=' * 78)
    print('3. DISCRIMINATION  -- by subset, on clean_segments only')
    print('=' * 78)
    rows = [(np.median(np.array(v), axis=0)[0], k, len(v))
            for k, v in by_sub.items() if len(v) >= 4]
    rows.sort(reverse=True)
    print(f'   {"subset":24s}{"files":>7s}{"d p50":>9s}')
    for m, k, n in rows[:9]:
        print(f'   {k:24s}{n:7d}{m:9.2f}')
    print('   ...')
    for m, k, n in rows[-5:]:
        print(f'   {k:24s}{n:7d}{m:9.2f}')

    # ---- 4. decomposition ----------------------------------------------
    print('\n' + '=' * 78)
    print('4. DECOMPOSITION  -- how much surprise can the 42 gangs express?')
    print('=' * 78)
    print(f'   median d_total = {np.median(dec["tot"]):.3f}')
    print(f'   median d_gang  = {np.median(dec["gang"]):.3f}   '
          f'(along the declared gang span)')
    print(f'   median d_free  = {np.median(dec["free"]):.3f}   '
          f'(orthogonal to ALL 42 gangs)')
    print(f'   median share of surprise the gangs CANNOT express: '
          f'{np.median(dec["frac_free"]):.1%}')
    print(f'   gang span covers {basis.shape[1]}/{prior.n_live} dimensions '
          f'= {basis.shape[1] / prior.n_live:.0%}; a random subspace of that '
          f'rank would capture ~{1 - basis.shape[1] / prior.n_live:.0%} free')


if __name__ == '__main__':
    main()
