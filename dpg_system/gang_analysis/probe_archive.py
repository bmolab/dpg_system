"""Second-pass probe of AMASS_Torque: what the two arrays are, and which
channels are structurally dead.

Checks:
  1. Is combined_effort just torque normalized per joint-axis? If so, recover
     the implied max-torque denominator and compare to gang_core's table.
  2. Which of the 22x3 channels are identically zero, and is that structural
     (same channels every file) or motion-dependent?
  3. Frame rates present, total frame count, subset coverage.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import dpg_system.gang_core as gc

ROOT = '/Users/drokeby/dpg_system/AMASS_Torque'


def walk(root, limit=None):
    out = []
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.npz'):
                out.append(os.path.join(dirpath, fn))
                if limit and len(out) >= limit:
                    return sorted(out)
    return sorted(out)


def main():
    n_probe = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = walk(ROOT)
    print(f'{len(files)} files under {ROOT}')

    # subset coverage
    subsets = Counter()
    for f in files:
        rel = os.path.relpath(f, os.path.join(ROOT, 'SMPL_H'))
        subsets[rel.split(os.sep)[0]] += 1
    print(f'\n{len(subsets)} subsets:')
    for k, v in sorted(subsets.items()):
        print(f'   {k:26s} {v:6d}')

    local = '/Users/drokeby/dpg_system/AMASS'
    if os.path.isdir(local):
        loc = {d for d in os.listdir(local)
               if os.path.isdir(os.path.join(local, d))}
        print(f'\nlocal AMASS subsets not in torque archive: '
              f'{sorted(loc - set(subsets)) or "none"}')
        print(f'torque subsets not in local AMASS: '
              f'{sorted(set(subsets) - loc) or "none"}')

    # sample spread across the corpus, not just the first files
    idx = np.linspace(0, len(files) - 1, min(n_probe, len(files))).astype(int)
    probe = [files[i] for i in idx]

    print(f'\n--- probing {len(probe)} files spread across the corpus ---')

    zero_always = None          # channels zero in EVERY probed file
    zero_ever = None            # channels zero in ANY probed file
    frac_zero_acc = np.zeros((22, 3))
    fps_counter = Counter()
    total_frames = 0
    ratio_samples = []
    n_ok = 0

    for path in probe:
        try:
            d = np.load(path, allow_pickle=True)
            tq = np.asarray(d['torque'], dtype=np.float64)
            ce = np.asarray(d['combined_effort'], dtype=np.float64)
        except Exception as e:
            print(f'  !! {os.path.basename(path)}: {e}')
            continue
        n_ok += 1
        total_frames += tq.shape[0]
        for key in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
            if key in d:
                fps_counter[float(d[key])] += 1
                break

        zmask = np.all(tq == 0.0, axis=0)          # (22,3) dead this file
        zero_always = zmask if zero_always is None else (zero_always & zmask)
        zero_ever = zmask if zero_ever is None else (zero_ever | zmask)
        frac_zero_acc += (tq == 0.0).mean(axis=0)

        # implied denominator, where torque is non-trivial
        live = np.abs(tq) > 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(live, tq / np.where(ce == 0, np.nan, ce), np.nan)
        ratio_samples.append(ratio)

    print(f'  read {n_ok} files, {total_frames} frames')
    print(f'  frame rates: {dict(fps_counter)}')

    frac_zero = frac_zero_acc / max(n_ok, 1)
    print(f'\nmean fraction of zero samples per channel: '
          f'{frac_zero.mean():.1%}')
    print(f'channels dead in EVERY probed file: {int(zero_always.sum())} / 66')
    names = {v: k for k, v in gc.JOINT_INDEX.items()}
    if zero_always.any():
        for j, a in zip(*np.where(zero_always)):
            print(f'    {names[j]:16s} axis {a}')
    print(f'channels dead in AT LEAST ONE probed file: {int(zero_ever.sum())} / 66')

    print('\nper-joint mean zero fraction (torque):')
    for j in range(22):
        bar = ''.join('#' if frac_zero[j, a] > 0.5 else
                      ('+' if frac_zero[j, a] > 0.05 else '.') for a in range(3))
        print(f'   {j:2d} {names[j]:16s} [{bar}]  '
              f'{frac_zero[j, 0]:.2f} {frac_zero[j, 1]:.2f} {frac_zero[j, 2]:.2f}')

    # --- is combined_effort torque / denom, with denom constant per channel? --
    allr = np.concatenate([r.reshape(-1, 22, 3) for r in ratio_samples], axis=0)
    print('\n--- torque / combined_effort, per channel ---')
    print('(constant within a channel => combined_effort is normalized torque)')
    med = np.nanmedian(allr.reshape(-1, 22, 3), axis=0)
    iqr = (np.nanpercentile(allr.reshape(-1, 22, 3), 75, axis=0)
           - np.nanpercentile(allr.reshape(-1, 22, 3), 25, axis=0))
    mt = gc.max_torque_array()
    print(f'{"joint":16s} {"axis":4s} {"median ratio":>13s} {"IQR":>10s} '
          f'{"gang max_torque":>16s}')
    for j in range(22):
        for a in range(3):
            if np.isnan(med[j, a]):
                continue
            print(f'{names[j]:16s} {a:<4d} {med[j, a]:13.4f} {iqr[j, a]:10.5f} '
                  f'{mt[j, a]:16.3f}')
        if j > 6:
            break
    rel_iqr = np.nanmean(iqr / np.abs(np.where(med == 0, np.nan, med)))
    print(f'\nmean relative IQR of the ratio: {rel_iqr:.6f}  '
          f'(~0 => strictly a per-channel constant)')


if __name__ == '__main__':
    main()
