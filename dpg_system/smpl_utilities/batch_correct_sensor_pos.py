"""Apply segment-based correction to every original take in a directory.

Each file gets its own independent fit, since the sensor mount may have
shifted between takes.  Skips files whose names contain `_<seg>fix` (already
corrected outputs) and skips a file if the matching output already exists
(use --overwrite to force re-fit).

Usage:
    python batch_correct.py <directory>
        [--segment pelvis|lhip|rhip] [--smooth 0.5] [--full]
        [--glob '*_b.npz'] [--overwrite]
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct_thigh_offset import (
    correct_take, pick_best_segment,
    L_HIP, R_HIP, PELVIS, SMOOTH_WINDOW_S,
)


SKIP_TAGS = (
    '_pelvisfix', '_lhipfix', '_rhipfix', '_thighfix',
    '_pelvisfullfix', '_lhipfullfix', '_rhipfullfix',
    '_corrected',     # exclude already-corrected outputs from external tools
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    p.add_argument('--segment', default='pelvis',
                   help='pelvis | lhip | rhip | auto[:s1,s2,...].  '
                        'auto picks per-file based on rotation diversity '
                        '(default candidates: pelvis,lhip,rhip).  Use e.g. '
                        '--segment auto:pelvis,lhip to constrain.')
    p.add_argument('--smooth', type=float, default=SMOOTH_WINDOW_S)
    p.add_argument('--full', action='store_true',
                   help='Use the 9-coef full-rotation model (captures axial '
                        'strap slip).  Outputs end in _<seg>fullfix.npz.')
    p.add_argument('--glob', default='*.npz',
                   help='Glob pattern for input files (default: *.npz). '
                        'Use *_b.npz to limit to beta-augmented takes only.')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-fit and overwrite existing fix files.')
    args = p.parse_args()

    seg_table = {'pelvis': PELVIS, 'lhip': L_HIP, 'rhip': R_HIP}
    inv_seg_table = {v: k for k, v in seg_table.items()}
    if args.segment.startswith('auto'):
        cand_str = (args.segment.split(':', 1)[1]
                    if ':' in args.segment else 'pelvis,lhip,rhip')
        auto_cands = [seg_table[s.strip()] for s in cand_str.split(',')]
        seg = None
    else:
        auto_cands = None
        seg = seg_table[args.segment]
    src_dir = Path(args.directory)
    inputs = sorted(p_ for p_ in src_dir.glob(args.glob)
                    if not any(tag in p_.stem for tag in SKIP_TAGS))

    print(f'Found {len(inputs)} candidate files in {src_dir}')
    print(f'Segment: {args.segment}   Model: {"full (9 coef)" if args.full else "rigid (3 coef)"}'
          f'   Smooth window: {args.smooth} s')
    print('=' * 78)

    summary = []
    for in_path in inputs:
        # Resolve segment per file (auto mode picks the most diverse)
        if auto_cands is not None:
            print(f'\n[auto] {in_path.name}')
            picked = pick_best_segment(in_path, auto_cands, verbose=True)
        else:
            picked = seg
        seg_label = inv_seg_table[picked]
        suffix = f'_{seg_label}{"full" if args.full else ""}fix'
        out_path = in_path.with_name(in_path.stem + suffix + '.npz')
        if out_path.exists() and not args.overwrite:
            print(f'[skip] {in_path.name}  ->  output exists ({out_path.name}). '
                  f'Use --overwrite to redo.')
            summary.append((in_path.name, f'skipped ({seg_label})', None))
            continue
        print(f'[run]  {in_path.name}  segment={seg_label}')
        try:
            res = correct_take(in_path, out_path, segment=picked,
                               smooth_window_s=args.smooth,
                               full=args.full, verbose=True)
            res['segment'] = seg_label
            summary.append((in_path.name, 'ok', res))
        except Exception as exc:
            print(f'  ERROR: {exc}')
            summary.append((in_path.name, f'error: {exc}', None))

    # Compact end-of-run summary
    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(f'{"file":<48} {"status":<8} {"seg":>7} {"||coefs||":>10}  {"c":>8}')
    for name, status, res in summary:
        if res is not None:
            import numpy as np
            d = res['delta']
            nm = float(np.linalg.norm(d))
            c = res.get('c', 0.0)
            sg = res.get('segment', '?')
            print(f'{name:<48} {status:<8} {sg:>7} {nm:>10.4f}  {c:+8.4f}')
        else:
            print(f'{name:<48} {status:<8}')


if __name__ == '__main__':
    main()
