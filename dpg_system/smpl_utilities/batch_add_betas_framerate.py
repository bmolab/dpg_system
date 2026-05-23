"""Batch-add `betas` (and optionally `gender` / `mocap_framerate`) to a folder
of npz mocap files.

For each input `take.npz` writes `take_beta.npz` in the same folder, copying
all original arrays and overwriting (or adding) `betas`. `gender` and
`mocap_framerate` are only written when explicitly passed on the CLI; any
pre-existing values in the file are otherwise left untouched. Files whose
name already ends in `_beta` are skipped so re-running is idempotent.

Examples:
    python batch_add_betas_framerate.py /path/to/folder \\
        --betas 0.1,-0.2,0,0,0,0,0,0,0,0
    python batch_add_betas_framerate.py /path/to/folder --framerate 120
    python batch_add_betas_framerate.py /path/to/folder --framerate 60 \\
        --betas-file shape.npy --recursive
"""

import argparse
import json
import os
import sys

import numpy as np


def _extract_betas(obj):
    # Unwrap 0-d object arrays (np.load on pickled dict returns one).
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, dict):
        if 'betas' not in obj:
            raise ValueError(f'dict has no "betas" key (keys: {list(obj)})')
        obj = obj['betas']
    return np.asarray(obj, dtype=np.float32).flatten()


def parse_betas(spec, betas_file):
    if betas_file is not None:
        ext = os.path.splitext(betas_file)[1].lower()
        if ext == '.npy':
            obj = np.load(betas_file, allow_pickle=True)
            return _extract_betas(obj)
        if ext == '.npz':
            with np.load(betas_file, allow_pickle=True) as src:
                if 'betas' not in src.files:
                    raise ValueError(
                        f'{betas_file} has no "betas" key (keys: {src.files})')
                return _extract_betas(src['betas'])
        if ext == '.json':
            with open(betas_file) as f:
                return _extract_betas(json.load(f))
        raise ValueError(f'unsupported betas file extension: {ext}')
    if spec is not None:
        parts = [p.strip() for p in spec.split(',') if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=np.float32)
    return np.zeros(10, dtype=np.float32)


def iter_npz(root, recursive):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith('.npz'):
                    yield os.path.join(dirpath, name)
    else:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith('.npz'):
                yield path


def process_file(path, betas, framerate, gender):
    stem, ext = os.path.splitext(path)
    if stem.endswith('_beta'):
        return 'skip (already _beta)'
    out_path = stem + '_beta' + ext
    with np.load(path, allow_pickle=True) as src:
        data = {k: src[k] for k in src.files}
    data['betas'] = betas
    if gender is not None:
        data['gender'] = gender
    if framerate is not None:
        data['mocap_framerate'] = np.float64(framerate)
    np.savez(out_path, **data)
    return f'wrote {os.path.basename(out_path)}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('folder', help='folder containing .npz mocap files')
    ap.add_argument('--framerate', type=float, default=None,
                    help='mocap_framerate to write (e.g. 60, 120); '
                         'omit to leave any existing value untouched')
    ap.add_argument('--gender', choices=['neutral', 'male', 'female'],
                    default=None,
                    help='gender string; omit to leave any existing value untouched')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--betas', help='comma-separated floats, e.g. "0,0,0,..."')
    g.add_argument('--betas-file',
                   help='path to .npy/.npz/.json with a betas vector '
                        '(npy/npz may be a dict with a "betas" key)')
    ap.add_argument('--recursive', action='store_true',
                    help='walk subfolders too')
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f'not a directory: {args.folder}', file=sys.stderr)
        sys.exit(1)

    betas = parse_betas(args.betas, args.betas_file)
    print(f'betas (len={len(betas)}): {betas.tolist()}')
    print(f'gender: {args.gender if args.gender is not None else "(unchanged)"}')
    print(f'mocap_framerate: {args.framerate if args.framerate is not None else "(unchanged)"}')

    n_written = n_skipped = n_failed = 0
    for path in iter_npz(args.folder, args.recursive):
        try:
            result = process_file(path, betas, args.framerate, args.gender)
        except Exception as e:
            n_failed += 1
            print(f'  FAIL {path}: {e}')
            continue
        if result.startswith('skip'):
            n_skipped += 1
        else:
            n_written += 1
        print(f'  {path}: {result}')

    print(f'\ndone — wrote {n_written}, skipped {n_skipped}, failed {n_failed}')


if __name__ == '__main__':
    main()
