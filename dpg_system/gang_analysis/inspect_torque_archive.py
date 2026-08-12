"""Inspect a precomputed torque archive and report what it retained.

Answers, without assuming a format:
  - which arrays are in it, their shapes/dtypes/ranges
  - whether per-joint 3-vectors survived, or only magnitudes/scalars
  - which torque streams are present (net / dynamic / gravity / passive)
  - frame counts, coverage, on-disk size
  - any recorded processing options

Usage:
    python inspect_torque_archive.py <path> [--max-files N] [--deep]

<path> may be a single file or a directory tree. Handles .npz/.npy, HDF5
(.h5/.hdf5), pickle (.pkl/.pickle), and torch (.pt/.pth) if those libraries
are importable; anything else is reported as an unknown extension rather
than guessed at.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Names we expect to mean a torque stream, mapped to the canonical stream.
# Matching is substring-based and case-insensitive, longest key first, so
# 'torques_dyn_vec' resolves to dynamic before the bare 'torque' catches it.
STREAM_HINTS = (
    ('passive', 'passive'),
    ('gravity', 'gravity'),
    ('grav', 'gravity'),
    ('dynamic', 'dynamic'),
    ('dyn', 'dynamic'),
    ('net', 'total'),
    ('total', 'total'),
    ('torques_vec', 'total'),
)

ARCHIVE_EXTS = ('.npz', '.npy', '.h5', '.hdf5', '.pkl', '.pickle', '.pt',
                '.pth', '.json')


def classify_stream(name):
    low = name.lower()
    for hint, canonical in STREAM_HINTS:
        if hint in low:
            return canonical
    return None


def describe_axes(shape):
    """Guess what a torque-ish array's trailing axes mean."""
    if len(shape) >= 2 and shape[-1] == 3 and shape[-2] in (22, 24, 52):
        return f'per-joint 3-vectors ({shape[-2]} joints x 3 axes)  << FULL VECTORS'
    if len(shape) >= 2 and shape[-1] in (22, 24, 52):
        return f'per-joint scalars ({shape[-1]} joints)  << REDUCED, no axis structure'
    if len(shape) >= 2 and shape[-1] in (66, 72, 156):
        return f'flattened per-joint 3-vectors ({shape[-1] // 3} joints x 3)  << FULL VECTORS'
    if len(shape) == 1:
        return 'scalar per frame (or a 1-D table)'
    return 'shape not recognised as a joint field'


def summarize_array(arr, deep):
    """Cheap stats; on huge arrays sample rather than scan unless --deep."""
    out = {'shape': tuple(arr.shape), 'dtype': str(arr.dtype),
           'nbytes': int(arr.nbytes)}
    if arr.dtype.kind not in 'fiu' or arr.size == 0:
        return out
    flat = arr.reshape(-1)
    if not deep and flat.size > 2_000_000:
        idx = np.linspace(0, flat.size - 1, 2_000_000).astype(np.int64)
        sample = flat[idx]
        out['sampled'] = True
    else:
        sample = flat
    sample = np.asarray(sample, dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        out['all_nonfinite'] = True
        return out
    out['min'] = float(finite.min())
    out['max'] = float(finite.max())
    out['mean'] = float(finite.mean())
    out['p99abs'] = float(np.percentile(np.abs(finite), 99))
    out['frac_zero'] = float((finite == 0).mean())
    out['frac_negative'] = float((finite < 0).mean())
    out['n_nonfinite'] = int(sample.size - finite.size)
    return out


def read_npz(path):
    d = np.load(path, allow_pickle=True)
    items = {}
    for k in d.files:
        try:
            items[k] = d[k]
        except Exception as e:
            items[k] = f'<unreadable: {e}>'
    return items


def read_npy(path):
    return {'<array>': np.load(path, allow_pickle=True)}


def read_hdf5(path):
    try:
        import h5py
    except ImportError:
        return {'<error>': 'h5py not installed in this env'}
    items = {}

    def visit(name, obj):
        if hasattr(obj, 'shape'):
            items[name] = obj
    with h5py.File(path, 'r') as f:
        f.visititems(visit)
        # materialize small things, keep big ones lazy-read
        return {k: (v[()] if v.size < 50_000_000 else np.asarray(v[:1000]))
                for k, v in items.items()}


def read_pickle(path):
    import pickle
    with open(path, 'rb') as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        return obj
    return {'<object>': obj}


def read_torch(path):
    try:
        import torch
    except ImportError:
        return {'<error>': 'torch not installed in this env'}
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        return {k: (v.numpy() if hasattr(v, 'numpy') else v)
                for k, v in obj.items()}
    return {'<object>': obj}


def read_json(path):
    import json
    with open(path) as fh:
        return {'<json>': json.load(fh)}


READERS = {'.npz': read_npz, '.npy': read_npy, '.h5': read_hdf5,
           '.hdf5': read_hdf5, '.pkl': read_pickle, '.pickle': read_pickle,
           '.pt': read_torch, '.pth': read_torch, '.json': read_json}


def inspect_file(path, deep):
    ext = os.path.splitext(path)[1].lower()
    reader = READERS.get(ext)
    if reader is None:
        return None, f'no reader for extension {ext}'
    try:
        items = reader(path)
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'
    return items, None


def report_one(path, deep, verbose=True):
    items, err = inspect_file(path, deep)
    if err:
        print(f'  !! {err}')
        return {}
    found = {}
    for key, val in sorted(items.items()):
        if isinstance(val, str):
            print(f'  {key:34s} {val}')
            continue
        if isinstance(val, np.ndarray) and val.dtype == object:
            try:
                unwrapped = val.item()
            except Exception:
                unwrapped = None
            if isinstance(unwrapped, dict):
                print(f'  {key:34s} dict with {len(unwrapped)} keys '
                      f'(likely recorded options/metadata)')
                for mk, mv in list(unwrapped.items())[:40]:
                    print(f'      {mk:32s} {mv!r:.90}')
                continue
            print(f'  {key:34s} object array {val.shape}: {unwrapped!r:.100}')
            continue
        if not isinstance(val, np.ndarray):
            print(f'  {key:34s} {type(val).__name__}: {val!r:.100}')
            continue

        s = summarize_array(val, deep)
        stream = classify_stream(key)
        axes = describe_axes(s['shape'])
        tag = f'[{stream}]' if stream else ''
        print(f'  {key:34s} {str(s["shape"]):22s} {s["dtype"]:8s} '
              f'{s["nbytes"] / 1e6:8.2f} MB {tag}')
        print(f'  {"":34s} {axes}')
        if 'min' in s:
            print(f'  {"":34s} range [{s["min"]:.4g}, {s["max"]:.4g}]  '
                  f'p99|x|={s["p99abs"]:.4g}  '
                  f'zero={s["frac_zero"]:.1%}  neg={s["frac_negative"]:.1%}'
                  + (f'  nonfinite={s["n_nonfinite"]}' if s['n_nonfinite'] else '')
                  + ('  (sampled)' if s.get('sampled') else ''))
        found[key] = (s, stream, axes)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='archive file or directory tree')
    ap.add_argument('--max-files', type=int, default=3,
                    help='how many files to open in detail (default 3)')
    ap.add_argument('--deep', action='store_true',
                    help='scan full arrays for stats instead of sampling')
    args = ap.parse_args()

    path = os.path.abspath(args.path)

    if os.path.isfile(path):
        files = [path]
        root = os.path.dirname(path)
    else:
        root = path
        files = []
        for dirpath, _dirnames, filenames in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in ARCHIVE_EXTS:
                    files.append(os.path.join(dirpath, fn))
        files.sort()

    if not files:
        print(f'no recognisable archive files under {path}')
        print('contents:')
        for entry in sorted(os.listdir(path))[:40]:
            print('   ', entry)
        return

    total_bytes = 0
    by_ext = Counter()
    by_subdir = Counter()
    for f in files:
        try:
            total_bytes += os.path.getsize(f)
        except OSError:
            pass
        by_ext[os.path.splitext(f)[1].lower()] += 1
        rel = os.path.relpath(f, root)
        by_subdir[rel.split(os.sep)[0] if os.sep in rel else '.'] += 1

    print('=' * 78)
    print(f'ARCHIVE: {path}')
    print('=' * 78)
    print(f'files          : {len(files)}')
    print(f'on-disk size   : {total_bytes / 1e9:.2f} GB')
    print(f'extensions     : {dict(by_ext)}')
    print(f'top-level dirs : {len(by_subdir)}')
    for name, n in by_subdir.most_common(30):
        print(f'    {name:36s} {n:6d} files')

    # Coverage against the local AMASS tree, if it looks like a mirror.
    amass = '/Users/drokeby/dpg_system/AMASS'
    if os.path.isdir(amass):
        local_subsets = {d for d in os.listdir(amass)
                         if os.path.isdir(os.path.join(amass, d))}
        archive_subsets = set(by_subdir)
        overlap = local_subsets & archive_subsets
        if overlap:
            print(f'\nlooks like an AMASS mirror: {len(overlap)}/{len(local_subsets)} '
                  f'subsets present')
            missing = sorted(local_subsets - archive_subsets)
            if missing:
                print(f'  subsets NOT in archive: {", ".join(missing)}')

    print('\n' + '=' * 78)
    print(f'CONTENTS (first {args.max_files} files in detail)')
    print('=' * 78)

    stream_presence = Counter()
    vector_verdicts = Counter()      # arrays whose name identifies a stream
    unnamed_verdicts = Counter()     # joint-shaped arrays we could not name
    for f in files[:args.max_files]:
        print(f'\n--- {os.path.relpath(f, root)}  '
              f'({os.path.getsize(f) / 1e6:.2f} MB)')
        found = report_one(f, args.deep)
        for _key, (_s, stream, axes) in found.items():
            # Only arrays that name a torque stream count toward the verdict --
            # a 'poses' array is also full 3-vectors and would otherwise vote.
            bucket = vector_verdicts if stream else unnamed_verdicts
            if stream:
                stream_presence[stream] += 1
            if 'FULL VECTORS' in axes:
                bucket['full'] += 1
            elif 'REDUCED' in axes:
                bucket['reduced'] += 1

    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    print(f'torque streams identified : '
          f'{dict(stream_presence) if stream_presence else "NONE recognised by name"}')
    for want in ('total', 'dynamic', 'gravity', 'passive'):
        mark = 'yes' if stream_presence.get(want) else 'no '
        print(f'    {want:10s} {mark}')
    if vector_verdicts:
        print(f'\nform of the stream arrays : {dict(vector_verdicts)}')
        if vector_verdicts.get('full') and not vector_verdicts.get('reduced'):
            print('  -> full per-joint 3-vectors retained. Co-variation analysis '
                  'and the gang bank run directly on this.')
        elif vector_verdicts.get('reduced') and not vector_verdicts.get('full'):
            print('  -> REDUCED to per-joint scalars. Sign/axis structure is gone; '
                  'gang presets cannot be evaluated from this.')
        else:
            print('  -> mixed; see per-array notes above.')
    else:
        print('\nno array NAME identified a torque stream.')
        if unnamed_verdicts:
            print(f'  but {dict(unnamed_verdicts)} joint-shaped array(s) are present '
                  f'under names this script does not recognise --')
            print('  read the listing above; the data may be there under a '
                  'different naming scheme.')


if __name__ == '__main__':
    main()
