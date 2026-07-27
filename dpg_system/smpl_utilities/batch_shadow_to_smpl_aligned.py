"""Run shadow_to_smpl_aligned on every Shadow .npz in a directory.

Each file is converted independently.  Skips files whose names already carry
a converted-output tag, skips a file if its output already exists (use
--overwrite to force re-conversion), and skips non-Shadow files (no `quats`
key) with a notice.

Usage:
    python batch_shadow_to_smpl_aligned.py <directory>
        [--fps 100] [--gender male|female|neutral] [--betas betas.npy]
        [--floor 0.0] [--output-dir DIR] [--glob '*.npz'] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_to_smpl_aligned import convert_shadow_to_smpl_aligned


SKIP_TAGS = (
    '_smpl_poses_aligned',   # this script's own outputs
    '_smpl_poses',           # outputs of the basic (un-aligned) converter
)

OUTPUT_SUFFIX = '_smpl_poses_aligned.npz'


def load_betas(betas_path: str | None):
    if not betas_path:
        return None
    p = Path(betas_path)
    if not p.is_file():
        raise FileNotFoundError(f'betas file not found: {betas_path}')
    betas = np.load(p, allow_pickle=True)
    if isinstance(betas, np.ndarray) and betas.ndim == 0:
        betas = betas.item()
    if isinstance(betas, dict):
        for key in ('betas', 'mean', 'robust_mean'):
            if key in betas:
                betas = betas[key]
                break
        else:
            for v in betas.values():
                betas = np.asarray(v)
                break
    betas = np.array(betas, dtype=np.float64).flatten()
    print(f'Loaded betas: {betas.shape[0]} values')
    return betas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('directory')
    p.add_argument('--fps', type=float, default=None,
                   help='Framerate of the Shadow recordings.  If omitted, '
                        'each file\'s own `mocap_framerate` is used (else 100).')
    p.add_argument('--gender', choices=('male', 'female', 'neutral'),
                   default=None,
                   help='Subject gender for the SMPLH rest pose.  If omitted, '
                        'each file\'s own `gender` is used (else neutral).')
    p.add_argument('--betas', default=None,
                   help='Path to a .npy file with SMPL beta parameters '
                        '(scalar dict or array), applied to every take.  If '
                        'omitted, each file\'s own `betas` is used (else zeros).')
    p.add_argument('--floor', type=float, default=None,
                   help='Floor height subtracted from trans Z each frame '
                        '(forwarded to the converter).')
    p.add_argument('--output-dir', default=None,
                   help='Where to write the converted files (default: '
                        'alongside the input).')
    p.add_argument('--glob', default='*.npz',
                   help="Glob pattern for input files (default: '*.npz').")
    p.add_argument('--overwrite', action='store_true',
                   help='Re-convert and overwrite existing output files.')
    args = p.parse_args()

    src_dir = Path(args.directory)
    if not src_dir.is_dir():
        sys.exit(f'Not a directory: {src_dir}')

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    betas = load_betas(args.betas)

    inputs = sorted(p_ for p_ in src_dir.glob(args.glob)
                    if not any(tag in p_.stem for tag in SKIP_TAGS))

    print(f'Found {len(inputs)} candidate files in {src_dir}')
    print(f'fps={args.fps if args.fps is not None else "(from file)"}  '
          f'gender={args.gender if args.gender is not None else "(from file)"}  '
          f'betas={"(cli)" if betas is not None else "(from file)"}  '
          f'floor={args.floor if args.floor is not None else "(none)"}  '
          f'output_dir={out_dir if out_dir else "(alongside input)"}')
    print('=' * 78)

    summary: list[tuple[str, str, int | None]] = []
    for in_path in inputs:
        out_path = (out_dir if out_dir else in_path.parent) / \
                   (in_path.stem + OUTPUT_SUFFIX)

        if out_path.exists() and not args.overwrite:
            print(f'[skip] {in_path.name}  ->  output exists ({out_path.name}). '
                  f'Use --overwrite to redo.')
            summary.append((in_path.name, 'skipped', None))
            continue

        # Quick check that this is actually a Shadow file
        try:
            with np.load(in_path, allow_pickle=True) as d:
                if 'quats' not in d.files:
                    print(f'[skip] {in_path.name}  ->  not a Shadow file '
                          f'(no `quats` key).')
                    summary.append((in_path.name, 'not-shadow', None))
                    continue
                n_frames = int(d['quats'].shape[0])
        except Exception as exc:
            print(f'[skip] {in_path.name}  ->  could not read: {exc}')
            summary.append((in_path.name, 'unreadable', None))
            continue

        print(f'[run]  {in_path.name}  ({n_frames} frames)')
        try:
            convert_shadow_to_smpl_aligned(
                str(in_path), str(out_path),
                fps=args.fps, gender=args.gender, betas=betas,
                floor=args.floor, verbose=True,
            )
            summary.append((in_path.name, 'ok', n_frames))
        except Exception as exc:
            print(f'  ERROR: {exc}')
            traceback.print_exc()
            summary.append((in_path.name, f'error: {exc}', None))

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(f'{"file":<56} {"status":<14} {"frames":>8}')
    for name, status, frames in summary:
        fcell = f'{frames:>8d}' if frames is not None else f'{"-":>8}'
        print(f'{name:<56} {status:<14} {fcell}')


if __name__ == '__main__':
    main()
