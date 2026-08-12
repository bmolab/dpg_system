"""What is actually available per frame, under the live-node option set?

Verifies the exact keys/attributes the reprocessing script would read, with
their real shapes and dtypes, rather than trusting the source reading.
"""
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_torque import make_options, load, MODEL_PATH

ATTRS = ('_local_ang_vel', '_current_ang_vel', 'current_com',
         'prob_prev_com_vel', 'prob_prev_com_acc', '_raw_com_acc',
         '_v2_raw_com_acc', 'contact_pressure', 'max_torque_array',
         'total_mass_kg')


def show(label, v):
    if v is None:
        print(f'  {label:26s} None')
    elif isinstance(v, np.ndarray):
        rng = ''
        if v.size and v.dtype.kind in 'fiu':
            f = v[np.isfinite(v)]
            if f.size:
                rng = f'  range [{f.min():.4g}, {f.max():.4g}]'
        print(f'  {label:26s} {str(v.shape):18s} {str(v.dtype):10s}{rng}')
    elif isinstance(v, (int, float, np.floating)):
        print(f'  {label:26s} scalar {v}')
    else:
        print(f'  {label:26s} {type(v).__name__}')


def main():
    path = sys.argv[1]
    poses, trans, fps, betas, gender = load(path, 90)
    p = SMPLProcessor(framerate=fps, betas=betas, gender=gender,
                      total_mass_kg=75.0, model_path=MODEL_PATH)
    p.set_axis_permutation('x, z, -y')
    opts = make_options(1.0 / fps)
    print(f'options.torque_output_frame = '
          f'{getattr(opts, "torque_output_frame", "<unset>")!r}')
    print(f'options.world_frame_dynamics = {opts.world_frame_dynamics!r}\n')

    res = None
    for t in range(poses.shape[0]):
        res = p.process_frame(poses[t:t + 1], trans[t:t + 1], opts)

    print('--- result dict keys ---')
    for k in sorted(res):
        v = res[k]
        if isinstance(v, np.ndarray):
            show(k, v)
        else:
            print(f'  {k:26s} {type(v).__name__}')

    print('\n--- processor attributes ---')
    for a in ATTRS:
        show(a, getattr(p, a, None))

    # does the local angular velocity track the pose derivative?
    av = getattr(p, '_local_ang_vel', None)
    if av is not None:
        print(f'\n_local_ang_vel is the LOCAL (parent-relative) angular velocity, '
              f'{av.shape} per streamed frame')


if __name__ == '__main__':
    main()
