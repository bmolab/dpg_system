"""Is acc_smooth_window's fixed frame count a real rate-dependent artifact?

acc_smooth_window is a Savitzky-Golay derivative window measured in FRAMES, so
its effective time constant scales with 1/fps: 7 frames is 58 ms at 120 Hz and
117 ms at 60 Hz. Dynamic torque is inertia x angular acceleration, so it sits
directly downstream of that window.

Controlled test -- same motion, only the rate changes:

  A  native 120 Hz, window 7        (58 ms)
  B  decimated to 60 Hz, window 7   (117 ms)  <- what a 60 Hz capture gets
  C  native 120 Hz, window 13       (108 ms)  <- B's time constant at A's rate

A vs B isolates the artifact. If C matches B, the difference is entirely the
fixed-frame window and not the sample rate itself -- which is the claim, and
also tells us the fix is to derive the window from milliseconds.

Comparison is at matched time points (every 2nd sample of the 120 Hz runs).
"""
import os
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpg_system.smpl_processor import SMPLProcessor
from probe_torque import make_options, MODEL_PATH

ARCH = '/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H'
N_JOINTS = 22


def run(poses, trans, fps, betas, gender, win, max_frames, ms=0.0):
    p = SMPLProcessor(framerate=fps, betas=betas, gender=gender,
                      total_mass_kg=75.0, model_path=MODEL_PATH)
    p.set_axis_permutation('x, z, -y')
    o = make_options(1.0 / fps)
    o.acc_smooth_window = win
    o.acc_smooth_ms = ms
    T = min(max_frames, poses.shape[0])
    dyn = np.zeros((T, N_JOINTS, 3))
    for t in range(T):
        try:
            r = p.process_frame(poses[t:t + 1], trans[t:t + 1], o)
        except Exception:
            continue
        v = r.get('torques_dyn_vec')
        if v is not None:
            dyn[t, :min(N_JOINTS, v.shape[1])] = v[0, :min(N_JOINTS, v.shape[1])]
    return dyn


def stats(dyn, dt):
    m = np.linalg.norm(dyn, axis=-1).ravel()
    m = m[m > 0]
    if m.size == 0:
        return None
    d = np.abs(np.diff(np.linalg.norm(dyn, axis=-1), axis=0)) / dt
    return {
        'p50': float(np.percentile(m, 50)),
        'p90': float(np.percentile(m, 90)),
        'p99': float(np.percentile(m, 99)),
        # roughness: how much the signal moves between samples relative to its
        # size -- the direct read on how much high frequency survived
        'rough': float(np.percentile(d, 50) / (np.percentile(m, 90) + 1e-12)),
    }


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    files = []
    for dp, _d, fns in os.walk(os.path.join(ARCH, 'CMU')):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()
    picked = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if round(float(d['mocap_framerate'])) == 120 and d['poses'].shape[0] > 400:
            picked.append(f)
        if len(picked) >= n_files:
            break

    agg = {k: [] for k in ('A', 'B', 'C', 'D')}
    for f in picked:
        d = np.load(f, allow_pickle=True)
        poses, trans = np.asarray(d['poses']), np.asarray(d['trans'])
        betas = np.asarray(d['betas'], np.float64).flatten()[:10]
        g = str(d['gender'])
        g = g if g in ('male', 'female', 'neutral') else 'neutral'

        # legacy: fixed 7 frames at both rates
        A = run(poses, trans, 120.0, betas, g, 7, max_frames)
        B = run(poses[::2], trans[::2], 60.0, betas, g, 7, max_frames // 2)
        # FIXED: acc_smooth_ms=70 -> 9 frames at 120 Hz, 5 at 60 Hz
        C = run(poses, trans, 120.0, betas, g, 7, max_frames, ms=70.0)
        D = run(poses[::2], trans[::2], 60.0, betas, g, 7, max_frames // 2, ms=70.0)

        # compare at matched time points
        n = min(A.shape[0] // 2, B.shape[0], C.shape[0] // 2)
        sa = stats(A[:2 * n:2], 1 / 60.0)
        sb = stats(B[:n], 1 / 60.0)
        sc = stats(C[:2 * n:2], 1 / 60.0)
        sd = stats(D[:n], 1 / 60.0)
        if not (sa and sb and sc and sd):
            continue
        for k, s in (('A', sa), ('B', sb), ('C', sc), ('D', sd)):
            agg[k].append(s)
        print(f'  {os.path.basename(f)[:34]:36s} '
              f'p99 A={sa["p99"]:7.2f} B={sb["p99"]:7.2f} C={sc["p99"]:7.2f}   '
              f'rough A={sa["rough"]:.3f} B={sb["rough"]:.3f} C={sc["rough"]:.3f}')

    print('\n' + '=' * 78)
    print('MEAN OVER FILES  (dynamic torque magnitude, N.m, at matched 60 Hz points)')
    print('=' * 78)
    lab = {'A': 'LEGACY 120 Hz, win 7   ( 58 ms)',
           'B': 'LEGACY  60 Hz, win 7   (117 ms)',
           'C': 'FIXED  120 Hz, ms 70 -> 9 ( 75 ms)',
           'D': 'FIXED   60 Hz, ms 70 -> 5 ( 83 ms)'}
    for k in ('A', 'B', 'C', 'D'):
        if not agg[k]:
            continue
        m = {q: float(np.mean([s[q] for s in agg[k]]))
             for q in ('p50', 'p90', 'p99', 'rough')}
        print(f'  {lab[k]}   p50={m["p50"]:7.3f}  p90={m["p90"]:7.3f}  '
              f'p99={m["p99"]:8.3f}  roughness={m["rough"]:.4f}')

    if agg['A'] and agg['B']:
        g = lambda k, q: float(np.mean([s[q] for s in agg[k]]))
        print(f'\n  RATE CONSISTENCY (60 Hz vs 120 Hz on identical motion; 1.0 = consistent)')
        print(f'    legacy   p99 60/120 = {g("B","p99") / g("A","p99"):.3f}    '
              f'roughness 60/120 = {g("B","rough") / g("A","rough"):.3f}')
        print(f'    fixed    p99 60/120 = {g("D","p99") / g("C","p99"):.3f}    '
              f'roughness 60/120 = {g("D","rough") / g("C","rough"):.3f}')


if __name__ == '__main__':
    main()
