"""Smooth the root-position jolt at stomps of the leg that carries the position sensor.

Background
----------
Shadow takes `trans` from a position sensor strapped to a thigh (the right one for
Subject 7). At each stomp of that leg the sensor bounces on its strap and the root
position shakes for ~150 ms: in Subject 7 the fast (> ~15 Hz) part of the root
position is 2.7x its pre-stomp level after right stomps and unchanged after left
stomps, while each thigh's own rotation shakes equally on its own stomp. Shaking
that fast is not whole-body movement, so it can be smoothed away without hiding
anything real.

This runs after correct_thigh_offset_flexion.py (the hip fix models a slow,
angle-following mount error and is smoothed over 0.5 s, so it cannot remove this).
Only `trans` changes; poses are untouched, so pose-only corrections (the video arm
correction) can come before or after.

Method
------
1. Stomps of the sensor leg are found from the pose alone (ankle height relative
   to the pelvis): fast downward ankle motion that stops abruptly; the strike is the
   frame of peak deceleration. The root position plays no part in finding them.
2. A soft weight is 1 from --pre s before each strike to --post s after it, with
   raised-cosine fades of --ramp s either side (overlapping stomps take the max).
3. trans_out = trans + weight * (gaussian_smoothed(trans) - trans).

Output filename suffix: `_stomp.npz`.

Usage:
    python smooth_stomp_jolt.py <input.npz> [<output.npz>]
        [--side R|L] [--sigma 0.03] [--pre 0.02] [--post 0.18] [--ramp 0.04]
"""
from __future__ import annotations

import sys
from pathlib import Path
import pickle
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct_thigh_offset import (
    shape_joints, fk_batch, N_BODY, PELVIS,
    DEFAULT_MODEL_FEMALE, DEFAULT_MODEL_MALE,
)

ANKLE = {'L': 7, 'R': 8}  # SMPL ankles


def find_strikes(z, fps, down_speed=0.6, stop_speed=0.15, min_drop=0.03):
    """Strike frames from root-relative ankle height z (m).

    A stomp = downward ankle speed above `down_speed` m/s that falls below
    `stop_speed` within 80 ms, after a drop of at least `min_drop` m.
    """
    v = np.gradient(np.convolve(np.pad(z, 1, mode='edge'), np.ones(3) / 3, 'valid')) * fps
    a = np.gradient(v) * fps
    out = []
    i = 0
    while i < len(z) - 20:
        if v[i] < -down_speed:
            j = i + int(np.argmin(v[i:i + 15]))                 # peak downward speed
            seg = slice(j, min(len(z), j + int(0.08 * fps)))
            if (np.abs(v[seg]) < stop_speed).any():             # stops within 80 ms
                k = j + int(np.argmax(a[seg]))                  # peak deceleration
                drop = z[max(0, j - int(0.15 * fps)):j + 1].max() - z[k]
                if drop > min_drop:
                    out.append(k)
                i = k + int(0.25 * fps)
                continue
        i += 1
    return np.array(out, int)


def stomp_weight(strikes, T, fps, pre, post, ramp):
    """Soft valve: 1 over [strike - pre, strike + post], raised-cosine fades of `ramp`."""
    t = np.arange(T) / fps
    w = np.zeros(T)
    for s in strikes:
        ts = s / fps
        lo, hi = ts - pre, ts + post
        d = np.maximum(lo - t, t - hi)                          # >0 outside the window
        w = np.maximum(w, np.where(d <= 0, 1.0, np.where(d < ramp, 0.5 * (1 + np.cos(np.pi * d / ramp)), 0.0)))
    return w


def smooth_take(in_path, out_path, side='R', sigma_s=0.03, pre_s=0.02, post_s=0.18,
                ramp_s=0.04, verbose=True):
    data = np.load(in_path, allow_pickle=True)
    poses = data['poses']; trans = data['trans']
    betas = np.asarray(data['betas']).flatten()
    gender = str(data['gender']).lower().strip().strip("'\"")
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 100.0
    T = poses.shape[0]

    mp = DEFAULT_MODEL_FEMALE if gender == 'female' else DEFAULT_MODEL_MALE
    with open(mp, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1; parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas[:model['shapedirs'].shape[2]])[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    z = np.empty(T)
    for s in range(0, T, 4000):
        e = min(T, s + 4000)
        tw, _ = fk_batch(J_rest, body_pose[s:e], parents)
        z[s:e] = tw[:, ANKLE[side], 2] - tw[:, PELVIS, 2]

    strikes = find_strikes(z, fps)
    w = stomp_weight(strikes, T, fps, pre_s, post_s, ramp_s)
    smooth = gaussian_filter1d(trans, sigma_s * fps, axis=0, mode='nearest')
    trans_out = trans + w[:, None] * (smooth - trans)

    if verbose:
        moved = np.linalg.norm(trans_out - trans, axis=1)
        print(f'Take: {Path(in_path).name}  T={T}  fps={fps}  side={side}  '
              f'stomps={len(strikes)}  frames weighted={np.mean(w > 0) * 100:.1f}%  '
              f'root moved: median {np.median(moved[w > 0.5]) * 1000:.1f} mm, '
              f'max {moved.max() * 1000:.1f} mm')

    out = {k: data[k] for k in data.files}
    out['trans'] = trans_out
    out['_stomp_smooth_side'] = np.array(side)
    out['_stomp_smooth_strikes'] = strikes
    out['_stomp_smooth_sigma_s'] = np.array(sigma_s)
    out['_stomp_smooth_window_s'] = np.array([pre_s, post_s, ramp_s])
    np.savez(out_path, **out)
    if verbose:
        print(f'Saved: {out_path}')
    return strikes, w


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input')
    p.add_argument('output', nargs='?', default=None)
    p.add_argument('--side', default='R', choices=['R', 'L'],
                   help='leg carrying the position sensor')
    p.add_argument('--sigma', type=float, default=0.03, help='smoothing width, s')
    p.add_argument('--pre', type=float, default=0.02, help='full weight starts this long before the strike, s')
    p.add_argument('--post', type=float, default=0.18, help='full weight lasts this long after the strike, s')
    p.add_argument('--ramp', type=float, default=0.04, help='fade in/out, s')
    a = p.parse_args()
    inp = Path(a.input)
    out = Path(a.output) if a.output else inp.with_name(inp.stem + '_stomp.npz')
    smooth_take(inp, out, a.side, a.sigma, a.pre, a.post, a.ramp)


if __name__ == '__main__':
    main()
