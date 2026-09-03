"""Sanity-check AMASS_Dynamic before any analysis is built on it.

Each check catches a specific way the reprocessing could have gone quietly
wrong -- shape mis-slotting, a stream that is not what its name says, or a
first-frame derivative artifact that would poison range statistics.
"""
import json
import os
import random
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import dpg_system.gang_core as gc

V2 = '/Users/drokeby/dpg_system/AMASS_Dynamic'
V1 = '/Users/drokeby/dpg_system/AMASS_Torque'
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}


def walk(root):
    out = []
    for dp, _d, fns in os.walk(root):
        for fn in fns:
            if fn.endswith('.npz'):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    files = walk(V2)
    random.seed(1)
    probe = random.sample(files, min(n, len(files)))

    # ---- 0. recorded options -------------------------------------------
    d0 = np.load(probe[0], allow_pickle=True)
    print('=' * 74)
    print('RECORDED PROCESSING OPTIONS')
    print('=' * 74)
    raw = str(d0['processing_options'])
    try:
        opts = json.loads(raw)
        for k in sorted(opts):
            print(f'  {k:34s} {opts[k]!r}')
    except Exception:
        print(raw[:2000])
        opts = {}

    # ---- 1. decomposition identity -------------------------------------
    print('\n' + '=' * 74)
    print('CHECK 1  which combination of streams reproduces net `torque`?')
    print('=' * 74)
    combos = {
        'dyn - grav - passive': lambda D, G, P: D - G - P,
        'dyn - grav':           lambda D, G, P: D - G,
        'dyn + grav + passive': lambda D, G, P: D + G + P,
        'dyn + grav':           lambda D, G, P: D + G,
    }
    acc = {k: [] for k in combos}
    for p in probe:
        d = np.load(p, allow_pickle=True)
        T = np.asarray(d['torque'], np.float64)
        D = np.asarray(d['torques_dyn_vec'], np.float64)
        G = np.asarray(d['torques_grav_vec'], np.float64)
        P = np.asarray(d['torques_passive_vec'], np.float64)
        scale = np.percentile(np.abs(T), 99) + 1e-9
        for k, f in combos.items():
            acc[k].append(np.median(np.abs(f(D, G, P) - T)) / scale)
    for k in combos:
        v = float(np.mean(acc[k]))
        flag = '  <== MATCH' if v < 0.02 else ''
        print(f'  {k:24s} median|err|/p99|net| = {v:.4f}{flag}')
    print('  (no match => net carries a contact term the archive did not keep;')
    print('   see smpl_processor.py:6855, net = dyn - grav - contact)')

    # ---- 2. structural zeros -------------------------------------------
    print('\n' + '=' * 74)
    print('CHECK 2  hinge/foot structural zeros (corroborates axis convention)')
    print('=' * 74)
    zero_always = None
    for p in probe:
        d = np.load(p, allow_pickle=True)
        T = np.asarray(d['torque'])
        z = np.all(T == 0.0, axis=0)
        zero_always = z if zero_always is None else (zero_always & z)
    print(f'  channels zero in all {len(probe)} probed files: '
          f'{int(zero_always.sum())}/66')
    for j, a in zip(*np.where(zero_always)):
        print(f'     {NAMES[j]:16s} axis {a}')
    elb = [gc.JOINT_INDEX['left_elbow'], gc.JOINT_INDEX['right_elbow']]
    ok = all(zero_always[j, 0] and zero_always[j, 2] and not zero_always[j, 1]
             for j in elb)
    print(f'  elbows live on axis 1 only (arm flex = Y): '
          f'{"YES" if ok else "NO  <== axis convention mismatch"}')

    # ---- 3. angular velocity sanity ------------------------------------
    print('\n' + '=' * 74)
    print('CHECK 3  angular velocity: first-frame artifact? magnitude sane?')
    print('=' * 74)
    for p in probe[:6]:
        d = np.load(p, allow_pickle=True)
        W = np.asarray(d['angular_velocity'], np.float64)
        mag = np.linalg.norm(W, axis=-1)          # (T,22)
        per_frame = mag.max(axis=1)
        head = per_frame[:3]
        rest = per_frame[3:]
        print(f'  {os.path.basename(p)[:44]:46s} '
              f'f0..2 max={np.array2string(head, precision=1)}  '
              f'rest p99={np.percentile(rest, 99):6.2f} max={rest.max():7.2f}')
    print('  (a huge value confined to frames 0-2 is a derivative start-up')
    print('   artifact -- must be trimmed before range statistics)')

    # ---- 4. v1/v2 agreement on net -------------------------------------
    print('\n' + '=' * 74)
    print('CHECK 4  net torque agrees with v1 archive, and effort normalizes')
    print('=' * 74)
    n_cmp = 0
    for p in probe:
        rel = os.path.relpath(p, V2)
        q = os.path.join(V1, rel)
        if not os.path.exists(q):
            continue
        a = np.asarray(np.load(p, allow_pickle=True)['torque'], np.float64)
        b_d = np.load(q, allow_pickle=True)
        b = np.asarray(b_d['torque'], np.float64)
        if a.shape != b.shape:
            print(f'  shape differs {a.shape} vs {b.shape} for {rel}')
            continue
        scale = np.percentile(np.abs(b), 99) + 1e-9
        err = np.median(np.abs(a - b)) / scale
        # v1 combined_effort vs v2 torque/max_torque
        mt = np.asarray(np.load(p, allow_pickle=True)['max_torque'], np.float64)[:22]
        ce_v2 = a / np.where(mt == 0, np.nan, mt)
        ce_v1 = np.asarray(b_d['combined_effort'], np.float64)
        eerr = np.nanmedian(np.abs(ce_v2 - ce_v1))
        n_cmp += 1
        if n_cmp <= 5:
            print(f'  {rel[-52:]:54s} net rel-err={err:.2e}  effort abs-err={eerr:.2e}')
    print(f'  compared {n_cmp} files present in both archives')

    # ---- 5. gang bank runs on it ---------------------------------------
    print('\n' + '=' * 74)
    print('CHECK 5  gang bank evaluates, and max_torque matches gang_core')
    print('=' * 74)
    mt_file = np.asarray(np.load(probe[0], allow_pickle=True)['max_torque'],
                         np.float64)
    mt_gang = gc.max_torque_array()
    print(f'  max_torque file vs gang_core: max abs diff = '
          f'{np.abs(mt_file - mt_gang).max():.6g}')

    d = np.load(probe[0], allow_pickle=True)
    T = d['torque'].shape[0]

    def pad(arr):
        out = np.zeros((T, 24, 3), np.float32)
        out[:, :22] = arr
        return out

    bundle = {'total': pad(np.asarray(d['torque'])),
              'gravity': pad(np.asarray(d['torques_grav_vec'])),
              'dynamic': pad(np.asarray(d['torques_dyn_vec'])),
              'passive': pad(np.asarray(d['torques_passive_vec']))}
    stacked = gc.stack_streams(bundle)
    for stream in ('total', 'gravity', 'dynamic', 'passive'):
        specs = [gc.spec_from_preset(pn, side=s, stream=stream)
                 for pn in gc.preset_names()
                 for s in (gc.sides_for(pn) or ['none'])]
        prog = gc.compile_specs(specs)
        net, tot, coh = prog.evaluate(stacked)
        live = tot.max(axis=0) > 1e-9
        print(f'  stream={stream:8s} gangs={len(prog.names):3d} '
              f'live={int(live.sum()):3d}  '
              f'p99|net|={np.percentile(np.abs(net), 99):8.4f}  '
              f'median coherence={np.median(coh):.3f}')


if __name__ == '__main__':
    main()
