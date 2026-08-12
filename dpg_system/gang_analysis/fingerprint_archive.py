"""Fingerprint AMASS_Dynamic so v2 can be checked against v3 after replacement.

The reprocess sets acc_smooth_ms=70.0. At 100 Hz that resolves to exactly 7
frames -- the legacy value -- so KIT and EKUT should come out BIT-IDENTICAL.
At 120 Hz it resolves to 9 and at 60 Hz to 5, so those must change. Both
halves of that are worth verifying, and the disk cannot hold both archives, so
the evidence has to be captured before the old one is overwritten.

Records a sha256 of each stream plus robust statistics, per sampled file.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H'
KEYS = ('torque', 'torques_dyn_vec', 'torques_grav_vec',
        'torques_passive_vec', 'angular_velocity', 'com_acc')

# 100 Hz subsets must not change; 120/60 Hz must.
EXPECT = {100: 'identical', 120: 'changed', 60: 'changed', 250: 'changed'}


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


def main():
    per_rate = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = sys.argv[2] if len(sys.argv) > 2 else 'fingerprint_v2.json'

    files = []
    for dp, _d, fns in os.walk(ROOT):
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(dp, fn))
    files.sort()

    picked = defaultdict(list)
    for p in files:
        if all(len(v) >= per_rate for v in picked.values()) and len(picked) >= 3:
            break
        try:
            d = np.load(p, allow_pickle=True)
            fps = round(float(d['mocap_framerate']))
        except Exception:
            continue
        if fps not in EXPECT or len(picked[fps]) >= per_rate:
            continue
        if d['torque'].shape[0] < 200:
            continue
        picked[fps].append(p)

    recs = []
    for fps in sorted(picked):
        for p in picked[fps]:
            d = np.load(p, allow_pickle=True)
            rel = os.path.relpath(p, ROOT)
            r = {'path': rel, 'fps': fps, 'expect': EXPECT[fps],
                 'frames': int(d['torque'].shape[0]), 'streams': {}}
            for k in KEYS:
                if k not in d:
                    continue
                a = np.asarray(d[k])
                m = np.abs(a.astype(np.float64))
                r['streams'][k] = {
                    'sha': digest(a),
                    'p50': float(np.percentile(m, 50)),
                    'p99': float(np.percentile(m, 99)),
                    'sum': float(a.astype(np.float64).sum()),
                }
            opt = d.get('processing_options')
            if opt is not None:
                try:
                    o = json.loads(str(opt))
                    r['acc_smooth_window'] = o.get('acc_smooth_window')
                    r['acc_smooth_ms'] = o.get('acc_smooth_ms', '<absent>')
                except Exception:
                    pass
            recs.append(r)
            print(f"  {fps:4d} Hz  {rel[-58:]:60s} "
                  f"torque sha={r['streams']['torque']['sha']}")

    with open(out, 'w') as fh:
        json.dump({'archive': ROOT, 'n': len(recs), 'records': recs}, fh, indent=1)
    print(f'\nwrote {out}: {len(recs)} files '
          f'({", ".join(f"{k}Hz:{len(v)}" for k, v in sorted(picked.items()))})')
    print('after the reprocess, run compare_fingerprint.py')


if __name__ == '__main__':
    main()
