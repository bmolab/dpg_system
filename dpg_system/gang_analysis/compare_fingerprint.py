"""Check the reprocessed archive against the v2 fingerprint.

The acc_smooth_ms=70.0 change touches ONE thing: the Savitzky-Golay window that
turns angular velocity into angular acceleration. That gives a specific and
falsifiable pattern, which is a much stronger test than "something changed":

  by stream
    torques_grav_vec     built from positions, not acceleration  -> UNCHANGED
    torques_passive_vec  built from pose                          -> UNCHANGED
    angular_velocity     upstream of the SG window                -> UNCHANGED
    torques_dyn_vec      inertia x angular acceleration           -> rate dependent
    torque               net = dyn - grav - passive               -> rate dependent
    com_acc              same SG window on CoM velocity           -> rate dependent

  by rate, for the acceleration-derived streams
    100 Hz   70 ms -> 7 frames, the legacy value  -> BIT-IDENTICAL
    120 Hz   70 ms -> 9 frames                    -> must change
     60 Hz   70 ms -> 5 frames                    -> must change

Anything outside that pattern means something other than the intended change
happened, and is worth chasing before trusting the archive.

Usage:  python compare_fingerprint.py fingerprint_v2.json [archive_root]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint_archive import digest, KEYS

RATE_SENSITIVE = ('torque', 'torques_dyn_vec', 'com_acc')
RATE_INVARIANT = ('torques_grav_vec', 'torques_passive_vec', 'angular_velocity')


def main():
    fp = json.load(open(sys.argv[1]))
    root = sys.argv[2] if len(sys.argv) > 2 else fp['archive']
    print(f'v2 fingerprint : {sys.argv[1]}  ({fp["n"]} files)')
    print(f'new archive    : {root}\n')

    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # stream->rate->[same,diff]
    problems = []
    missing = 0
    opts_seen = set()

    for rec in fp['records']:
        p = os.path.join(root, rec['path'])
        if not os.path.exists(p):
            missing += 1
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            problems.append(f'{rec["path"]}: unreadable ({e})')
            continue

        o = d.get('processing_options')
        if o is not None:
            try:
                oo = json.loads(str(o))
                opts_seen.add((oo.get('acc_smooth_window'),
                               oo.get('acc_smooth_ms', '<absent>')))
            except Exception:
                pass

        fps = rec['fps']
        for k, old in rec['streams'].items():
            if k not in d:
                problems.append(f'{rec["path"]}: stream {k} missing in new archive')
                continue
            same = digest(np.asarray(d[k])) == old['sha']
            tally[k][fps][0 if same else 1] += 1

            expect_same = (k in RATE_INVARIANT) or (fps == 100)
            if same != expect_same:
                a = np.abs(np.asarray(d[k]).astype(np.float64))
                problems.append(
                    f'{rec["path"][-46:]:48s} {k:20s} {fps:4d}Hz  '
                    f'{"UNCHANGED but should differ" if same else "CHANGED but should match"}'
                    f'  p99 {old["p99"]:.4g} -> {np.percentile(a, 99):.4g}')

    print('=' * 92)
    print('RECORDED OPTIONS IN NEW ARCHIVE')
    print('=' * 92)
    for w, ms in sorted(opts_seen, key=str):
        flag = '  <== expected' if ms == 70.0 else '  <== NOT the intended value'
        print(f'  acc_smooth_window={w!r}  acc_smooth_ms={ms!r}{flag}')
    if not opts_seen:
        print('  none recorded')

    print('\n' + '=' * 92)
    print('CHANGED / UNCHANGED BY STREAM AND RATE   (files same|different)')
    print('=' * 92)
    rates = sorted({r['fps'] for r in fp['records']})
    print(f'{"stream":22s}' + ''.join(f'{str(r) + " Hz":>16s}' for r in rates)
          + '   expectation')
    for k in KEYS:
        if k not in tally:
            continue
        cells = ''
        for r in rates:
            s, dch = tally[k][r]
            cells += f'{f"{s} same|{dch} diff":>16s}'
        exp = ('unchanged at every rate' if k in RATE_INVARIANT
               else 'same @100, differ @60/120')
        print(f'{k:22s}{cells}   {exp}')

    print('\n' + '=' * 92)
    if missing:
        print(f'{missing} fingerprinted file(s) absent from the new archive')
    if problems:
        print(f'{len(problems)} DEVIATION(S) FROM THE EXPECTED PATTERN:')
        for s in problems[:40]:
            print('   ' + s)
    else:
        print('PASS — every stream changed exactly where it should and nowhere else.')
        print('  100 Hz bit-identical, 60/120 Hz shifted, gravity/passive/omega untouched.')


if __name__ == '__main__':
    main()
