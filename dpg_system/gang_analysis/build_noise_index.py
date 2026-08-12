"""Build a compact index from the noise-estimation results.

Maps each AMASS_Dynamic file to its quality verdict and the frame ranges the
noise work judged unusable, so characterization can be recomputed under
different filtering regimes without re-reading 570 MB of JSON.

Two levels of exclusion are captured:
  - surgery.segments marked usable=False  (whole stretches after a stream break)
  - surgery.excision.zones                (localized corrupted joint episodes)

Subset names differ in case between the noise tree and the archive
(BMLmovi vs BMLMovi), so matching is case-insensitive on the subset component.
"""
import json
import os
import sys

NOISE = ('/Users/drokeby/dpg_system/dpg_system/noise_estimation/'
         'noise_results_lenses_2026_06_17')
ARCHIVE = '/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'noise_index.json'
AMASS_PREFIX = '/Users/drokeby/dpg_system/AMASS/'


def main():
    # archive files, keyed by a case-folded relative path for robust matching
    arch = {}
    for dp, _d, fns in os.walk(ARCHIVE):
        for fn in fns:
            if fn.endswith('.npz'):
                rel = os.path.relpath(os.path.join(dp, fn), ARCHIVE)
                arch[rel.lower()] = rel
    print(f'{len(arch)} files in archive')

    index = {}
    n_seen = n_matched = 0
    for jf in sorted(os.listdir(NOISE)):
        if not jf.endswith('.json') or jf == 'checkpoint.json':
            continue
        d = json.load(open(os.path.join(NOISE, jf)))
        for path, r in d['files'].items():
            n_seen += 1
            rel = path[len(AMASS_PREFIX):] if path.startswith(AMASS_PREFIX) else path
            key = rel.lower()
            if key not in arch:
                continue
            n_matched += 1
            drop = []
            surg = r.get('surgery') or {}
            for seg in (surg.get('segments') or []):
                if not seg.get('usable', True):
                    drop.append([int(seg['start']), int(seg['end'])])
            exc = surg.get('excision') or {}
            for z in (exc.get('zones') or []):
                drop.append([int(z['start']), int(z['end'])])
            # clean_segments is the noise work's own per-frame verdict, and it
            # exists for problematic files too -- 32% of their frames sit inside
            # one. Filtering by classification alone discards all of that.
            keep = [[int(s0['start']), int(s0['end'])]
                    for s0 in (r.get('clean_segments') or [])]
            index[arch[key]] = {
                'cls': r.get('classification', '?'),
                'n_frames': int(r.get('n_frames', 0)),
                'drop': drop,
                'keep': keep,
                'noise_score': float(r.get('noise_score', 0.0)),
            }
        print(f'  {jf:28s} cumulative matched {n_matched}/{n_seen}')

    missing = sorted(set(arch.values()) - set(index))
    print(f'\nmatched {len(index)} / {len(arch)} archive files')
    print(f'{len(missing)} archive files have NO noise verdict')
    from collections import Counter
    miss_sub = Counter(m.split(os.sep)[0] for m in missing)
    for k, v in miss_sub.most_common(10):
        print(f'    {k:24s} {v}')

    with open(OUT, 'w') as fh:
        json.dump(index, fh)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
