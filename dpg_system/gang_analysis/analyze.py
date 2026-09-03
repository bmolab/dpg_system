"""Read the characterization output and answer the two questions.

A. CONDITIONING -- what range does each gang occupy, and what would it take
   to make it usable by a mapping in any direction.

B. STRUCTURE -- what does the torque field's own co-variation look like, and
   how do the 15 hand-written presets stand up against it. The presets are
   treated as a hypothesis to be scored, not as ground truth.
"""
import json
import os
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import dpg_system.gang_core as gc

OUT = sys.argv[1] if len(sys.argv) > 1 else 'full'
NAMES = {v: k for k, v in gc.JOINT_INDEX.items()}
AXES = ('x', 'y', 'z')
CH = [f'{NAMES[j]}.{AXES[a]}' for j in range(22) for a in range(3)]
CHI = {c: i for i, c in enumerate(CH)}


def preset_vectors():
    """Each preset/side as a unit vector in the 66-channel space."""
    vecs = {}
    for preset in gc.preset_names():
        for side in (gc.sides_for(preset) or ['none']):
            spec = gc.spec_from_preset(preset, side=side, stream='total')
            v = np.zeros(66)
            for j, a, w in spec.terms:
                if j < 22:
                    v[j * 3 + a] += w
            n = np.linalg.norm(v)
            if n > 0:
                vecs[f'{preset}|{side}'] = (v / n, spec.terms)
    return vecs


def main():
    cond = json.load(open(os.path.join(OUT, 'conditioning.json')))
    corr = np.load(os.path.join(OUT, 'correlation.npz'), allow_pickle=True)
    print(f"corpus: {cond['total_frames']:,} frames\n")

    rows = {(r['stream'], r['gang']): r for r in cond['rows']}
    vecs = preset_vectors()

    # ---------------------------------------------------------------- A ---
    print('=' * 96)
    print('A. CONDITIONING  -- dynamic range of each gang (stream=total)')
    print('=' * 96)
    print('crest = p99/p50: how many times the median a loud moment is.')
    print('A linear map of a high-crest signal sits near zero almost always.\n')
    print(f"{'gang':32s}{'p50':>8s}{'p90':>8s}{'p99':>8s}{'p99.9':>8s}"
          f"{'crest':>7s}{'coh10':>7s}{'coh50':>7s}{'coh90':>7s}{'terms':>6s}")
    tr = [r for r in cond['rows'] if r['stream'] == 'total']
    for r in sorted(tr, key=lambda z: -z['net_p99']):
        nt = len(vecs.get(r['gang'], (None, ()))[1])
        crest = r['net_p99'] / max(r['net_p50'], 1e-9)
        print(f"{r['gang']:32s}{r['net_p50']:8.4f}{r['net_p90']:8.4f}"
              f"{r['net_p99']:8.4f}{r['net_p999']:8.4f}{crest:7.1f}"
              f"{r['coh_p10']:7.3f}{r['coh_p50']:7.3f}{r['coh_p90']:7.3f}{nt:6d}")

    print('\ncrest factor by stream:')
    for s in ('total', 'gravity', 'dynamic'):
        rr = [r for r in cond['rows'] if r['stream'] == s]
        cr = np.array([r['net_p99'] / max(r['net_p50'], 1e-9) for r in rr])
        p99 = np.array([r['net_p99'] for r in rr])
        print(f'  {s:9s} median crest={np.median(cr):6.1f}   '
              f'p99 spans [{p99.min():.4f}, {p99.max():.4f}] '
              f'= {p99.max() / max(p99.min(), 1e-9):.0f}x across gangs')

    # coherence, split by term count -- single-term gangs are 1.0 by
    # construction and must not be counted as evidence either way
    print('\ncoherence, separating single-term gangs (coherence==1 trivially):')
    for s in ('total', 'gravity', 'dynamic'):
        multi = [r for r in cond['rows'] if r['stream'] == s
                 and len(vecs.get(r['gang'], (None, ()))[1]) > 1]
        single = [r for r in cond['rows'] if r['stream'] == s
                  and len(vecs.get(r['gang'], (None, ()))[1]) == 1]
        m10 = np.nanmedian([r['coh_p10'] for r in multi])
        m50 = np.nanmedian([r['coh_p50'] for r in multi])
        print(f'  {s:9s} multi-term (n={len(multi)}): '
              f'median coh_p10={m10:.3f} coh_p50={m50:.3f}   '
              f'single-term (n={len(single)}): all ~1.0')

    # ---------------------------------------------------------------- B ---
    print('\n' + '=' * 96)
    print('B. STRUCTURE  -- the field\'s own co-variation')
    print('=' * 96)

    for stream in ('total', 'dynamic'):
        key = f'__all____{stream}'
        if key not in corr:
            continue
        C = np.array(corr[key], np.float64)
        live = ~np.isnan(np.diag(C))
        idx = np.where(live)[0]
        Cl = C[np.ix_(idx, idx)]
        w, V = np.linalg.eigh(Cl)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        var = w / w.sum()

        print(f'\n--- stream={stream}: {len(idx)} live channels of 66 ---')
        print(f'variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}  '
              f'PC3={var[2]:.1%}  PC1-5={var[:5].sum():.1%}  '
              f'PC1-10={var[:10].sum():.1%}')
        eff = np.exp(-(var * np.log(var + 1e-30)).sum())
        print(f'participation ratio (effective dimensionality): {eff:.1f} of {len(idx)}')

        print('\n  data-derived gangs (top 3 PCs, channels with |loading|>0.15):')
        for k in range(3):
            load = V[:, k]
            big = np.argsort(-np.abs(load))[:9]
            terms = ', '.join(f'{CH[idx[b]]}{"+" if load[b] > 0 else "-"}'
                              f'{abs(load[b]):.2f}'
                              for b in big if abs(load[b]) > 0.15)
            print(f'    PC{k + 1} ({var[k]:5.1%}): {terms}')

        # -- score each preset against the data --------------------------
        print(f'\n  presets scored against this correlation structure:')
        print(f'    {"gang":30s}{"coh_data":>10s}{"var_ratio":>11s}  verdict')
        scores = []
        rng = np.random.default_rng(0)
        for name, (v, terms) in vecs.items():
            vv = v[idx]
            nrm = np.linalg.norm(vv)
            if nrm < 1e-9 or len(terms) < 2:
                continue
            vv = vv / nrm
            # variance captured by this direction, relative to a random
            # direction in the same space: >1 means the preset points along
            # real structure, ~1 means it is no better than arbitrary
            q = float(vv @ Cl @ vv)
            rand = rng.normal(size=(400, len(idx)))
            rand /= np.linalg.norm(rand, axis=1, keepdims=True)
            base = float(np.mean(np.einsum('ij,jk,ik->i', rand, Cl, rand)))
            # mean signed pairwise correlation among the preset's own terms:
            # does the data agree these channels move together in the sense
            # the weights assert?
            ii = [j * 3 + a for j, a, _w in terms if j < 22]
            ww = [w_ for j, _a, w_ in terms if j < 22]
            pair, n_pair = 0.0, 0
            for p in range(len(ii)):
                for q2 in range(p + 1, len(ii)):
                    if ii[p] in idx and ii[q2] in idx:
                        a_ = np.where(idx == ii[p])[0][0]
                        b_ = np.where(idx == ii[q2])[0][0]
                        pair += np.sign(ww[p] * ww[q2]) * Cl[a_, b_]
                        n_pair += 1
            pair = pair / n_pair if n_pair else float('nan')
            scores.append((name, pair, q / base))
        for name, pair, ratio in sorted(scores, key=lambda z: -z[2]):
            if ratio > 2.0:
                verdict = 'strong -- data agrees'
            elif ratio > 1.2:
                verdict = 'weak'
            elif pair < -0.05:
                verdict = 'CONTRARY -- terms anti-correlate as weighted'
            else:
                verdict = 'no better than a random direction'
            print(f'    {name:30s}{pair:10.3f}{ratio:11.2f}  {verdict}')

        # -- strong co-variation the presets do not capture ---------------
        print(f'\n  strongest channel pairs NOT covered by any preset:')
        covered = set()
        for _n, (_v, terms) in vecs.items():
            ii = [j * 3 + a for j, a, _w in terms if j < 22]
            for p in range(len(ii)):
                for q2 in range(p + 1, len(ii)):
                    covered.add(frozenset((ii[p], ii[q2])))
        cand = []
        for a_ in range(len(idx)):
            for b_ in range(a_ + 1, len(idx)):
                if frozenset((idx[a_], idx[b_])) in covered:
                    continue
                cand.append((abs(Cl[a_, b_]), Cl[a_, b_], idx[a_], idx[b_]))
        cand.sort(reverse=True)
        for _m, c, a_, b_ in cand[:14]:
            print(f'    {CH[a_]:22s} {CH[b_]:22s} r={c:+.3f}')


if __name__ == '__main__':
    main()
