"""Compare full (torque+lenses) vs lenses-only classifications from a batch run.
Answers: is the torque pass redundant? Usage: python3 torque_ablation_diff.py <results_dir>"""
import json, os, sys, glob
from collections import defaultdict

results_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = []
for jp in glob.glob(os.path.join(results_dir, '*.json')):
    if os.path.basename(jp) in ('checkpoint.json',): continue
    try:
        d = json.load(open(jp))
    except Exception:
        continue
    if 'files' not in d: continue
    for path, r in d['files'].items():
        if not isinstance(r, dict) or 'classification' not in r: continue
        full = r.get('classification', '')
        lens = r.get('classification_lenses_only', '')
        tb   = r.get('classification_torque_base', '')
        if not lens:  # older record without the field
            continue
        rows.append((d.get('dataset', '?'), path, full, lens, tb,
                     r.get('noise_score', 0)))

N = len(rows)
print(f"TORQUE-ABLATION COMPARISON  —  {N} files\n")
if N == 0:
    print("no comparable records found"); sys.exit(0)

order = {'clean': 0, 'moderate': 1, 'problematic': 2}
def dist(idx):
    c = defaultdict(int)
    for r in rows: c[r[idx]] += 1
    return c

print("classification distribution:")
print(f"  {'verdict':12s} {'FULL(torque+lens)':>18s} {'LENSES-ONLY':>13s} {'TORQUE-BASE':>13s}")
for v in ('clean', 'moderate', 'problematic'):
    df, dl, dt = dist(2), dist(3), dist(4)
    print(f"  {v:12s} {df[v]:18d} {dl[v]:13d} {dt[v]:13d}")

# agreement full vs lenses-only
agree = sum(1 for r in rows if r[2] == r[3])
print(f"\nFULL == LENSES-ONLY: {agree}/{N}  ({100*agree/N:.2f}%)")

# torque escalations: full strictly more severe than lenses-only
esc = [r for r in rows if order.get(r[2],0) > order.get(r[3],0)]
print(f"\nfiles where TORQUE escalates beyond the lenses (full > lenses-only): {len(esc)}  ({100*len(esc)/N:.2f}%)")
cells = defaultdict(int)
for r in esc: cells[(r[3], r[2])] += 1
for (l, f), n in sorted(cells.items(), key=lambda x:-x[1]):
    print(f"   lenses={l:11s} -> full={f:11s} : {n}")

# the critical set: torque uniquely calls PROBLEMATIC
tonly_prob = [r for r in esc if r[2] == 'problematic']
print(f"\n*** TORQUE-ONLY problematic (full=problematic, lenses-only<problematic): {len(tonly_prob)} ***")
print("   (these are the files that justify keeping torque — inspect whether they are real)")
for r in sorted(tonly_prob, key=lambda x:-x[5])[:40]:
    print(f"   {r[0]:22s} lens={r[3]:9s} score={r[5]:5.1f}  {os.path.basename(r[1])}")

# reverse direction (should be ~zero by construction; sanity check)
rev = [r for r in rows if order.get(r[3],0) > order.get(r[2],0)]
print(f"\nsanity (lenses > full, should be 0): {len(rev)}")

# save the torque-only list
json.dump([{'dataset': r[0], 'path': r[1], 'full': r[2], 'lenses_only': r[3],
            'torque_base': r[4], 'noise_score': r[5]} for r in tonly_prob],
          open(os.path.join(results_dir, 'torque_only_problematic.json'), 'w'), indent=2)
print(f"\nfull torque-only-problematic list -> {results_dir}/torque_only_problematic.json")
