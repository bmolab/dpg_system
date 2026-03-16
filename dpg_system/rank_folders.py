import json
import glob
import os

rows = []

for f in glob.glob("noise_reports/result_*.json"):
    with open(f) as fp:
        data = json.load(fp)

    scores = [d["noise_score"] for d in data if "noise_score" in d]

    if not scores:
        continue

    name = os.path.basename(f).replace("result_", "").replace(".json", "")

    mean = sum(scores) / len(scores)
    p95 = sorted(scores)[int(0.95*(len(scores)-1))]
    worst = max(scores)

    rows.append((name, mean, p95, worst, len(scores)))

# rank datasets (robust choice = P95)
rows.sort(key=lambda x: -x[2])

print(f"{'Dataset':<20} {'Files':>6} {'Mean':>8} {'P95':>8} {'Max':>8}")
for r in rows:
    print(f"{r[0]:<20} {r[4]:6d} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f}")