import json
import glob
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))


def iter_reports(data):
    """Yield per-file report dicts, tolerant of both result schemas.

    Old noise reports are a flat list of report dicts.  Torque results are
    {"dataset": name, "files": {npz_path: report_dict, ...}}.
    """
    if isinstance(data, dict) and isinstance(data.get('files'), dict):
        return list(data['files'].values())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'noise_score' in data:
        return [data]
    return []


def dataset_name(json_path, data):
    if isinstance(data, dict) and data.get('dataset'):
        return data['dataset']
    return os.path.basename(json_path).replace('result_', '').replace('.json', '')


def stats(values):
    """(mean, p95, max) using the original P95 index convention."""
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    mean = sum(s) / len(s)
    p95 = s[int(0.95 * (len(s) - 1))]
    return mean, p95, s[-1]


def discover_json_files(args):
    """Resolve CLI args (files/dirs) to a list of JSON files.

    With no args, prefer the newest torque_results_* dir, else noise_reports.
    """
    if args:
        files = []
        for a in args:
            if os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, '*.json'))))
            else:
                files.append(a)
        return files

    torque_dirs = sorted(glob.glob(os.path.join(_this_dir, 'torque_results_*')))
    torque_dirs = [d for d in torque_dirs if os.path.isdir(d)]
    if torque_dirs:
        return sorted(glob.glob(os.path.join(torque_dirs[-1], '*.json')))
    return sorted(glob.glob(os.path.join(_this_dir, 'noise_reports', 'result_*.json')))


def main():
    files = discover_json_files(sys.argv[1:])
    if not files:
        print('No result JSON files found. Pass files or a directory as arguments.')
        return

    rows = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        reports = iter_reports(data)
        noise = [r['noise_score'] for r in reports if 'noise_score' in r]
        clean = [r['clean_section_score'] for r in reports if 'clean_section_score' in r]
        if not noise:
            continue

        rows.append((dataset_name(f, data), len(noise), stats(noise), stats(clean)))

    # rank datasets by whole-file noise (robust choice = P95)
    rows.sort(key=lambda r: -r[2][1])

    print(f"{'Dataset':<20} {'Files':>6} "
          f"{'Mean':>8} {'P95':>8} {'Max':>8}   "
          f"{'cMean':>8} {'cP95':>8} {'cMax':>8}")
    print(f"{'':<20} {'':>6} {'─ noise_score ─':>26}   {'─ clean_section_score ─':>26}")
    for name, n, (nm, np95, nmx), (cm, cp95, cmx) in rows:
        print(f"{name:<20} {n:6d} "
              f"{nm:8.3f} {np95:8.3f} {nmx:8.3f}   "
              f"{cm:8.3f} {cp95:8.3f} {cmx:8.3f}")


if __name__ == '__main__':
    main()
