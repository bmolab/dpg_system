import json
import argparse
import matplotlib.pyplot as plt


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


def load_scores(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    reports = iter_reports(data)
    noise = [r["noise_score"] for r in reports if "noise_score" in r]
    clean = [r["clean_section_score"] for r in reports if "clean_section_score" in r]
    return noise, clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Result JSON file")
    parser.add_argument("--name", default="Dataset")
    parser.add_argument("--out", default=None, help="Output image filename")
    args = parser.parse_args()

    noise, clean = load_scores(args.json)
    print(f"{len(noise)} files loaded "
          f"({len(clean)} with clean_section_score)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    def draw(a):
        a.hist(noise, bins=60, alpha=0.6, label="noise_score")
        if clean:
            a.hist(clean, bins=60, alpha=0.6, label="clean_section_score")
        a.set_xlabel("Score")
        a.legend()

    # ---- Linear scale ----
    draw(ax[0])
    ax[0].set_title(f"Score Distribution — {args.name} (Linear)")
    ax[0].set_ylabel("Count")

    # ---- Log scale ----
    draw(ax[1])
    ax[1].set_yscale("log")
    ax[1].set_title(f"Score Distribution — {args.name} (Log Count)")

    plt.tight_layout()

    out_path = args.out if args.out else f"{args.name}_hist.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved histogram {out_path}")


if __name__ == "__main__":
    main()
