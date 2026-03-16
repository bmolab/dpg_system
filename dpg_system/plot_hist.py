import json
import argparse
import matplotlib.pyplot as plt


def load_scores(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    # extract noise scores
    scores = [d["noise_score"] for d in data if "noise_score" in d]
    return scores


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--json", required=True, help="Result JSON file")
    # parser.add_argument("--name", default="Dataset")
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Result JSON file")
    parser.add_argument("--name", default="Dataset")
    parser.add_argument("--out", default=None, help="Output image filename")
    args = parser.parse_args()

    scores = load_scores(args.json)

    print(f"{len(scores)} files loaded")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Linear scale ----
    ax[0].hist(scores, bins=60)
    ax[0].set_title("Noise Score Distribution (Linear)")
    ax[0].set_xlabel("Noise Score")
    ax[0].set_ylabel("Count")

    # ---- Log scale ----
    ax[1].hist(scores, bins=60)
    ax[1].set_yscale("log")
    ax[1].set_title("Noise Score Distribution (Log Count)")
    ax[1].set_xlabel("Noise Score")

    plt.tight_layout()

    # plt.figure(figsize=(8,5))
    # plt.hist(scores, bins=60)
    # plt.xlabel("Noise Score")
    # plt.ylabel("Count")
    # plt.yscale("log")
    # plt.title(f"Noise Score Distribution — {args.name}")

    # plt.show()
    # plt.savefig("CMU_hist.png", dpi=200)

    # choose output filename automatically
    out_path = args.out if args.out else f"{args.name}_hist.png"

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved histogram {out_path}")


if __name__ == "__main__":
    main()