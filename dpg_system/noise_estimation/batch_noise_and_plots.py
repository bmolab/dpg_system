import argparse
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amass_root", default="../AMASS", help="AMASS root folder")
    p.add_argument("--outdir", default="noise_reports", help="Where to save JSON + plots")
    p.add_argument("--pattern", default="*", help="Which top-level datasets to include (glob)")
    args = p.parse_args()

    amass_root = Path(args.amass_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    estimate_py = Path("estimate_noise_Lucy.py").resolve()
    plot_py = Path("plot_hist.py").resolve()

    # Top-level AMASS datasets = subfolders under AMASS
    dataset_dirs = sorted([d for d in amass_root.glob(args.pattern) if d.is_dir()])

    print(f"Found {len(dataset_dirs)} dataset folders under {amass_root}")

    for d in dataset_dirs:
        name = d.name
        json_out = outdir / f"result_{name}.json"
        png_out = outdir / f"hist_{name}.png"

        print(f"\n=== {name} ===")

        # 1) Run noise estimation recursively
        cmd1 = ["python", str(estimate_py), "--dir", str(d), "--json", str(json_out)]
        print("Running:", " ".join(cmd1))
        subprocess.run(cmd1, check=False)

        # 2) Plot histogram from JSON (plot script should save + close)
        cmd2 = ["python", str(plot_py), "--json", str(json_out), "--name", name, "--out", str(png_out)]
        # ^ if your plot script doesn't have --out, see note below
        print("Plotting:", " ".join(cmd2))
        subprocess.run(cmd2, check=False)

    print(f"\nDone. Outputs in: {outdir}")


if __name__ == "__main__":
    main()