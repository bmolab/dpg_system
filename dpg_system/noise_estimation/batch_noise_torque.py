import os
import json
import subprocess
import numpy as np

AMASS_ROOT = os.path.abspath("../../AMASS")
CHECKPOINT = "torque_checkpoint.json"
TEMP_JSON = "temp_single_result.json"
RESULTS_DIR = "torque_results_by_folder"


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"processed_files": {}}


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def collect_npz_files(root):
    files = []

    skip_dirs = {
        "venv",
        ".venv",
        "__pycache__",
        "site-packages",
        "tests",
        "test",
        "support_data",
        "github_data",
        "sample_data",
        "samples",
        "downloads",
        "dowloads",
        ".git",
        ".idea",
        ".vscode",
    }

    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for f in sorted(fs):
            if f.endswith(".npz"):
                files.append(os.path.abspath(os.path.join(r, f)))

    return files


def is_valid_mocap_npz(filepath):
    try:
        with np.load(filepath, allow_pickle=True) as d:
            return ("poses" in d.files) and ("trans" in d.files)
    except Exception:
        return False


def get_dataset_name(filepath, amass_root):
    rel = os.path.relpath(filepath, amass_root)
    parts = rel.split(os.sep)

    if len(parts) < 1:
        raise ValueError(f"Cannot determine dataset folder for {filepath}")

    if parts[0] in {"SMPL_H", "SMPL_X", "SMPL"}:
        if len(parts) < 2:
            raise ValueError(f"Cannot determine dataset folder for {filepath}")
        return parts[1]

    return parts[0]


def save_result_by_folder(filepath, report, amass_root, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    dataset = get_dataset_name(filepath, amass_root)
    out_path = os.path.join(results_dir, f"{dataset}.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            folder_data = json.load(f)
    else:
        folder_data = {
            "dataset": dataset,
            "files": {}
        }

    folder_data["files"][filepath] = report

    sorted_items = sorted(
        folder_data["files"].items(),
        key=lambda x: (-x[1].get("noise_score", 0), x[0])
    )
    folder_data["files"] = dict(sorted_items)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(folder_data, f, indent=2)
    os.replace(tmp_path, out_path)


def load_processed_from_folder_jsons(results_dir):
    processed = {}

    if not os.path.exists(results_dir):
        return processed

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)

            files_dict = data.get("files", {})
            if isinstance(files_dict, dict):
                for filepath, report in files_dict.items():
                    processed[filepath] = report
        except Exception as e:
            print(f"Warning: could not read {fpath}: {e}")

    return processed


def main():
    checkpoint = load_checkpoint(CHECKPOINT)
    done_from_checkpoint = checkpoint.get("processed_files", {})

    done_from_folder_jsons = load_processed_from_folder_jsons(RESULTS_DIR)

    # folder JSONs are source of truth; checkpoint fills in anything extra
    done = dict(done_from_folder_jsons)
    for k, v in done_from_checkpoint.items():
        if k not in done:
            done[k] = v

    checkpoint["processed_files"] = done
    save_checkpoint(CHECKPOINT, checkpoint)

    all_candidates = collect_npz_files(AMASS_ROOT)
    all_files = [f for f in all_candidates if is_valid_mocap_npz(f)]
    remaining = [f for f in all_files if f not in done]

    if len(remaining) == 0:
        print("\nAll files have been processed. Exiting.")
        return

    print(f"Total .npz files found: {len(all_candidates)}")
    print(f"Valid mocap files found: {len(all_files)}")
    print(f"Already processed: {len(done)}")
    print(f"Remaining: {len(remaining)}")

    for i, filepath in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Processing {filepath}")

        if os.path.exists(TEMP_JSON):
            os.remove(TEMP_JSON)

        # NOTE FOR FUTURE DETECTOR INTEGRATION:
        # This batch runner assumes the detector script:
        #   1. accepts a single input filepath as a positional argument
        #   2. supports "--json <output_path>" to save one-file results
        #   3. exits with return code 0 on success
        #   4. writes JSON in the same general structure expected below
        #
        # If a new noise detector is introduced later, this command may need to be
        # updated to call the new script name and/or use different CLI arguments.

        cmd = [
            "python", "estimate_noise_torque.py",  # change this to new noise detector
            filepath,
            "--json", TEMP_JSON
        ]

        try:
            result = subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

        if result.returncode != 0:
            print(f"Failed: {filepath}")
            continue

        if not os.path.exists(TEMP_JSON):
            print(f"No temp JSON produced: {filepath}")
            continue

        # NOTE FOR FUTURE DETECTOR INTEGRATION:
        # This batch runner currently assumes the detector writes a non-empty JSON list
        # for a single-file run, so we read data[0] below.
        #
        # If a future detector writes:
        #   - a single dict instead of a list
        #   - a differently named top-level field
        #   - additional wrapper metadata
        # then this parsing logic must be updated accordingly.

        with open(TEMP_JSON, "r") as f:
            data = json.load(f)

        if not data:
            print(f"Empty result for: {filepath}")
            continue

        # Assumption: single-file detector run returns exactly one report entry.
        # If a future detector changes output shape, update this assignment.

        done[filepath] = data[0]
        checkpoint["processed_files"] = done

        save_result_by_folder(filepath, data[0], AMASS_ROOT, RESULTS_DIR)
        print(f"Saved result to folder JSON for {get_dataset_name(filepath, AMASS_ROOT)}")

        save_checkpoint(CHECKPOINT, checkpoint)
        print(f"Saved checkpoint ({len(done)} done)")

        print(f"\nProcessed total now: {len(done)} / {len(all_files)}")

    print("\nRun finished.")


if __name__ == "__main__":
    main()