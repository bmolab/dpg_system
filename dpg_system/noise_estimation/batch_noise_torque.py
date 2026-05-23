"""
In-process batch runner for estimate_noise_torque.

Imports `analyze_file` directly and calls it in a loop — avoids the
~2-3 sec Python+numpy+scipy import cost per file that a subprocess
runner would pay.  Supports parallel processing across multiple
worker subprocesses via --workers N.

Safety note: a fresh SMPLProcessor is created inside analyze_file()
for every file, so EMAs and other stateful filters (prob_prev_*,
prev_zmp_smooth, _ang_sg_cache, _com_sg_cache) reset between files.
NEVER reuse a SMPLProcessor instance across files — the EMAs carry
state and would contaminate downstream files.

Parallel mode: each worker is its own Python subprocess with its own
SMPLProcessor instances and its own smplx model cache.  Workers do
not share any mutable state with each other or the main process.

Usage:
    python batch_noise_torque.py [--amass-root PATH] [--output-dir PATH]
                                  [--workers N] [--resume] [--limit N]
"""

import os
import json
import argparse
import datetime
import time
from dataclasses import asdict
from multiprocessing import Pool, cpu_count

import numpy as np


DEFAULT_AMASS_ROOT = os.path.abspath("../../AMASS")


def collect_npz_files(root):
    files = []
    skip_dirs = {
        "venv", ".venv", "__pycache__", "site-packages",
        "tests", "test", "support_data", "github_data",
        "sample_data", "samples", "downloads", "dowloads",
        ".git", ".idea", ".vscode",
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


def save_result_by_folder(filepath, report_dict, amass_root, results_dir):
    """Append a single report to the per-dataset JSON file.
    Only safe to call from the main process (no cross-process locking)."""
    os.makedirs(results_dir, exist_ok=True)
    dataset = get_dataset_name(filepath, amass_root)
    out_path = os.path.join(results_dir, f"{dataset}.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            folder_data = json.load(f)
    else:
        folder_data = {"dataset": dataset, "files": {}}

    folder_data["files"][filepath] = report_dict

    # Sort by descending noise_score so noisiest files surface first
    sorted_items = sorted(
        folder_data["files"].items(),
        key=lambda x: (-x[1].get("noise_score", 0), x[0]),
    )
    folder_data["files"] = dict(sorted_items)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(folder_data, f, indent=2)
    os.replace(tmp_path, out_path)


def load_processed_from_folder_jsons(results_dir):
    """Treat the per-folder JSONs as source of truth for what's done."""
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
                processed.update(files_dict)
        except Exception as e:
            print(f"Warning: could not read {fpath}: {e}")
    return processed


def format_eta(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


# ───────────────────────────────────────────────────────────────────────────
# Worker process function.  Runs in a subprocess; must be picklable.
# Imports estimate_noise_torque inside the worker so each worker has its own
# independent smplx model cache (loaded once per worker, reused across files
# within that worker).  Workers do NOT share mutable state with each other
# or the main process.
# ───────────────────────────────────────────────────────────────────────────

def _worker_analyze(args):
    """Analyze one file in a worker subprocess.

    Args is a tuple (filepath, smooth_window).  Returns
    (filepath, report_dict, error_str, elapsed_seconds).
    error_str is None on success.

    EMA safety: estimate_noise_torque.analyze_file() creates a fresh
    SMPLProcessor instance per call, so EMAs and sequential-filter
    state reset between files even within the same worker.
    """
    filepath, smooth_window = args
    t0 = time.time()
    try:
        # Imported inside the function so it's also imported per-worker
        # under multiprocessing 'spawn' semantics.  Module-level imports
        # in the parent process are NOT inherited by spawn workers.
        import estimate_noise_torque as ent
        from dataclasses import asdict
        report = ent.analyze_file(
            filepath,
            smooth_input_window=smooth_window,
            verbose=False,
        )
        report_dict = ent._to_jsonable(asdict(report))
        return filepath, report_dict, None, time.time() - t0
    except Exception as e:
        return filepath, None, f"{type(e).__name__}: {e}", time.time() - t0


def _sequential_loop(remaining, smooth_window, results_dir, amass_root,
                     checkpoint_path, done):
    """Run files one at a time in this process (workers=1 path)."""
    import estimate_noise_torque as ent

    start_time = time.time()
    n_ok = 0
    n_fail = 0
    failures = []

    for i, filepath in enumerate(remaining, 1):
        elapsed = time.time() - start_time
        rate = (i - 1) / elapsed if elapsed > 0 and i > 1 else 0
        eta = (len(remaining) - i + 1) / rate if rate > 0 else 0
        print(f"[{i}/{len(remaining)}] ({n_ok} ok, {n_fail} fail, "
              f"ETA {format_eta(eta)}) {filepath}", flush=True)

        try:
            t0 = time.time()
            report = ent.analyze_file(
                filepath, smooth_input_window=smooth_window, verbose=False)
            dt = time.time() - t0
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            n_fail += 1
            err = f"{type(e).__name__}: {e}"
            failures.append((filepath, err))
            print(f"  FAILED ({err})")
            continue

        try:
            report_dict = ent._to_jsonable(asdict(report))
        except Exception as e:
            n_fail += 1
            failures.append((filepath, f"serialize: {e}"))
            print(f"  SERIALIZE FAILED: {e}")
            continue

        save_result_by_folder(filepath, report_dict, amass_root, results_dir)
        done[filepath] = {"noise_score": report_dict.get("noise_score", 0)}
        save_checkpoint(checkpoint_path, {"processed_files": done})

        n_ok += 1
        print(f"  OK  cls={report.classification:11s}  "
              f"noise={report.noise_score:6.2f}  "
              f"clean_sec={report.clean_section_score:6.2f}  ({dt:.1f}s)")

    return n_ok, n_fail, failures, time.time() - start_time


def _parallel_loop(remaining, smooth_window, n_workers, results_dir,
                   amass_root, checkpoint_path, done):
    """Run files in parallel across N worker subprocesses.

    Workers stream results back via imap_unordered; the main process
    handles all disk writes (results JSON + checkpoint) so there's no
    cross-process write contention.
    """
    print(f"Running with {n_workers} parallel workers.")
    print(f"(First file per worker pays the smplx-model load cost ~5-10s; "
          f"subsequent files in that worker hit the cache.)\n")

    work_items = [(f, smooth_window) for f in remaining]

    start_time = time.time()
    n_ok = 0
    n_fail = 0
    n_done = 0
    failures = []

    try:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_worker_analyze, work_items, chunksize=1):
                filepath, report_dict, err, dt = result
                n_done += 1
                elapsed = time.time() - start_time
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - n_done) / rate if rate > 0 else 0

                if err is not None:
                    n_fail += 1
                    failures.append((filepath, err))
                    print(f"[{n_done}/{len(remaining)}] ({n_ok} ok, "
                          f"{n_fail} fail, ETA {format_eta(eta)})  "
                          f"FAILED: {filepath}\n  {err}", flush=True)
                    continue

                save_result_by_folder(filepath, report_dict, amass_root, results_dir)
                done[filepath] = {"noise_score": report_dict.get("noise_score", 0)}
                # Save checkpoint every 10 successful files to avoid
                # disk thrash at high parallelism.
                if n_ok % 10 == 0:
                    save_checkpoint(checkpoint_path, {"processed_files": done})

                n_ok += 1
                cls = report_dict.get("classification", "?")
                ns = report_dict.get("noise_score", 0)
                css = report_dict.get("clean_section_score", 0)
                print(f"[{n_done}/{len(remaining)}] ({n_ok} ok, "
                      f"{n_fail} fail, ETA {format_eta(eta)})  "
                      f"cls={cls:11s} noise={ns:6.2f} clean_sec={css:6.2f}  "
                      f"({dt:.1f}s)  {filepath}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Final checkpoint save
    save_checkpoint(checkpoint_path, {"processed_files": done})
    return n_ok, n_fail, failures, time.time() - start_time


def main():
    p = argparse.ArgumentParser(
        description="In-process batch noise analysis over AMASS.")
    p.add_argument("--amass-root", default=DEFAULT_AMASS_ROOT,
                   help=f"AMASS root directory (default: {DEFAULT_AMASS_ROOT})")
    p.add_argument("--output-dir",
                   help="Output directory for results (default: torque_results_<DATE>)")
    p.add_argument("--checkpoint",
                   help="Checkpoint file (default: <output-dir>/checkpoint.json)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker subprocesses (default: 1 = sequential).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing run; skip files already in the output directory.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N files this run (for testing).")
    p.add_argument("--smooth-window", type=int, default=0,
                   help="Input smoothing window (0=off, 3/5/7).")
    args = p.parse_args()

    today = datetime.date.today().isoformat()
    if args.output_dir is None:
        args.output_dir = os.path.abspath(f"torque_results_{today}")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, "checkpoint.json")

    if args.workers < 1:
        args.workers = 1
    if args.workers > cpu_count():
        print(f"Warning: requested {args.workers} workers but only "
              f"{cpu_count()} CPUs available.  Clamping.")
        args.workers = cpu_count()

    amass_root = args.amass_root
    results_dir = args.output_dir
    checkpoint_path = args.checkpoint

    print(f"AMASS root:    {amass_root}")
    print(f"Output dir:    {results_dir}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Workers:       {args.workers}")
    print(f"Resume mode:   {args.resume}")
    print()

    if not os.path.isdir(amass_root):
        print(f"Error: AMASS root does not exist: {amass_root}")
        return 2

    print("Scanning AMASS for .npz files ...")
    all_candidates = collect_npz_files(amass_root)
    print(f"  found {len(all_candidates)} candidate .npz files")
    print("Filtering to valid mocap files (poses + trans) ...")
    all_files = [f for f in all_candidates if is_valid_mocap_npz(f)]
    print(f"  {len(all_files)} valid mocap files")
    print()

    done = {}
    if args.resume:
        done = load_processed_from_folder_jsons(results_dir)
        ckpt = load_checkpoint(checkpoint_path)
        for k, v in ckpt.get("processed_files", {}).items():
            done.setdefault(k, v)
        print(f"Resume: {len(done)} files already processed, skipping those.")

    remaining = [f for f in all_files if f not in done]
    if args.limit is not None:
        remaining = remaining[:args.limit]
        print(f"Limit:  processing at most {args.limit} files this run.")

    if not remaining:
        print("Nothing to do.")
        return 0
    print(f"Remaining: {len(remaining)} files to process")
    print()

    os.makedirs(results_dir, exist_ok=True)

    if args.workers == 1:
        n_ok, n_fail, failures, elapsed = _sequential_loop(
            remaining, args.smooth_window, results_dir, amass_root,
            checkpoint_path, done)
    else:
        n_ok, n_fail, failures, elapsed = _parallel_loop(
            remaining, args.smooth_window, args.workers, results_dir,
            amass_root, checkpoint_path, done)

    print()
    print(f"Run finished in {format_eta(elapsed)}")
    print(f"  ok:    {n_ok}")
    print(f"  fail:  {n_fail}")
    if failures:
        print(f"  First few failures:")
        for fp, err in failures[:5]:
            print(f"    {fp}")
            print(f"      {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
