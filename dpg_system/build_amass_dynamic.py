#!/usr/bin/env python
"""
build_amass_dynamic.py

Walk an entire AMASS SMPL-H dataset and, for every `*.npz` motion file, stream
it through the SMPLProcessor using the default `smpl_torque` node settings.
For each input file a parallel output `.npz` is written into a mirror directory
tree (default: AMASS_Dynamic). The output preserves ALL of the source file's
original keys/values (poses, trans, betas, gender, dmpls, mocap_framerate, ...)
and adds these arrays:

  Per-frame streams:
    torque              (T, 22, 3)  net joint torque vectors   (res['torques_vec'])
    torques_grav_vec    (T, 22, 3)  gravitational component    (res['torques_grav_vec'])
    torques_dyn_vec     (T, 22, 3)  dynamic component          (res['torques_dyn_vec'])
    torques_passive_vec (T, 22, 3)  passive-limit component    (res['torques_passive_vec'])
    contact_pressure    (T, J_cp)   per-joint supported mass   (res['contact_pressure'])
    angular_velocity    (T, 22, 3)  per-joint angular velocity (processor._current_ang_vel;
                                    world-frame because world_frame_dynamics=True)
    com_pos             (T, 3)      body CoM position  (processor.current_com)
    com_vel             (T, 3)      body CoM velocity  (processor.prob_prev_com_vel)
    com_acc             (T, 3)      body CoM accel     (processor.prob_prev_com_acc)

  File-level metadata:
    max_torque          (24, 3)     processor.max_torque_array — the per-joint
                                    max-torque profile that converts torque
                                    into effort (effort = torque / max_torque)
    total_mass_kg       ()          body mass used for dynamics
    processing_options  ()          JSON dump of the SMPLProcessingOptions used

combined_effort is intentionally NOT saved (derivable: torque / max_torque).

The processor is driven frame-by-frame in streaming mode with a FRESH
SMPLProcessor per file (it carries per-frame EMA/streaming state, so reusing it
across files would leak state and spike the frame-0 torque).

The option block mirrors the live `smpl_torque` node widget defaults, the same
block used by build_amass_torque.py and noise_estimation/estimate_noise_torque.py.

Resumable: files whose output already exists are skipped unless --overwrite.

Usage:
    python build_amass_dynamic.py                    # full run, single process
    python build_amass_dynamic.py --workers 8        # parallel
    python build_amass_dynamic.py --limit 5 -v       # smoke test on 5 files
"""

import os
import sys
import json
import time
import argparse
import traceback
import dataclasses

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
N_JOINTS = 22  # SMPLProcessor.target_joint_count

# model_path must be the directory that CONTAINS the `smplh/` folder
# (SMPLH_{GENDER}.pkl). This file lives in the dpg_system/ package dir,
# alongside smplh/.
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
# smpl_processor imports its siblings as `dpg_system.*`, so the REPO ROOT
# (parent of the package dir) must be importable.
REPO_ROOT = os.path.dirname(MODEL_PATH)

DEFAULT_SRC = "/home/bmolab/Projects/AMASS/SMPL_H"
DEFAULT_OUT = "/home/bmolab/Projects/AMASS_Dynamic/SMPL_H"
DEFAULT_MASS = 75.0

# SMPLProcessor / options are imported lazily inside build_options()/worker so
# that argument parsing and --help stay fast and so each multiprocessing worker
# imports them in its own process.


def build_options(dt):
    """Options matching the smpl_torque node's live widget defaults.

    Mirrors noise_estimation/estimate_noise_torque.py (the canonical offline
    batch config), NOT the raw SMPLProcessingOptions dataclass defaults.
    """
    from dpg_system.smpl_processor import SMPLProcessingOptions

    return SMPLProcessingOptions(
        input_type='axis_angle',
        input_up_axis='Y',
        axis_permutation='x, z, -y',
        quat_format='wxyz',
        return_quats=False,
        dt=dt,

        # Physics
        add_gravity=True,
        enable_passive_limits=True,
        enable_apparent_gravity=True,
        use_s_curve_spine=True,
        world_frame_dynamics=True,

        # Floor / contact
        floor_enable=True,
        floor_height=0.0,
        floor_tolerance=0.15,
        contact_method='logodds_valved',
        enable_body_contacts=True,

        # LogOdds evidence streams (match node defaults)
        logodds_enable_height=True,
        logodds_enable_kinematic=True,
        logodds_enable_structural=True,
        logodds_enable_divergence=True,
        logodds_weight_height=1.0,
        logodds_weight_kinematic=0.5,
        logodds_weight_structural=1.0,
        logodds_weight_divergence=1.0,
        logodds_decay_rate=0.90,
        logodds_struct_force_ema_alpha=1.0,
        logodds_struct_relief_logodds=0.3,

        # All rate limiting / filtering OFF (raw torques)
        enable_rate_limiting=False,
        enable_jitter_damping=False,
        enable_kf_smoothing=False,
        enable_velocity_gate=False,
        enable_one_euro_filter=False,
        smooth_input_window=0,
        smooth_contact_forces=False,

        # Acceleration smoothing (Savitzky-Golay derivative, fixed in TIME so
        # results are capture-rate independent; 70 ms ≈ the old window of 7
        # at Shadow's 100 Hz)
        acc_smooth_ms=70.0,
        acc_smooth_window=0,
        torque_smooth_window=0,

        # CoM One Euro filter params — match node widget defaults
        com_pos_min_cutoff=999.0,   # Position filter OFF
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,    # Velocity: light smoothing
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,     # Acceleration
        com_acc_beta=0.8,
    )


def options_to_json(options):
    """Serialize the SMPLProcessingOptions dataclass to a JSON string."""
    d = dataclasses.asdict(options)

    def _clean(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return v.item()
        return v

    return json.dumps({k: _clean(v) for k, v in d.items()}, default=str)


def _read_gender(d, override=None):
    gender = 'neutral'
    if override:
        return override
    if 'gender' in d:
        g = d['gender']
        if hasattr(g, 'item'):
            g = g.item()
        if isinstance(g, bytes):
            g = g.decode('utf-8')
        gender = str(g)
        if gender not in ('male', 'female', 'neutral'):
            gender = 'neutral'
    return gender


def process_file(src_file, out_file, total_mass=DEFAULT_MASS,
                 gender_override=None, verbose=False):
    """Process one AMASS npz → dynamics-annotated npz.

    Returns (status, message) where status is 'ok' | 'skip' | 'fail'.
    """
    from dpg_system.smpl_processor import SMPLProcessor

    d = np.load(src_file, allow_pickle=True)
    if 'poses' not in d or 'trans' not in d:
        return 'fail', 'missing poses/trans'

    poses = d['poses']
    trans = d['trans']
    T = poses.shape[0]
    if T == 0:
        return 'fail', 'zero frames'

    # Framerate: try multiple keys (matches estimate_noise_torque.py)
    fps = 60.0
    for key in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if key in d:
            fps = float(d[key])
            break
    dt = 1.0 / fps

    # Betas: optional (constructor truncates to 10 internally)
    betas = None
    if 'betas' in d:
        betas = np.array(d['betas'], dtype=np.float64)
        if betas.ndim > 1:
            betas = betas.flatten()

    gender = _read_gender(d, gender_override)

    # Fresh processor per file — it carries streaming/EMA state.
    processor = SMPLProcessor(
        framerate=fps,
        betas=betas,
        gender=gender,
        total_mass_kg=total_mass,
        model_path=MODEL_PATH,
    )
    processor.set_axis_permutation('x, z, -y')

    options = build_options(dt)

    torque = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    torques_grav = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    torques_dyn = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    torques_passive = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    angular_velocity = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    com_pos = np.zeros((T, 3), dtype=np.float32)
    com_vel = np.zeros((T, 3), dtype=np.float32)
    com_acc = np.zeros((T, 3), dtype=np.float32)
    contact_pressure = None  # allocated lazily once its joint-width is known
    n_failed_frames = 0

    def _grab_joints(res, key, dst, t):
        v = res.get(key, None)
        if v is not None:
            n = min(N_JOINTS, v.shape[1])
            dst[t, :n] = v[0, :n]

    for t in range(T):
        frame_pose = poses[t:t + 1]    # (1, 156)
        frame_trans = trans[t:t + 1]   # (1, 3)
        try:
            res = processor.process_frame(frame_pose, frame_trans, options)

            _grab_joints(res, 'torques_vec', torque, t)
            _grab_joints(res, 'torques_grav_vec', torques_grav, t)
            _grab_joints(res, 'torques_dyn_vec', torques_dyn, t)
            _grab_joints(res, 'torques_passive_vec', torques_passive, t)

            cp = res.get('contact_pressure', None)
            if cp is not None:
                cp = np.asarray(cp)
                if cp.ndim == 2:
                    cp = cp[0]
                if contact_pressure is None:
                    contact_pressure = np.zeros((T, cp.shape[0]), dtype=np.float32)
                n = min(contact_pressure.shape[1], cp.shape[0])
                contact_pressure[t, :n] = cp[:n]

            av = getattr(processor, '_current_ang_vel', None)
            if av is not None:
                av = np.asarray(av)
                if av.ndim == 3:
                    av = av[0]
                n = min(N_JOINTS, av.shape[0])
                angular_velocity[t, :n] = av[:n]

            c = getattr(processor, 'current_com', None)
            if c is None:
                c = getattr(processor, 'prob_prev_com', None)
            if c is not None:
                c = np.asarray(c)
                com_pos[t] = c[0] if c.ndim == 2 else c

            cv = getattr(processor, 'prob_prev_com_vel', None)
            if cv is not None:
                cv = np.asarray(cv)
                com_vel[t] = cv[0] if cv.ndim == 2 else cv

            ca = getattr(processor, 'prob_prev_com_acc', None)
            if ca is not None:
                ca = np.asarray(ca)
                com_acc[t] = ca[0] if ca.ndim == 2 else ca
        except Exception:
            # First few frames may produce partial results; leave zeros.
            n_failed_frames += 1

        if verbose and (t + 1) % 500 == 0:
            print(f"      frame {t + 1}/{T}", flush=True)

    if contact_pressure is None:
        contact_pressure = np.zeros((T, N_JOINTS), dtype=np.float32)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Carry over ALL original keys/values, then add the new arrays.
    out_data = {k: d[k] for k in d.files}
    out_data['torque'] = torque
    out_data['torques_grav_vec'] = torques_grav
    out_data['torques_dyn_vec'] = torques_dyn
    out_data['torques_passive_vec'] = torques_passive
    out_data['contact_pressure'] = contact_pressure
    out_data['angular_velocity'] = angular_velocity
    out_data['com_pos'] = com_pos
    out_data['com_vel'] = com_vel
    out_data['com_acc'] = com_acc
    out_data['max_torque'] = processor.max_torque_array.astype(np.float32)
    out_data['total_mass_kg'] = np.float32(total_mass)
    out_data['processing_options'] = np.array(options_to_json(options))
    # Write atomically (tmp then rename) so an interrupted run never leaves a
    # half-written npz that the resume logic would treat as "done".
    tmp = out_file + '.tmp.npz'
    np.savez_compressed(tmp, **out_data)
    os.replace(tmp, out_file)

    msg = f"{T} frames @ {fps:.0f}fps, gender={gender}"
    if n_failed_frames:
        msg += f" ({n_failed_frames} frame errs)"
    return 'ok', msg


def collect_npz(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            # `shape.npz` is a per-subject betas/gender file with no motion —
            # not a sequence, so it is not part of the work list.
            if f.endswith('.npz') and f != 'shape.npz':
                files.append(os.path.join(dirpath, f))
    files.sort()
    return files


def out_path_for(src_file, src_root, out_root):
    rel = os.path.relpath(src_file, src_root)
    return os.path.join(out_root, rel)


# ── Multiprocessing worker ───────────────────────────────────────────────
# Config is stashed in a module global so the worker only needs to pass the
# lightweight src_file path across the process boundary.
_WORKER_CFG = {}


def _worker_init(src_root, out_root, mass, gender_override, overwrite, verbose):
    _WORKER_CFG.update(dict(src_root=src_root, out_root=out_root, mass=mass,
                            gender_override=gender_override,
                            overwrite=overwrite, verbose=verbose))
    # Ensure `import dpg_system.smpl_processor` resolves in the worker process.
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def _worker(src_file):
    cfg = _WORKER_CFG
    # AMASS ships one `shape.npz` per subject — betas/gender only, no motion
    # (no poses/trans). Nothing to compute torque on; skip cleanly.
    if os.path.basename(src_file) == 'shape.npz':
        return src_file, 'skip', 'shape.npz (no motion)'
    out_file = out_path_for(src_file, cfg['src_root'], cfg['out_root'])
    if not cfg['overwrite'] and os.path.exists(out_file):
        return src_file, 'skip', 'exists'
    try:
        status, msg = process_file(
            src_file, out_file, total_mass=cfg['mass'],
            gender_override=cfg['gender_override'], verbose=cfg['verbose'])
        return src_file, status, msg
    except Exception as e:
        return src_file, 'fail', f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=DEFAULT_SRC,
                    help=f'AMASS source root (default: {DEFAULT_SRC})')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help=f'Parallel output root (default: {DEFAULT_OUT})')
    ap.add_argument('--mass', type=float, default=DEFAULT_MASS,
                    help='Body mass in kg (default: 75)')
    ap.add_argument('--gender', default=None, choices=['male', 'female', 'neutral'],
                    help='Override gender from file metadata')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel worker processes (default: 1)')
    ap.add_argument('--overwrite', action='store_true',
                    help='Reprocess files even if output already exists')
    ap.add_argument('--limit', type=int, default=0,
                    help='Process at most N files (0 = all; for testing)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='Per-frame progress within each file')
    args = ap.parse_args()

    # Make `import dpg_system.smpl_processor` work in the main process too.
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    if not os.path.isdir(args.src):
        print(f"ERROR: source root not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.src} ...", flush=True)
    all_files = collect_npz(args.src)
    print(f"Found {len(all_files)} npz files.", flush=True)

    # Pre-filter already-done files for an accurate work count / ETA.
    if not args.overwrite:
        todo = [f for f in all_files
                if not os.path.exists(out_path_for(f, args.src, args.out))]
        n_pre_skipped = len(all_files) - len(todo)
    else:
        todo = list(all_files)
        n_pre_skipped = 0

    if args.limit:
        todo = todo[:args.limit]

    print(f"Output root: {args.out}")
    print(f"To process: {len(todo)}  (already done, skipped: {n_pre_skipped})")
    print(f"Workers: {args.workers}, mass: {args.mass}kg"
          + (f", gender override: {args.gender}" if args.gender else ""), flush=True)
    if not todo:
        print("Nothing to do.")
        return

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    failures = []

    def report(i, src_file, status, msg):
        nonlocal n_ok, n_skip, n_fail
        if status == 'ok':
            n_ok += 1
        elif status == 'skip':
            n_skip += 1
        else:
            n_fail += 1
            failures.append((src_file, msg))
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - i) / rate if rate > 0 else 0
        rel = os.path.relpath(src_file, args.src)
        tag = {'ok': 'OK  ', 'skip': 'SKIP', 'fail': 'FAIL'}[status]
        short = msg.splitlines()[0] if msg else ''
        print(f"[{i}/{len(todo)}] {tag} {rel}  ({short})  "
              f"| {rate:.2f} files/s, ETA {remaining/60:.1f} min", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.workers,
                      initializer=_worker_init,
                      initargs=(args.src, args.out, args.mass, args.gender,
                                args.overwrite, args.verbose),
                      maxtasksperchild=50) as pool:
            for i, (src_file, status, msg) in enumerate(
                    pool.imap_unordered(_worker, todo, chunksize=1), 1):
                report(i, src_file, status, msg)
    else:
        _worker_init(args.src, args.out, args.mass, args.gender,
                     args.overwrite, args.verbose)
        for i, src_file in enumerate(todo, 1):
            src_file, status, msg = _worker(src_file)
            report(i, src_file, status, msg)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {elapsed/60:.1f} min. "
          f"ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    if failures:
        log_path = os.path.join(args.out, 'build_amass_dynamic_failures.log')
        os.makedirs(args.out, exist_ok=True)
        with open(log_path, 'w') as fh:
            for src_file, msg in failures:
                fh.write(f"{src_file}\n{msg}\n{'-'*40}\n")
        print(f"{n_fail} failures logged to {log_path}")


if __name__ == '__main__':
    main()
