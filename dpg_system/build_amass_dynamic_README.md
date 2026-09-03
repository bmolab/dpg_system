# build_amass_dynamic.py — usage notes

Batch tool that walks an AMASS SMPL-H dataset and, for every motion `.npz`,
streams it through the `dpg_system` SMPLProcessor to produce a parallel output
tree annotated with joint dynamics.

## What it produces

The output mirrors the source directory tree exactly. Each output `.npz`
preserves **all** of the source file's original keys (`poses`, `trans`, `betas`,
`gender`, `dmpls`, `mocap_framerate`, ...) and adds:

Per-frame streams, where `T` is the frame count:

| key | shape | meaning |
| --- | --- | --- |
| `torque` | (T, 22, 3) | net joint torque vectors |
| `torques_grav_vec` | (T, 22, 3) | gravitational component |
| `torques_dyn_vec` | (T, 22, 3) | dynamic component |
| `torques_passive_vec` | (T, 22, 3) | passive joint-limit component |
| `contact_pressure` | (T, J) | per-joint supported mass |
| `angular_velocity` | (T, 22, 3) | per-joint angular velocity, world frame |
| `com_pos` | (T, 3) | whole-body centre-of-mass position |
| `com_vel` | (T, 3) | centre-of-mass velocity |
| `com_acc` | (T, 3) | centre-of-mass acceleration |

File-level metadata:

| key | meaning |
| --- | --- |
| `max_torque` | (24, 3) per-joint max-torque profile; effort = torque / max_torque |
| `total_mass_kg` | body mass used for the dynamics |
| `processing_options` | JSON dump of the exact options used for this file |

Combined effort is deliberately *not* stored, since it is derivable as
`torque / max_torque`.

Angular velocity is world-frame because the run uses `world_frame_dynamics=True`.

## Running it

```
python build_amass_dynamic.py --src <AMASS/SMPL_H> --out <destination> --workers 12
```

The `--src` and `--out` defaults hardcoded in the script are absolute paths on
the machine it was written on, so both should be given explicitly anywhere else.

| flag | default | meaning |
| --- | --- | --- |
| `--src` | (hardcoded) | AMASS source root; walked recursively for `*.npz` |
| `--out` | (hardcoded) | output root; mirrors the source tree |
| `--mass` | 75.0 | body mass in kg |
| `--gender` | off | force `male`/`female`/`neutral`, overriding file metadata |
| `--workers` | 1 | parallel worker processes |
| `--overwrite` | off | reprocess files that already have output |
| `--limit N` | 0 (all) | process at most N files — use for a smoke test |
| `-v` | off | per-frame progress within each file |

Smoke test first:

```
python build_amass_dynamic.py --src <...> --out <...> --limit 5 -v
```

A reference full run over 14,279 files took 60.6 minutes at `--workers 12` with
no failures.

## Two things that will bite you

**1. The script must live inside the `dpg_system/` package directory.**

It derives the SMPL model path from its own file location and expects the
`smplh/` folder (containing `SMPLH_MALE.pkl`, `SMPLH_FEMALE.pkl`, and
`SMPLH_NEUTRAL.pkl` if any of your files report neutral gender) as a sibling. It
then puts the *parent* of that directory on `sys.path`, because
`smpl_processor` imports its siblings as `dpg_system.*`. Putting the script at
the repo root or in a scratch directory will not work.

Note the gender fallback: any file whose metadata is missing, or reports
something other than male/female, is processed as `neutral`. If you have no
neutral model those files will fail — either supply the model or force a gender
with `--gender`.

**2. The physics options are hardcoded, not exposed as flags.**

`build_options()` mirrors the live `smpl_torque` node's widget defaults, which is
the same block used by the offline noise-estimation batch config. Notably:

- world-frame dynamics on
- gravity, passive limits, apparent gravity, S-curve spine on
- floor contact on, `logodds_valved` contact method, body contacts on
- **all rate limiting, jitter damping, Kalman smoothing, velocity gating and
  One-Euro filtering off** — these are raw torques
- acceleration smoothing is a Savitzky-Golay derivative fixed in *time*
  (`acc_smooth_ms=70`) rather than in frames, so results are capture-rate
  independent
- input axis permutation `x, z, -y`, input up-axis `Y`, axis-angle input

Changing the physics means editing that function. Whatever was actually used
gets serialized into every output file as `processing_options`, so any given
`.npz` records the configuration that produced it.

## Implementation details worth knowing

**A fresh SMPLProcessor is constructed per file.** It carries per-frame EMA and
streaming state, so reusing one across files would leak state between sequences
and spike the torque on frame 0.

**Resumable.** Files whose output already exists are filtered out *before* the
work count is computed, so the progress count and ETA stay honest on a restart.
Re-running the same command continues where it left off; add `--overwrite` to
force a rebuild.

**Atomic writes.** Each output goes to a `.tmp.npz` and is then renamed, so an
interrupted run never leaves a half-written file that the resume logic would
mistake for completed work.

**`shape.npz` is skipped.** AMASS ships one per subject holding betas and gender
only, with no motion, so it is excluded from the work list rather than counted
as a failure.

**Per-frame errors are tolerated.** If a frame raises, its slot is left as zeros
and the count is reported in that file's log line rather than failing the whole
sequence — the first few frames can legitimately produce partial results while
the streaming state spins up.

**Failure log.** Any files that fail outright are written to
`build_amass_dynamic_failures.log` in the output root, with tracebacks.

## Framerate handling

Framerate is read from `mocap_framerate`, `motioncapture_framerate`, or
`framerate`, in that order, defaulting to 60 fps if none is present. The
timestep passed to the processor is derived from it per file, so mixed-rate
datasets are handled correctly.
