# smpl_utilities

Pipeline for cleaning Shadow mocap takes and converting them to AMASS-style
SMPL `.npz` files for downstream use (`smpl_torque`, AMASS-format consumers,
etc.).

## TL;DR pipeline

```
Shadow .npz  (quats + positions, Y-up)
   │
   │  batch_add_betas_framerate.py        stamp betas / gender / mocap_framerate
   ▼
Shadow .npz  (now with metadata)
   │
   │  batch_shadow_to_smpl_aligned.py     Shadow → SMPL, with T-pose retarget
   ▼
SMPL .npz  (_smpl_poses_aligned.npz)
   │
   │  batch_correct_sensor_pos.py         fix Vive-tracker mount offset in trans
   ▼
SMPL .npz  (_<seg>fix.npz)
   │
   │  align_worlds.py                     IMU-world vs tracker-world yaw + offset
   ▼
   │  yaw_align_smpl.py                   residual yaw cleanup via foot-skate
   ▼
SMPL .npz  (fully aligned)
```

You can skip stages when not needed — e.g., for clean takes that already have
metadata, jump straight to `batch_shadow_to_smpl_aligned.py`.

## Three "alignment" problems (don't confuse them)

| # | Problem                                                  | Solved by                                          | Mechanism                                                            |
|---|----------------------------------------------------------|----------------------------------------------------|----------------------------------------------------------------------|
| 1 | Skeleton T-pose mismatch: Shadow rest ≠ SMPL rest        | `shadow_to_smpl_aligned.py`                        | Per-bone global-orientation retargeting                              |
| 2 | Sensor mount offset (Vive tracker not where solver thinks) | `correct_thigh_offset.py` / `batch_correct_sensor_pos.py` | Iterative planted-foot LS: `trans_err = R_segment @ delta + [0,0,c]` |
| 3 | World-frame yaw misalignment + back-surface offset       | `align_worlds.py`, `yaw_align_smpl.py`             | (a) walking-velocity correlation, (b) foot-skate minimization        |

Problems 1 and 3 sound similar but are different: (1) is a *skeletal*
rest-pose mismatch; (3) is a *world-frame* mismatch between the IMU's
orientation world and the 3D-tracker's translation world.

## Per-script reference

### Conversion (Shadow → SMPL)

**`shadow_to_smpl.py`** — basic converter.
- Input: Shadow `.npz` with `quats: (T,37,4)` local wxyz (Y-up) and
  `positions: (T,37,3)` (Y-up).
- Reindexes Shadow → bmolab-active → SMPL joint order.
- Root orientation and `trans`: Y-up → Z-up via −90° X rotation.
- Writes AMASS axis-angle `poses` + `trans`. **No skeleton retargeting** —
  assumes Shadow's rest matches SMPL's exactly (it doesn't, quite).
- Output suffix: `_smpl_poses.npz`.

**`shadow_to_smpl_aligned.py`** — preferred converter.
- Same conversion job as above, **plus** T-pose structural retargeting:
  per-frame Shadow-bone global orientations → deviation from Shadow rest →
  applied to SMPL rest → back-projected to SMPL local quats.
- Reads bone offsets from `../definition.xml`; loads SMPLH from `../smplh/`.
- Output suffix: `_smpl_poses_aligned.npz`.

**`batch_shadow_to_smpl_aligned.py`** — directory wrapper around the aligned
converter.
- Skip-tags for already-converted outputs, skip-if-output-exists with
  `--overwrite`, `--glob` filter, per-file try/except, end-of-run summary
  table.
- All CLI flags pass through; omitted flags fall back to per-file auto-
  detection (see "Auto-detection" below).

### Metadata stamping

**`batch_add_betas_framerate.py`** — add `betas`, `gender`, `mocap_framerate`
to existing `.npz` files (Shadow or SMPL). Output suffix `_beta.npz`. Idempotent
(skips files already ending in `_beta`). After this, the converters and the
sensor-correction can auto-pick up these values without CLI flags.

### Sensor-mount correction

**`correct_thigh_offset.py`** — corrects `trans` for an offset error in the
Vive-tracker mount.
- Model: `trans_recorded[t] = trans_true[t] + R_segment_world[t] @ delta` plus
  a constant floor offset `c`.
- Algorithm: SMPL FK → inlier mask → iterate { pick "planted" foot frames →
  ridge-regularized LS refit with MAD outlier trim } → pick `c` from bottom-
  quantile of corrected lower-foot z → Savitzky-Golay smooth the per-frame
  shift → apply to all frames.
- Knobs: `--segment pelvis|lhip|rhip|auto[:cands]`, `--full` (9-coef model
  for axial strap slip), `--fit-range LO:HI`, `--smooth`.
- Output suffix: `_<seg>fix.npz` / `_<seg>fullfix.npz`.

**`batch_correct_sensor_pos.py`** — directory wrapper. Each file fit
independently (strap pose may drift between takes). Same knobs as the
single-file script, plus `--glob` and `--overwrite`.

### World alignment

**`align_worlds.py`** — solves two calibrations from a converted SMPL file:
1. **Yaw alignment** between the quaternion world (IMU-derived rotations)
   and the translation world (3D tracker). Infers the yaw offset by
   correlating pelvis-forward direction with ground-plane velocity over
   walking segments.
2. **Sensor-to-root offset** for the back-mounted tracker: applies a
   configurable local-frame offset rotated by pelvis orientation per frame.

**`yaw_align_smpl.py`** — *residual* yaw cleanup, post-everything.
- Runs `SMPLProcessor` (the `smpl_torque` pipeline) to get per-frame contact
  pressure + world joint positions.
- 2-D Procrustes solve over planted-foot frames to find the yaw of `trans`
  (poses untouched) that minimizes foot skating.
- Iterates because the closed-form solve recovers only ~90% per pass
  (non-linear filter interactions inside the processor that don't commute
  exactly with rotation).
- Use when you have residual heading drift after `align_worlds.py`, or when
  you don't have a clean walking segment for the velocity-correlation method.

### Other

**`prepare_amass_data.py`** — MPG's reference AMASS data-prep code
(third-party, license header in the file).

## Auto-detection of fps / gender / betas

Both `shadow_to_smpl.py` and `shadow_to_smpl_aligned.py` (and the batch
wrapper) auto-pick up `mocap_framerate`, `gender`, and `betas` from the input
`.npz` when the corresponding CLI flag is omitted. Precedence:

1. Explicit CLI flag → wins (applies to every file in batch mode).
2. File's own field, if present.
3. Hard fallback: 100 fps, `neutral`, zeros.

Companion: `batch_add_betas_framerate.py` stamps these values once; the
converters pick them up automatically afterward.

## Filename suffix conventions

| Suffix                                  | Produced by                          |
|-----------------------------------------|--------------------------------------|
| `_beta.npz`                             | `batch_add_betas_framerate.py`       |
| `_smpl_poses.npz`                       | `shadow_to_smpl.py`                  |
| `_smpl_poses_aligned.npz`               | `shadow_to_smpl_aligned.py`          |
| `_pelvisfix.npz` / `_lhipfix.npz` / …   | `correct_thigh_offset.py` (rigid)    |
| `_<seg>fullfix.npz`                     | `correct_thigh_offset.py --full`     |
| `_corrected.npz`                        | reserved tag for external tools      |

## External resources expected one level up

- `../definition.xml` — Shadow skeleton definition (bone offsets).
- `../smplh/SMPLH_MALE.pkl`, `../smplh/SMPLH_FEMALE.pkl` — SMPLH model files.

The aligned converter's `load_shadow_offsets` and `load_smpl_offsets` look
in both `./` and `../`, so the scripts work whether or not the resources are
co-located.

## Conda environment

Run everything inside `dpg_system_2025`. The aligned converter pulls in
`torch`, `smplx`, and `dpg_system.body_defs` (for the joint-name maps).
