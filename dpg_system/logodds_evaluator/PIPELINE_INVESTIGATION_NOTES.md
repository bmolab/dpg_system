# SMPL Torque Pipeline — Investigation & Refinement Notes

Working notes for the long-running project of validating, challenging, and
improving the `smpl_torque` pipeline (torque, effort, floor contact, ZMP,
log-odds) by examining specific frame ranges of specific SMPL motion-capture
files.

This document is **the single source of truth across sessions** for this
project. Update it as we go.

## Goal

Build a body of validated understanding of the pipeline's behavior. For a
user-supplied (file, frame-range, question) tuple, run targeted diagnostics,
decide whether the pipeline's output is correct, and — when it isn't — find
the root cause and improve the code. Each investigation should leave the
codebase or our understanding measurably better.

## How a session works

The loop is:

1. **User names a case.** "In file X, at frames A–B, the right knee torque
   looks too high during the foot plant. Is it right?"
2. **Pick or write a diagnostic.** Use the existing `TESTS` registry in
   `muscle_activation_tester.py`, or add a new test function that exposes
   the specific internals the question needs.
3. **Run it.** Capture relevant per-frame numbers (torques, contact forces,
   surface distances, log-odds, CoM, ZMP).
4. **Judge.** Is the output physically plausible? Compare against expected
   behavior (kinematic intuition, equivalent simpler model, prior validated
   cases).
5. **If wrong, drill down.** Inspect the underlying signals along the path
   from input → output: pose smoothing → world_pos → CoM/velocity → contact
   detection → forces → dynamic torques.
6. **Decide action.** Code fix, threshold change, "this is correct and our
   intuition was wrong," or "need more data — investigate further."
7. **Log it below.** Even null results are useful — they tell us a
   behavior was checked and validated.

## Diagnostic harness reference

**Template / canonical entry point:**
`dpg_system/logodds_evaluator/muscle_activation_tester.py`

This script is the model for all pipeline investigation. It:
- Loads an NPZ, makes a `SMPLProcessor` with the canonical option set
  (matching the live `smpl_torque` node's defaults).
- Runs `process_frame` in streaming mode (no batch shortcuts) so the
  pipeline behaves identically to live patches.
- Exposes per-frame internals via `processor.last_world_pos`,
  `_inferred_floor_height`, `_prev_com_for_stability`, `_logodds_result`,
  `contact_pressure`, `_frame_eval_result`, `_joint_surface_extents`,
  `_joint_surface_min_dists`, `_prev_global_rots`, `_logodds_estimator`.

**Existing test modes** (extend or write new ones):
- `torque_jitter` — frame-to-frame torque vector magnitude jitter.
- `body_contact` — body-contact gating: CoM height, surface distances,
  per-joint intensities.
- `zmp_pipeline` — ZMP decomposition: position vs. acceleration noise.
- `pressure` — per-joint contact pressure (kg equivalent).
- `logodds_streams` — per-stream log-odds increments per group.

**When to add a new test:** if a question needs internals not exposed by
the existing tests. Keep new tests in the same file under the `TESTS`
registry so they're invocable as `python muscle_activation_tester.py
--test <name>`.

## Where we left off

_(Update at end of each session — one short paragraph)_

- **2026-05-19:** Investigation infrastructure set up (this doc + memory
  pointer). No cases investigated yet. Next: user supplies the first
  case (file, frame range, question).

## Investigations log

For each investigation, record file/range/question/finding/action.
Even null results matter — they document validated behaviors.

### Template

```
### <YYYY-MM-DD> — <one-line summary>
- File: <path>
- Frames: <start–end> (t=<...>s)
- Question: <what does the user want to know / challenge>
- Diagnostic(s) used: <test names / new test added>
- Observation: <relevant numbers>
- Judgement: CORRECT / INCORRECT / AMBIGUOUS
- Root cause (if INCORRECT): <where in the pipeline it goes wrong>
- Action: <code change / threshold change / no action — already correct / pending>
- Files touched: <if any>
```

_(no investigations yet)_

## Validated behaviors

Things we've checked and confirmed work correctly. Each entry is a
guard against re-investigating the same case.

_(none yet)_

## Known issues / open investigations

Pending work — things we noticed but haven't resolved.

_(none yet)_

## Pipeline change log

Code changes made as a direct result of investigations. Date, file,
what changed, why, and which investigation entry motivated it.

_(none yet)_

## Conventions when extending the tester

- Match the live node: use `make_options(fps)` and only override what
  the test specifically needs. Don't drift from the node defaults.
- Stream frame-by-frame via `process_frame`; never use batch shortcuts
  that bypass the per-frame state we want to inspect.
- When exposing a new processor internal, prefer reading it off the
  processor between calls (e.g. `getattr(pr, '_some_attr', None)`)
  rather than modifying the processor to return more.
- One-shot probes are fine, but if a check is worth re-running, put it
  in `TESTS`.

## References

- Diagnostic template: `dpg_system/logodds_evaluator/muscle_activation_tester.py`
- Pipeline source: `dpg_system/smpl_processor.py`,
  `dpg_system/dynamic_frame_evaluator.py`
- Default test files (from tester):
  - `dpg_system/Subject_81_F_19_poses.npz` (clean walking gait reference)
  - `HS_take6_smpl_poses_b.npz` (alt)
- Candidate stress-test file: `take 3_smpl_poses.npz` (see
  `smpl_contact_test_data` memory)
- Related: noise estimation work in `dpg_system/noise_estimation/`
  (`VALIDATION_NOTES.md`) — those flags drive *which* sections deserve
  investigation here.
