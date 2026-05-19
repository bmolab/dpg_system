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

- **2026-05-19:** First investigation complete + fix applied. Confirmed
  upper-limb over-filtering, applied effort-based threshold to
  `adaptive_effort_smooth`, re-validated. Wrist/elbow alpha now varies
  with actual effort instead of being locked at alpha_min. Open
  follow-up: per-joint `acc_smooth_window` with global scalar (see
  Known Issues). Next: user to spot-check fix in live patch, then
  pick next case to investigate.

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

### 2026-05-19 — Upper-limb over-filtering on popping_1 (frames 500-1000)
- File: `/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/smpl_mocap_files/RP/popping_1_aligned_smpl_poses_aligned_yaw_aligned.npz`
- Frames: 500–1000 (5.0 s at 100 fps) — popping dance, has jerky wrist/lower-arm motion
- Question: are real jerky lower-arm/wrist movements being filtered out, or
  only noise? User notes wrist/elbow effort visualization looks smooth even
  in cartwheels because max_torque is high enough that effort isn't clipped.
- Diagnostic: new `upper_limb_filtering` test in `muscle_activation_tester.py`.
  Runs two passes — canonical vs (smooth_input_window=0, acc_smooth_window=0,
  adaptive_effort_smooth=False). Reports RMS, frame-to-frame jitter, fraction
  of frames in alpha_min regime, and top-K most divergent frames per joint.
- Observation:
  | joint     | raw_rms | filt_rms | ret% | jit_ret% | <lo% | alpha_est |
  |-----------|---------|----------|------|----------|------|-----------|
  | L_shldr   |   16.35 |     6.38 |  39% |       9% |  46% |     0.124 |
  | R_shldr   |   19.82 |     7.19 |  36% |       8% |  47% |     0.135 |
  | L_elbow   |    5.37 |     2.07 |  39% |       3% |  94% |     0.055 |
  | R_elbow   |    5.37 |     1.36 |  25% |       4% |  92% |     0.055 |
  | L_wrist   |    1.15 |     0.27 |  24% |       3% | 100% |     0.050 |
  | R_wrist   |    0.45 |     0.19 |  42% |       3% | 100% |     0.050 |
  Single-frame examples (real movement, not noise): R_wrist frame 674 raw=1.32
  Nm filt=0.19 Nm at kinematic angular displacement 12.4°/frame (~1240°/s
  raw motion). R_shoulder frame 617 raw=97 Nm filt=6.9 Nm at 4.6°/frame.
- Judgement: INCORRECT — the pipeline is over-filtering real lower-arm motion.
- Root cause: `adaptive_effort_smooth` uses a torque-magnitude threshold
  (`adaptive_effort_lo=10 Nm`) that's above the wrist's maximum possible
  torque on every axis (`wrist max_torque = [8, 15, 10]`, min axis = 8 Nm).
  So wrists are mathematically incapable of escaping the alpha_min=0.05
  heavy-smoothing regime, regardless of how fast the motion actually is.
  Elbows have the same issue on the twist/abduction axes (max [10, 40, 8])
  and on the flexion axis only escape during big swings. The filter design
  was built around proximal joints where torque magnitudes are large.
- Action: APPLIED fix #1. Switched `adaptive_effort_smooth` threshold from
  torque magnitude (Nm) to effort magnitude (||τ/τ_max||). Field semantics
  changed (`adaptive_effort_lo` / `_hi` now in effort units), defaults
  retuned from 10.0/40.0 Nm to 0.1/0.5 effort. Re-ran diagnostic on same
  range, results:
  | joint     | ret%       | jit%        | <lo%        | alpha_est       |
  |-----------|------------|-------------|-------------|-----------------|
  | L_shldr   | 39→**47**  |  9→**13**   |  46→**20**  |  0.12→**0.22**  |
  | R_shldr   | 36→**46**  |  8→**13**   |  47→**17**  |  0.14→**0.25**  |
  | L_elbow   | 39→**52**  |  3→**6**    |  94→**62**  |  0.06→**0.10**  |
  | R_elbow   | 25→**42**  |  4→**9**    |  92→**65**  |  0.06→**0.09**  |
  | L_wrist   | 24→**30**  |  3→**5**    | 100→**72**  |  0.05→**0.09**  |
  | R_wrist   | 42→42      |  3→3        | 100→**96**  |  0.05→0.05      |
  L_shoulder/R_shoulder/elbows now see real per-frame alpha variation on
  popping accents (top-divergence frames hit alpha_max). L_wrist's most
  energetic frames now alpha=0.4-0.8 instead of permanently locked at 0.05.
  R_wrist barely moved because its peak raw torque (2.26 Nm) sits just
  above the lo=0.1 floor — correct behavior, that wrist genuinely isn't
  doing much in this range.
- Files touched: `dpg_system/smpl_processor.py` (lines 337-341 defaults +
  comments, lines 7625-7641 docstring, lines 7700-7706 magnitude → effort),
  `dpg_system/logodds_evaluator/muscle_activation_tester.py` (added
  `test_upper_limb_filtering`, updated to track effort fraction).

## Open follow-ups from this investigation

- `acc_smooth_window` is a global SG window applied uniformly to all
  joints (currently 7 at the live node). User wants this to be per-joint
  with a single global scalar that multiplies the per-joint windows
  (analogous to `adaptive_effort_alpha_max` being a global scalar on
  per-joint alpha_max). This wasn't addressed in this session.

## Validated behaviors

Things we've checked and confirmed work correctly. Each entry is a
guard against re-investigating the same case.

_(none yet)_

## Known issues / open investigations

Pending work — things we noticed but haven't resolved.

- **Per-joint `acc_smooth_window`** (2026-05-19): currently global SG
  window of 7 frames applied uniformly. Should become per-joint (smaller
  for wrist/elbow, larger for proximal high-inertia joints) with a single
  global scalar that multiplies the per-joint values. Surfaced during
  the upper-limb over-filtering investigation as the secondary
  contributor to lost wrist/elbow signal. Not addressed in that session.

## Pipeline change log

### 2026-05-19 — `adaptive_effort_smooth` threshold: Nm → effort
- Files: `dpg_system/smpl_processor.py`
- What: `adaptive_effort_lo` / `_hi` switched from torque-magnitude
  thresholds (Nm) to effort-magnitude thresholds (||τ/τ_max||). Defaults
  retuned from 10.0/40.0 Nm to 0.1/0.5 effort. The blend uses
  `||efforts_net||` instead of `||torques_vec||`.
- Why: the Nm threshold was above the wrist's maximum possible torque
  on every axis, locking wrist/hand permanently in alpha_min=0.05
  heavy-smoothing regime regardless of motion speed. Effort is naturally
  joint-scaled. See investigation entry 2026-05-19 (upper-limb).
- Motivated by: 2026-05-19 upper-limb over-filtering investigation.

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
