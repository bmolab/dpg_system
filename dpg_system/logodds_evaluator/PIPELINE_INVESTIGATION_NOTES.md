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

- **2026-05-19:** Three changes applied in one session. (1) Upper-limb
  over-filtering fixed via effort-based threshold in
  `adaptive_effort_smooth`. (2) R_knee oscillation case (take 3 frames
  5492-5532) drove design of signal-change-aware alpha — median-of-K
  |Δeff| over a sliding window, combined with effort-blend via max.
  Spike-resistant by construction (a single-frame spike contributes 2
  large deltas out of K=5; median picks the 3rd, small, value). (3)
  Both fixes validated on diagnostic. Side finding: `acc_smooth_window`
  does negligible useful work when the EMA is active. Next: user to
  spot-check in live viz on both files; if knee shows pulsing-during-
  oscillation, design works; if user wanted sustained-activation
  visualization, separate magnitude-EMA approach needed.

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

### 2026-05-19 — R_knee fast back-and-forth on take 3 frames 5492-5532
- File: `/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/smpl_mocap_files/LR/take 3_smpl_poses_aligned.npz`
- Frames: 5492–5532 (~0.67 s at 60 fps) — quick rhythmic knee flexion/extension
- Question: user reports the right lower leg's quick back-and-forth at
  the knee doesn't appear in the muscle visualization. Is the effort
  fix from earlier enough? If not, what's the dominant filter here?
- Diagnostic: `test_lower_limb_filtering` (new wrapper around shared
  `_joint_filtering_diagnostic` helper, defaults to `smooth_input_window=3`
  for this 60-fps file). Plus an inline ablation script that varies one
  filter at a time.
- Observation:
  - Cascade (R_knee, all values starting from raw rms=20.90 djit=12.0):
    | filter added                       |  rms   | djit | comment                      |
    |------------------------------------|--------|------|------------------------------|
    | + smooth_input_window=3            | 14.58  | 5.35 | -30% rms, -55% djit          |
    | + acc_smooth_window=7              | 12.53  | 3.53 | -14% rms, -34% djit          |
    | + adaptive_effort EMA (canonical)  | 12.23  | 0.51 | -2% rms,  -86% djit ← killer |
  - Even with `adaptive_effort_alpha_max=1.0` the canonical config gave
    djit=0.49 — alpha_max doesn't help because effort is too low
    (eff_max=0.20) to escape the threshold.
  - 12–14 Hz oscillation visible in kin° column (peaks at 5510, 5520-22,
    5525-26, 5531).
- Judgement: INCORRECT — fast low-effort oscillations get squashed by
  the EMA because alpha responds only to magnitude. AMBIGUOUS for
  `acc_smooth_window`: marginal benefit relative to EMA — user observed
  it "does not seem to do useful work in the current overall filtering
  regime."
- Root cause: EMA's alpha is magnitude-only. A 12 Hz oscillation at 50 Nm
  peak / 250 Nm max never crosses the effort threshold; alpha stays near
  alpha_min; oscillation is wiped. Magnitude alone is the wrong signal —
  *rate of change* also needs to count.
- Action: APPLIED. Designed and implemented signal-change-aware alpha.
  New fields on SMPLProcessingOptions: `adaptive_effort_change_window`
  (K=5), `adaptive_effort_change_lo` (0.02 |Δeff|/frame),
  `adaptive_effort_change_hi` (0.10). In the EMA block, compute
  `deff_now = ||en_cur - prev_en||` per joint, push into a (K, J) ring,
  take median, map to `chg_blend`. Final blend = `max(eff_blend,
  chg_blend)`. Spike-resistant: a single-frame spike contributes 2
  large deltas (in + out) out of K=5; the median picks the 3rd value
  which is small, so a single glitch doesn't open the gate.
- Validation post-fix:
  - R_knee: filt_rms 12.23 → 7.10 (lower because EMA on a direction-
    reversing vector produces smaller magnitudes near zero crossings —
    physically correct), filt_djit **0.51 → 1.45 (+184%)**. Oscillation
    now coming through as visible jitter in the magnitude.
  - popping_1 upper limb (sanity, frames 500-1000): every joint
    improved on jit_retained. L_wrist 5%→12%, R_wrist 3%→9%, L_elbow
    6%→15%, R_elbow 9%→18%. Effort-only fix still in effect, change
    branch adds on top.
- Files touched: `dpg_system/smpl_processor.py` (added 3 fields lines
  342-348, modified EMA block ~lines 7707-7757),
  `dpg_system/logodds_evaluator/muscle_activation_tester.py` (refactored
  upper/lower limb tests around shared `_joint_filtering_diagnostic`,
  added `test_lower_limb_filtering` with `smooth_input_window=3`).
- Pending: user spot-check in live viz. If knee oscillation now reads
  as pulsing activation, design is good. If user wanted *sustained*
  activation through the oscillation, that's a separate design (apply
  EMA to magnitude not vector).

### 2026-05-19 — Pec L/R asymmetry from per-axis effort skew + shoulder slot bug
- File: any with shoulder motion (user-observed in symmetric-movement footage)
- Question: pec activation asymmetric between L and R for "apparently
  symmetric" movements. Is the activation formula too sensitive to
  small direction differences in torque?
- Diagnostic(s): code reading. Walked the chain from `smpl_torque`
  `combined_effort` output → heatmap `torques` input → flex_axis dot
  product. Inspected per-joint max_torque table and v4 muscle atlas
  flex_axes.
- Observation: two compounding issues.
  1. **Per-axis effort vector skew**: `efforts_net = τ / max_torque`
     (per-axis division). Whichever axis has the smallest max_torque
     gets the largest contribution to the effort vector → the vector
     direction is tilted toward that axis. The flex_axis dot product
     then projects onto this skewed direction instead of the raw τ
     direction.
  2. **Shoulder slot mis-ordering**: comment says "Arms: X=Twist,
     Y=Flex/Ext, Z=Abd/Add" and elbow/wrist values match it, but
     shoulder is `[70, 30, 60]` — twist torque (~30) is in the X slot
     would mean X=70, but biomechanically shoulder twist is ~30 and
     flex is ~70. Values appear to be in `[Flex, Twist, Abd]` order
     rather than the `[X=Twist, Y=Flex, Z=Abd]` order they're stored in.
  3. **L/R pecs have opposite-sign Y in v4 flex_axes**:
     L_Pec=[+1, -0.3, -0.5], R_Pec=[+1, +0.3, +0.5]. So the Y axis is
     precisely the axis where the two pecs respond to torque with
     OPPOSITE signs. The shoulder Y slot (currently 30, the smallest)
     is also the most-amplified axis in `efforts_net`. So any Y-axis
     noise → over-amplified → projected onto opposite-sign L/R pec
     flex_axes → large L/R asymmetry.
- Judgement: INCORRECT — `efforts_net` direction is not what should be
  projected onto flex_axes. The flex_axis lives in the same space as
  joint-local torques, not per-axis-skewed efforts.
- Action: APPLIED two fixes.
  1. New helper `SMPLProcessor._effort_with_raw_direction(τ, denom)`:
     magnitude = `||τ / max_torque||` (unchanged capacity-relative
     semantic), direction = `τ / ||τ||` (raw). Both `efforts_net`
     computation sites updated to use it.
  2. Shoulder max_torque corrected: `[70, 30, 60]` → `[30, 70, 60]`.
- Side effects:
  - `efforts_net` magnitude semantic unchanged → adaptive_effort_smooth
    behavior unchanged (uses ||en_cur|| and ||Δen_cur||).
  - `combined_effort` output direction is now meaningful w.r.t. raw
    torque direction. Heatmap dot product against flex_axes is now
    geometrically correct.
  - Smoothing diagnostics re-run on popping_1 upper limb and take 3
    lower limb: numbers essentially unchanged (≤0.02 shift in any
    cell). Confirms the change is direction-only, not magnitude.
- Files touched: `dpg_system/smpl_processor.py` (lines 3479, 3540 for
  shoulder slot fix; lines 6583-6594 + 7634 + new helper at ~6593 for
  effort vector reconstruction).
- Pending: user spot-check pec L/R symmetry in live viz on symmetric
  movement. If still asymmetric, investigate (a) torque vectors
  themselves differing between L/R shoulders, (b) v4 flex_axis values
  themselves being wrong.

### 2026-05-19 — v4 atlas flex_axis comprehensive audit + empirical polarity rule
- File: dpg_system/smpl_muscle_editor/generate_muscle_atlas_v4.py
- Question: are the v4 atlas flex_axes biomechanically correct for all
  upper-body muscles?
- Diagnostic: full muscle-by-muscle audit (76 muscles), then user ran a
  hand-rotation visualizer to empirically determine joint-local axis
  polarities for shoulder / elbow / wrist.
- Empirical findings (user's hand-rotation test, 2026-05-19):
  1. **L/R polarity rule** for shoulder/elbow/wrist: X axis (twist) has
     SAME polarity between L and R; Y and Z axes have OPPOSITE polarity.
     So mirror between L and R muscles is: X same, Y/Z negated.
  2. **Wrist axis convention differs from shoulder/elbow.** The
     smpl_processor.py:3457 comment says "Arms: X=Twist, Y=Flex/Ext,
     Z=Abd/Add" but for the wrist specifically, Y=Abd/Add (radial/ulnar
     deviation) and Z=Flex/Ext (wrist flexion/extension). This is
     because the wrist's local frame inherits from the forearm-twisted
     elbow frame, ending up rotated 90° around X.
  3. Elbow is correct (hinge joint, only Y axis active).
  4. Right shoulder muscles had correct signs after first round of
     fixes; left shoulder muscles needed Y and Z negated (mirror).
- Action: APPLIED. Two-pass fix:
  - Pass 1 (incorrect): hypothesized smpl_processor comment applied
    uniformly to wrist → swapped Y↔Z for wrist muscles. Wrong per
    empirical test.
  - Pass 2 (final): for L shoulder muscles (Pec, Lat, DeltAnt,
    DeltPost) negate X and Y, keep Z. For wrist muscles, restore v4
    originals (flex on Z, deviation on Y). Elbow muscles unchanged.
- Verification: after re-running generator, R shoulder values match
  user-validated R muscle behavior; L now properly mirrors R per
  polarity rule (X same, Y/Z opposite). Wrist matches user's
  empirical finding (flex/ext on Z, deviation on Y).
- Files touched:
  - `dpg_system/smpl_muscle_editor/generate_muscle_atlas_v4.py`
    (16 flex_axis literals across pec/lat/deltoid and wrist muscles)
  - `dpg_system/smpl_muscle_editor/muscle_atlas_v4_meta.npy`
    (regenerated)
  - `dpg_system/smpl_muscle_editor/muscle_atlas_v4.npy` (regenerated,
    contents unchanged — atlas matrix doesn't depend on flex_axes)
- Documentation TODO: update smpl_processor.py:3457 comment to clarify
  that the wrist's local axis convention differs from shoulder/elbow
  (wrist: Y=Abd/Add, Z=Flex/Ext).

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

- **Soft-tissue vibration ringing on impacts** (2026-05-19):
  high-frequency vibration of sensors mounted on muscle mass during
  floor contacts or sudden decelerations (e.g. popping moves) shows up
  in the torque/effort signal as 100-150 ms of 15-25 Hz oscillation.
  Locally indistinguishable from real fast voluntary movement by a
  memoryless filter — the change-aware alpha (correctly) lets it
  through alongside legitimate motion. User flagged this as
  potentially-acceptable noise. Three angles for a future fix:
    1. **Contact-event-gated smoothing**: use existing floor-contact
       detection to apply a 100-150 ms post-impact attenuation window
       on the impacted side's chain. Exploits causality (ringing
       always follows impact). Most promising — uses information the
       current filter ignores.
    2. **Physiological ceiling**: voluntary muscle activation can't
       exceed ~10-12 Hz; current alpha_max=0.8 at 60 fps is already
       near that cap. Raising it lets in more vibration, not less.
    3. **Median filter after EMA**: 3- or 5-tap median to attenuate
       isolated one-cycle transients without smearing sustained
       signal. Doesn't help with multi-cycle decay.

- **Retire `acc_smooth_window`?** (2026-05-19, updated): originally
  planned as a per-joint refactor with a global scalar. Subsequent probe
  on `take 3` R_knee (frames 5492-5532, 12-14 Hz oscillation at 60 fps)
  showed that with the EMA active, `acc_smooth_window=7` provides only a
  ~9% rms / ~22% djit marginal benefit — the EMA flattens what the SG
  was going to flatten anyway. User observation: "acc_smooth_window does
  not seem to do useful work in the current overall filtering regime."
  Decision: don't refactor; consider defaulting to 0 and removing as a
  tuning knob. Validate the "no useful work" claim on a more diverse set
  of cases before deleting.

- **Signal-change-aware alpha for adaptive_effort EMA** (2026-05-19):
  The effort-based threshold fix handled the wrist/elbow case (motion
  scales with effort), but didn't fix the R_knee case where a 12-14 Hz
  oscillation happens at low effort (eff_max=0.20). Even with
  `adaptive_effort_alpha_max=1.0` the EMA stays near alpha_min for that
  oscillation because effort never crosses the threshold. Proposed:
  `alpha = max(effort_blend, change_blend)` where `change_blend`
  measures frame-to-frame torque-vector delta — catches fast oscillation
  regardless of magnitude. Discuss with user before implementing.

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

### 2026-05-19 — efforts_net direction now raw τ direction (not per-axis skewed)
- Files: `dpg_system/smpl_processor.py`
- What: new helper `_effort_with_raw_direction` rebuilds `efforts_net`
  as `(||τ/max_torque||) * (τ/||τ||)`. Magnitude unchanged (still
  capacity-relative), direction = raw torque direction.
- Why: the prior `efforts_net = τ/max_torque` (per-axis) skewed the
  effort vector toward whichever axis had the smallest max_torque. For
  the heatmap's flex_axis dot product this skew amplified L/R muscle
  asymmetry (most visible on pecs whose v4 flex_axes have
  opposite-sign Y while shoulder Y has the smallest max_torque). The
  flex_axes live in raw-torque-direction space; dotting them against
  a per-axis-skewed vector was geometrically wrong.
- Motivated by: 2026-05-19 pec L/R asymmetry investigation.

### 2026-05-19 — Shoulder max_torque slot order fix
- Files: `dpg_system/smpl_processor.py`
- What: shoulder `max_torque` `[70, 30, 60]` → `[30, 70, 60]`.
- Why: the axis convention comment is `[X=Twist, Y=Flex/Ext, Z=Abd]`
  and elbow/wrist values match it, but shoulder values were in the
  order `[Flex, Twist, Abd]` instead. Biomechanically shoulder twist
  is ~30 Nm and flex is ~40-80 Nm. Fix matches comment and the rest
  of the arm chain.
- Motivated by: 2026-05-19 pec L/R asymmetry investigation.

### 2026-05-19 — `adaptive_effort_smooth` adds signal-change-aware alpha
- Files: `dpg_system/smpl_processor.py`
- What: alpha is now `alpha_min + max(eff_blend, chg_blend) *
  (per_joint_alpha_max - alpha_min)`, where `chg_blend` is the median
  of |Δeff| over a K-frame sliding window mapped through
  `adaptive_effort_change_lo` / `_hi` thresholds. New fields on
  `SMPLProcessingOptions`: `adaptive_effort_change_window` (default 5),
  `adaptive_effort_change_lo` (0.02 |Δeff|/frame),
  `adaptive_effort_change_hi` (0.10). Shared per-joint alpha_max cap.
  No widget exposes the new fields — existing patches unaffected.
- Why: the effort-only blend doesn't catch fast oscillations that
  happen at low effort (e.g. R_knee 12 Hz back-and-forth at eff~0.2).
  Median (not mean) is the spike-rejection mechanism — a single-frame
  glitch contributes 2 large |Δeff| values out of K=5, so the median
  picks the 3rd (small) value and the gate doesn't open. Sustained
  oscillation overwhelms the median and triggers.
- Motivated by: 2026-05-19 R_knee oscillation investigation
  (take 3 frames 5492-5532).

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
