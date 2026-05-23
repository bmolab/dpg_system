# Noise Estimation Validation Notes

Working notes for validating `estimate_noise_torque.py` against real files,
with the eventual goal of filtering AMASS for clean training data.

This document is **the single source of truth across sessions** for what's
been examined, what was learned, and what's next. Update it as we go.

## Goal

Verify that `estimate_noise_torque.py` accurately flags noise (and only noise)
across the kinds of motion we care about, then run it across all of AMASS to
filter out noisy sections and stream breaks. AMASS was originally built for
static poses, so it contains motion-specific noise its creators didn't care
about but which is problematic for torque analysis.

## Current phase

**Phase 1 — bootstrap.** Setting up validation infrastructure. No files
re-run with refined script yet. Prior batch outputs in
`torque_results_by_folder/` and `torque_checkpoint.json` are **stale** —
they predate refinements that better handle real dynamic movement, so
they cannot be used for validation.

## Where we left off

_(Update at end of each session — one short paragraph)_

- **2026-05-19:** Validation infrastructure set up (this doc + pointer
  memory). Next: pick first batch and decide inspection method.
- **2026-05-22:** Walked spike frames in C6 (ACCAD/Male2Running) and C24
  (ACCAD/Male1Running) by hand. Built a per-frame "spike severity"
  metric (weighted, p95-normalised, with a core-joint gate) that
  cleanly separates the TPs we identified from the FPs — see new
  "Spike severity metric" section below. Also found and verified
  a corruption-zone false-positive pattern on short martial-arts
  punch files — see tuning-log entry.
- **2026-05-22 (later):** First fix attempt
  (`CORRUPTION_MIN_FRAC_OF_P95 = 0.8`) eliminated punch FPs but
  dropped Maritsa from 12 zones → 6 and re-classified it moderate
  → wrong. Root cause: whole-file p95 is inflated by pervasive
  corruption, defeating the gate. **Reverted and replaced with
  `CORRUPTION_MIN_FILE_S = 5.0`**: corruption detection is skipped
  on files shorter than 5 s where the rolling baseline can't
  settle. All four punch FPs still eliminated; Maritsa restored to
  problematic. Then closed a separate gap where `_find_clean_segments`
  ignored `spike_frames` entirely: spike-fold via fold-all was too
  aggressive (fragmented C24/E15 on minor wrist twitches), so
  switched to **severity-aware folding** via a new in-script
  `_significant_spike_frames` helper (gate: wsev ≥ 10 AND ≥ 2
  qualifying core/mid joints). Finally, made `CLEAN_MIN_DURATION`
  adaptive (`min(1.0, T/4)`) so short clean files actually get
  clean segments. Then raised the bad-mask threshold from
  SUSPECT_THRESH to GLITCH_THRESH, then dropped the per-frame score
  from clean-segment fragmentation entirely (the torque scorer
  produces FPs on dynamic core motion — C24's spine-impulse
  glitches at 100–108 / 127–133), then added an arm-chain
  spike-density gate (1-s window, 6 unique frames) to catch the
  Maritsa "shoulders / pecs / arms" flickers. Final state on the
  calibration set: E15 100 % clean (matches classification), C24
  single segment `[0–264]` ending right before MAJOR event at 268
  (74.4 %), C6 single segment `[0–412]` broken at the cadence
  glitch (59.2 % — region [420–678] correctly fragmented because
  of TP wrist corruption that the corruption-zone detector
  missed), Maritsa 22.0 % with the 8.43 s opening preserved. Next:
  audit AMASS for sub-5-s files with real corruption, then
  re-batch.
- **2026-05-22 (later still):** Initial density-gate threshold (6)
  over-clipped C6 — user pointed out post-416 is "the real meat of
  the file" with vigorous-but-legitimate arm swings, not flickers.
  Profiled actual density distributions across all four
  calibration files: C6 max=8, Maritsa median=8 / max=27 — natural
  gap above 8. Raised threshold to 10. C6 post-416 segment restored
  (96.3 % clean), Maritsa at 30.4 % (flickery middle still
  fragmented, cleanest 8.78 s opening preserved).
- **2026-05-23:** Tested on Shadow file (Subject2_take3_beta_smpl_
  poses_rhipflexstat.npz, 1040 s, 60 fps captured at 100 Hz).
  Detector flagged 23 stream breaks; user identified only 2 as real
  (sensor recalibrations during stillness). The other 18 were all
  `multi_joint` type during rapid motion — cadence-induced drops
  from the 60/100 Hz resampling that the static per-joint threshold
  couldn't distinguish from real teleportation. **Fix: adaptive
  per-joint threshold using a 1-s rolling local median** (analog of
  the corruption detector's approach). 23 → 5 breaks, all real
  recalibrations preserved, clean total 97.5 → 98.6 %, no impact
  on AMASS calibration files. Ready for AMASS re-batch.

## Spike severity metric (proposed 2026-05-22)

Per-joint spike detection (`_detect_spike_frames` in
`estimate_noise_torque.py:1418`) emits one entry per joint per
suspect frame. Walking these by hand showed three failure modes:

1. **Joint-mobility blindness.** A wrist hitting 14 rad/s is normal
   during a quick arm motion; the same velocity in pelvis would be
   anomalous. The raw detector treats both identically.
2. **Local-context blindness.** A pelvis spike in an already-noisy
   running region (C6 frame 591) is just the apex of a gait cycle —
   not a glitch — but `vel[t] / max(vel[t-1], vel[t+1])` doesn't see
   the wider context.
3. **Single-joint vs synchronized events.** Real cadence glitches and
   splices fire on many joints simultaneously (8 at C6/416, 3 at
   C6/682, 13 at C24/268). Wrist-only "spikes" almost always turn
   out to be normal fast arm motion.

The proposed severity ranking addresses (1) and (3):

```
per-joint contribution = joint_weight × max(0, vel / joint_p95 − 1) × neighbor_ratio
weighted_severity      = Σ contributions across joints at that frame
n_core_mid_qualifying  = count of joints with weight ≥ 0.5 AND contribution ≥ 2.0
```

Joint weights:

| joint group | weight | rationale |
|---|---|---|
| pelvis, spine1/2/3 | 1.0 | core only moves when whole body moves |
| neck, head | 0.7 | usually correlated with body motion |
| hips, knees, ankles, shoulders, elbows | 0.5 | limb roots, semi-independent |
| collars, wrists | 0.2 | most mobile, often move independently |

Tier assignment:

| tier | gate | example outcomes |
|---|---|---|
| MAJOR | wsev ≥ 20 AND nC ≥ 2 | C24/268 (62.3, 6 nC), C6/682 (42.4, 3 nC), C6/416 (24.9, 4 nC) |
| major-distal | wsev ≥ 20 AND nC < 2 | (rare — investigate) |
| strong | wsev 10–20 AND nC ≥ 2 | (real but small) |
| review-distal | wsev 10–20 AND nC < 2 | C24/93 (FP), C24/100 (FP), boundary frames |
| minor | wsev 3–10 | C6/591 FP gait apex → 6.3 |
| noise | wsev < 3 | C6/477, C6/624, C6/646 (wrist-only minor events) |

Validated against the walkthrough:

- All confirmed TP-major events sit at the top in MAJOR tier
- All identified FPs (C24/93, C24/100, C6/591, C6/624) drop into
  `review-distal`, `minor`, or `noise`
- Wrist-only corruption (C6/646, C6/477) drops to `noise` — consistent
  with the working view that single-wrist corruption is not significant
  for downstream training

Next step is to either compute on-the-fly in `noise_review_node.py`
(cheap, reversible) or promote into the JSON output of
`estimate_noise_torque.py` (durable, requires re-batching).

## Per-file observations

## Sample selection plan

Files to run with the refined script. Goal is to span:
- Known-noisy with known noise type (shadow planted-foot)
- Clean reference (typical AMASS walking gait)
- Dynamic motion that previously got false-positived
- Static-pose-style files representative of AMASS bulk

### Candidate files

| File | Type | Notes |
|------|------|-------|
| `/Users/drokeby/Projects/BMO Lab/GRANTS/NFRF 2023/smpl_data/LR_June_24_betas/take 3_smpl_poses.npz` | shadow / dynamic | Foot rolls, weight transfers, ball-of-foot stance. From `smpl_contact_test_data` memory. |
| `ACCAD/Male2Running_c3d/C6 - stand to run backwards_poses.npz` | dynamic running | Reviewed 2026-05-22. Has TP cadence-glitch (416), TP end-of-capture splice (682), and FPs (591). |
| `ACCAD/Male1Running_c3d/Run C24 - quick side step left_poses.npz` | dynamic, quick lateral | Reviewed 2026-05-22. Has TP MAJOR at frame 268 plus arm-only FPs (93, 100). |
| `ACCAD/Male2MartialArtsPunches_c3d/E15 - body jab right_poses.npz` | short, high-action | Reviewed 2026-05-22. False-positive corruption zone on the punching elbow. Confirmed across 4 sibling files (E14, E16, E2). |
| _add as we go_ | | |

## Per-file observations

For each file run: record file path, top-line score/classification, then
walk through flagged regions and label each as TP (true positive, real
noise), FP (false positive, real motion), or AMBIG (need viewer).

### Template

```
### <filename>
- Run date: YYYY-MM-DD
- Score / classification:
- Stream breaks:
- Corruption zones:
- Glitch clusters reviewed:
  - [start–end] frames at t=X.Xs — TP / FP / AMBIG — short reason
- Notes / what this teaches us:
```

### ACCAD/Male2Running_c3d/C6 - stand to run backwards_poses.npz
- Run date: 2026-05-22
- Source: `noise_estimation_results_May_20/ACCAD.json`
- Score / classification: noise_score 46.5, moderate
- n_frames 698, duration 5.82 s, fps 120
- Stream breaks: none reported (but frame 682 should arguably be one)
- Corruption zones: none reported
- Cadence: not detected (period 3, strength 0.189, coverage 0.261)
- Spike frames reviewed (per-frame severity in parens):
  - frame 416 (wsev 24.9, MAJOR) — 8-joint synchronized one-frame
    step. Pose midpoint-error 10× normal at this frame; whole-body
    delta 2.2× baseline; total angular velocity 78.7 vs ~30 rad/s
    surrounding. **TP — looks like a single-sample upsampling phase
    glitch** (delta exactly 2× baseline). Not a global 2-frame
    cadence: bumps elsewhere in the file are irregular (gaps 1, 2,
    3, 4, 7, 11) and parity distribution is mixed.
  - frame 477 (wsev 1.4, noise) — single-joint left_wrist trajectory
    break. Pose snaps to a new trajectory between 477 and 478 on the
    wrist only; rest of body smooth. **TP minor dropout/splice on
    one joint** — not significant for downstream use.
  - frame 591 (wsev 6.3, minor) — single-joint pelvis vel peak (6.5
    rad/s) inside an already-noisy running region. Surrounding
    pelvis vel oscillates 0.6–6.5; whole-body delta only 1.3×
    baseline; translation perfectly smooth. **FP — gait-cycle apex,
    not a glitch.**
  - frame 624 (wsev 0.0, noise) — right_ankle Z-axis-only 1-frame
    step. No kinetic chain echo; ankle vel = 6 r/s but joint p95 =
    6.5 so below envelope. **AMBIG, leaning toward minor TP** —
    real one-axis pose discontinuity, but probably not significant.
  - frame 646 (wsev 2.3, noise) — right_wrist peak (26 r/s) inside
    a sustained corruption pattern. The wider region (frames 642–
    655) has multiple right_wrist spikes (642 / 646 / 649 / 655);
    pose values bounce around with multiple 0.05–0.15 magnitude
    discontinuities. **TP wrist corruption that should have been
    caught by the corruption-zone detector**, not by 4 separate
    spike-frame reports.
  - frame 682 (wsev 42.4, MAJOR) — synchronized whole-lower-body
    step. right_knee, right_ankle, left_ankle all jump in unison
    between 682 and 683 (X-component changes 0.13–0.32). Running
    cadence stops dead at 683; translation continues at constant
    velocity. **TP — end-of-capture splice (running take + settled
    pose end-frame stitched together)** — should have been caught
    by the stream-break detector.
- Notes / what this teaches us:
  - Spike channel is doing the work of corruption-zone (frame 646)
    and stream-break (frame 682) detectors — the surgery thresholds
    are too high for these signatures.
  - Single-wrist spikes are essentially never the right thing to act
    on; severity metric correctly drops them.
  - The detector's two-frame neighbourhood is too local in
    already-noisy regions (591 / 624) → FPs.
  - "2-frame cadence" intuition from user about frame 416 confirmed
    by midpoint test, but it's a single-point phase glitch, not a
    sustained file-wide cadence.

### ACCAD/Male1Running_c3d/Run C24 - quick side step left_poses.npz
- Run date: 2026-05-22
- Score / classification: noise_score 41.66, problematic
- n_frames 356, duration 2.97 s, fps 120
- Spike frames reviewed:
  - frame 268 (wsev 62.3, MAJOR) — 13 joints firing simultaneously,
    including multiple core (spine1/2/3) at 1.7–3.8× their own p95.
    Max joint vel 18 r/s with max ratio 12×. **TP MAJOR** — almost
    certainly a take splice or major glitch (not visually confirmed
    yet but the multi-core synchronized signature is unmistakable).
  - frame 93 (wsev 11.9, review-distal) — top contributor is
    left_wrist at 14 r/s (3.88× wrist p95). Only 1 qualifying
    core/mid joint (spine3 at 3.0). **FP — arm-only fast motion**
    during a quick side step, NOT visible as a glitch (per user).
  - frame 100 (wsev 11.1, review-distal) — same pattern as 93.
    left_wrist 10 r/s, shoulders barely qualify. **FP — arm-only.**
  - frame 4 (wsev 3.7, minor) — file boundary, low velocities with
    high ratios because rolling baseline hasn't stabilised. Likely
    spurious.
  - frames 133, 154, 248, 98 — all single-joint, wsev < 4,
    correctly demoted to `minor` or `noise` by the metric.
- Notes / what this teaches us:
  - "Problematic" classification + few spike entries (33 across 8
    unique frames) can still mean one HUGE event (268) overshadowed
    by minor noise.
  - The metric's joint-weighting cleanly separates arm-flailing
    (93, 100) from real events (268).

### ACCAD/Male2MartialArtsPunches_c3d/E15 - body jab right_poses.npz
- Run date: 2026-05-22
- Score / classification: noise_score 38.69, **problematic**
- n_frames 206, duration 1.72 s, fps 120
- Stream breaks: none. Corruption zones: **1, recommended excise (52% of file)**.
- Cadence: detected (period 2, strength 0.55, coverage 0.83)
- All spike entries (4) score wsev = 0 — they're sub-p95 wrist/head
  twitches, not real spikes.
- **The "problematic" label is driven entirely by a false-positive
  corruption zone on the right_elbow during the punch.** See
  detailed analysis in "Corruption detector false-positives on
  short high-action files" tuning-log entry below.
- Notes / what this teaches us:
  - "Problematic" doesn't mean noisy — it can mean the detector is
    confused. Always check what's driving the classification.
  - 4 short punch files in ACCAD/Male2MartialArtsPunches_c3d show
    the identical pathology (see tuning log).
  - Cadence detector successfully finds the 2-frame cadence on
    this file (it was missed on C6, where the apparent cadence is
    only a local artifact, not file-wide).

## Threshold / parameter tuning log

When we change a constant in `estimate_noise_torque.py`, record:
- Date
- Constant name and old → new value
- Files / observations that motivated the change
- What got better, what (if anything) got worse

### 2026-05-22 — `CORRUPTION_MIN_FILE_S` introduced, value 5.0

**First attempt** (since reverted): added `CORRUPTION_MIN_FRAC_OF_P95 = 0.8`
gating frames where `short_mean ≤ 0.8 × joint_file_p95`. This eliminated
all four punch FPs but **dropped Maritsa from 12 zones → 6** and flipped
its classification problematic → moderate. Wrong outcome: Maritsa is known
to have severe corruption throughout, and the user confirmed problematic
is the correct label.

**Root cause of the failure**: whole-file p95 is a corruption-fragile
reference. When corruption is pervasive, it inflates the p95 itself,
which raises the gate threshold and hides the corruption from
detection. Self-defeating.

**Replacement fix** (now in place): added
`CORRUPTION_MIN_FILE_S = 5.0` constant and an early return in
`_detect_corruption_zones` when `T / fps < CORRUPTION_MIN_FILE_S`.

Rationale: the punch FPs occur specifically because the long-window
rolling baseline collapses to ~the length of the activity itself on
short files. Below ~5 s, the rolling baseline simply cannot give a
meaningful "is this anomalous?" signal, so corruption detection is
skipped entirely. Above 5 s, the original local-ratio test (which
uses a *median*, robust to corruption) works correctly — Maritsa
proves this.

**Verified outcomes (re-run with the file-length gate):**

| file | dur | orig cls / score / zones | new cls / score / zones |
|---|---|---|---|
| E15 body jab right | 1.7 s | problematic / 38.69 / 1 FP | **clean / 19.01 / 0** |
| E14 body cross right | 2.5 s | problematic / 34.26 / 1 FP | **clean / 20.15 / 0** |
| E16 body jab left | 2.5 s | problematic / 31.97 / 1 FP | **clean / 20.11 / 0** |
| E2 Jab right | 2.8 s | problematic / 31.79 / 1 FP | **clean / 21.03 / 0** |
| C6 stand to run backwards | 5.8 s | moderate / 46.50 / 0 | moderate / 47.46 / 0 |
| C24 quick side step left | 3.0 s | problematic / 41.66 / 0 | problematic / 39.79 / 0 |
| **Maritsa Relaxed** | 142 s | **problematic / 36.85 / 12** | **problematic / 36.03 / 12** |

All four punch FPs eliminated; Maritsa preserves all 12 zones and
its problematic classification. Note C24 is also under the 5 s
threshold so corruption detection is now skipped there — no change
in observable output (C24 never produced zones), but worth flagging
that any sub-5-s file is now blind to corruption.

**Open**: do any AMASS files exist that are < 5 s AND have real
marker corruption? If so, they will now be silently missed. Worth
auditing the dataset for short files with extreme max-velocity
joint signatures before considering the fix final.

### 2026-05-22 — spike frames folded into clean-segment bad mask

`_find_clean_segments` now accepts a `spike_frame_indices` parameter
and marks each provided frame as bad before margin expansion runs.
This closes a real gap: the per-frame torque score's smoothing
absorbs many single-frame discontinuities (that's why
`_detect_spike_frames` exists as a complementary detector), so
spikes were previously invisible to the clean-segment logic. A
user filtering for "clean" data could get a chunk containing a
synchronized whole-body glitch (e.g. C24 frame 268).

**Initial implementation (conservative fold-all) over-clipped on
files where the spike detector emits many sub-envelope wrist
twitches** — Maritsa dropped 88.1 % → 4.9 %, C24 dropped 61 % → 0 %,
E15 (classified clean) had 0 clean segments. The user pushed back
that this fragmented files based on motion that wasn't actually
problematic.

**Replaced with severity-aware folding.** The script now computes a
per-frame weighted severity score (the metric described in the
"Spike severity metric" section above, but now in-script via a new
`_significant_spike_frames` helper) and only folds frames that pass
the `MAJOR or strong` gate (`wsev ≥ 10 AND ≥ 2 qualifying core/mid
joints`).

New constants in `estimate_noise_torque.py`:

```
SPIKE_SEVERITY_JOINT_WEIGHT        — per-joint weight dict
SPIKE_SEVERITY_QUALIFY_CONTRIB = 2.0
SPIKE_SEVERITY_STRONG_THRESHOLD = 10.0
SPIKE_SEVERITY_MIN_QUALIFYING = 2
```

**Impact on clean-segment totals** (no impact on classification or
noise_score):

| file | clean orig | clean (fold-all) | clean (severity-aware) | significant spikes folded |
|---|---|---|---|---|
| C6 | ~93 % | 55.7 % | **90.7 %** | 2 (416, 682) |
| C24 | 61 % | 0 % | **62.1 %** | 1 (268) |
| Maritsa | 88.1 % | 4.9 % | **44.6 %** | many |
| E15 | 0 % | 0 % | 0 % then 40.8 % (see below) | 0 |

C24 now produces exactly what was expected: `[0–94]` and
`[139–264]`, both ending before frame 268. C6 keeps its two big
clean segments, broken only by the MAJOR events at 416 and 682.
Maritsa is between the over-clip and the original — an honest
middle ground that reflects only the spikes severe enough to be
real glitches.

### 2026-05-22 — `CLEAN_MIN_DURATION` made adaptive on short files

After the severity fix, E15 (clean classification) still showed
zero clean segments because of 5 suspect-tier (score ≥ 0.2) frames
spread through its 1.72-second length, leaving no 1.0 s gap.
Logical inconsistency: file is "clean" yet has no clean segments.

Fix: `_find_clean_segments` now computes
`effective_min_duration = min(CLEAN_MIN_DURATION, T_seconds / 4)`
so on short files the minimum scales down. For 1.72 s files the
min is 0.43 s; for 2.97 s files (C24) it's 0.74 s. Above ~4 s the
1.0 s cap takes over (unchanged behaviour).

### 2026-05-22 — clean segments no longer use per-frame score for fragmentation

After raising the threshold to GLITCH_THRESH, C24 was still broken
into two pieces because score-based glitches at 100–108 and 127–133
fragmented the file. Investigation showed those are all spine-joint
torque impulses (scores 0.5–1.1, worst joint spine1/2/3) from the
side-step motion itself — not pose anomalies.

The torque scorer has a known bias on dynamic motion: real
movement impulses on core joints produce sustained scores in the
glitch range without there being any actual pose discontinuity.
This is fine for noise_score and classification (where the file
*is* dynamic), but it shouldn't fragment clean-segment output
because the underlying poses are clean.

**Fix:** `_find_clean_segments` no longer consults `scores` at all.
Bad mask now comes from three sources only:
1. `corruption_mask` (corruption zones + stream breaks via the
   structural_mask the caller assembles)
2. `spike_frame_indices` (significant spikes — multi-joint
   above-envelope, filtered by `_significant_spike_frames`)
3. Margin expansion around the above

This treats clean segments semantically as "stretches with no
identified pose-level anomalies." Per-frame score is a softer
torque-impulse signal that goes into noise_score and
classification but not segment boundaries.

**Results:**

| file | clean before | clean after | segments |
|---|---|---|---|
| E15 | 100 % | 100 % | `[0–205]` 1.72 s |
| C6 | 92.6 % | 96.3 % | `[0–412]` 3.44 s + `[420–678]` 2.16 s |
| C24 | 63.2 % | **74.4 %** | `[0–264]` 2.21 s (single segment, ends at 268) |
| Maritsa | 45.7 % | 45.9 % | unchanged |

C24 now reads as a single clean stretch right up to its only real
event at 268, matching the user's expectation. C6 expands slightly
where the score-margin around frame 416 was wider than the spike
margin. Maritsa is essentially unchanged because its fragmentation
already comes from corruption zones + significant spikes.

### 2026-05-22 — arm-chain spike-density gate added for clean segments

Maritsa's clean segments still contained many sub-significant
arm-chain spikes ("flickers of torque in shoulders, pecs, and
arms"). These don't pass the spike-severity gate individually
(wrist weight is 0.2; requires nC ≥ 2 core/mid joints), so they sit
inside clean sections. The new density gate fragments clean
segments wherever the density of unique arm-chain spike frames in a
rolling window crosses a threshold.

**New constants** in `estimate_noise_torque.py`:

```
ARM_CHAIN_JOINT_INDICES = {13, 14, 16, 17, 18, 19, 20, 21}
    # collars + shoulders + elbows + wrists (covers "shoulders / pecs / arms")
SPIKE_DENSITY_WINDOW_S = 1.0
SPIKE_DENSITY_MIN_COUNT = 10
```

(Initial value of 6 over-clipped C6's "vigorous-arm-action" post-416
region — user pushed back that this is real running motion, not
flickers. Empirical density profile across calibration files:

| file | max density | p99 | p95 | p90 | p50 |
|---|---|---|---|---|---|
| E15 | 3 | 3 | 2 | 2 | 2 |
| C24 | 4 | 4 | 4 | 4 | 2 |
| C6 | 8 | 8 | 7 | 5 | 1 |
| Maritsa | **27** | 23 | 19 | 16 | 8 |

C6's vigorous-arm peak is 8; Maritsa's *median* is 8 and its peak
is 27. The gap above 8 is the natural cutoff; threshold 10
cleanly separates "fast arm motion" from "arm-marker flicker".)

**Implementation** in new helper `_arm_spike_density_mask`:

1. Mark every frame with ≥ 1 arm-chain spike entry as 1, else 0.
2. Rolling sum over a 1-second window.
3. Bad mask = (rolling count >= 6).

OR'd into the bad-mask passed to `_find_clean_segments`. Does not
affect noise_score or classification — clean-segment-only signal.

**Crucially, this counts UNIQUE FRAMES, not entries.** A single
multi-joint event (e.g. C24 frame 100 with 4 arm-chain joint
entries) only contributes 1 to the count. So a brief multi-joint
motion burst doesn't trigger; only sustained flickering does.

**Results across the calibration files (threshold = 10):**

| file | clean (no density gate) | clean (thr=10) | impact |
|---|---|---|---|
| E15 | 100 % | 100 % | unchanged |
| C24 | 74.4 % | 74.4 % | unchanged |
| C6 | 96.3 % | **96.3 %** | unchanged — vigorous arm action during running peaks at density 8, well below threshold 10 |
| Maritsa | 45.9 % | **30.4 %** | flickery middle fragmented; 8.78 s clean opening + several 3–4 s segments preserved |

Originally tried threshold 6 (over-clipped C6) then 10. The C6
walkthrough's "TP wrist corruption" labelling at frames 477 / 523
/ 554 / 559 / 581 / 587 was wrong — user clarified post-416 is
the vigorous-running portion of the file, with legitimate arm
swings, not corruption.

### 2026-05-22 — pure-arm-cluster gate added

Maritsa frame 994 highlighted a remaining gap. Spike entries:
`left_shoulder(v=3.9 r=6.8), left_elbow(v=16.3 r=10.3),
left_wrist(v=15.1 r=8.5)` — 3 arm-chain joints firing
synchronously on the same side, with two of them well above their
own p95. The severity gate missed it (wsev=8.68, just below 10;
nC=1) because Maritsa's p95 is itself corruption-inflated. The
density gate missed it (5 unique arm frames in window, below 10).

But these "pure-arm clusters" have a clean signature: **N+ arm-chain
joint entries AND zero non-arm-chain entries at the same frame**.
Real coordinated motion always involves spine, pelvis, or legs;
pure-arm activity is suspicious.

**Empirical check across calibration files** (count of frames with
≥ 3 arm-chain joints and 0 non-arm-chain joints in the same spike
entry list):

| file | pure-arm 3+ frames |
|---|---|
| E15 | 0 |
| C24 | 0 |
| C6 | 0 |
| Maritsa | **131** |

Clean separation. Added `PURE_ARM_CLUSTER_MIN_JOINTS = 3` constant
and a second gate in `_significant_spike_frames`:

```python
if n_arm_chain >= PURE_ARM_CLUSTER_MIN_JOINTS and n_non_arm_chain == 0:
    significant.add(f)
```

A frame is significant if it passes EITHER the severity gate OR
the pure-arm-cluster gate.

**Final results across calibration files:**

| file | duration | clean total | first segment |
|---|---|---|---|
| E15 | 1.7 s | 100 % | `[0–205]` 1.72 s (whole file) |
| C24 | 3.0 s | 74.4 % | `[0–264]` 2.21 s |
| C6 | 5.8 s | 96.3 % | `[0–412]` 3.44 s + `[420–678]` 2.16 s |
| Maritsa | 142 s | 23.9 % | `[0–990]` 8.26 s (ends right before pure-arm cluster at frame 994) |

### 2026-05-22 — clean-fraction penalty added to noise_score (Component 8)

User observation: "A long file with only a very few short clean
segments is automatically very problematic." Maritsa demonstrated
the problem: noise_score = 36, just barely "problematic," even
though only 24% of its 142 seconds is extractable as clean. The
score should reflect how little usable material survives, not just
which channels fired.

Added new score component and a classification rule. New constants:

```
CLEAN_FRAC_PENALTY_MIN_DURATION_S = 10.0   # only apply to long files
CLEAN_FRAC_PENALTY_THRESHOLD = 0.5         # penalise below 50%
CLEAN_FRAC_PENALTY_SCALE = 60.0            # multiplier on the gap to threshold
CLEAN_FRAC_PROBLEMATIC_THRESHOLD = 0.5     # force problematic below this
```

Component 8 formula:
```
if file_seconds >= 10 and clean_frac < 0.5:
    score += (0.5 - clean_frac) * 60
```

Plus a classification override: long files with clean_frac < 50 %
are forced to "problematic" regardless of which signals fired.

**Impact on calibration files:**

| file | duration | clean frac | score before | score after | penalty |
|---|---|---|---|---|---|
| E15 | 1.7 s | 100 % | 19.01 | 19.01 | exempt (< 10 s) |
| C24 | 3.0 s | 74.4 % | 39.79 | 39.79 | exempt (< 10 s) |
| C6 | 5.8 s | 96.3 % | 47.46 | 47.46 | exempt (< 10 s) |
| Maritsa | 142 s | 23.9 % | 36.03 | **51.68** | +15.66 |

The 10-second exemption matters: a short file with a single spike
near a boundary could reasonably have a low clean fraction without
being "problematic." Long files have no such excuse.

Scale of 60 was chosen so:
- 25 % clean (Maritsa-like) → ~15 point penalty
- 10 % clean → 24 point penalty
- 0 % clean → 30 point penalty (saturates at 100 anyway)

### 2026-05-22 — score_spikes made length-relative and uncapped

User: "score_spikes should probably be relative to file length, and
not capped." Old formula `min(n_spike_frames * 3, 15)` saturated
at 5 spikes — Maritsa's 1574 spikes contributed the same 15 points
as a file with 5 spikes.

New formula: `score_spikes = (n_spike_frames / T) * 300` — scales
linearly with spike density (fraction of file), no cap.

| file | T | n_spikes | old | new |
|---|---|---|---|---|
| E15 | 206 | 4 | 12.0 (cap) | 5.83 |
| C24 | 356 | 8 | 15.0 (cap) | 6.74 |
| C6 | 698 | 22 | 15.0 (cap) | 9.46 |
| Maritsa | 17055 | 1574 | 15.0 (cap) | 27.68 |

### 2026-05-22 — clean_section_score added (dual-score model)

User proposed two complementary scores: noise_score rates the file
overall, clean_section_score rates whether the identified clean
segments contain usable data.

Added as a required field `clean_section_score: float` on
TorqueFileReport. Also displayed in print_report alongside
noise_score.

### 2026-05-22 — hard-glitch corroboration concept

User: "I still feel we are not successfully differentiating
sufficiently between real dynamic torque and clearly impossible
glitches." The torque scorer treats both the same: dynamic core
motion impulses (C24 spine during a side step) and impossible
pose discontinuities (Maritsa marker explosions) both produce
high frame_scores.

**Discriminator: pose-level evidence.** Significant spikes,
corruption zones, and stream breaks are reliable channels for
flagging real physical anomalies. The per-frame torque score is
unreliable on its own (FPs on dynamic motion).

**Definition: a "hard glitch" requires BOTH the score elevation
AND independent pose-level corroboration:**

```python
significant_spike_frame_set = _significant_spike_frames(spike_frames, poses, fps)
corroboration_mask = structural_mask.copy()
for f in significant_spike_frame_set:
    corroboration_mask[f] = True
glitch_mask = (frame_scores >= GLITCH_THRESH) & corroboration_mask
```

Applied throughout:
- `n_g`, `g_frac`, `glitch_list`, `glitch_clusters` — use hard mask
- `peak`, `p99` — restricted to hard glitch frames (default 0 if none)
- `max_raw_surprise`, `max_raw_effort` — restricted to hard frames

Plus a classification bump: if classified clean but has any pose-
level anomalies (`n_significant_spikes + n_corruption_zones +
n_stream_breaks >= 1`), bump to moderate. Prevents a file with
a real event but no torque-score corroboration from being labelled
clean.

### 2026-05-22 — clean_section_score simplified to spike density

After the hard-glitch concept landed, user noticed clean_section_score
was still using the old per-frame-score formula, inflating C24's
(36) and C6's (40) clean_section_scores with the same FP spine
glitches that no longer count toward noise_score.

By construction, clean segments contain no significant spikes,
no corruption zones, no stream breaks — so under the hard-glitch
rule there are zero hard glitches within clean. Per-frame torque
score has the same dynamic-motion FP problem within clean as it
does for the whole file.

**Final formula** (pure sub-significant spike density within clean):

```python
clean_section_score = (n_spikes_in_clean / n_clean_frames) * 300
```

This measures exactly the "flicker" signal — frames flagged by the
spike detector but didn't pass the severity gate. Single-joint
wrist twitches, isolated arm blips, etc.

**Final results across calibration files:**

| file | classification | noise_score | clean_section_score | clean frac |
|---|---|---|---|---|
| E15 | clean | 5.83 | 5.83 | 100 % |
| C6 | moderate | 9.46 | 8.04 | 96.3 % |
| C24 | moderate | 6.74 | 7.92 | 74.4 % |
| Maritsa | problematic | **76.65** | **7.72** | 23.9 % |

**Maritsa contrast (76.65 → 7.72) is the headline result of all
this work.** noise_score answers "is this file usable as-is?" (no,
it's broken). clean_section_score answers "if I extract just the
clean sections, is that data usable?" (yes, the extracted data is
very clean).

For reference: Maritsa has 568 significant spike frames, 12
corruption zones, and 9 stream breaks. If a score component were
added based on raw pose-anomaly count (n_pose_anomalies × 5),
unsaturated value would be 2945. Saturation at 30 was therefore
warranted as an absolute cap — but a length-relative formula
provides better behavior. See next entry.

### 2026-05-22 — Component 8: pose-anomaly density (length-relative, uncapped)

User: "Anomalies value is only relevant if relative to file_size.
Kill 30 cap, but scale by length."

Implementation: density formula matches the other density components
in the score:

```python
n_pose_anomalies = (
    len(significant_spike_frame_set)
    + len(corruption_zones)
    + len(stream_breaks)
)
score_anomalies = (n_pose_anomalies / T) * 300
```

No explicit cap — only saturates via the natural `ns ≤ 100` ceiling.
Same scaling factor (300) as the spike-density component, so the two
"density × 300" terms read consistently.

Impact on calibration files:

| file | n_anom | T | density × 300 | noise_score before | noise_score after |
|---|---|---|---|---|---|
| E15 | 0 | 206 | 0 | 5.83 | 5.83 |
| C6 | 2 | 698 | 0.86 | 9.46 | 10.32 |
| C24 | 1 | 356 | 0.84 | 6.74 | 7.58 |
| Maritsa | 589 | 17055 | 10.36 | 76.65 | **87.01** |

The Maritsa case is the headline: 568 significant spikes + 12 zones
+ 9 breaks distributed across 142 seconds → +10.36 from this
component alone. Short files with few significant events get a
proportionally small contribution.

**Final calibration set (after all changes):**

| file | classification | noise_score | clean_section_score | ratio |
|---|---|---|---|---|
| E15 | clean | 5.83 | 5.83 | 1.0 |
| C6 | moderate | 10.32 | 8.04 | 1.28 |
| C24 | moderate | 7.58 | 7.92 | 0.96 |
| Maritsa | problematic | **87.01** | **7.72** | **11.3** |

The 11.3× ratio for Maritsa is the dual-score model demonstrating
its full value: file is heavily compromised overall (87) yet the
extractable clean material is genuinely usable (7.72, in the same
range as the clean/moderate calibration files).

### 2026-05-23 — adaptive per-joint break threshold (Shadow-file FPs)

Validated detector on Shadow-derived file `Subject2_take3_beta_smpl
_poses_rhipflexstat.npz` (Subject2 Contemporary, flex_stat, 1040 s,
60 fps — hardware captured at 100 Hz, saved at 60 Hz so the file
has BOTH a 3-frame cadence in the original data AND a 5-frame
interference pattern from 60/100 Hz resampling).

Initial output flagged **23 stream breaks**. User identified only
2 as real (frames 14679, 15273 — sensor recalibrations during
stillness). The other 18 were all `multi_joint` type, clustered in
fast-motion sections of the file.

**Diagnostic:** plotted local body angular velocity at each break:

| break type | count | mean local body vel | vs file p90 (49 r/s) |
|---|---|---|---|
| translation (real recal) | 3 | 27 rad/s | 55 % (low motion) |
| trans_shift | 2 | 15 rad/s | 30 % |
| **multi_joint** | **18** | **62 rad/s** | **126 %** (above p90) |

Smoking gun: every `multi_joint` break happens during body velocity
above the file's 90th percentile. They're cadence-induced drops
during fast movement, not real discontinuities.

**Root cause:** the per-joint break threshold was static
(`joint_median × JOINT_BREAK_FACTOR`). During rapid motion, joint
pose displacements are naturally elevated; cadence-induced phase
errors during such motion push enough joints over the static
threshold to trigger the multi-joint criterion.

**Fix:** adaptive threshold using rolling local median per joint.

```python
JOINT_BREAK_LOCAL_WINDOW_S = 1.0
JOINT_BREAK_LOCAL_FACTOR = 5.0

# Per-frame, per-joint threshold:
local_joint_medians = rolling_median(pose_disp, window=1s)
local_joint_thresholds = local_joint_medians * 5
joint_thresholds[t] = max(global_threshold, local_threshold[t])
```

**Result on Shadow file:**

| | before | after |
|---|---|---|
| stream breaks | 23 | **5** |
| clean total | 97.5 % | 98.6 % |
| noise_score | 2.60 | 2.51 |
| user-identified real breaks (14679, 15273) | preserved | preserved ✓ |
| cadence-induced FPs eliminated | 0 | 18 |

The 5 surviving breaks are all translation/trans_shift type with
meaningful root displacement (0.03–0.13 m) — the recalibration
signature. The fix is the multi-joint analog of the corruption-
detector's local-baseline approach; both use rolling local
statistics so dynamic motion doesn't get mistaken for noise.

### Earlier: clean-segment threshold raised from SUSPECT_THRESH to GLITCH_THRESH

After the adaptive-duration fix, E15 was still only 40.8 % clean
because `_find_clean_segments` used `bad = scores >= SUSPECT_THRESH`
(0.2). The user pushed back: a file with 0 glitch frames and score
19 should be essentially entirely clean. Suspect frames are a soft
signal — they shouldn't fragment otherwise-clean files.

Fix: changed the bad-mask in `_find_clean_segments` from
`scores >= SUSPECT_THRESH` to `scores >= GLITCH_THRESH`. The
two-tier suspect/glitch system was designed to distinguish soft
(informational) from hard (actionable) anomalies — clean segments
should use the hard threshold. `n_suspect_frames` is still reported
separately for diagnostic purposes.

**Final clean-segment results across the four calibration files:**

| file | duration | classification | clean total | clean segments |
|---|---|---|---|---|
| E15 | 1.72 s | clean | **100 %** | `[0–205]` 1.72 s |
| C6 | 5.82 s | moderate | 92.6 % | `[0–404]` 3.38 s + `[438–678]` 2.01 s |
| C24 | 2.97 s | problematic | 63.2 % | `[0–96]` 0.81 s + `[137–264]` 1.07 s |
| Maritsa | 142 s | problematic | 45.7 % | 5+ multi-second segments |

C24 still ends its second clean segment exactly before frame 268
(the MAJOR event). C6 still breaks at frames 416 and 682. Maritsa
preserves its 12 corruption zones and severity-folded major spikes.
E15 is now correctly 100 % clean — matching its classification.

### Earlier proposal (now applied — see above)

**Corruption detector false-positives on short high-action files.**

Discovered while looking at E15 - body jab right and 3 sibling files.
Every short ACCAD martial-arts punch file is being mis-classified
`problematic` because the corruption-zone detector flags the punching
arm's elbow as a 100-frame corruption zone covering 30–52 % of the
file.

Verified pathology, 4-for-4:

| file | dur | flagged joint | zone mean v | joint p95 |
|---|---|---|---|---|
| E15 body jab right | 1.7 s | right_elbow | 4.9 | 13.7 |
| E14 body cross right | 2.5 s | right_elbow | 5.0 | 10.5 |
| E16 body jab left | 2.5 s | left_elbow | 4.4 | 10.1 |
| E2 Jab right | 2.8 s | right_elbow | 4.4 | 11.4 |

In every case **zone mean velocity is below the joint's own p95** —
the opposite of what real marker corruption looks like. The
`CORRUPTION_LOCAL_RATIO = 4.0` test fires because the punching
movement (~5 rad/s sustained for ~1 s) is much larger than the
long-term median over the rest of the file (the body is mostly still
between punches, joint median ~0.5 rad/s) → ratio 6–12×.

Worsened by the long-window clamp: for these short files,
`CORRUPTION_LONG_WINDOW_S = 10.0` collapses to `(T-1)//2 ≈ 2 s`,
which is roughly the same length as the punching activity itself. So
the "baseline" the punch is compared against IS the punch.

**Proposed fix** (estimate_noise_torque.py corruption-zone test):
require that the zone's mean velocity exceed the joint's whole-file
p95 (or some fraction like 0.8 ×) before flagging. A zone with
`mean_vel < joint_p95` cannot be sustained corruption — by
definition, corruption produces above-normal velocity. This single
test would eliminate all four false-positives above without
weakening the detector for real corruption (where `zone_mean >> p95`
is the usual signature).

Tentative threshold: `zone_mean_vel > 0.8 × joint_file_p95`.

To apply: edit `_detect_corruption_zones` (estimate_noise_torque.py
around line 1462) and re-run the batch. Need to verify it still
catches the real corruption cases (e.g. shadow planted-foot files).

## Known noise types

What we expect the script to catch (and what it should NOT flag):

**Should be flagged as noise:**
- Marker dropout / arm-chain corruption (sustained garbage on shoulder/elbow/wrist)
- Stream breaks (concatenated takes, capture resets)
- Single-frame teleportation glitches
- Direction-flip jitter in static poses

**Should NOT be flagged (legitimate physics):**
- Foot contact torque spikes during normal gait
- Whole-body torque elevation during cartwheels, jumps, falls
- Hand plant during floor work
- Airborne phase inertial effects

**Edge cases worth specifically testing:**
- Shadow planted-foot movement (foot stays planted but moves a little) —
  the dynamic-motion and contact-event discounts must handle this without
  flagging it as noise.
- AMASS static-pose-style files where the performer is mostly still:
  small movements should not produce huge surprise scores.

## Open questions

- Are the corruption-zone thresholds (`CORRUPTION_LOCAL_RATIO=4.0`,
  `CORRUPTION_VEL_FLOOR=3.0`) appropriate across AMASS sub-datasets, or do
  they need per-dataset tuning?
- Does the additive scoring composition (frac + peak + p99 + raw + count
  + corruption) double-count in ways that distort `clean` / `moderate` /
  `problematic` thresholds?
- Joint importance weights down-weight contact-affected joints heavily —
  do we lose real ankle/foot noise in shadow files because of this?
- Spike-severity tier thresholds (`MAJOR ≥ 20`, `strong ≥ 10`,
  `minor ≥ 3`) were calibrated on 2 files. Do they generalise across
  more sub-datasets? Need to check at least 3–5 more before treating
  them as stable.
- C6 frame 694 sits at `review-distal` (wsev 10.2, only 1 qualifying
  core/mid joint) but was tentatively labelled "likely TP" without
  visual verification. Worth confirming.
- The corruption FP on short punch files (see tuning log) — is the
  same FP appearing on real-corruption files (i.e. zones that pass
  the proposed `mean_vel > 0.8×p95` test)? Need to verify the fix
  doesn't break true corruption detection.
- Cadence detector worked on E15 (period 2, strength 0.55) and gave
  null on C6 (period 3, strength 0.19). Does its `strength` field
  correlate with severity-of-spike-events as an independent quality
  signal? Worth a scatter plot across the whole dataset.

## References

- Script: `dpg_system/noise_estimation/estimate_noise_torque.py`
- Stale prior runs: `dpg_system/noise_estimation/torque_results_by_folder/`,
  `torque_checkpoint.json`, `result_*.json`
- Related memory entries: `smpl_contact_test_data`, `project_axis_permutation_design`