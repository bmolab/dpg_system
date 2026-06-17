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
- **2026-06-14:** Reviewed the first full lens run (06-13, lenses-only). Long
  investigation into spike significance on rub100/0022_throwing_hard1 (f211 vs
  f215) — see the dated section "Spike significance, impact metrics, and the
  severity/dial architecture" below for the full reasoning. Headline:
  physical impact (exact SMPLH mesh displacement) does NOT separate a visible
  glitch from an imperceptible one (both ~0.5 cm); the visible one is
  kinematically indistinguishable from legitimate fast motion, so that class
  shouldn't be chased. Built and shipped a severity/dial architecture instead:
  per-cluster lens severity in the run, an offline `clean_segment_tools`
  re-derivation (tunable tolerance + segment-joining, no re-run), and live
  dials in NoiseReviewNode. Fresh full run in
  `noise_results_lenses_2026_06_14/` (14,182 files, 0 fail). At the default
  dial, 81.5% of flagged files recover a usable clean middle.
- **2026-06-14 (continued — very long session):** After the severity/dial work,
  the session continued through: flicker is NOT the over-firing culprit (the
  moderate flood is the single-significant-spike rule; tuned it → +1,023 files
  recovered to clean, recall held); severity-aware join fix (never bridge a hard
  glitch); dev×speed corruption lens (diagnostic, complementary); the
  **excursion lens** (off_dev + regime_change — the first real glitch-vs-
  ballistic-motion discriminator; built, complementary not redundant); a long
  **perceptibility investigation** concluding perceptibility must be LOCALIZED
  per-limb and is DIRECTIONAL not a global scalar (do NOT wire a threshold
  filter — it can't separate the subtle regime); and an **attribution pass**
  (region + corrected frame + source joint, `attribution.py`, wired + re-run,
  100% coverage, frame moved in 35% of clusters). The product goal is
  re-centred: extract motion-valid SUBSEQUENCES (clean-section excision), not
  just classify files — so excursion is now in the clean-segment bad-mask
  (prevents glitch leak). See the dated sections below for full reasoning.
  **Where we ARE:** attribution corpus-wide in `noise_results_lenses_2026_06_14/`.
  **Next:** node navigation (emit peak_frame + region); re-validate the localized
  ~0.30 cm perceptibility line now that flags are attributed; extend attribution
  to spike_clusters; (future) repair-in-place for repairable excursions.

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

### 2026-06-11 — spike-frame consolidation into contaminated sections

Assessment of the June 1 batch (`noise_estimation_results_June_1_2026/`)
showed spike frames cluster heavily: 2.86M per-joint spike records →
920K distinct spike frames → 210K clusters at gap ≤ 30 frames (4.4×).
17.8% of clusters contain ≥ 5 spike frames; extreme cases (DanceDB
`Theodora_Mix`: 2,938 spike frames → 8 sections covering 97% of file)
are clearly contaminated sections, not independent dropped frames.

Added `SpikeCluster` + `_cluster_spike_frames()`: spike frames within
`SPIKE_CLUSTER_GAP_S = 0.25 s` merge into one cluster; clusters with
≥ `SPIKE_CLUSTER_CONTAMINATED_MIN = 3` distinct spike frames are
reported as contaminated sections (start/end/duration/density/joints/
max_velocity), the rest as isolated spikes. New `spike_clusters` field
in the report JSON (raw `spike_frames` retained for compatibility);
`classification_detail` now reads "N contaminated section(s) (Xs) +
M isolated spike(s)" instead of enumerating every spike frame.
Reporting/JSON only — noise_score and classification are unchanged.

Validated on Maritsa (1,574 spike frames → 64 contaminated sections
+ 47 isolated; sections align with the known corruption zones) and
ACCAD Walk B17 (29 → 4 sections + 8 isolated, including the frame-4
multi-joint onset spike correctly kept as one isolated event).

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

## 2026-06-14 — Spike significance, impact metrics, and the severity/dial architecture

Long session driven by reviewing the first full lens run
(`noise_results_lenses_2026_06_13/`, lenses-only via `--skip-torque`).
Worked from one file —
`AMASS/BioMotionLab_NTroje/rub100/0022_throwing_hard1_poses.npz`
(470 f @120, "problematic": heavy flicker bracketing a fast throw) — out to
a corpus-wide design change. Findings, in order:

**1. Spike frames are kinematic, not torque-derived (corrects a prior
assumption).** `_detect_spike_frames` runs on the pose stream (neighbour-ratio
on joint angular velocity) *before* the torque pass, so it is unaffected by
`--skip-torque`. The torque scorer produces glitch/suspect frames, surprise,
effort — all zero under skip-torque. So "spike frames" were never the torque
pipeline's output. `n_spike_frames` is not a serialized scalar (only
`spike_clusters[].n_spike_frames` and the `spike_frames` list), hence `None`
on `.get`.

**2. The neighbour-ratio is anti-correlated with perceptibility.** Two
ground-truthed frames in the throw's "clean" middle:
- f211 (real, visible sudden wrist rotation): wrist 18.9 rad/s, ratio 4.0
  (LOW — a real fast move has moving neighbours).
- f215 (a real but imperceptible artifact): 5 right-side joints, ratios
  3.1–11.3 (HIGH — isolated 1-frame pops against a still baseline).
So raising `SPIKE_NEIGHBOR_RATIO` would drop the visible one and keep the
imperceptible one. The `SPIKE_VEL_FLOOR=2.0` is also low enough that
sub-perceptual micro-jitter clears it. Ratio finds the *isolated-frame shape*
but says nothing about magnitude or perceptibility.

**3. Physical IMPACT does NOT separate f211 from f215 — they are equal.**
Tested with exact manual SMPLH skinning (LBS from `SMPLH_NEUTRAL.npz`:
v_template + shapedirs·β, posedirs, J_regressor, weights), measuring rendered
mesh-vertex displacement of each frame vs a quadratic-predicted clean pose
(neighbours excluding the frame; root removed → body-local):
- f211: 0.64 cm worst-vertex, 0.20 cm mean
- f215: 0.48 cm worst-vertex, 0.25 cm mean (clean neighbour frames ~0.15 cm)
Essentially identical. **The user's parent/child compensation insight is
confirmed**: at f215 the right-leg chain (ankle 5.75, knee 4.62 rad/s) spikes
hard in angular terms yet the foot barely moves — the chain rotations cancel.
f211 is small for the opposite reason: a huge wrist twist of a small distal
segment (big angle × tiny lever). The composed full-chain skinning captures
both automatically — making mesh/descendant-displacement the correct
"significance" weighting (unifies leverage + compensation), but it does not
discriminate these two because they are genuinely equally low-impact.
Per-joint isolated `ω × downstream-reach` is wrong: it double-counts (ignores
that descendants also rotate) and inverts the ranking.

**4. What DOES discriminate (per the user): "a sudden deviation from a FAST
trajectory."** Tested across controls (the throw file; DanceDB
`Sophie_Excited` as genuine corruption; ACCAD `E15 body-jab` + BML
`0021_throwing_hard1` as clean fast motion):
- Reversal-at-speed FAILS — f211 is an *acceleration* (cos 0.95), not a
  reversal; the reversal lens flags f215's ankle instead.
- Speed-weighted jerk `|Δω|·√(|ω_in|·|ω_out|)` separates f211 (137) from f215
  (45), 3×. BUT on the full controls, **clean punches/throws score HIGHER**
  (740–943 raw; 2.9–3.9 on the transience-weighted dev×speed) than the f211
  glitch (137 / 0.57). The transience gate (deviation from a quadratic arc ×
  speed) did not save it — a real throw release genuinely deviates from a
  smooth fit too.
- **Conclusion:** f211 is kinematically indistinguishable from — milder than
  — legitimate vigorous arm motion. This is the wall already noted for the
  teleport lens. Any threshold catching f211 floods on clean punches/throws.
  Since f211/f215 are not severe (~0.5 cm), the right call is NOT to chase
  this class. Speed-weighted-jerk / dev×speed IS, however, an excellent
  high-threshold **corruption detector**: DanceDB sits 100–1000× above
  everything (dev×speed 359 vs 0.5–3.9; raw swj 88,386 vs hundreds). Candidate
  lens if a high threshold (e.g. dev×speed > 20) is used.

**5. "Dirty-clean-dirty" is one mode of a general clean/glitch/clean problem,
common in long files.** Across the 11,974 flagged files in the 06-13 run,
~50% had a recoverable clean run (≥2 s and ≥40% of file); the specific
edge-bracketed pattern (≥60% of bad frames in the outer 15%/15%) was 14.6%
(1,753 files). Edge-bracketing is physically expected (T-pose calibration /
sensor settling / actor entry-exit at capture boundaries). Example: longest
CMU file (22,948 f ≈ 191 s) has an 8.5 s clean opening then scattered
single-frame flickers — clean/glitch/clean. So file-level dropping discards a
usable middle in ~half of flagged files.

### Architecture built this session (all three + a fresh run)

**(a) Per-cluster lens severity** (`estimate_noise_torque.py`). New
`LensCluster` dataclass `{start, end, n_frames, peak, severity}`. Each lens
detector (teleport/flicker/ROM/reversal; zigzag already) now returns a
per-frame severity track; `_lens_clusters()` gap-merges flagged frames and
attaches the run's peak severity, normalised to the lens flag threshold
(`severity ≈ peak/ref`; ~1.0 = just flag-worthy). Refs: teleport
`TELEPORT_VEL_HARD=40`, flicker `FLICKER_SYNC=3`, zigzag `ZIGZAG_FLOOR=0.12`,
ROM `ROM_SEVERITY_REF=45°`, reversal `REVERSAL_HARD=25`. **zigzag now also
emits clusters** (it had none — was continuous-only). Cluster fields changed
`List[Tuple]` → `List[LensCluster]`. (Bug fixed post-run: `_detect_reversals`
short-file `T<4` early-return missed `sev_track` → arity crash on 2 sub-4-frame
files; fixed and reprocessed.)

**(b) Offline re-derivation** (`clean_segment_tools.py`, NEW).
`rederive_clean_segments(report, lens_severity_tol, spike_density_tol,
join_gap_s, min_clean_s, margin)` — a pure function over the serialized JSON
(ms per file, no pipeline re-run). Tunes how much minor spiking is tolerated
inside a clean segment and joins segments across sub-tolerance breaks. **Key
design split:** HARD lenses (teleport/ROM/reversal — physically-impossible
excisions) ALWAYS break; SOFT lenses (flicker/zigzag — graded "minor spiking")
are absorbed when `severity ≤ lens_severity_tol`. Backward-compatible with
old bare `[start,end]` tuples (treated as hard breaks). `clean_fraction()`
helper.

**(c) Live dials in NoiseReviewNode.** Four `drag_float` options — `lens
severity tol` (default 1.25), `spike density tol` (1.0), `clean join gap (s)`
(0.0), `clean min dur (s)` (1.0) — re-derive clean segments live from the
loaded JSON. Lens-flag display shows per-cluster severity; handles dict + old
tuple clusters. (Imports `clean_segment_tools` by adding the non-package
`noise_estimation/` dir to `sys.path`.)

**Calibration of the default `lens severity tol = 1.25`:**
- clean punch E15 → 100% clean (zigzag sev 1.17 absorbed)
- rub100/0022 → recovers its clean throw sections as TWO honest segments
  `[107–181]` and `[218–372]`; the contaminated spike at 184–215 (the real
  f211 region) is a HARD boundary that join_gap must never bridge (see the
  severity-aware-join fix below — an earlier version wrongly bridged it).
- DanceDB Sophie → 0% clean (hard teleport/reversal always break)

**Fresh full run:** `noise_results_lenses_2026_06_14/` (14,182 files, 0 fail,
~5 min @ 8 workers `--skip-torque`). 100,436 severity-bearing soft clusters
corpus-wide. At the default dial (tol 1.25, join 0.3 s, min 2 s), **81.5% of
flagged files (9,755 / 11,974) recover a ≥40% clean middle** — vs the ~50%
length-only estimate, because severity-aware joining absorbs minor
flicker/zigzag.

### Flicker is NOT the over-firing culprit — the moderate flood is single-spike (resolved)

Investigated "tune the over-firing flicker thresholds" and the premise was
wrong. Measured on the 06-14 run:
- The "81% active" figure is the harmless per-frame flicker MASK (excises
  individual frames; does not change the label). The flicker LOCK
  (`rate≥2` or `peak≥8` → problematic) fires on only **4.4%** of files.
- Of 10,342 `moderate` files, **exactly 1** has a flicker lock — flicker
  drives `moderate` essentially never. 99.9% of moderate files have spike
  clusters; 53% have a contaminated cluster.
- The 172 flicker-locked files in clean-expected datasets (CMU/BMLmovi/…) sit
  on a **still baseline (p50 0.6 rad/s)** — the textbook jitter signature, so
  most are likely REAL marker jitter, not FPs. Clean-ish `flicker_rate` p99
  (2.23) overlaps DanceDB's median (2.22): no safe threshold gap, and a
  still-baseline gate would invert (DanceDB jitter sits at p50 1.6). So
  flicker was left as-is.

**The real `clean→moderate` driver is line ~1720**: a SINGLE significant
spike frame (`n_pose_anomalies ≥ 1`) demoted the file. That is the f211/f215
case — a sub-perceptual ~0.5 cm single-frame event — and it alone demoted
~17% of moderate files.

**Fix applied:** spike contribution to `moderate` now requires SUSTAINED
evidence — `len(significant_spike_frame_set) >= SIGNIFICANT_SPIKE_MIN_FRAMES`
(=2) OR a contaminated spike cluster. Corruption zones / stream breaks still
demote on a single occurrence. Applied to both `cls` and `cls_lenses_only`.
Re-ran the corpus (06-14, in place):

```
            BEFORE     AFTER
clean        2,207  →  3,230   (+1,023)
moderate    10,342  →  9,308   (-1,034)
problematic  1,632  →  1,632   (unchanged — recall fully preserved)
DanceDB                98.7% problematic (151/153) — corruption recall intact
```

1,023 genuinely-clean files (single harmless spike) recovered from moderate to
clean; zero problematic lost; DanceDB unaffected. Conservative — a `>=3`
threshold would recover more but risks dropping real multi-spike contamination.

### Severity-dial sweep — 1.25 confirmed (but it's a weak knob)

Swept `lens_severity_tol` against the full 06-14 corpus:
- **Soft-cluster severity does NOT separate quality classes.** flicker/zigzag
  cluster severity p50 is ~1.67 (clean) / 1.36 (moderate) / 1.33 (problematic)
  — essentially the same range across all three. (Flicker problematic-ness is
  rate/density via the lock, not per-cluster severity; severity is quantized
  sync_count/3.) So there is no clean/problematic gap to pin the default to.
- **Clean-fraction response is nearly flat in `lens_severity_tol`**: clean
  files re-derive to 0.95→0.98 across tol 0.5→2.0; moderate 0.69→0.77. So 1.25
  is safe and non-critical — keep it (absorbs the weakest 3-joint flicker at
  sev 1.0, breaks on denser).
- **The dominant recovery lever is `join_gap_s`, not lens tol** — but
  SEVERITY-AWARE (see fix below). Mean clean_fraction of moderate files
  (tol 1.25, min 2 s): join_gap 0.0 → 0.53, 0.3 s → 0.69, plateau at 0.69
  beyond 0.3 s (soft gaps are short; the rest are hard boundaries).
  lens_tol 0.5→2.0 only ~+0.15. So for recovering clean middles, reach for
  **join gap** first, then lens severity as a fine-trim.

### Severity-aware join fix (join must NOT bridge regardless of severity)

The first `join_gap` bridged ANY gap shorter than the threshold — including a
brief genuine glitch (teleport / corruption / contaminated spike) — pulling it
into a "clean" segment. That was wrong (and was inflating the recovery numbers
above: the old unsafe version reached 0.85 on moderate files; ~16 points of
that was bridging real glitches, e.g. rub100/0022's contaminated f211 region
at 184–215). Fixed: `clean_segment_tools` now tracks a HARD mask (teleport /
ROM / reversal / corruption zones / stream breaks / contaminated spikes / old
tuple clusters with no severity) separately from a SOFT mask (above-tolerance
flicker / zigzag / uncontaminated spikes). `join_gap` bridges a gap only if it
is short AND contains zero HARD frames — so a genuine glitch is never swallowed
no matter how brief. rub100/0022 now stays as two honest segments
`[107–181]` + `[218–372]` at every join_gap. Node `join_gap` default left at
0.0 (purity-first; raising it trades purity for coverage, the user's call).

**Remaining follow-ups:**
- DONE: `keep contaminated spikes` exposed as a NoiseReviewNode checkbox
  (default On = contaminated regions are hard boundaries; Off = absorbable).

### dev×speed lens built + disagreement analysis (keep diagnostic, do NOT lock)

Built `_detect_dev_speed` (signature-agnostic gross-corruption lens): per-joint
geodesic deviation from a quadratic-predicted smooth arc × local speed, full
body; serialized as `dev_speed_max` / `dev_speed_rate` / `dev_speed_clusters`
(DIAGNOSTIC — not wired into classification or clean-segment derivation).
`DEVSPEED_HARD=20`. Re-ran corpus (06-14). `dev_speed_max` by class:
clean p50 0.17 / p99 4.2; moderate p50 0.81 / p99 9.8; problematic p50 11.0
/ p99 402; DanceDB p50 148. It tracks the verdict strongly.

Disagreement analysis (gate τ=20, where clean is essentially all below):
- **99% of high-dev×speed files (542/548) are ALREADY problematic** — it
  corroborates, it doesn't find new corruption.
- **Set A (high dev×speed but NOT problematic) = only 6 files** (5 moderate,
  1 clean), all low-rate (0.03–0.33/s) isolated events on FAST LOCOMOTION
  (CMU 111_21 dev 53.6; run-backwards / shuffle-forward / run). Almost
  certainly dev×speed's own fast-motion borderline, not dramatic missed
  corruption. So the existing lenses already have full recall on gross
  trajectory corruption.
- **Set B (problematic but dev×speed < 5) = 547 files** — but these are NOT
  over-flags. They're flagged by ORTHOGONAL signatures dev×speed structurally
  can't see: flicker (337; low-amplitude jitter by design, FLICKER_VCAP=25),
  teleport ceiling (62; >60 rad/s, a different signature), reversal (3). Low
  dev×speed here means "not gross trajectory deviation", not "clean".
- dev×speed agrees with only 33% of problematic files; the other 67% are the
  orthogonal-signature set above. So the lens is **complementary, not
  redundant** — but it adds essentially no NEW detection.

**Verdict:** keep dev×speed as a DIAGNOSTIC (serialized) — its value is a
CONTINUOUS corruption-severity ranking (0→1000+) for prioritising review and
the AMASS-filter, plus independent corroboration of the gross-corruption
subset. Do NOT promote it to a classification lock: 99% overlap = no recall
gain, and its 6 unique hits are fast-motion FPs that a lock would wrongly flag.
DONE: surfaced as a NoiseReviewNode `sort by` key ('dev×speed' ranks
worst-corruption-first; also 'noise score' / 'classification' / 'file order'),
and `dev×speed N.N` is shown on each file's console line.
- `NoiseReviewNode` resolves stored absolute `.npz` paths with an `amass root`
  fallback (re-root by dataset tail / AMASS anchor / basename+n_frames) and a
  path FILTER (all whitespace terms must substring-match the path); both added
  this session.

### Excursion discriminator — glitch vs legitimate ballistic motion (BREAKTHROUGH, prototype validated)

The first signal all session that actually separates a glitch from legitimate
fast motion. From the user's morphology of BMLNTroje glitches: a GLITCH is a
local EXCURSION from a trajectory that is self-consistent before AND after the
event (remove the bad frame(s) and the motion continues unchanged — reversals,
kinks-with-continuation, pops in minimal movement). A real maneuver (throw
release) is the BOUNDARY between two regimes (fast forward → follow-through),
so the motion after is genuinely NOT a continuation of the motion before.

Formalised with two per-joint geodesic quantities at a candidate pose t:
- `off_dev` = residual of R[t] from the geodesic midpoint of R[t-1],R[t+1]
  (×fps) — how far OFF the smooth path the pose is (direction-aware; catches
  reversals the magnitude neighbour-ratio misses, drops the f58-style settling
  tails and smooth fast motion).
- `regime_change` = |mean angular-velocity vector AFTER the event − mean BEFORE|
  (skipping the event frames) — did the motion regime actually shift?
- A glitch = HIGH off_dev with LOW regime_change (big excursion, motion
  recovers). excursion_ratio = off_dev / (regime_change + 2).

Prototype validation (per-file glitch-frame counts, off_dev≥8 & ratio≥4):
clean punch E15 = 0, clean front kick = 0 (NO firing on real ballistic motion —
the wall every prior metric hit); walk rub036 catches f56 (f13/f114 just under
at ratio 3.3 → drop ratio to ≥3); DanceDB = 999 (rate 10.9/s); jump
rub056/0029_jumping2 (264f) flags left_wrist f75/104/109/114 — **user confirmed
ALL true reversal glitches**, in a file the current lenses classed "clean".
Cross-check: of 67 same-named jump files, only rub056 shows the f104 off_dev
spike (others 0.1–1.8) — so it is a real arm glitch, NOT a universal
jump-landing artifact (which would appear across files at the landing frame).

Status: discriminator validated on a handful of files; NOT yet built into the
pipeline. To do: tune ratio→3, recast `_detect_spike_frames` (or add a new
lens) to emit excursion events keyed by full path (navigable in
NoiseReviewNode), re-run + corpus-validate (clean-file rate, DanceDB recall,
classification stability). Open question to watch: does it fire on genuine
high-impact ground contact (landings/foot-strikes on LEG joints)? — needs an
impact/ground-contact guard if so (the jump hits were arm, not leg, so clear).

### FUTURE: repairable glitches (user idea, 2026-06-14)

The excursion definition IS a repairability test. Because an excursion is a
deviation from a trajectory that is self-consistent before and after, the
glitch frame(s) can be REPLACED by the geodesic interpolation of their
neighbours WITHOUT destroying real motion — the real motion is, by definition,
recoverable across the event. (A regime-change event is NOT repairable: you'd
be deleting genuine motion.) So `off_dev` high + `regime_change` low flags
exactly the glitches that are safe to repair in place.

Implication for the AMASS-filtering goal: files with isolated repairable
excursions (e.g. rub056's 4 left-wrist reversals in an otherwise-clean jump)
need not be TRASHED — repair the excursion frames (LERP/SLERP the joint
rotations across the event) and keep the file. This would recover far more
usable training data than excise-and-segment, and goes beyond the
clean-segment recovery work (repair-in-place vs route-around). Not built;
captured for future thinking.

### Excursion lens BUILT + redundancy analysis (complementary, not consolidating)

Built `_detect_excursion` into the pipeline (off_dev = geodesic residual from
the quadratic-predicted smooth arc; regime_change = |mean ang-vel after − before|
skipping the event; flag = high off_dev AND low regime_change, ratio ≥ 3).
Serialized `excursion_max/rate/clusters`. DIAGNOSTIC (not in classification).
Tuned ratio 4→3 to catch walk f13/f114 (not just f56).

Checked whether it makes existing lenses redundant — **confirm-before-retiring
caught a mistake.** Surface overlap looked like redundancy (reversal 88% frame
recall, dev×speed 90%), but focused confirmation showed otherwise:
- Reversal: of the frames excursion misses, **93% are high-off_dev
  REGIME-CHANGE events** — excursion's regime gate fails in CHAOTIC corruption
  (DanceDB), where everything is regime-change. So excursion does NOT replace
  reversal.
- Significant spikes: excursion covers only **18%** — doesn't subsume them.
So excursion is **complementary, not consolidating** — its niche is isolated
recovering glitches in otherwise-clean content (found real left-wrist reversals
in rub056/0029_jumping2, a file the suite called "clean"; user confirmed all
true). Nothing retired.

### Perceptibility is LOCALIZED + DIRECTIONAL, not a global scalar (the big lesson)

Long investigation into "which flagged spikes actually matter" for extracting
motion-valid sequences (the real product goal — excellent clean-section
excision matters as much as classification). Conclusions, each forced by user
correction:

1. **Perceptibility ≠ velocity.** First proxy (single-joint, vel<12 = "sub-
   perceptual") was leverage-blind and **wrong 38% of the time** vs the mesh
   measure (a low-velocity shoulder spike IS perceptible; the lever matters).
2. **Perceptibility ≠ a global scalar.** Exact SMPLH mesh-vertex displacement
   (LBS, actual vs quadratic-predicted clean pose, cm) calibrated cleanly on 11
   obvious frames (~0.55 cm line) — but on a diverse 16-frame adjudication
   batch it FAILED: perceptible frames at 0.35 cm interleaved with imperceptible
   at 0.47 cm. Root causes (user diagnosed): (a) **global MAX over all 6890
   vertices = background noise** — a frame flagged on l_hip scored on an *arm*
   vertex elsewhere; (b) **the scalar discards direction** — any movement makes
   displacement; the glitch signature is the displacement DIRECTION wobbling.
3. **Localizing fixes it.** Scoping the mesh measure to the flagged joint's
   kinematic SUBTREE collapsed the imperceptibles (l_shoulder 0.45→0.26,
   l_hip 0.47→0.29) below the perceptibles (≥0.33) — a clean ~0.30 cm line
   reappeared. So perceptibility must be measured PER-LIMB, not globally.
4. **Direction-wobble alone also fails** (real motion changes direction too —
   a salute reverses). The working discriminator is direction-deviation from the
   SMOOTH path + transience — which excursion already encodes; but excursion is
   tuned for bigger events and fires on none of the small ones.
5. **The discriminator works once LOCALIZED + ATTRIBUTED (verdict updated).**
   First pass (global mesh, flagged joint/frame) failed to separate the 16-frame
   batch — but that was the global scalar + wrong attribution. After building
   attribution, re-ran it: attribute each frame → (limb, source joint, corrected
   frame), then measure localized subtree-max mesh displacement THERE. Result:
   imperceptibles (f313 0.27, f1608 0.26) drop cleanly below perceptibles
   (0.40–1.78); a **~0.30 cm localized line separates 15/16** (only miss: f17,
   "just perceptible", 0.27). (Note f606 was a mislabel on my part — the user
   said 603–604 IS perceptible; attribution moved it to f604, measure reads 0.69
   = perceptible, correct.) So a perceptibility filter IS viable — but ONLY as a
   per-limb, attributed, positional measure, NEVER the global scalar. Caveats:
   16 frames, thin 0.27→0.40 margin, f17 a genuine miss → firm with more
   boundary frames before trusting the exact threshold.
7. **FIRMING FAILED — the threshold does NOT generalize (final verdict).** A
   second, region-diverse boundary batch (14 frames, leg/spine-heavy: handball,
   beam, locomotion) overlapped almost completely: imperceptible localized
   0.23–0.45, perceptible 0.23–0.44; the HIGHEST value (knee, 0.45) was
   imperceptible; spine 0.33 (P) vs 0.34 (I) tied with opposite calls. Root
   cause: fast CURVED leg/spine motion leaves a localized residual the quadratic
   prediction can't remove, so real motion mimics glitch displacement — a
   region/activity-dependent real-motion floor that no fixed (or per-region)
   threshold separates. The earlier 15/16 was an ARM-BIASED sample. And the user
   "seems like real movement" call (f552) is the crux: displacement MAGNITUDE
   cannot separate a perceptible glitch from perceptible real motion. **FINAL:
   do NOT wire a fixed-threshold perceptibility filter** — keep the localized
   mesh measure only as a soft severity/ranking signal; lens DETECTION is the
   relevance. (Also surfaced: attribution's ±3 window missed f665's event at
   660–661, ~5 frames off — widen the window.)
6. **Orientation axis validated then demoted.** Tested an axial/orientation
   axis for the mesh blind spot (axial bone twist → low vertex displacement).
   Empirically the axial blind spot is RARE (3/806 frames; corr(mesh,orient)
   = 0.86) — real twists still swing offset descendants. The WORSE gap is the
   other way: orient (local rotation) is leverage-blind (29% mesh-perceptible
   but orient-low). So **mesh (leverage-aware) is the better single axis**;
   orient is at most a cheap backstop. Sustained ROM is invisible to BOTH
   vs-neighbour measures → stays on its absolute anatomical cap (ungated).

Net: perceptibility/validity is NOT a single global number. Mesh displacement
is meaningful only LOCALIZED to the flagged limb and is one (positional) axis;
direction/transience is the other (= what the lenses already compute). For the
product, keep lens flags as fragmenting (most are real), trim only the f7-class
tiny tail. (`mesh ≠ validity`: an axial joint-angle error can be invalid
training data even when mesh-imperceptible — the lenses, in rotation space,
catch those.)

### Attribution pass — region + corrected frame (BUILT)

The adjudication exposed two attribution bugs independent of perceptibility:
a **±2–3 frame wobble** between the flag and the perceived event (f606 flagged
606, real 603–604), and **joint mis-attribution** (f577 "shoulder not elbow";
f432 flagged head, event on the arm). New `attribution.py`: re-attributes a
flag to the (region, source joint, frame) maximising `rotation_deviation ×
bone_length` (the joint whose glitch swings the most limb), reported as LIMB
REGION (sidesteps the ill-posed elbow-vs-shoulder call) with the peak joint as
sub-detail. Pure rotation math, no model/LBS dependency. Exact-joint
attribution is genuinely ambiguous (the user's own f577 call was tentative);
region + corrected frame is what navigation, localization, and clean-segment
boundaries actually need.

Wired as a pipeline pass over all lens clusters (LensCluster gains
`region`/`peak_frame`/`peak_joint`); re-ran corpus (06-14, 14,182 files, 0
fail). 100% of excursion clusters attributed; **78% are arm** (right 24,300 /
left 23,906 — confirms arm-marker glitches dominate); **attribution moved the
frame in 35% of clusters** (mean 0.92, p90 3) — the wobble was real and is now
corrected at source.

**Open follow-ups (end of 2026-06-14):**
- DONE: NoiseReviewNode shows the excursion lens (was missing), displays each
  lens flag's `region/peak_joint @peak_frame`, and navigates Prev/Next to the
  corrected `peak_frame` (not the cluster start).
- Re-validate the localized ~0.30 cm perceptibility line now that flags are
  attributed (f432/f606 confounds should resolve) — or confirm it's only good
  for trimming the < ~0.25 cm tail.
- DONE: attribution extended to `spike_clusters`; NoiseReviewNode now shows
  attributed spike CLUSTERS (not noisy per-frame flags) navigating to the
  corrected peak frame — e.g. the rub036 cluster [7-58] (spanning the trivial
  f7/f58) now lands on f56, the substantive reversal.
- Repairable-excursion repair-in-place (see FUTURE section) — highest-leverage
  for dataset yield.

### Off-trajectory contextual residual — the perceptibility dial (RESOLVED + built)

The perceptibility saga resolved, reversing the earlier "no threshold" verdict
— that verdict was a methodological artifact (the user caught it): both
adjudication batches had been SELECTED on the metric's own margin (borderline
mesh / localized 0.22–0.45), so testing separation there was circular. Across
the FULL range the metric works.

The final metric (`perceptibility.py`): **off-trajectory contextual residual** —
per-vertex deviation of the actual mesh from a QUADRATIC fit of each vertex's
position trajectory (the quadratic absorbs smooth acceleration/curvature, so
the **direction fit is baked into the prediction** — a glitch deviates off the
smooth path; smooth fast motion is absorbed), localized to the attributed
joint's subtree, p90, **absolute cm**. Two corrections the user forced:
- Direction is NOT discarded — it's the decomposition that fixes the f1732
  case (smooth knee: residual 0.45 cm but it's ALONG the path / real
  acceleration; magnitude alone wrongly flagged it). The quadratic-position
  prediction encodes direction.
- Activity-normalisation HURTS (over-corrects low-activity smooth motion) — use
  ABSOLUTE residual; the contextual fit is already in the quadratic.

Full-range validation: gross corruption 15.8 cm, clear glitches (f56, jump)
2.0–2.5, smooth-fast / clean / imperceptible 0.0–0.7. **~1 cm separates them**;
0.7–1.5 is the acknowledged fuzzy margin (where the user's own perception is
angle-dependent, so no metric should be crisp).

Built + wired: serialized as `off_traj_cm` per spike + excursion cluster (hard
lenses ungated). `clean_segment_tools` gains `off_traj_cm_tol` (default 1.0 cm)
as the PRIMARY spike/excursion relevance dial — fragment iff residual ≥ dial,
else absorbed; falls back to contaminated/severity when off_traj_cm is -1.
`keep_contaminated_spikes` is now an OVERRIDE on the off-traj dial: On (default)
= a contaminated (dense-jitter) cluster fragments even if its off-traj residual
is below the dial (dense jitter = motion-invalid regardless); Off = trust
off-traj alone, recovering contaminated-but-imperceptible regions (e.g. walk
rub036 [232-252], off_traj 0.4 cm: On → clean stops at 229 / 0.29; Off → clean
extends through to 285 / 0.43).  NoiseReviewNode exposes the dial and shows
`off=N.NNcm` per event. Verified:
walk rub036 goes 0 → 43% → 95% clean as the dial moves 0 → 1 → 2, with the
real glitches fragmenting and the imperceptible (incl. a contaminated-but-tiny
0.13 cm cluster) absorbed.

**Performance (the dial needs LBS — stalled, then fixed).** Computing mesh for
EVERY cluster stalled the corpus run: full-body (6890-vertex) LBS at 26 s on
cluster-heavy DanceDB files (600 excursions/file), plus a posedir precache that
blew worker memory. Fixed three ways: (1) skin only the attributed subtree's
vertices; (2) slice posedirs on the fly (no precache); (3) **short-circuit the
mesh** (the user's idea — "if a spike is large enough we needn't dig deeper").

**Short-circuit v1 (retired) — bone-length score.** First cut gated on the
cheap attribution score (rotation_dev × *bone length*): score<0.5 → skip as
imperceptible, score≥2.0 → skip as glitch, ambiguous middle → mesh. corr(score,
off_traj)=0.68; DanceDB worst 26 s → 4.2 s. **Bug found in review (rub011 f419):**
a clear elbow glitch (mesh 1.04 cm) scored 0.34 and was wrongly absorbed as 0.4.
Bone length (elbow→forearm, 0.25 m) misses the forearm+hand lever the mesh sees,
so distal-subtree joints (elbow/shoulder/hip) systematically UNDER-read → the
low band false-negatives perceptible glitches. The flat 0.4/2.5 sentinels were
also misleading in the node.

**Short-circuit v2 (current) — leverage-aware geometric proxy.**
`perceptibility.proxy_cm` = max over body joints of (geodesic-dev from the
quadratic prediction) × (VERTEX subtree reach, incl. hand/foot mesh) × 100. This
is a strict UPPER BOUND on the mesh off-traj residual (skinning blend + child
compensation only reduce it — validated proxy ≥ mesh on all 7 test cases:
419 1.05/1.04, f56 3.64/1.97, f1732 0.50/0.44, f215-compensation 2.03/0.77, f7
1.05/0.68, gross 29.6/15.8, jump 3.84/2.50). So **proxy < 0.5 cm ⟹ mesh < 0.5
guaranteed → absorb with no LBS and ZERO risk of missing a glitch** (`SC_OFFTRAJ_
ABSORB=0.5`, safe for any dial ≥0.5). proxy ≥ 0.5 → exact mesh — including the
compensation cases (high proxy, low mesh: f215 2.03→0.77) that a *high*
short-circuit would mis-fragment, so there is NO high short-circuit. f419 now
correctly reads 1.038 cm. Stored off_traj is the real proxy/mesh value (no
sentinels). Cost: DanceDB worst 7.5 s (600/603 mesh — corrupt file, most are
real); clean CMU 0.7 s (9/13 absorbed by proxy). Corpus est. ~12–15 min.

Corpus re-run v1 (bone-length) DONE 2026-06-15, 14,182 files, 9m29s: 43% low /
13% high / 43% mesh; 39% fragment at the 1 cm dial. **v2 (leverage-aware proxy)
DONE 2026-06-15, 14,182 files, 0 fail, 11m22s → noise_results_lenses_2026_06_15/**
(~2 min slower than v1: no high short-circuit + a provably-safe absorb band skip
only 37% of LBS vs v1's 56%). 173,903 clusters: 37% proxy-absorbed (<0.5 cm, no
LBS) / 63% exact mesh. At the 1 cm dial: **42% fragment / 58% absorb** — the
extra ~3% over v1 (~5,200 clusters) are the recovered false negatives, distal-
limb glitches v1 wrongly absorbed at the 0.4 sentinel (e.g. rub011 f419: v1 0.40
→ v2 1.038 cm). This run is what the node reads; clean-segment derivation is now
correct. **Clean-recovery payoff:** flagged (moderate/problematic)
files recover mean clean_frac 0.78 at tol=1.0 (vs 0.28 at tol=0), and 84% yield
a usable ≥40% clean section — the perceptibility dial recovers most clean motion
from flagged files instead of discarding them. The off-trajectory dial is the
working resolution of the whole magnitude-vs-direction / perceptibility thread.

## References

- Script: `dpg_system/noise_estimation/estimate_noise_torque.py`
- Stale prior runs: `dpg_system/noise_estimation/torque_results_by_folder/`,
  `torque_checkpoint.json`, `result_*.json`
- Related memory entries: `smpl_contact_test_data`, `project_axis_permutation_design`