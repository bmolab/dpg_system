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

_(no files reviewed yet)_

## Threshold / parameter tuning log

When we change a constant in `estimate_noise_torque.py`, record:
- Date
- Constant name and old → new value
- Files / observations that motivated the change
- What got better, what (if anything) got worse

_(no changes since validation began)_

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

## References

- Script: `dpg_system/noise_estimation/estimate_noise_torque.py`
- Stale prior runs: `dpg_system/noise_estimation/torque_results_by_folder/`,
  `torque_checkpoint.json`, `result_*.json`
- Related memory entries: `smpl_contact_test_data`, `project_axis_permutation_design`