# Shadow IMU Magnetometer Assessment — 2026-06-18 (verification re-sweep)

Full re-sweep of all 3 suits × 17 physical IMU sensors (51 total), using the
`shadow_sensor` + `mag_offset` nodes in `dpg_system`. Each sensor is rotated
through many orientations; a least-squares sphere is fit to the calibrated
(`<sensor>`) magnetometer cloud.

**Purpose:** verify that the 2026-06-17 recalibration campaign is holding (the 14
recalibrated sensors should still be recentered) and catch any fresh drift on the
sensors that were already clean. Success per sensor = offset ≲3% of field, radius
~53–55 µT, residual ≲1.5%.

**Status: COMPLETE** — 51/51 sensors swept; the one new offset (suit-2 left hip)
recalibrated same day. **All 51 clean across all 3 suits.**

## Executive summary

- **The recalibration is holding.** All 14 sensors recalibrated on 06-17 remain
  recentered — every one ≤1.73% of field (most <1%), vs the 7–33% offsets they
  carried before. Includes both reproduced-across-controllers outliers (suit-3
  right ankle 17.9%→0.40%, left shoulderblade 16.2%→1.25%) and the worst sensor
  in the whole assessment (suit-2 head 33.1%→0.63%).
- **Finding 1 (all-3-suits head contamination) stays resolved** — all three head
  sensors clean (suit 1 0.35%, suit 2 0.63%, suit 3 0.90%).
- **Finding 4 (elevated-residual arm sensors) has cleared** — all four (suit-2
  right/left shoulder, suit-2 left elbow, suit-1 left ankle) now read clean
  residuals ≤0.66%, where they were ~2% before.
- **Scale healthy everywhere** — radii 51.9–54.2 µT, consistent with the Toronto
  geomagnetic field. No low-radius/coverage artifacts this run.
- **One new offset, now fixed: suit-2 left hip drifted to 4.46% (Y-dominated,
  clean sphere) from its 2.09% baseline** — a real +2.4-pt rise, well outside the
  session noise floor (the worst worsening on any unchanged sensor was +0.9 pt).
  Recalibrated same day → **0.51%** (center [0.24, −0.12, 0.03], r52.82, residual
  1.80% — mildly high from lighter sweep coverage; offset clean), then re-swept
  with full coverage → **1.04% / residual 0.49%** (radius recovered 52.82→53.42,
  confirming the elevated residual was coverage not soft-iron). Confirms it was
  stable hard-iron, same mechanism as the 06-17 campaign. Individual to that
  sensor (suit-3 left hip pristine at 0.21%).
- **Session-to-session stability (37 non-recalibrated sensors):** offset %r
  reproduces to mean |Δ| 0.57 pt / median 0.44 pt / σ 0.73 pt; radius stable to
  ±1–2 µT. All large swings except the left hip were toward *cleaner* (fuller
  sweeps); max worsening on any unchanged sensor was +0.9 pt.

## Baseline column

`Baseline %r` is the value to beat for each sensor:
- **(recal)** — the 06-17 post-recalibration reading. These were freshly
  recentered yesterday; watch for any regression.
- **(06-01)** / **(06-04)** — last clean reading for sensors never recalibrated
  (suits 1 & 2 from the 06-01 assessment; suit 3 from the 06-04 recheck).

## How to read these numbers

- **center / offset** — fitted sphere center = hard-iron offset vector (µT).
- **offset % of radius** — severity metric for hard-iron. Clean ≈ 0.3–2.7%.
- **radius** — local geomagnetic field magnitude (µT); should match Toronto
  (~54.5 µT). Low radius usually = incomplete sweep coverage, not scale error.
- **residual** — RMS sphere-fit deviation; elevated with a small offset =
  soft-iron or (more often) incomplete orientation coverage.

## Results

### Suit 1
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag | Baseline %r | Δ |
|---|---|---|---|---|---|---|---|---|
| Chest (MidVertebrae) | 1.63 | 3.02% | 53.91 | 0.283 | 0.52% | clean (high) | 2.71% (06-01) | +0.31 |
| Base of skull (Head) | 0.19 | 0.35% | 52.67 | 0.413 | 0.78% | clean | 0.15% (recal) | +0.20 ✓ |
| Pelvis anchor | 0.18 | 0.34% | 53.09 | 0.287 | 0.54% | clean | 1.23% (06-01) | −0.89 |
| Right hip | 0.60 | 1.11% | 53.87 | 0.305 | 0.57% | clean | 1.43% (06-01) | −0.32 |
| Right knee | 0.08 | 0.16% | 53.12 | 0.228 | 0.43% | clean | 0.18% (recal) | −0.02 ✓ |
| Right ankle | 0.37 | 0.68% | 53.84 | 0.270 | 0.50% | clean | 0.99% (06-01) | −0.31 |
| Left hip | 0.88 | 1.63% | 53.84 | 0.415 | 0.77% | clean | 2.42% (06-01) | −0.79 |
| Left ankle | 0.51 | 0.95% | 53.74 | 0.356 | 0.66% | clean | 0.72% (06-01) | +0.23 ✓ resid resolved |
| Left knee | 0.68 | 1.25% | 54.13 | 0.544 | 1.00% | clean | 1.48% (06-01) | −0.23 |
| Left shoulderblade base | 0.16 | 0.29% | 53.22 | 0.160 | 0.30% | clean | 0.19% (recal) | +0.10 ✓ |
| Left shoulder | 0.19 | 0.36% | 53.40 | 0.341 | 0.64% | clean | 0.19% (recal) | +0.17 ✓ |
| Left elbow | 0.12 | 0.23% | 53.38 | 0.276 | 0.52% | clean | 0.06% (recal) | +0.17 ✓ |
| Left wrist | 0.11 | 0.20% | 53.01 | 0.308 | 0.58% | clean | 2.45% (06-01) | −2.25 ↓ |
| Right shoulderblade base | 0.95 | 1.75% | 53.99 | 0.181 | 0.34% | clean | 1.77% (06-01) | −0.02 |
| Right shoulder | 0.77 | 1.42% | 54.12 | 0.390 | 0.72% | clean | 1.50% (06-01) | −0.08 |
| Right elbow | 0.68 | 1.25% | 54.11 | 0.331 | 0.61% | clean | 0.93% (06-01) | +0.32 |
| Right wrist | 0.19 | 0.36% | 52.56 | 0.293 | 0.56% | clean | 0.40% (recal) | −0.04 ✓ |

### Suit 2
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag | Baseline %r | Δ |
|---|---|---|---|---|---|---|---|---|
| Right shoulderblade base | 1.10 | 2.12% | 52.02 | 0.281 | 0.54% | clean | 1.55% (06-01) | +0.57 |
| Right hip | 0.83 | 1.54% | 53.95 | 0.482 | 0.89% | clean | 1.47% (06-01) | +0.07 |
| Right knee | 0.79 | 1.47% | 54.12 | 0.468 | 0.86% | clean | 0.57% (06-01) | +0.90 |
| Right ankle | 1.54 | 2.87% | 53.86 | 0.325 | 0.60% | clean | 2.48% (06-01) | +0.39 |
| Right shoulder | 0.22 | 0.42% | 52.67 | 0.262 | 0.50% | clean | 0.98% (06-01) | −0.56 ✓ resid resolved |
| Right elbow | 0.83 | 1.58% | 52.64 | 0.309 | 0.59% | clean | 1.28% (06-01) | +0.30 |
| Right wrist | 0.16 | 0.30% | 52.28 | 0.280 | 0.54% | clean | 2.05% (06-01) | −1.75 |
| Left shoulderblade base | 0.30 | 0.56% | 53.22 | 0.333 | 0.63% | clean | 0.33% (06-01) | +0.23 |
| Left shoulder | 0.94 | 1.73% | 54.28 | 0.319 | 0.59% | clean | 2.67% (06-01) | −0.94 ✓ resid resolved |
| Left elbow | 1.53 | 2.83% | 54.29 | 0.329 | 0.61% | clean | 2.06% (06-01) | +0.77 ✓ resid resolved |
| Left wrist | 0.25 | 0.48% | 53.11 | 0.325 | 0.61% | clean | 0.21% (recal) | +0.27 ✓ |
| Pelvis anchor | 0.91 | 1.73% | 52.73 | 0.246 | 0.47% | clean | 0.09% (recal) | +1.64 ✓ |
| Base of skull (Head) | 0.33 | 0.63% | 51.87 | 0.312 | 0.60% | clean | 0.43% (recal) | +0.20 ✓ |
| Chest (MidVertebrae) | 1.47 | 2.79% | 52.66 | 0.237 | 0.45% | clean | 2.37% (06-01) | +0.42 |
| Left hip | 2.37 | 4.46% | 53.14 | 0.375 | 0.71% | recal'd → 1.04% (resid 0.49%) ✓ | 2.09% (06-01) | **+2.37 ↑ then fixed** |
| Left knee | 0.56 | 1.06% | 53.13 | 0.388 | 0.73% | clean | 1.36% (06-01) | −0.30 |
| Left ankle | 0.49 | 0.93% | 53.11 | 0.475 | 0.89% | clean | 0.96% (06-01) | −0.03 |

### Suit 3
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag | Baseline %r | Δ |
|---|---|---|---|---|---|---|---|---|
| Pelvis anchor | 0.82 | 1.52% | 54.15 | 0.244 | 0.45% | clean | 1.30% (06-04) | +0.22 |
| Left hip | 0.11 | 0.21% | 53.32 | 0.319 | 0.60% | clean | 0.36% (recal) | −0.15 ✓ |
| Left knee | 0.59 | 1.09% | 54.16 | 0.353 | 0.65% | clean | 1.56% (06-04) | −0.47 ✓ resid resolved |
| Left ankle | 0.12 | 0.23% | 53.79 | 0.269 | 0.50% | clean | 1.99% (06-04) | −1.76 |
| Right hip | 0.62 | 1.17% | 53.32 | 0.164 | 0.31% | clean | 2.11% (06-04) | −0.94 |
| Right knee | 0.99 | 1.86% | 53.35 | 0.285 | 0.53% | clean | 1.33% (06-04) | +0.53 |
| Right ankle | 0.21 | 0.40% | 52.87 | 0.249 | 0.47% | clean | 0.33% (recal) | +0.07 ✓ |
| Right shoulder | 0.28 | 0.53% | 52.62 | 0.494 | 0.94% | clean | 1.05% (06-04) | −0.52 |
| Right elbow | 0.71 | 1.35% | 52.37 | 0.505 | 0.96% | clean | 0.73% (06-04) | +0.62 |
| Right wrist | 0.34 | 0.64% | 52.43 | 0.540 | 1.03% | clean | 0.36% (06-04) | +0.28 |
| Left shoulder | 0.27 | 0.51% | 53.79 | 0.229 | 0.43% | clean | 1.05% (06-04) | −0.54 |
| Left elbow | 0.49 | 0.92% | 53.67 | 0.286 | 0.53% | clean | 1.11% (06-04) | −0.19 |
| Left wrist | 0.20 | 0.38% | 53.67 | 0.203 | 0.38% | clean | 1.32% (06-04) | −0.94 |
| Left shoulderblade base | 0.66 | 1.25% | 52.71 | 0.327 | 0.62% | clean | 0.14% (recal) | +1.11 ✓ |
| Right shoulderblade base | 0.94 | 1.75% | 53.81 | 0.369 | 0.69% | clean | 1.19% (06-04) | +0.56 |
| Chest (MidVertebrae) | 0.26 | 0.49% | 52.48 | 0.342 | 0.65% | clean | 0.40% (recal) | +0.09 ✓ |
| Base of skull (Head) | 0.48 | 0.90% | 53.45 | 0.319 | 0.60% | clean | 0.75% (recal) | +0.15 ✓ |

## Running log

- 2026-06-18 — Verification re-sweep started; full 51-sensor scope. Baselines:
  14 recalibrated sensors from 06-17 post-recal; remaining 37 from their last
  clean reading (06-01 suits 1&2, 06-04 suit 3).
- 2026-06-18 — **Suit 1 COMPLETE (17/17 clean).** All offsets ≤3.02% of field;
  highest is chest 3.02% (clean-high, consistent with its 2.71% history, never
  recalibrated). All 6 recalibrated suit-1 sensors (head, right wrist, right knee,
  left shoulder, left shoulderblade, left elbow) are holding their recentering
  (≤0.36%). Two prior soft flags resolved: suit-1 left ankle residual 2.44%→0.66%.
  No regressions.
- 2026-06-18 — **Suit 2 COMPLETE (16/17 clean, 1 watch).** All 3 recalibrated
  suit-2 sensors holding: head 0.63% (was the 33.11% worst-overall), left wrist
  0.48% (was 20.18% outlier), pelvis 1.73% (was 6.48%). All three suit-2
  Finding-4 elevated-residual arm sensors (right/left shoulder, left elbow) now
  read clean residuals (≤0.61%). **One new watch flag: left hip 4.46% (Y-dominated,
  clean sphere), up from 2.09% baseline** — fresh hard-iron developing; not severe;
  re-sweep to confirm reproducibility.
- 2026-06-18 — **Suit 3 COMPLETE (17/17 clean).** All 6 recalibrated suit-3
  sensors holding: right ankle 0.40% (was 17.9% #4-outlier), left shoulderblade
  1.25% (was 16.2% #5-outlier), head 0.90% (was 10.1%), left hip 0.21%, chest
  0.49%, all ≤1.25%. The 06-04 left-knee residual re-sweep flag resolved to 0.65%.
- 2026-06-18 — **RUN COMPLETE: 51/51 swept, 50 clean, 1 watch.** Recalibration
  campaign confirmed holding across all 3 suits one day on; all 14 recalibrated
  sensors ≤1.73%. Only new development is suit-2 left hip drift (2.09%→4.46%, Y,
  clean sphere) — the single watch item.
- 2026-06-18 — **Suit-2 left hip RECALIBRATED → 0.51%** (was 4.46% on this run's
  sweep), center [0.24, −0.12, 0.03], r52.82, residual 1.80% (mildly high =
  lighter sweep coverage; offset clean). Watch item resolved → **all 51 sensors
  now clean across all 3 suits.** Stable hard-iron captured by a calibration
  refresh, consistent with Finding 2b.
- 2026-06-18 — **Suit-2 left hip re-swept (full coverage) → 1.04%, residual
  0.49%**, center [0.39, −0.39, −0.07], r53.42. Residual dropped 1.80%→0.49% and
  radius recovered 52.82→53.42, confirming the elevated residual was sweep
  coverage, not soft-iron. Left hip fully settled: 4.46% → 1.04%, clean sphere,
  clean residual.
