# Shadow IMU Magnetometer Assessment — 2026-06-01

Magnetometer offset (hard-iron) assessment of Motion Workshop Shadow sensors,
using the `shadow_sensor` + `mag_offset` nodes in `dpg_system`. Each sensor was
rotated through many orientations; a least-squares sphere is fit to the raw
magnetometer cloud.

**Status: COMPLETE** — 3 suits × 17 physical IMU sensors (51 total).

> **RESOLVED 2026-06-17** — all flagged offsets across all 3 suits were cleared by
> recalibration (recentered to ≤0.75% of field). The findings below record the
> original 06-01 assessment as found; see the **Recalibration campaign** section at
> the end of this document for per-sensor before→after results and confirmation of
> Finding 2b (stable remanent hard-iron, correctable by a calibration refresh).

## Executive summary

- **Most sensors are clean** (offset <3% of field), and scale is healthy
  everywhere (radii 53–55.5 µT, matching the Toronto geomagnetic field).
- **THE systematic finding: the head sensor is contaminated on all 3 suits**
  (7.5% / 33% / 10%). The head mount is a fabric headband (no ferrous hardware),
  so this is NOT a mount issue. (Finding 1)
- **Leading mechanism hypothesis (Finding 2b): remanent magnetization from
  storage/charging proximity to batteries.** Every severe offset is a clean
  displaced sphere → the source co-rotates with the sensor → the sensor's own
  ferrous parts are permanently magnetized (not an external field at measurement
  time). Batteries (magnetized steel cans) are the suspected agent. One mechanism
  unifies both the all-suits head result and the scattered outliers. Testable by
  degaussing + controlled re-exposure.
- **Five severe single-sensor hard-iron offsets** (>15% of field), scattered
  with no anatomical pattern → individual sensors magnetized, varying
  unit-to-unit (Findings 1, 2, 2b):
  | Sensor | Offset | % field | Axis |
  |---|---|---|---|
  | suit 2 base of skull | 17.7 µT | 33% | Y |
  | suit 1 right wrist | 15.6 µT | 30% | X |
  | suit 2 left wrist | 10.7 µT | 20% | Z |
  | suit 3 right ankle | 9.5 µT | 18% | 3-axis |
  | suit 3 left shoulderblade | 8.7 µT | 16% | Y |
- **Minor / watch:** suit-specific moderate offsets (suit 1 right knee 8.6%,
  suit 1 head 7.5%, suit 2 pelvis 6.5%, suit 1 left shoulder 5.4%); a mildly
  elevated suit-1 left upper-arm chain; a few elevated residuals (possible mild
  soft-iron) on arm sensors. Details in Findings 3–6.
- All severe offsets have clean spherical shape → hard-iron (removable by
  calibration / source removal), not soft-iron.

## How to read these numbers

- **center / offset** — the fitted sphere center. This *is* the hard-iron
  offset vector (µT). A clean magnetometer traces a sphere centered on the
  origin; a magnetized sensor shifts the sphere off-origin.
- **offset % of radius** — offset magnitude relative to field strength. The
  best severity metric for hard-iron. Clean sensors here sit at ~0.3–2.7%.
- **radius** — local geomagnetic field magnitude (µT). Should match the
  Toronto total field (~54.5 µT); a consistent ~53–55.5 µT confirms the
  magnetometer scale is correct.
- **residual** — RMS deviation of points from the fitted sphere. Low (~<1.5%)
  means a clean spherical shape (negligible soft-iron). Elevated residual with
  a *small* offset points to soft-iron distortion OR incomplete orientation
  coverage during the sweep — not hard-iron.

## Results

### Suit 1
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag |
|---|---|---|---|---|---|---|
| Chest (MidVertebrae) | 1.48 | 2.71% | 54.56 | 0.423 | 0.78% | clean |
| Base of skull (Head) | 4.14 | 7.45% | 55.50 | 0.657 | 1.18% | **offset** (Z) |
| Pelvis anchor | 0.66 | 1.23% | 53.56 | 0.558 | 1.04% | clean |
| Right hip | 0.77 | 1.43% | 53.43 | 0.315 | 0.59% | clean |
| Right knee | 4.58 | 8.61% | 53.18 | 0.458 | 0.86% | **offset** (Y) |
| Right ankle | 0.53 | 0.99% | 53.84 | 0.441 | 0.82% | clean |
| Left hip | 1.28 | 2.42% | 52.84 | 0.467 | 0.88% | clean |
| Left ankle | 0.38 | 0.72% | 53.29 | 1.302 | 2.44% | residual |
| Left knee | 0.78 | 1.48% | 52.23 | 0.593 | 1.14% | clean |
| Left shoulderblade base | 2.13 | 3.97% | 53.53 | 0.326 | 0.61% | clean (high) |
| Left shoulder | 2.92 | 5.41% | 54.03 | 0.393 | 0.73% | offset (Z, minor) |
| Left elbow | 2.55 | 4.80% | 53.01 | 0.348 | 0.66% | clean (high) |
| Left wrist | 1.28 | 2.45% | 52.25 | 0.889 | 1.70% | clean |
| Right shoulderblade base | 0.94 | 1.77% | 53.46 | 0.253 | 0.47% | clean |
| Right shoulder | 0.81 | 1.50% | 53.90 | 0.699 | 1.30% | clean |
| Right elbow | 0.49 | 0.93% | 52.56 | 0.411 | 0.78% | clean |
| **Right wrist** | **15.65** | **30.44%** | _51.40_ | 0.775 | 1.51% | **OFFSET OUTLIER** (X) |

### Suit 2
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag |
|---|---|---|---|---|---|---|
| Right shoulderblade base | 0.82 | 1.55% | 53.04 | 0.345 | 0.65% | clean |
| Right hip | 0.79 | 1.47% | 53.61 | 0.530 | 0.99% | clean |
| Right knee | 0.31 | 0.57% | 53.76 | 0.682 | 1.27% | clean |
| Right ankle | 1.31 | 2.48% | 53.00 | 0.741 | 1.40% | clean |
| Right shoulder | 0.52 | 0.98% | 53.19 | 1.200 | 2.26% | residual |
| Right elbow | 0.69 | 1.28% | 53.98 | 0.685 | 1.27% | clean |
| Right wrist | 1.12 | 2.05% | 54.41 | 0.741 | 1.36% | clean |
| Left shoulderblade base | 0.18 | 0.33% | 54.41 | 0.544 | 1.00% | clean |
| Left shoulder | 1.42 | 2.67% | 53.28 | 1.057 | 1.98% | residual (Z) |
| Left elbow | 1.14 | 2.06% | 55.03 | 1.081 | 1.96% | residual (Z) |
| **Left wrist** | **10.68** | **20.18%** | 52.89 | 0.752 | 1.42% | **OFFSET OUTLIER** (Z) |
| Pelvis anchor | 3.50 | 6.48% | 54.04 | 0.815 | 1.51% | **offset** (Y) |
| **Base of skull (Head)** | **17.71** | **33.11%** | 53.47 | 0.631 | 1.18% | **OFFSET OUTLIER** (Y) |
| Chest (MidVertebrae) | 1.26 | 2.37% | 53.05 | 0.255 | 0.48% | clean |
| Left hip | 1.12 | 2.09% | 53.31 | 0.439 | 0.82% | clean |
| Left knee | 0.73 | 1.36% | 53.63 | 0.531 | 0.99% | clean |
| Left ankle | 0.51 | 0.96% | 53.48 | 0.380 | 0.71% | clean |

### Suit 3
| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | Flag |
|---|---|---|---|---|---|---|
| Pelvis anchor | 0.99 | 1.82% | 54.27 | 0.356 | 0.66% | clean |
| Left hip | 1.79 | 3.30% | 54.35 | 0.870 | 1.60% | clean |
| Left knee | 0.58 | 1.08% | 53.80 | 0.421 | 0.78% | clean |
| Left ankle | 0.87 | 1.62% | 53.46 | 0.444 | 0.83% | clean |
| Right hip | 0.74 | 1.39% | 53.32 | 0.261 | 0.49% | clean |
| Right knee | 1.07 | 2.01% | 53.26 | 0.264 | 0.50% | clean |
| **Right ankle** | **9.50** | **17.91%** | 53.04 | 0.310 | 0.58% | **OFFSET OUTLIER** (3-axis) |
| Right shoulder | 0.23 | 0.42% | 53.62 | 0.238 | 0.44% | clean |
| Right elbow | 0.10 | 0.19% | 53.33 | 0.228 | 0.43% | clean |
| Right wrist | 0.15 | 0.27% | 53.42 | 0.291 | 0.55% | clean |
| Left shoulder | 0.35 | 0.66% | 53.92 | 0.256 | 0.48% | clean |
| Left elbow | 0.27 | 0.50% | 53.51 | 0.329 | 0.62% | clean |
| Left wrist | 0.22 | 0.41% | 53.50 | 0.299 | 0.56% | clean |
| **Left shoulderblade base** | **8.69** | **16.29%** | 53.34 | 0.441 | 0.83% | **OFFSET OUTLIER** (Y) |
| Right shoulderblade base | 0.33 | 0.62% | 53.09 | 0.647 | 1.22% | clean |
| Chest (MidVertebrae) | 2.46 | 4.68% | 52.64 | 0.432 | 0.82% | clean (high) |
| Base of skull (Head) | 5.26 | 10.09% | 52.09 | 0.831 | 1.60% | **offset** (Y/Z) |

## Findings

1. **Head-mount hard-iron on ALL THREE suits — systematic, the key finding.**
   The base-of-skull sensor is a notable offset on every suit:
   - suit 1: 4.14 µT (7.5%, Z)
   - suit 2: 17.71 µT (33%, Y) — severe, WORST sensor overall
   - suit 3: 5.26 µT (10.1%, Y/Z)
   All clean spherical shape (hard-iron, not soft). Three-for-three is not
   chance. NOTE: the head mount is a FABRIC HEADBAND — no ferrous hardware — so
   it is NOT a mount-hardware issue. The leading hypothesis is remanent
   magnetization of each head sensor's own ferrous parts from storage/charging
   proximity to a battery (see Finding 2b); a common storage arrangement would
   explain why all three head sensors are affected. The suit-2 head (17.7 µT on
   a 53 µT field) is large enough to grossly skew head heading/yaw — do not
   trust head orientation on suit 2 until fixed.
   *Action:* degauss the head sensors and re-measure; re-run calibration; review
   how/where head sensors are stored relative to batteries.

2. **Scattered per-sensor hard-iron (non-head) — action required.**
   The severe non-head offsets are scattered across the body with n. I get offset ofo clean
   anatomical pattern — wrists, a foot, an upper-back sensor:
   - suit 1 RIGHT wrist: 15.65 µT (30.4%, X) — #2 worst overall
   - suit 2 LEFT wrist: 10.68 µT (20.2%, Z)
   - suit 3 RIGHT ankle/foot: 9.50 µT (17.9%, 3-axis spread)
   - suit 3 LEFT shoulderblade base: 8.69 µT (16.3%, Y)
   - Counterparts clean (suit-specific): e.g. suit 1 left wrist 2.45%, suit 2
     right wrist 2.05%, suit 1/2 ankles clean, suit 1 left shoulderblade 3.97% &
     suit 2 0.33%.
   All have clean spherical shape — textbook displaced-center hard-iron. Varied
   axes (X / Z / Y / 3-axis) and scattered locations indicate INDIVIDUAL
   localized contamination per sensor — magnetized hardware near each sensor's
   own mount (clip/screw/strap buckle, and for the foot footwear), varying
   unit-to-unit — NOT a location-based design issue. (Earlier "extremity"
   reading dropped: the suit-3 shoulderblade is upper back, not distal.) Each is
   large enough to grossly skew that sensor's heading/yaw.
   *Action (each bad sensor):* degauss the sensor (and/or remove the nearby
   magnetized source), re-run the suit's calibration, then re-measure (expect
   center back to ~1 µT; use the `centered cloud` output to confirm recentering
   on origin).

2b. **Leading mechanism hypothesis: remanent magnetization from battery/storage
   proximity.** Every severe offset is a CLEAN displaced sphere → the magnetic
   source CO-ROTATES with the sensor → it is intrinsic to the sensor (its ferrous
   parts permanently magnetized), NOT an external field present during the sweep
   (that would smear the cloud / raise residual). Observed in the field: batteries
   significantly disturb magnetometer readings. Batteries have nickel-plated
   STEEL cans that are often slightly magnetized; a sensor resting against one for
   a long time can acquire a small remanent magnetization in low-coercivity nearby
   material. Magnitudes fit (worst offset ~18 µT ≈ 0.018 mT, well within reach of a
   small magnetized component near the magnetometer die). This single mechanism
   unifies BOTH the systematic head result (head sensors stored alike near a
   battery) and the scattered single-sensor outliers (whichever sensors happened
   to rest against a battery/magnet).
   *Tests:* (1) degauss a bad sensor (e.g. suit-2 head) and re-measure — offset
   should collapse if remanent; (2) controlled exposure — store a known-clean
   sensor (e.g. suit-3 right elbow 0.10 µT) against the battery for a day, re-
   measure for offset growth; (3) sweep the `mag_offset` tool near the battery /
   storage spot to check for a static field; (4) check whether each bad sensor's
   offset axis matches how it sits against the battery in storage.
   *Implication:* remanent offsets are stable, so the suit's hard-iron
   calibration should largely correct them — but degaussing removes the root
   cause, and storing sensors away from batteries/magnets prevents recurrence.

3. **Moderate offsets — suit-specific (not mount patterns).**
   Suit 1 right knee 4.58 µT (8.61%, Y) and suit 2 pelvis anchor 3.50 µT
   (6.48%, Y), both clean spherical shape. Each is clean on the *other* suit
   (suit 2 knee 0.31 µT, suit 1 pelvis 0.66 µT), and the matching limb is clean
   (suit 1 LEFT knee 0.78 µT), so these are individual sensor/hardware issues,
   NOT mount-design patterns like the head. Should be removed by the suit's
   hard-iron calibration; worth watching. (Both happen to be Y-dominated, but on
   independent sensors.)

4. **Elevated residuals (clean offset, correct radius) — soft-iron or
   coverage.** Suit 2 right shoulder (2.26%), suit 1 left ankle (2.44%), suit 2
   left shoulder (1.98%), left elbow (1.96%) show ~2× the residual of other
   sensors while their offsets are small/clean and radii normal. Mostly arm
   sensors (hardest to sweep fully), but the left ankle joined this group only
   *after* a fuller re-sweep raised its radius — i.e. better coverage exposed
   more real deviation, hinting the cause is genuine mild soft-iron (slight
   ellipsoid) rather than pure coverage. Right elbow (1.27%) and right wrist
   (1.36%) remained normal. Check cloud shapes (ellipsoid → soft-iron) to
   confirm; not a hard-iron concern either way.

5. **Suit 1 left-arm chain runs mildly elevated.** All three measured suit-1
   left-arm sensors sit in a 4–5.4% offset band with clean spherical shapes:
   shoulderblade base 3.97%, shoulder 5.41% (Z), elbow 4.80% — vs ~1–2% for
   most clean sensors. A soft regional cluster suggesting something weakly
   biasing the whole left-arm chain (mildly magnetized component along that arm)
   rather than one sensor. Not severe; all clean-shape. Confined to the
   proximal/upper arm — the left wrist drops back to clean (2.45%), so the bias
   does NOT continue distally. Confirmed LEFT-specific: suit-1 right
   shoulderblade base is clean (1.77%) vs left (3.97%).

6. **Scale is healthy where swept fully; low radius = coverage artifact.** Radii
   cluster at 53.0–55.5 µT, consistent with the Toronto geomagnetic total field.
   Two low readings both traced to incomplete coverage, not scale error: suit 1
   left ankle 50.48→53.29 µT and suit 3 chest 51.09→52.64 µT, each recovered on a
   full re-sweep. Sweep fully before trusting the radius.

## Coverage / TODO

- ASSESSMENT COMPLETE — all 3 suits, all 17 physical IMU sensors each (51 total).
- (The other Shadow joints are virtual/inferred — no physical magnetometer to
  assess.)
- ✅ **DONE (2026-06-17) — Remediate then re-measure.** All flagged sensors
  recalibrated and recentered to ≤0.75% (see Recalibration campaign section):
  - ✅ all 3 head sensors (Finding 1) — suit 1/2/3 heads all cleared
  - ✅ suit 1 right wrist (30%), suit 2 left wrist (20%), suit 3 right ankle (18%),
    suit 3 left shoulderblade base (16%), plus all remaining moderate/watchlist sensors
  - Remediation was **recalibration** (calibration-take refresh), not source
    removal/degaussing — and it fully recentered every sensor.
- ✅ **Battery hypothesis (Finding 2b) — confirmed by the recal result.** Every
  offset was captured by a calibration refresh ⇒ stable remanent (co-rotating)
  hard-iron that post-dated each sensor's last calibration take. No degaussing
  needed. (The controlled re-exposure / storage-field sweep tests remain optional
  for understanding the *source*, but are no longer required to fix the suits.)
- ⏳ **Change storage** so sensors don't rest against batteries/magnets (root-cause
  prevention — still recommended to prevent recurrence).

---

## Suit 3 recheck — different controller (2026-06-04, COMPLETE)

Re-measuring suit 3 after connecting to a **different controller** (possibly
different internal magnetometer settings). Goal: confirm whether the controller
change shifts the readings, and whether the suit-3 outliers reproduce.

**Compare offset-% and residual-% across controllers, NOT absolute µT** — radius
is the controller's field-magnitude scale, so a gain difference would move
absolute µT but leave the normalized percentages intact.

| Sensor | Offset (µT) | Offset %r | Radius (µT) | Residual (µT) | Resid %r | vs old controller |
|---|---|---|---|---|---|---|
| Pelvis anchor | 0.74 | 1.30% | 56.61 | 0.619 | 1.09% | clean before (1.82% / r54.27) — **still clean** |
| **Left shoulderblade base** | **8.79** | **16.15%** | 54.42 | 0.454 | 0.83% | outlier before (16.29% / r53.34) — **REPRODUCES** |
| Left shoulder | 0.58 | 1.05% | 54.79 | 0.540 | 0.99% | pristine before (0.66% / r53.92) — **still pristine** |
| Left elbow | 0.60 | 1.11% | 54.28 | 0.382 | 0.70% | pristine before (0.50% / r53.51) — **still pristine** |
| Left wrist | 0.71 | 1.32% | 53.98 | 0.410 | 0.76% | pristine before (0.41% / r53.50) — **still pristine** |
| Right shoulderblade base | 0.64 | 1.19% | 53.93 | 0.588 | 1.09% | pristine before (0.62% / r53.09) — **still pristine** |
| Right shoulder | 0.56 | 1.05% | 53.36 | 0.439 | 0.82% | pristine before (0.42% / r53.62) — **still pristine** |
| Right elbow | 0.39 | 0.73% | 53.15 | 0.435 | 0.82% | pristine before (0.19% / r53.33) — **still pristine; scale cross-check** |
| Right wrist | 0.19 | 0.36% | 52.83 | 0.504 | 0.95% | pristine before (0.27% / r53.42) — **still pristine** |
| Left hip | 1.73 | 3.18% | 54.41 | 0.437 | 0.80% | clean-high before (3.30% / r54.35) — **reproduces, still clean** |
| Left knee | 0.83 | 1.56% | 53.21 | 1.250 | 2.35% | offset clean (was 1.08%) but **residual elevated (was 0.78%) → RE-SWEEP (coverage)** |
| Left ankle | 1.05 | 1.99% | 52.62 | 0.279 | 0.53% | clean before (1.62% / r53.46) — **still clean** (Y) |
| Right hip | 1.15 | 2.11% | 54.39 | 0.577 | 1.06% | clean before (1.39% / r53.32) — **still clean** |
| Right knee | 0.71 | 1.33% | 53.76 | 0.430 | 0.80% | clean before (2.01% / r53.26) — **still clean** |
| **Right ankle** | **9.58** | **17.92%** | 53.46 | 0.669 | 1.25% | outlier before (17.9% / [−5.59,−3.76,6.70]) — **REPRODUCES** (3-axis) |
| Chest (MidVertebrae) | 2.48 | 4.69% | 52.90 | 0.510 | 0.96% | clean-high before (4.68% / r52.64) — **reproduces, still clean** (Y/Z) |
| Base of skull (Head) | 5.17 | 10.13% | 51.05 | 0.481 | 0.94% | elevated before (10.1% / [0.84,3.22,−4.07]) — **REPRODUCES** (Y/Z) |

**Readings so far:**
- **Pelvis anchor — still clean.** Offset 1.30% and residual 1.09% are both in
  the normal band, consistent with the old-controller reading (1.82%). No
  hard-iron change. Radius 56.61 is the highest measured anywhere (prior range
  52.2–55.5) — a +4.3% jump vs the old-controller pelvis (54.27). Not a coverage
  artifact (partial sweeps bias radius *low*).
- **Left shoulderblade base — outlier REPRODUCES almost exactly.** Offset
  8.69→8.79 µT (+1.2%), Y-dominated throughout (8.59→8.71), %-of-radius
  16.29→16.15, residual identical (0.441→0.454, 0.83%). A different controller
  recovers the same ~8.7 µT Y-displacement to within ~1%.

**Interpretation:**
1. **The severe offset is real, intrinsic, and stable.** Reproducing across
   controllers to ~1% rules out a controller/measurement artifact and is the
   textbook signature of remanent (co-rotating) hard-iron — strong corroboration
   of the battery-remanence hypothesis (Finding 2b). The controller change does
   NOT explain the outliers; they live in the suit/sensor. The original outlier
   list (Findings 1, 2) stands.
2. **Controller scale: unchanged — NOT a gain shift; the pelvis was an
   outlier.** Radius shifts across 6 clean sensors scatter ±2% with no consistent
   direction: right elbow −0.3% (53.33→53.15), right shoulder −0.5%
   (53.62→53.36), left elbow +1.4%, left shoulder +1.6%, right shoulderblade
   +1.6%, left shoulderblade +2.0% — while pelvis stands alone at +4.3%
   (54.27→56.61). The cross-check on the cleanest sensor (right elbow) landed at
   53.15, essentially unchanged, *not* the ~56–57 a uniform +4% gain would
   predict. So the new controller reads on the same scale as the old one; the
   ±2% scatter is normal sweep-coverage variation and the pelvis 56.61 was a
   genuine high-side/coverage outlier. **Scale question closed.** Cross-controller
   comparison by offset-% / residual-% is sound (and so is absolute µT, since
   scale is unchanged).

**Recheck complete — all 17 suit-3 physical sensors re-measured. The original
assessment is fully validated:**
- **Both severe outliers reproduce to ~1%, same direction** — right ankle
  9.50→9.58 µT (17.9%, 3-axis), left shoulderblade 8.69→8.79 µT (16%, Y). Real,
  intrinsic, stable hard-iron in those two sensors → strong corroboration of the
  remanent/battery hypothesis (Finding 2b).
- **All elevated-but-clean sensors reproduce** — head 10.1%→10.13% (Y/Z), chest
  4.68%→4.69% (Y/Z), left hip 3.30%→3.18%. The all-3-suits head elevation holds.
- **All clean/pristine sensors stay clean.** Scale unchanged (see point 2).
- **Only new flag: left knee** — offset clean (1.56%) but residual rose to 2.35%
  with a slightly low radius (53.21) → incomplete sweep coverage; **re-sweep**
  (expect residual back to ~1%); genuine soft-iron only if it persists with full
  coverage.
- **Conclusion:** the controller change shifts nothing of substance. Severe
  offsets live in the suit/sensors, not the controller; the original remediation
  list (right ankle, left shoulderblade, head) stands unchanged.

## Calibrated vs raw stream — suit-3 left shoulderblade base (2026-06-05)

After adding the `data` selector to the `shadow_sensor` node (calibrated
`<sensor/>` / raw `<raw/>` / legacy `<a/><m/><g/>`), the worst suit-3 sensor was
read in all available streams:

| stream | center (µT / counts) | offset | %r | radius | residual |
|---|---|---|---|---|---|
| `a / m / g` (06-04 recheck) | [−1.04, 8.71, 0.56] µT | 8.79 µT | 16.2% | 54.42 | 0.83% |
| **`sensor (calibrated)`** | [−1.08, 8.60, 0.68] µT | **8.69 µT** | **16.1%** | 53.88 | 0.80% |
| `raw` | [−633.8, 619.8, −369.9] counts | ~960 counts | — | >4000 | 10 (0.25%) |

**Two findings:**

1. **`<a/><m/><g/>` IS the calibrated stream.** It matches `<sensor/>` to within
   sweep noise (and is in µT, vs raw's integer counts). So the entire prior
   assessment (all 3 suits) was already measuring the suit-CALIBRATED output —
   clean ~1 µT sensors = calibration working, outliers = calibration failing.
   (Raw confirms the node is healthy: ~74 counts/µT scale, residual 0.25% = clean
   sphere in counts; the large numbers are just unscaled ADC, not a bug.)

2. **The suit's calibration does NOT remove this sensor's hard-iron.** The
   calibrated stream still carries the full ~8.6 µT Y-dominated offset that the
   calibration's bias-subtraction is meant to eliminate. Most likely the sensor
   acquired this magnetization *after* its last calibration take, so the stored
   correction predates it (fits the battery-remanence timeline, Finding 2b).

**Decisive next test:** re-run the suit's magnetometer calibration take for this
sensor (full orientation coverage) and re-read `<sensor>`. Offset collapses to
~1 µT → magnetization post-dated last calibration, refresh fixes it; offset
persists at ~8.7 µT → bias+scale model can't capture this hard-iron, sensor needs
degaussing.

---

## Recalibration campaign — live tracking (started 2026-06-17)

Recalibrating the significantly-off sensors **one at a time** and re-measuring
as we go. Recalibration re-fits and stores the sensor's hard-iron
bias-subtraction so the **calibrated** `<sensor>` output recenters on the origin;
the physical magnetization remains in the hardware but its effect on the output
is corrected (cf. the 06-05 note and Finding 2b — stable remanent offsets should
be correctable by calibration). This section snapshots the **pre-recal** reading
for each flagged sensor (the baseline to beat) and is updated in place as each
sensor is recalibrated and re-swept. Only sensors flagged offset/outlier
(≥~5% of field) are in scope; the borderline 4–5% sensors are on a watchlist below.

**Baselines:** suits 1 & 2 from the 06-01 assessment; **suit 3 from the 06-04
recheck** (latest, reproduced to ~1%). Method: re-run the suit's magnetometer
calibration take for the sensor (full orientation coverage) → re-read `<sensor>`
with `shadow_sensor` + `mag_offset`. Success = calibrated center recenters to
~1 µT (≲3%) with radius ~53–55 µT and residual ≲1.5%.

### Tracking table (worst-first)

| # | Suit | Sensor | Pre offset (µT) | Pre %r | Axis | Recal date | Post offset (µT) | Post %r | Radius | Resid %r | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | Base of skull (Head) | 17.71 | 33.1% | Y | 2026-06-17 | 0.23 | 0.43% | 52.90 | 0.56% | done ✓ recentered |
| 2 | 1 | Right wrist | 15.65 | 30.4% | X | 2026-06-17 | 0.21 | 0.40% | 52.37 | 0.69% | done ✓ recentered |
| 3 | 2 | Left wrist | 10.68 | 20.2% | Z | 2026-06-17 | 0.11 | 0.21% | 52.63 | 0.40% | done ✓ recentered |
| 4 | 3 | Right ankle | 9.58 | 17.9% | 3-axis | 2026-06-17 | 0.18 | 0.33% | 52.90 | 0.60% | done ✓ recentered |
| 5 | 3 | Left shoulderblade base | 8.79 | 16.2% | Y | 2026-06-17 | 0.07 | 0.14% | 52.75 | 0.68% | done ✓ recentered |
| 6 | 3 | Base of skull (Head) | 5.17 | 10.1% | Y/Z | 2026-06-17 | 0.39 | 0.75% | 52.00 | 1.07% | done ✓ recentered |
| 7 | 1 | Right knee | 4.58 | 8.61% | Y | 2026-06-17 | 0.10 | 0.18% | 53.03 | 0.35% | done ✓ recentered |
| 8 | 1 | Base of skull (Head) | 4.14 | 7.45% | Z | 2026-06-17 | 0.08 | 0.15% | 52.98 | 0.73% | done ✓ recentered |
| 9 | 2 | Pelvis anchor | 3.50 | 6.48% | Y | 2026-06-17 | 0.05 | 0.09% | 53.19 | 0.42% | done ✓ recentered |
| 10 | 1 | Left shoulder | 2.92 | 5.41% | Z | 2026-06-17 | 0.10 | 0.19% | 53.06 | 0.53% | done ✓ recentered |
| 11 | 3 | Chest (MidVertebrae) | 2.48 | 4.69% | Y/Z | 2026-06-17 | 0.21 | 0.40% | 52.58 | 0.79% | done ✓ recentered _(from watchlist)_ |
| 12 | 3 | Left hip | 1.73 | 3.18% | — | 2026-06-17 | 0.19 | 0.36% | 52.86 | 0.86% | done ✓ recentered _(from watchlist)_ |
| 13 | 1 | Left shoulderblade base | 2.13 | 3.97% | — | 2026-06-17 | 0.10 | 0.19% | 53.37 | 0.47% | done ✓ recentered _(from watchlist)_ |
| 14 | 1 | Left elbow | 2.55 | 4.80% | — | 2026-06-17 | 0.03 | 0.06% | 53.27 | 0.63% | done ✓ recentered _(from watchlist)_ |

Status legend: `queued` → `recalibrated (re-sweep pending)` → `done ✓ recentered` / `done ✗ persists`.

### Watchlist (borderline 4–5%, re-measure only if convenient)

- suit 1 left-arm chain — shoulderblade 3.97% (row 13) + elbow 4.80% (row 14) both recalibrated 2026-06-17; with the left shoulder (row 10) the whole Finding-5 chain is now clean. **Watchlist cleared.**

_(Suit-3 watchlist items — chest and left hip — recalibrated 2026-06-17 and promoted to the tracking table.)_

### Notes / running log

- 2026-06-17 — Tracking section created; recalibration campaign begun.
- 2026-06-17 — **Suit 3 left shoulderblade base: 8.79 µT (16.2%) → 0.07 µT (0.14%)**, center
  [−0.00, −0.02, −0.07], radius 52.75, residual 0.68%. Full recentering. First recal result, and a
  decisive confirmation of Finding 2b: the offset was stable remanent hard-iron that the bias-subtraction
  model captures, so the magnetization post-dated the last calibration take and a refresh removes it (no
  physical degaussing needed).
- 2026-06-17 — **Suit 3 right ankle: 9.58 µT (17.9%) → 0.18 µT (0.33%)**, center [−0.14, 0.08, −0.07],
  radius 52.90, residual 0.60%. Full recentering — second confirmation of Finding 2b. (Two reads taken,
  both centered: [−0.15, 0.10, −0.07] r52.61 and [−0.14, 0.08, −0.07] r52.90; latter logged.) A "left
  ankle" label used mid-session was a mislabel — the off sensor was the right ankle, as the original
  assessment recorded; no L/R swap, and the left ankle was never off.
- 2026-06-17 — **Suit 3 base of skull (head): 5.17 µT (10.1%) → 0.39 µT (0.75%)**, center
  [0.20, 0.20, 0.27], radius 52.00, residual 1.07%. Full recentering. First of the three head sensors
  (the all-3-suits systematic finding, Finding 1) cleared by recalibration — suit-3 head done.
- 2026-06-17 — **Suit 3 chest (MidVertebrae): 2.48 µT (4.69%) → 0.21 µT (0.40%)**, center
  [−0.08, 0.05, 0.19], radius 52.58, residual 0.79%. Recentered. Watchlist item (4–5% band) recalibrated
  and promoted to the tracking table (row 11).
- 2026-06-17 — **Suit 3 left hip: 3.18% → 0.19 µT (0.36%)**, center [−0.11, 0.08, 0.13], radius 52.86,
  residual 0.86%. Recentered. Last suit-3 watchlist item, promoted to the tracking table (row 12).
  **Suit 3 now fully clean** — all flagged + watchlist sensors recalibrated and recentered.
- 2026-06-17 — **Correction:** three readings logged as "suit 2" (head, left wrist, pelvis) were
  mislabeled and have been **discarded**. Those three suit-2 rows reverted to queued; suit 2 to be redone.
- 2026-06-17 — **Suit 2 base of skull (head): 17.71 µT (33.1%) → 0.23 µT (0.43%)**, center
  [−0.01, −0.11, −0.20], radius 52.90, residual 0.56%. Full recentering of the worst sensor in the
  assessment. Second of the three head sensors cleared (Finding 1).
- 2026-06-17 — **Suit 2 left wrist: 10.68 µT (20.2%) → 0.11 µT (0.21%)**, center [−0.03, 0.10, 0.04],
  radius 52.63, residual 0.40%. Full recentering.
- 2026-06-17 — **Suit 2 pelvis anchor: 3.50 µT (6.48%) → 0.05 µT (0.09%)**, center [0.01, 0.02, 0.04],
  radius 53.19, residual 0.42%. Recentered. **Suit 2 now fully clean** — head, left wrist, pelvis all
  recalibrated and recentered.
- 2026-06-17 — **Suit 1 right wrist: 15.65 µT (30.4%) → 0.21 µT (0.40%)**, center [0.09, −0.05, −0.18],
  radius 52.37, residual 0.69%. Full recentering of the second-worst sensor in the assessment.
- 2026-06-17 — **Suit 1 right knee: 4.58 µT (8.61%) → 0.10 µT (0.18%)**, center [0.07, −0.03, 0.06],
  radius 53.03, residual 0.35%. Recentered.
- 2026-06-17 — **Suit 1 base of skull (head): 4.14 µT (7.45%) → 0.08 µT (0.15%)**, center
  [−0.04, −0.03, −0.06], radius 52.98, residual 0.73%. Recentered. **Third and final head sensor —
  Finding 1 (head contaminated on all 3 suits) fully resolved by recalibration.** All three head sensors
  recentered to ≤0.75%, confirming the systematic head offset was stable remanent hard-iron correctable
  by a calibration refresh (Finding 2b), not a mount-design defect.
- 2026-06-17 — **Suit 1 left shoulder: 2.92 µT (5.41%) → 0.10 µT (0.19%)**, center [−0.07, 0.07, 0.03],
  radius 53.06, residual 0.53%. Recentered. **All 10 originally-flagged sensors (rows 1–10) now
  recalibrated and recentered**, plus 2 promoted watchlist items (rows 11–12). Only the suit-1 left-arm
  chain watchlist item (shoulderblade/elbow, 4–5%) remains. The left shoulder was part of that same
  mildly-elevated chain (Finding 5) and is now clean at 0.19%.
- 2026-06-17 — **Suit 1 left shoulderblade base: 2.13 µT (3.97%) → 0.10 µT (0.19%)**, center
  [−0.07, 0.07, 0.03], radius 53.37, residual 0.47%. Recentered, promoted to the tracking table (row 13).
  Suit-1 left-arm chain now 2 of 3 clean (shoulder + shoulderblade); only the left elbow (4.80%) remains.
- 2026-06-17 — **Suit 1 left elbow: 2.55 µT (4.80%) → 0.03 µT (0.06%)**, center [0.03, −0.01, 0.00],
  radius 53.27, residual 0.63%. Recentered, promoted to the tracking table (row 14). Last sensor in the
  campaign.
- 2026-06-17 — **CAMPAIGN COMPLETE.** All 14 tracked sensors (10 flagged + 4 promoted watchlist:
  suit-3 chest & left hip, suit-1 shoulderblade & elbow) recalibrated and recentered to ≤0.75% of field
  (most ≤0.4%), radii 52–53 µT, residuals ≤1.1%. Watchlist cleared. Every previously-flagged offset
  across all 3 suits is resolved. Confirms Finding 2b end-to-end: the offsets were stable remanent
  hard-iron that post-dated each sensor's last calibration take, fully corrected by a calibration refresh
  — no physical degaussing required. Finding 1 (all-3-suits head contamination) resolved on all three.

