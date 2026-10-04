# Shadow IMU Magnetometer — 2026-08-13 (new space: field survey, IN PROGRESS)

**Status: PAUSED mid-investigation.** Test 1 complete (6 sphere sweeps, suit 2 left
arm chain, two heights). Room mapping not yet started. Resume at "Next session"
below.

**This run is not a 51-sensor sweep.** It started as one (suit 2, all 17), but the
first readings redirected it into a survey of the *new space*. The suits turned
out to be fine; the room is the open question.

## Context — why this run happened

We moved to a new studio. Preliminary readings looked "significantly shifted"
against the 06-18 baselines, on sensors that were clean then. The working
hypothesis was that the 06-17 recalibration campaign — done in the old space,
on suits that had been stored with batteries in among the sensors — had **baked a
correction for a distortion that no longer exists**, so the stored calibration
would now be over-correcting.

**That hypothesis is not supported by the data below.** See Finding 1.

The space: sprung floor, raised 3–4 inches above a concrete slab. Interference is
known to be in the floor and known to be non-uniform — the Shadow app's
interference display shows discrete hotspots plus larger areas registering
nothing. All measurements below were taken in the section that reads cleanest,
after moving out of an office that showed significant interference.

## Test 1 — same sensor, two heights

The discriminating test: a sensor's own hard iron cannot know where in the room it
is. If the fitted center moves between locations, the offset is in the space, not
the suit.

Suit 2, left arm chain, calibrated (`<sensor>`) stream.

### Waist height

| Sensor | Center (µT) | Offset | %r | Radius | Residual | Resid %r |
|---|---|---|---|---|---|---|
| Left shoulder | (0.01, −0.54, 0.67) | 0.861 | 1.63% | 52.69 | 0.242 | 0.46% |
| Left elbow | (−0.07, 0.10, −1.39) | 1.395 | 2.65% | 52.75 | 0.199 | 0.38% |
| Left wrist | (−0.01, 0.05, 0.09) | 0.103 | 0.20% | 51.61 | 0.272 | 0.53% |

### At the floor

| Sensor | Center (µT) | Offset | %r | Radius | Residual | Resid %r |
|---|---|---|---|---|---|---|
| Left shoulder | (−0.40, −0.09, 1.42) | 1.478 | 2.89% | 51.18 | 1.031 | 2.01% |
| Left elbow | (0.13, 0.61, −1.68) | 1.792 | 3.50% | 51.17 | 0.898 | 1.75% |
| Left wrist | (0.08, −0.30, 0.63) | 0.702 | 1.40% | 50.01 | 0.904 | 1.81% |

### Floor − waist

| Sensor | Center moved | Radius | Residual |
|---|---|---|---|
| Left shoulder | 0.97 µT | 52.69 → 51.18 (−1.51) | 0.46% → 2.01% (×4.3) |
| Left elbow | 0.62 µT | 52.75 → 51.17 (−1.58) | 0.38% → 1.75% (×4.5) |
| Left wrist | 0.65 µT | 51.61 → 50.01 (−1.60) | 0.53% → 1.81% (×3.3) |

## Finding 1 — the suits are fine; the recalibration did NOT bake in old-space error

At waist height, all three sensors reproduce their 06-18 readings **component-wise,
same directions**, within 0.10–0.15 µT:

- left shoulder: (−0.10, −0.39, 0.85) @ 1.73% → (0.01, −0.54, 0.67) @ 1.63%
- left elbow: (−0.18, 0.20, −1.51) @ 2.83% → (−0.07, 0.10, −1.39) @ 2.65%
- left wrist: (0.15, −0.15, 0.14) @ 0.48% → (−0.01, 0.05, 0.09) @ 0.20%

All three came in *slightly cleaner* than baseline, and the deltas (−0.10, −0.18,
−0.28 pt) are well inside the June session-to-session noise floor (mean |Δ| 0.57,
σ 0.73 pt). The stored calibration is not over-correcting. The preliminary
"shifted" readings were almost certainly taken in the office or low to the floor.

**Consequence: no recalibration is indicated on this evidence.** Do not recalibrate
the suits in response to the move.

## Finding 2 — the floor carries a real gradient

Center movement of 0.62–0.97 µT between waist and floor, on all three sensors.
Sensor hard iron is location-independent by construction, so that is the room.

Crucially this is **not** the incomplete-sweep-coverage signature catalogued in
June. Coverage drops the radius but leaves the residual artificially *tight*. Here
the radius drops **and** the residual roughly quadruples, on all three sensors, by
nearly the same factor. Low radius + inflated residual = genuine field variation
across the volume the sweep passes through.

The consistent −1.5 µT radius drop across three independent sensors says the same
thing.

**Operational rule adopted: all calibration takes and all sphere sweeps at waist
height or above.** Calibrating a sensor low in this room would bake the floor
gradient into the suit's stored correction — precisely the failure mode we were
worried about, but aimed forward rather than backward. The risk is not behind us
in the old space; it is live, going forward, if anyone calibrates on the floor.

## Finding 3 — new-space field magnitude is ~1.5 µT below the old space

Working-height radius reads ~52.7 µT here vs ~54.3 µT in the old space, consistent
across all three sensors (−1.50, −1.54, −1.59). That is the building's steel shell.

Largely benign for heading: the fusion normalizes the magnetometer vector and uses
its direction. Our percent-of-radius metric normalizes it out too, so cross-space
comparisons of offset %r remain valid. Recorded so future low radii here are not
misread as sweep-coverage artifacts.

## The thing that actually matters: the capture volume, not the calibration

Ankle and knee sensors sit near the floor for an entire take — permanently inside
the gradient measured above. No sensor calibration can fix that; it is a property
of the space. Expect yaw wander concentrated on the lower legs.

### Uniform vs gradient — why the "clean section" guess is physically reasonable

What makes a floor source dangerous is its **spatial scale**, not its strength.

- A rebar mesh is periodic, and a periodic source's field decays exponentially with
  height, decay length set by the grid pitch. For a 6-inch mesh: ~15% of surface
  value at 4 inches up, ~1% at a foot. The sprung floor's 3–4 inch gap is already
  doing real work, so the *mesh* component is probably not what reads as hotspots.
- Large-scale sources (floor box, beam, conduit run, large magnetized plate) decay
  slowly — which is why they show up as hotspots at all. But the same slow decay
  means that away from them their field is locally **uniform**.

So "relatively uniform and benign in the cleanest section" is the physically
expected outcome, not wishful thinking. It is also directly measurable.

And uniform genuinely is benign in a way a gradient is not:

- **Uniform offset** — constant vector added everywhere. Sphere fits unaffected
  (center stays at origin, only radius moves). Heading gets a constant bias,
  consistent across the space, absorbed by a heading reset.
- **Gradient** — yaw wanders *as the performer walks*. Not correctable, because it
  is a function of where they are standing.

### The tolerance number

Total field 52.7 µT at ~70° dip → the horizontal component carrying heading is only
about **18 µT**. So 1 µT of horizontal variation = **3.2° of yaw**.

**To hold lower-leg yaw wander under 2°, the horizontal field must be uniform across
the working area to within about 0.6 µT.**

That is tight, and the floor-vs-waist center movement we measured was 0.6–1.0 µT —
same order. This is genuinely marginal rather than obviously fine or obviously
broken, which is why it is worth mapping properly.

## Next session — map the room

**The sphere sweep is the wrong instrument for this.** A sweep rotates *and*
translates at once, smearing position-dependence into a fit that assumes a single
ambient vector — exactly why the floor sweeps returned quadrupled residuals. To map
a room, do the opposite: **hold the sensor in a fixed orientation and move it.**
Then every change in the reading is purely spatial.

Two passes, both fixed-orientation, both in the clean section:

1. **Vertical profile**, one spot — sensor flat, readings at roughly 2 / 6 / 12 / 24
   / 40 inches above the sprung floor. Gives the decay curve, which identifies the
   source's scale (fast decay = mesh, slow = something large) and shows where foot
   sensors actually ride relative to the problem.
2. **Horizontal grid at foot height**, same fixed orientation throughout — corners
   and center of the rectangle actually performed in, ~2 m spacing. The number that
   matters is the spread in **heading direction** between points, not magnitude.

**Verdict metric: heading spread across the working area, in degrees.** Under ~2° →
the clean section is confirmed benign, stop thinking about it. Around 10° → we know
which part of the floor to keep performers off.

**Protocol caution:** the sensor's own hard iron is a constant in its body frame, so
it cancels out of every point-to-point comparison *as long as the sensor is not
rotated between points*. Same sensor, same orientation, for the whole map — mark a
heading on the floor and align to it at each point.

**Tooling to build:** reading nine points off the display by hand is error-prone and
orientation drift will corrupt it. Wanted: a probe path — `shadow_sensor` into an
averager taking a labelled few-second sample per point, appending to CSV, with the
heading-spread math done afterward. Not built yet. Makes the map repeatable, which
matters because it should be re-checked whenever the capture area moves.

## Still available, not done

- **Suit 2 full 17-sensor sweep** — the original plan for this session. Now optional
  insurance rather than the main event, given the left arm chain reproduced. Would
  give a fresh in-this-space baseline. Do it at waist height or above.
- **Raw vs calibrated** (`data` dropdown: a/m/g | sensor | raw) — the other test
  proposed for the over-correction hypothesis. Not needed now that Finding 1 settles
  it, but it remains the direct probe if the question ever comes back.
- **Outdoor ground-truth sweep**, well away from the building — Toronto total field
  ~54.5 µT. Would confirm Finding 3 is the building rather than the suits.

## Running log

- 2026-08-13 — Session opened intending a full suit-2 17-sensor sweep. Preliminary
  readings looked shifted (up to −2.16 µT on previously-clean sensors); over-
  correction hypothesis raised. Redirected to diagnosis.
- 2026-08-13 — Test 1 (same sensors, waist vs floor) run on suit-2 left shoulder /
  elbow / wrist. Result: suits reproduce 06-18 at waist height (Finding 1); floor
  carries a genuine gradient (Finding 2); new space reads −1.5 µT (Finding 3).
- 2026-08-13 — **PAUSED.** Room mapping (vertical profile + horizontal grid) is the
  next step, along with the probe/logging tooling for it.