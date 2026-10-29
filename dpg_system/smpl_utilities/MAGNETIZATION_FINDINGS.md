# Shadow IMU Magnetization — Findings & Tools Ledger

Clean status ledger for the magnetization-correction investigation (Subject7 Bharatanatyam session,
11 `*beta.npz` takes). The detailed chronological log lives in the auto-memory
`project_magnetometer_deviation.md`; THIS file is the curated summary of what is validated, what is
disproven, and which tools to trust. Update this when a trial concludes.

Status legend: **[V]** validated (cross-checked / multiple independent tests) · **[~]** tentative
(one instrument, plausible, not cross-checked) · **[X]** disproven / dead-end · **[O]** open.

---

## VALIDATED findings [V]

- **Forearm yaw magnetization.** L-forearm ~24°, R-forearm ~27° world-yaw (1st-harmonic, hard-iron).
  Cross-validated: the geometry-free **headlock** curve and the **twist landscape** agree on phase to
  ~2° (after the L/R bone-axis mirror), and the world-yaw/twist amplitude ratio >1 as geometry predicts.
- **Hand magnetization.** Both hands magnetized, LEFT strongly (~25°, 45–67% variance explained), with
  a phase distinct from the forearm ⇒ the hand sensor's own error, not inherited. Direct no-turn
  heading-binned method (`hand_magnetization.py`).
- **Roll-bleed (Mechanism B) on the left arm.** Magnetization bleeds into the fusion's roll/twist
  estimate: heading-locked twist persists at ~0° elevation where a clean yaw-only error predicts ~0;
  effect is LEFT-dominant, tracking the more-magnetized side (rules out symmetric choreography).
- **f4318 candy-wrapper is a SMPL skinning artifact, NOT a correctable orientation error.** The forearm
  sits ON its elbow-hinge arc (impossibility test passes), and de-twisting it *worsens* the crease
  (causal test). Multiple independent tests agree.
- **Magnetization is real & pervasive but NOT the dominant *visible* defect.** Yaw correction
  (`*_magfix`) and roll/twist correction (`*_twistfix`) both removed real heading-locked error yet did
  NOT improve the mesh crease — because world-yaw on a near-horizontal limb produces *swing*, and the
  crease is skinning-dominated.
- **Arm symmetry analysis is confounded by a core facing-yaw.** The L/R mirror asymmetry is flat by
  arm-heading but varies by pelvis-facing, and the residual is a vertical-axis yaw ⇒ a body-facing
  (core) yaw error rotating the mirror plane, not arm error. So symmetry-based arm fits are corrupted.
- **Separate-recalibration logic.** Each take is recalibrated separately, so a *constant* physical
  mounting/tilt offset would be zeroed every take. A consistent-across-takes error must therefore be
  heading/pose-dependent (magnetization), re-baked at a consistent calibration facing — OR real posture.
- **Measurement principle.** TILT (roll/pitch) is gravity-referenced ⇒ absolutely measurable per
  sensor. YAW has no absolute reference here (no Vive, no raw magnetometer) ⇒ only measurable RELATIVE
  to body/parent (gauge-limited).

## TENTATIVE / partial [~]

- **Upper-arm yaw magnetization** (headlock): L/R shoulders show large heading-dependent yaw (~42–46°),
  asymmetric. Supports the asymmetric-T-pose-calibration story BUT not cross-validated (shoulder twist
  too noisy to check); may be inflated by arm motion correlated with turning. Magnitude uncertain.
- **Asymmetric hang = ~26° elevation + ~20° yaw** (low-arm frames, session-consistent). Yaw part =
  upper-arm yaw magnetization [~]. Elevation part [O] (below). Confound: "low-arm" frames include
  active dance poses, not just true rest, so real asymmetric posture is mixed in.

- **T-pose elbow turn (2026-10-27).** Calibration zeroes how a sensor is strapped on, but it bakes
  in the T-pose POSTURE: shoulder and wrist can hold the T-pose while the elbow turns a long way
  (more in trained, mobile dancers), and the upper-arm sensor takes that turn as its zero. Measured
  per take from the direction a bent forearm points around the upper arm (a real elbow bends
  forward; AMASS/SMPL elbows bend about rest vertical, axis y 0.94). Moderate bend (45–80°), arm
  low: jathiswaram L 42.7 / R 42.3 (symmetric, as a postural habit should be), padam 45 / 53;
  steps between takes (25–60°), consistent with recalibration before most takes. This is most of
  the opening-hold candy-wrapper twist in jathiswaram (shoulder twist 98/−104 raw). The
  elbow-hinge test is blind to it (it turns the whole hinge), so "forearm on its hinge arc" does
  not rule it out. Node `elbow_turn`; `shadow_mag_correct` 'turn' inlet. **VIDEO check pending**:
  the T-pose is filmed — predicted inside-of-elbow turned up ~40° both arms in jathiswaram, ~13° L
  vs ~40° R in take 1. The forearm sensor takes part of the turn as its own twist (invisible to
  the hinge; shows at the hand — what `--wtwist` dialled).
- **Cause split of the varying roll error (forearm-azimuth lens, all 11 takes).** RIGHT upper-arm
  roll error grows with arm height (+15–17° raised) and with elbow bend (jathiswaram R turn 39 →
  76 from moderate to deep bend), identically in every facing ⇒ body-relative (strap/cable pull or
  muscle under the sensor), not magnetization; no hysteresis after overhead (not a slipped sensor).
  LEFT has a facing-dependent component (arm near horizontal: −16° facing front vs +12° facing
  away; few facing-away frames) ⇒ magnetization. Shoulder blades move little (sd 3–7°) and carry
  only 6–8° twist in the opening hold — not the wrapper's source; blade errors cannot show in the
  hinge anyway.
- **Learned correction with body terms (shadow_mag_correct, 40 weights).** Pooled training on 10
  takes, held-out padam: heading-only 9.1/9.5 → 9.3/9.0 (with turn); + arm height and elbow bend on
  the ROLL only → 7.4/7.1. Body features on all three corrections, or the full arm direction,
  overfit (held-out R worsened). Off-hinge rms in degrees, L/R.
- **Fitted-circle loss drifts in roll; straight arms see nothing (2026-10-27).** Trained against the
  take-FITTED hinge circle (turn + 40 weights, jathiswaram), scatter fell 10.2/9.7 → 6.1/6.4 but the
  shoulder twist got WORSE where the fitted circle is blind: opening hold 107/−79, straight arms
  −114/+88 (old cos-bend feature extrapolating to bend 0), all frames beyond ±40°: 65%/43%
  (raw 73/72; DanceDB 98% within ±31 straight, ±40 bent). Fixes: (1) loss against the FORWARD
  circle (elbow bends about local vertical) — pins roll at every bend: hold 59/−27, deep bend
  17/17, beyond ±40° 24%/15%; (2) bend feature = smooth max(cos 62.5° − cos bend, 0), zero
  below elbow_turn's band; (3) 'turn fade' (30°): the turn comes in over the first 30° of bend.
  Straight arms are UNOBSERVED by every hinge lens: jathiswaram raw straight shoulder twist is
  already typical (−13/+16; constant turn → −55/+58), naatadavu's is not (L +45, what a constant
  offset predicts). Fade 30 vs 0 is a judgement until VIDEO of straight-arm sections. Small-bend
  turn readings (12–30°) drift L up / R down — likely carrying angle, not roll; don't trust them.
- **Asymmetric hang = shared sideways shift of both lowered arms (2026-10-27).** David: "good
  T-pose, then when the arms are lowered a clear asymmetry to the hang" — the most common visible
  Shadow error. Measured as the mean of the two upper arms' angles in the chest's front plane
  (mirror-symmetric pair = 0, splay free): −16 to −20° (toward the dancer's RIGHT) with both arms
  >50° below level, in EVERY Subject 7 take and every facing (body-relative, not magnetization;
  unchanged across recalibrations, so not the T-pose turn). By arm height (all takes): above level
  0 to +2°, 0–30° below −1 to −2°, 30–50° below −6°, 50°+ below −16 to −18°. Elbow hinge cannot
  see it (the forearm hangs consistently; R's 28° "abduction" reads as a forward bend).
  Correction: `shadow_mag_correct` 'hang' term (whole arm about the chest forward axis, faded in
  25–55° below level, elbow/wrist locals unchanged); loss `hang_symmetry`. Trained on jathiswaram
  with David's flagged ranges: 16.3° per arm; held-out takes' hang shift −16…−19 → −1…−8, hinge
  unchanged. Cause unknown (suit/skeleton solver? both arms, same direction) [~].
  **Follow-up (same day).** (i) Zero at the T-pose CONFIRMED: Subject 2 take 3's three pretend
  T-poses at facings 144/−72/−23 read +1.0/−0.6/+1.9; the arms-down stretches around them −8.5 to
  −10.6 (beware: requiring both arms at the same height hides a bank; mean-height bins of
  choreography near level read −15 in S7 but are not T-poses). (ii) ARCHIVE, 48 takes / 6 subjects:
  arms-down bank toward the dancer's RIGHT in essentially every take (−20 to +2: S7 −16..−20, S2
  −8..−11, S1 −3..−8, S3 −1..−7, S6 −11..+2, S4 −7..+1); T-pose frames ≈0 everywhere. Same sign
  across suits/people ⇒ a body-fixed lopsided cause, not independent sensor magnetization (that
  would give random signs); facing-independent. [CORRECTED 2026-10-29: the same sign holds only for
  2025, the old Shadow fusion algorithm; 2026 sessions take both signs. See the timeline entry
  below.] (iii) TIMING (S2 pretend T-poses): most appears
  during the lowering, but it keeps growing ~5° over ~2 s with the arms still, and fades over ~1 s
  after returning to T-pose ⇒ a fusion-filter component being pulled toward a wrong reference.
  [CORRECTED 2026-10-29: the "growing" rested on 3 noisy drops in one take. Over all still holds
  (`hang_bank_analysis/creep.py`) the bank is complete ~0.5 s after the arms arrive: old fusion
  then RELAXES toward 0 by +1.3° over 4 s (23 holds, 78%, interval +0.9..+2.4); new fusion no
  change (−0.3, 116 holds, interval −1.0..+0.4). The bank arrives with the movement, not after.]
  A baked-in COMPASS HEADING error cannot make it (it would swing the hanging arms forward/back in
  opposite directions; not seen). Candidate fitting all of David's field notes: a magnetic or
  current-carrying part of the suit low and off-centre (hub/battery/cable bundle) disturbing the
  field near the HANGING arms only, with filters that use the full 3D field direction (Toronto dip
  ~70°). Untested — studio test plan in `HANG_BANK_STUDIO_TESTS.md` (belt reversal, field probe,
  root tilt, turn with arms down, sensor swap). Upper arms bank MORE than forearms in S7/S2
  (−17/−9, −11/−6), which argues against the belt for the large effect. [WITHDRAWN 2026-10-29:
  the forearm number mixes in a separate effect. In S4, S1 and S5 the forearms shift −10 to −18
  with the upper arms near 0; hanging forearms angle forward 30–45°, so a forearm heading error
  (the validated forearm magnetization) reads as a sideways shift. Not comparable to the upper arm.]
- **Hang bank timeline: algorithm, session drift, sensors (2026-10-29).** Upper-arm bank (same
  measure as above, both upper arms >50° below level) for every session with arm data, 2025–2026,
  scripts in `hang_bank_analysis/` (`bank_2026.py`, `bank_post.py`,
  `session_drift.py`, `s13_stilts.py`, `hang_foreaft.py`). Session medians, negative = toward the dancer's right:

  | Session | Recorded | Bank | Notes |
  |---|---|---|---|
  | S1–S7 | 2025-06-19 … 11-07 | −3 to −18 | all negative; S7 −16..−20 |
  | S11 | 2026-03-06 | −7 to −12 | |
  | S12 | 2026-03-17 | +7 to +14 | right arm 15–20° further forward in every take |
  | S13 | 2026-03-19 | −10 → +13 | stilts takes 1–4; drifts through the session |
  | S14, S16, Gaia suits 1+2 | 2026-04-09 … 05-29 | about 0 | |
  | S15, S17 | 2026-04-16, 05-08 | −5 to +3, −7 to 0 | S17 in steps (≈0 for takes 4–8) |
  | S18 | 2026-06-16 | +5 | day before the magnetometer recalibration |
  | S19 | 2026-07-23 | +6 | first session in the new space (moved 2026-07-06) |
  | S20 | 2026-08-14 | +3 | day after the 08-13 check found the arm sensors clean |
  | S21, S9 retakes, S22 | 2026-10-05 … 10-22 | −2, 0, −3 | |

  T-pose frames stay ≈0 throughout. Calibration-pose takes (controlled arms-down): S20 +3.5,
  S9 retakes +1.1. Take-to-take scatter in clean sessions is about ±3°, so single takes under ~4°
  mean little; session medians do.
  (i) **ALGORITHM [~, strong].** Shadow changed its fusion algorithm on 2025-12-18. Every
  pre-update session is negative (the large consistent right-leaning bank, which relaxes ~1–2°
  toward 0 in the seconds after the arms come down); post-update sessions range −8..+9 with both signs. The large systematic bank
  is a property of the OLD fusion; the update shrank it, but S11 (March 2026) still read −7..−12.
  (ii) **SESSION DRIFT [~].** Fitting bank against recording time: 13 of 18 sessions drift toward
  the dancer's LEFT, typically +4..+6° per hour (S2–S7, S12, S13, S15, S19, S20, S9r, S22; flat or
  slightly negative S1, S14, S17, S18, S21; S11 excluded, its files lost their recording times).
  Under both algorithms. Possible warm-up choreography confound. Candidates: sensor warming
  (magnetometer/gyro offsets shift with temperature), battery discharge changing currents in belt
  and cables, the filter's magnetic reference adapting. Test: protocol take at start/middle/end.
  (iii) **NOT THE SENSORS' OWN MAGNETIZATION (post-update residual).** The June 1 sweep had the
  upper-arm sensors at ≤2.9 µT (suit 1 left, 5%), others ≤1.4 µT; 08-13 confirmed them clean
  (<2%, ~1 µT ≈ 1° of tilt). S20, the next day, still read +3 with the right arm forward ⇒ that
  residual comes from the field around the arm, the solver, or the body. The 2025 sensor state is
  unknown (no sweep before 2026-06-01). Sign change between August (+3..+6) and October (−2..−3):
  new magnetization or a different suit, only a fresh sweep can tell.
  (iv) **STILTS (S13).** Root 0.6 m higher in takes 1–4 (stilts); the recalibration after removing
  them did not reset the bank (take 4 −4, takes 5–6 −3/−6), which then climbs steadily to about +4
  (+15°/h, steepest of any session). Not set by the calibration in this session, not flipped by
  the stilts.
  (v) **LOCATION.** No step at the 2026-07-06 move (S18 +5 old space; S19 +2 → +9 and S20 +3 new).
  **Floor position (2026-10-29, `floor_position.py`):** hanging frames, each take's median removed
  (so time drift and between-take choreography can't pose as position), 0.5 m cells of the
  tracker position (`trans_raw`, room-fixed, in the new space), null = position track slid in time
  within each take. Spread of the bank across cells 0.7–2.3°, beating chance only in S17, S20, S21
  and by tenths of a degree ⇒ floor position moves the bank ≤1–2° within a session; it does not
  make the 3–9° session offsets or the 2025 banks. FACING, same test: a once-per-turn variation in
  7 of 11 sessions (S12, S15, S17, S18, S19, S20, S21), 0.6–2.4° ⇒ sensor magnetization present
  but minor in 2026. One lead [O]: the new-space corner at tracker x ≈ +1.5..+3.5 m,
  z ≈ +0.5..+2.5 m reads negative in both sessions that went there (S20 −2 to −9, S9r 0 to −3.5);
  few seconds per cell, and assumes the tracker frame matched between sessions.
  (vii) **NO CREEP (2026-10-29, `creep.py`).** After the arms come down and hold still, the bank
  is complete within ~0.5 s. Old fusion: relaxes toward 0 by +1.3° over the next 3.5 s (23 holds,
  78% toward the left, interval +0.9..+2.4), consistent with an error made during the movement
  that gravity slowly corrects. New fusion: flat (−0.3°, 116 holds, 46%, interval −1.0..+0.4).
  The earlier "grows ~5° over 2 s" (3 drops, S2 take 3) is withdrawn.
  (vi) **FORWARD/BACK.** Hanging upper arms mostly sit forward together (choreography). The
  left/right difference changes sign between takes in 2025; in S12, S18, S19, S20 the RIGHT arm
  sits 5–20° further forward in nearly every take; October sessions show no consistent difference.

## DISPROVEN / dead-ends [X] — do not repeat

- **"Chest roll-bleed 68%"** — AXIS ERROR. Trunk-sensor local-x is LATERAL, not forward; the 68% was
  chest *pitch* (forward-lean, symmetric, possibly real posture). Real chest *roll* is small (2.6°,18%).
- **Core/trunk lateral roll as the hang cause** — lateral roll is small everywhere (≤6.5°, 14–36%);
  not the dramatic culprit.
- **"Constant mounting tilt offset" as the hang source** — ruled out: separate recalibration zeroes any
  constant offset. (It does NOT zero a T-pose posture offset — see the T-pose elbow turn, above.)
- **Symmetry-based upper-arm fit** (auto-crossings `fit_upper_arm_from_crossings.py`, pooled
  `pooled_arm_fit.py`) — FAILED: this dance is rarely truly bilateral, crossings are impure (~28°
  median residual), fitting them wrecks the genuinely-clean windows. Symmetry is exhausted as a
  reference for this session.
- **Generic per-sensor roll assessment** (`assess_sensor_magnetization.py` all-sensor mode) — FAILED:
  degenerates for near-vertical bones (roll≈yaw) and is swamped by motion for mobile limbs (gave 146°+).
  Tilt-bleed is only cleanly measurable for near-upright trunk sensors.
- **VPoser as a correction objective** — plausibility ≠ correctness; rewards plausible-but-wrong poses.
- **Rotated root (or any) sensor at calibration as the hang cause** (2026-10-28, `hang_bank_analysis/calib_sim.py`):
  per-sensor calibration absorbs how a sensor sits exactly; root and whole-body tilt are shared by
  chest and arms and cancel in the chest-relative bank. Only an arm-vs-torso turn about the forward
  axis during the T-pose produces it (1:1, invisible in T-pose).
- **Calibration posture as the hang cause** — that arm-vs-torso turn would need 5–20° of visible
  asymmetry. David checks arm alignment before every calibration and would notice 5°; S7 video
  T-poses are clean; S13's recalibration did not reset the bank.

## OPEN [O] — need video or more data

- **Elevation half of the hang (~26°, right arm too high):** gravity-referenced, so not yaw-mag. Real
  asymmetric posture vs tilt-bleed — **VIDEO adjudicates** (is the relaxed right arm really raised?).
- **Does any properly-combined heading-dependent correction improve the *rendered* defect?** Never
  cleanly demonstrated (yaw and twist fixes did not).
- **Hang bank: cause of the within-session leftward drift** (warming, battery, filter, or
  choreography) — start/middle/end protocol takes; see `HANG_BANK_STUDIO_TESTS.md`.
- **Hang bank: new magnetization since 2026-08-13?** Fresh `mag_offset` sweep of arm, shoulder-blade
  and chest sensors on all three suits.
- **Hang bank: the new-space corner** (tracker x ≈ +1.5..+3.5, z ≈ +0.5..+2.5 m) — is it one of the
  known interference areas? Studio test 3 (protocol take at marked spots) settles it.

---

## TOOLS (in `dpg_system/smpl_utilities/`)

| script | purpose | trust |
|---|---|---|
| `diag_magnetometer_deviation.py` | foundation: skeleton load, FK, quat helpers, joint indices | infra |
| `headlock_deviation.py` | yaw deviation δ(ψ) from body turns (geometry-free) | [V] forearm/hand; [~] upper-arm |
| `fit_session_deviation.py` | twist landscape, per-take const + session-shared harmonics | [V] (twist-deg, under-states world-yaw) |
| `hand_magnetization.py` | direct no-turn heading-binned hand δ | [V] |
| `mag_twist_elevation_test.py` | elevation-stratified twist → detects roll-bleed | [V] |
| `elbow_hinge_violation.py` | hard hinge impossibility constraint | [V] diagnostic |
| `which_dof_creases.py` | causal: neutralize a DOF, measure mesh crease | [V] |
| `mesh_crease_compare.py` | AREA-collapse crease (roll-sensitive) before/after | [V] (use this, not perp_spread) |
| `mesh_joint_distortion.py` | perp_spread crease — ROLL-INVARIANT | weak (don't use for roll) |
| `trunk_roll_assessment.py` | lateral-roll bleed, axes identified | [V] (core roll small) |
| `detect_sym_crossings.py` | symmetric-crossing detection (local minima) | works, but crossings impure in this dance |
| `correct_upper_arm_offset.py` | constant per-arm C + heading-yaw + dials; node delegates here | vertical-anchor BUG FIXED; C-fit weak (symmetry-limited) |
| `apply_forearm_magfix.py` | applies forearm yaw correction → `*_magfix.npz` | built; visually minor |
| `apply_twist_magfix.py` | applies child-frame roll/twist correction → `*_twistfix.npz` | built; crease wash |
| `fit_upper_arm_from_crossings.py`, `pooled_arm_fit.py` | symmetry-based upper-arm fit | [X] failed |
| `assess_sensor_magnetization.py` | per-sensor roll-bleed scan | [X] only trunk-upright valid |
| `correct_chest_roll_check.py` | chest-roll check | superseded (axis error) |

Nodes in `dpg_system/shadow_mag_nodes.py`: `elbow_turn` (per-take T-pose turn, measured) [~];
`elbow_hinge` (off-hinge loss for t.optimize; circle forward (default) or fitted) [V diagnostic];
`shadow_mag_correct` (turn with 'turn fade' + 42 learned weights: heading on roll/yaw/fyaw, arm
height + bend past 62.5° on roll, 1 hang per arm) [~]; `hang_symmetry` (shared-shift loss) [~]. Demo:
`examples/shadow_mag_learning.json`. Note: `correct_upper_arm_offset.correct()` with its heading-yaw
step raises the forearm-azimuth scatter on jathiswaram from ~14° to ~36° (breaks the hinge).

`shadow_arm_correct` node (motion_cap_nodes.py) = live/interactive wrapper that imports & calls
`correct_upper_arm_offset.py` — same math, sliders + render. Inherits its fixes.

## Gotchas / methodological lessons

- **Axis conventions:** trunk-sensor local-x = LATERAL (not forward); local-y = up; local-z = forward.
  Identify per sensor, never assume — the chest roll/pitch error came from assuming local-x=forward.
- **Gauge:** no absolute heading reference ⇒ yaw only measurable relative to body/parent.
- **Session-consistency ≠ sensor error for mobile limbs / pitch:** stylized choreography is itself
  session-consistent, so it can't be separated from a sensor bleed by consistency alone.
- **`perp_spread` is roll-invariant; use AREA-collapse for roll-sensitive crease.**
- **Headlock units:** feed ψ̇ in radians (coeffs come out in radians otherwise).
- **Decomposition that works:** per-take CONSTANT (calibration drift / re-strap) + session-shared
  heading SHAPE (magnetization). Don't conflate.

---

## ERROR LANDSCAPE MAP (fusion view — overlay lenses, converge where they agree)

Each lens is ONE soft hint. Read for CONVERGENCE (multiple independent lenses on the same sensor+DOF
= confident) vs single/divergent (= ambiguous). "systematic" (consistent across takes) narrows
choreographic-*variation* but NOT choreographic-*style* (a stance habit looks like a sensor bias);
where a DOF is non-postural (e.g. lateral roll) "systematic" leans sensor; where postural (forward
pitch, arm stance) it stays ambiguous. Lenses: HL=headlock yaw, TW=twist-landscape, RB=roll-bleed
(elev-stratified), MP=mean-pose(cross-take), HG=hang abduction/elev, CR=mesh area-crease, HV=hinge,
SY=symmetry(core-yaw-confounded).

| sensor | DOF | lenses & reading | converge? |
|---|---|---|---|
| **Forearm L** | yaw | HL 24° + TW phase-match (2 indep) | **CONVERGE → confident mag** [V] |
| Forearm L | roll/twist | RB strong (24→6 self-check) | [V] roll-bleed |
| **Forearm R** | yaw | HL 27° + TW phase-match | **CONVERGE → confident mag** [V] |
| Forearm R | roll/twist | RB ≈ clean-yaw (projection only) | [V] |
| Forearm L/R | crease | CR+HV+de-twist: f4318 on-arc, de-twist worsens | **CONVERGE → skinning, not error** [V] |
| **Hand L** | yaw | direct heading-bin 25° (45–67% expl), phase≠forearm | confident own mag [V] |
| Hand R | yaw | direct 16° (17%) | real, weaker [~] |
| Hand L/R | twist | MP common-mode ±40° pronation | single lens, new [~] |
| **Upper-arm L/R** | yaw | HL 46°/42°, asymmetric | large but NOT cross-validated [~] |
| Upper-arm | abduction(tilt-lat) | MP abdR −50°±6.6 (tight); HG R-out asym | systematic; sensor-vs-stance ambiguous [O] |
| Upper-arm | elevation(tilt) | MP L ~24° below R (sign consistent all takes); HG | systematic; ambiguous [O] |
| Upper-arm | elbow-twist | MP R 25° > L 9° | moderate [~] |
| **Chest** | roll | small 2.6°/18°; **MP roll −2.0°±1.5** | MP+body-tilt CONVERGE; non-postural→sensor-leaning [V-small] |
| Chest | pitch | heading-dep 6.6°/68%; MP +5.6°±2.9 | systematic but postural (aramandi)→likely REAL [O] |
| **Pelvis** | roll | lateral 4.8°/36%; **MP roll −2.6°±1.2** | MP+chest+body-tilt CONVERGE; sensor-leaning [V-small] |
| Pelvis | (constant) | per-take constant, varies −12..+5° | calibration/re-strap, squarable per take [V] |
| Pelvis/chest | pitch | MP +6° | systematic, postural→likely REAL [O] |
| Blades L/R | roll | 4–6.5°/20% | weak/tentative [~] |

**Convergences (confident):** forearm yaw (2 lenses), f4318=skinning (3 lenses), core roll ~−2°
(mean-pose + body-tilt observation, non-postural).
**Systematic but sensor-vs-posture ambiguous:** right-arm higher+more-abducted (mean-pose + hang);
core forward pitch ~+6° (postural — likely real Bharatanatyam aramandi, NOT an error to chase).
**Single-lens / new:** hand common-mode wrist pronation ~40°; upper-arm yaw (uncross-validated).
**ONE suit on this dancer (confirmed by user).** So "session-consistent across the 11 takes" =
genuinely THIS suit's magnetization signature — pooling is clean (not averaging two sensors). Take-to-
take variation = choreography + per-take recalibration constants, NOT a suit difference. Per-take
multi-metric PCA confirms: PC1 35%/PC2 28% (diffuse, no tight clusters), PC1 driven by ARM POSE-
ASYMMETRY (dance character), not sensor-mounting → no hidden grouping, as expected for one suit.
(Separating this dancer's stance HABITS from sensor bias still needs another subject — but that's a
posture-vs-sensor question, not a two-suit question.)

### Crease + hinge incidence lenses (added) — they SEPARATE skinning from genuine error
- **Crease** (`crease_incidence.py`, roll-sensitive area metric, by sensor heading, pooled): PERVASIVE
  skinning baseline (forearms 47–96% creased at all headings) + heading structure (R-forearm worst
  +0…+120° near its mag heading; L-forearm worst −120°; R upper-arm least creased).
- **Hinge violations** (off-arc = impossible, by forearm heading, pooled): heading-localized, ~mirror
  L/R — R-elbow +0…+60°/+150°, L-elbow −180…−120°/−30°; sustained 3–14%.
- **CONVERGENCE/separation:** at R-forearm +90° crease is WORST but hinge violations LOWEST → creased
  *but on-arc* = **skinning**; at +0…+60° it's *off-arc* = **genuine error**, overlapping the mag
  heading. ⇒ use hinge-violation (off-arc) to flag the magnetization-linked genuine errors, and
  crease-without-violation to flag skinning. Cleaner than either alone.
- **Hinge-violation × wrist-twist co-occurrence:** PARTIAL coupling — pooled corr R +0.20 / L +0.24
  (weak), but wrist-twist excess is ~1.5–1.8× higher at elbow-off-arc frames (R 41 vs 27°, L 40 vs
  22°). ⇒ a SHARED forearm-sensor component (forearm error throws both elbow-hinge & wrist) PLUS a
  SEPARATE hand component (hand independently magnetized, distinct phase). The forearm↔wrist errors
  are a CHAIN, partly coupled: correcting the forearm partially relieves the wrist, but the hand
  needs its own correction. (Per-take corr spread −0.22..+0.66, no two-suit split.)
- **Solver twist-flip glitch scan** (`solver_twist_glitch.py`; user-observed: shoulders snap between
  bone-twist configs with no pose change = solver redistributing twist). RESULT: CLEAN in beta.npz —
  shoulder twist smooth (max 15°/frame, p99.9 7–8°, ~0% >10°), elbows a few fast frames (max ~33°)
  but isolated, no toggle, endpoint moving = real motion. ⇒ no glitch contaminating our beta.npz
  analysis. CAVEAT: beta.npz is 100 Hz and very smooth = likely the cadence-reconstructed/LERP'd
  version, which would SMOOTH single-frame solver flips → this scan can't rule the artifact out at the
  SOURCE. To confirm the solver behavior, need RAW Shadow output (pre-LERP), not beta.npz.
- **Static-pose twist-toggle scan** (`static_twist_toggle.py`): shoulder/elbow TWIST smooth (range
  1–7° in static windows, no toggles). [CORRECTED below — this was scanning the wrong sensors.]
- **SOLVER GLITCHES CONFIRMED & LOCATED (2026-06-02, user was right).** My earlier "clean" was wrong:
  I only scanned shoulder/elbow TWIST (which IS smooth). A FULL 37-sensor full-ORIENTATION scan
  (geodesic >20°/frame, single-frame) found 163 glitch frames (0.0035% of sensor-frames), concentrated
  DISTALLY: RightAnkle 55, LeftAnkle 54, RightWrist 19 (16 REVERTING=toggle), LeftKnee 19, RightKnee
  12, Tracker0 3, LeftWrist 1. SHOULDERS/UPPER-ARMS = ZERO (genuinely smooth). The glitches ARE in raw
  beta.npz (SMPL-pose jump magnitude == raw-quat jump magnitude exactly → in data, not conversion).
  Single-frame spikes; RIGHT-WRIST 16 reverts = the 'flip between two states in static pose' the user
  saw (distal sensors under-constrained → toggle, NOT the shoulders — user's location memory was off).
- IMPLICATIONS: (a) proximal arm/core/magnetization findings NOT contaminated (shoulders smooth);
  (b) right-wrist toggles DO affect wrist-level measures (common-mode ±40° pronation, forearm→wrist
  co-occurrence) — despike before trusting wrist detail; (c) FIX = despike (detect geodesic>~20°/frame
  single-frame spikes, SLERP-replace from neighbors) — removes all 163. New error class: distal-sensor
  single-frame solver glitches, separate from magnetization, rare, removable.
- METHOD LESSON: scan ALL sensors + FULL orientation (geodesic), not one DOF on a few sensors — the
  glitches were in feet/knees/wrist, invisible to a shoulder-twist scan.
