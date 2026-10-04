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

## DISPROVEN / dead-ends [X] — do not repeat

- **"Chest roll-bleed 68%"** — AXIS ERROR. Trunk-sensor local-x is LATERAL, not forward; the 68% was
  chest *pitch* (forward-lean, symmetric, possibly real posture). Real chest *roll* is small (2.6°,18%).
- **Core/trunk lateral roll as the hang cause** — lateral roll is small everywhere (≤6.5°, 14–36%);
  not the dramatic culprit.
- **"Constant mounting tilt offset" as the hang source** — ruled out: separate recalibration zeroes any
  constant offset.
- **Symmetry-based upper-arm fit** (auto-crossings `fit_upper_arm_from_crossings.py`, pooled
  `pooled_arm_fit.py`) — FAILED: this dance is rarely truly bilateral, crossings are impure (~28°
  median residual), fitting them wrecks the genuinely-clean windows. Symmetry is exhausted as a
  reference for this session.
- **Generic per-sensor roll assessment** (`assess_sensor_magnetization.py` all-sensor mode) — FAILED:
  degenerates for near-vertical bones (roll≈yaw) and is swamped by motion for mobile limbs (gave 146°+).
  Tilt-bleed is only cleanly measurable for near-upright trunk sensors.
- **VPoser as a correction objective** — plausibility ≠ correctness; rewards plausible-but-wrong poses.

## OPEN [O] — need video or more data

- **Elevation half of the hang (~26°, right arm too high):** gravity-referenced, so not yaw-mag. Real
  asymmetric posture vs tilt-bleed — **VIDEO adjudicates** (is the relaxed right arm really raised?).
- **Does any properly-combined heading-dependent correction improve the *rendered* defect?** Never
  cleanly demonstrated (yaw and twist fixes did not).

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
