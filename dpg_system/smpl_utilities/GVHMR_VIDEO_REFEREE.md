# GVHMR video referee for Shadow arms

> **How-to for correcting a take (process and palette of techniques, from Subject 11 take 8): `SHADOW_CORRECTION_PROCESS.md`.**

Working notes, started 2026-10-30. Single-camera video → SMPL (GVHMR) set up on the M1 Max and used
as an independent check on Shadow data, first on Subject 7 jathiswaram. Related ledger:
`MAGNETIZATION_FINDINGS.md` (this bears on its OPEN item "elevation half of the hang … VIDEO
adjudicates").

---

## Result in one paragraph

Video and Shadow agree on timing (Shadow time = video time − 0.10 s, no drift over 6.8 min), body
heading (within ±6° all take after one fixed rotation between the two world frames) and legs (in the
normal range for single-camera estimates). They disagree on the arms. Measured against the arm
positions detected in the video, **Shadow's right upper arm reads 36° higher than the video shows**
(median over all usable frames; middle half +21° to +48°), while the left upper arm is unbiased
(+1°). It is one-sided and on the right only, so it is a real Shadow error, not a quirk of the
method. An old manual arm fix (`beta/…_armfix.npz`) removes most of it (+36° → +5°) but makes the
forearms worse.

---

## Setup (where things are)

- GVHMR clone: `~/Projects/GVHMR`, conda env `gvhmr` (separate; `dpg_system_2025` untouched)
- Weights: `~/Projects/GVHMR/inputs/checkpoints/` (SMPL linked from `~/Dev/smpl/models`; SMPL-X
  from the MPI site, only `SMPLX_NEUTRAL.npz` is used — the 1.6 GB of `.pkl` beside it can go)
- Mac patches: `hmr4d/utils/mac_compat.py` (CUDA calls → Apple GPU, CPU fallback for missing ops),
  one `.contiguous()` in `hmr4d/network/hmr2/hmr2.py`
- Driver: `tools/demo/demo_mac.py` (static camera only)
- Skeleton overlay video: `tools/demo/overlay_mac.py`
- Comparison scripts: `tools/shadow_compare/` (below)
- Jathiswaram run: `~/Projects/GVHMR/outputs/mac/take10_full/`
- Pictures referred to here: `gvhmr_video_referee/` beside this file

### Running a new take

```
conda activate gvhmr
cd ~/Projects/GVHMR
python -W ignore tools/demo/demo_mac.py --video TAKE.mov --name NAME --target_h 440 --target_x 850
```

`--target_h` / `--target_x` are the dancer's typical box height and horizontal centre in pixels
(440 / 850 for this camera position). They steer person selection; look at one frame first.

Outputs in `outputs/mac/NAME/`:
- `NAME_gvhmr.npz` — `poses` (22 joints, axis-angle), `trans`, `betas`, `reliable` mask;
  30 fps, **y up** (GVHMR's world frame), SMPL-X body (shares SMPL-H's first 22 joints)
- `hmr4d_results.pt` — GVHMR's full output, including in-camera parameters
- `features.pt` — ViTPose 2D keypoints + image features; `bbx.pt`, `detections.pt` — tracking

Timing for the 6.8-min 1080p take: person detection + keypoints + image features 30 min
(≈7 fps), GVHMR itself 1 min 50 s.

### Things that went wrong and how they are handled

- **Several people in shot.** The studio videos have a second suited woman (left edge, walks in
  front of the camera around 4:29–5:10), a man in a sweater at the right, and the projected avatars
  on the screen get detected as small people. The stock tracker (largest YOLO track) and a greedy
  frame-to-frame follower both jumped to other people. Fix: pick one detection per frame by the
  cheapest path through the whole take, with soft costs on height, touching the frame edge,
  horizontal position and frame-to-frame jumps, plus a "not visible" option. Frames where she is not
  found or another box overlaps hers are flagged `reliable = False` (10% of this take, mostly the
  4:29–5:10 break). Picture: `gvhmr_video_referee/viterbi_check.jpg`.
- **Long takes.** GVHMR only looks 120 frames around each frame but builds a full
  frame-count × frame-count attention table; the whole take ran over 10 min and was stopped.
  `demo_mac.py` runs 1000-frame windows with 200 overlap and stitches them (each window's world
  frame turned about vertical and shifted to match over the overlap). Joins are clean except one
  inside the 4:29–5:10 break.
- **Memory.** The stock demo loads the whole video (~30 GB for this take); the driver decodes once,
  in chunks.
- **Frame rate.** The stock demo labels every video 30 fps without resampling. This video is
  29.97 — negligible, but the comparison uses the true rate.
- **Mesh rendering** on the CPU at 1080p is unusably slow; use `overlay_mac.py` (skeleton only).
- **Focal length is guessed** from the frame size. It mostly affects distance from the camera and
  root travel, not joint angles. If the lens is known, pass `--f_mm`.

---

## The comparison (jathiswaram_a vs `Subject7_8_take10.mov`)

Shadow file: `Subject7_take_jathiswaram_a_beta_smpl_poses_rhipflexstat.npz` (100 Hz, 40,919
frames = 409.19 s; the video is 12,268 frames = 409.34 s). The hip fix doesn't touch the arms;
the plain `smpl_bare` conversion gives identical arm numbers.

### Timing
Cross-correlating total joint rotation speed: Shadow time = video time − 0.10 s, correlation 0.85
(next-best peak 0.59). Per-minute estimates −0.096 s for five minutes, −0.11 / −0.106 s in the last
two. No meaningful drift.

### Agreement
- Body heading: within ±6° of Shadow in every 30 s, after one fixed rotation between frames.
- Legs and head, pelvis-relative joint positions on a common neutral body, after best rotation per
  frame: knees 35–54 mm, ankles 57–67 mm, head 38 mm.
- Travel area similar (GVHMR 3.0 × 3.1 m, Shadow 2.9 × 3.4 m). Root position along the floor is
  off by 56 cm median — distance from the camera is the weak point of single-camera estimates.
- Left/right are not swapped (a mirrored Shadow fits worse, clearly in the legs).

### Arms
1. **Upper-arm twist convention.** Both shoulders differ by a steady 45–51° roll about the upper
   arm. Roll barely moves the arm, so this is how the two divide upper-arm twist, not a visible
   error.
2. **Right upper arm too high.** The referee: put Shadow's body pose on GVHMR's root and camera,
   project both skeletons into the image, and compare the direction of each upper arm and forearm
   with ViTPose's detected shoulder/elbow/wrist (confidence > 0.8, reliable frames only).
   Picture: `gvhmr_video_referee/arm_referee.jpg` (red = Shadow, green = GVHMR, dots = ViTPose).

| Arm segment height vs the video (+ = higher) | Shadow | GVHMR |
|---|---|---|
| Left upper arm | +1.2° | −1.0° |
| Right upper arm | **+35.9°** (middle half +21..+48) | −0.1° |
| Left forearm | +11.3° | +4.1° |
| Right forearm | +10.6° | +0.1° |

### Old manual arm fix (`beta/…_armfix.npz`)

These `armfix*` files are **old manual attempts** to fix the arms (`correct_upper_arm_offset.py`
dials tuned by eye against video, `ARM_FIX_TUNING.md`), not the current approach
(`shadow_mag_correct`). Plain `armfix.npz` was written last and differs from the
`tw-60_abl+12_abr+5_el+28_wr+15` variant only at the wrists, so it is taken as that recipe.
Converted with `shadow_to_smpl.py` (the unfixed file converts back to `smpl_bare` exactly).

| Segment | Height vs video, before → after | Direction error (median), before → after |
|---|---|---|
| Right upper arm | +35.9° → **+5.4°** | 36.6° → **11.9°** |
| Left upper arm | +1.2° → −1.7° | 10.9° → 8.5° |
| Right forearm | +10.6° → −3.8° (scattered −25..+15) | 13.8° → **21.4°** |
| Left forearm | +11.3° → **−10.4°** | 13.6° → **17.0°** |

Also: the upper-arm twist difference against GVHMR fell from 45–51° to 16–18° (the fix's −60°
twist matches the video model's convention), and the right shoulder keeps an ~18° steady
forward/back offset. The forearms got worse; the elbow +28 opening may be too much.

---

## Caveats

- **The referee favours GVHMR.** GVHMR is built on the same kind of 2D keypoints it is judged
  against. A large one-sided miss by Shadow is still a real error; a small Shadow miss does not
  prove Shadow right in depth.
- **"Height" is the segment's angle above horizontal in the image, ignoring left/right.** For a
  hanging arm, a sideways turn either way reads as "higher". So a whole-arm bank like the hang
  bank (both arms turned toward the dancer's right) would read as "higher" on BOTH arms when they
  hang. The +36° / +1° split is pooled over all arm poses, so it does not by itself separate the
  hang bank from a right-only tilt.
- Shadow's arms are drawn from GVHMR's root but Shadow's own spine and collar chain, so spine or
  shoulder-blade differences feed into the arm numbers (the left arm being unbiased says this is
  small).
- Projection angles depend on which way she faces the camera; the numbers are medians over all
  facings.

---

## Next steps

1. **Split the arm referee by arm height** (low / near level / raised), with a SIGNED sideways
   measure (toward or away from the body) instead of the left/right-blind height. That is what
   answers the ledger's open item "is the relaxed right arm really raised?" and separates the hang
   bank (both arms, toward the dancer's right, only when lowered) from a right-only tilt that grows
   with arm height (the ledger's "right upper-arm roll error grows with arm height").
2. **Check the current correction, not the old manual one:** export jathiswaram through
   `shadow_mag_correct` (hang 'fixed bank' −17.9, turn, learned terms), convert to SMPL, run
   `arm_referee_multi.py` on it beside the raw file.
3. **Pending video checks in the ledger** that this could now measure: the T-pose elbow turn
   (predicted inside-of-elbow turned up ~40° both arms in jathiswaram) and straight-arm sections
   (turn fade 30 vs 0). Forearm roll is hard to see in 2D keypoints alone — may need the HMR2/GVHMR
   mesh or a closer look at the hands.
4. Other Subject 7 takes with video: same pipeline, look at one frame to set `--target_h/x`.

---

## 2026-10-30 (later): measure first — which frame is the error fixed in?

Script `tools/shadow_compare/sensor_error_landscape.py OUT_DIR SHADOW_SMPL_NPZ [SAVE_NPZ]`: per segment, the
3D world bone direction of Shadow vs the video, split into forward / toward the dancer's left / up (dancer's
heading frame) and into camera depth vs image plane, gated to frames where GVHMR's projected segment is within
12° of ViTPose. Binned by true (video) elevation, facing, minute, speed. Frame test: scratchpad
`video_gt/frame_test.py`, `own_frame_detail.py` (fit on one half of the take, test on the other).

- **Noise floor** of the method ≈ 6–9° (left upper arm, spine, shoulder line, thighs).
- **Right upper arm is the outlier**: 31–42° off when hanging / low / level, mostly in the image plane (not a
  depth artefact). Hanging: swung OUTWARD 26° and up 13°. **Left upper arm hanging: 5.8° — correct.** So in this
  take the "hang bank" is the right arm alone, not both arms leaning; a symmetric (pair) correction moves a
  correct left arm.
- **Which frame the error is constant in** (held-out median error, right upper arm): none 33–34, constant in
  world 18–19, in chest 15, in parent (collar) 14, **in the segment's own (sensor) frame 10–12** — about the
  left arm's level. A world-fixed (magnetic field / gravity) error explains it worst.
- The own-frame offset is **32°, steady every minute (31–33°, within 0.7–5° of the whole-take value)**, mostly a
  rotation about the T-pose vertical: at a true T-pose Shadow would show the right arm ~30° behind it. Fixes the
  hanging arm (31 → 6.7°), low/level (42/34 → 15/14), but not overhead (16 → 20, n = 398) — a second, smaller
  effect there.
- Left forearm: own-frame offset 14° (11 → per-minute 10–19°, less steady); right forearm ~10°, not explained
  by any constant.
- Facing could not be tested: she faces within ±45° of the camera in >90% of gated frames.

Reading: a constant offset in the sensor's own frame is what a calibration-time error leaves behind (posture
at the T-pose, or a wrong heading of that sensor at the moment of calibration), or a strap rotated before the
take. Not a field that changes as she moves. Test: the other Subject 7 takes with video (`Subject7_8_take1..10.mov`
beside the Shadow files) — a calibration-time cause should give a DIFFERENT right-arm offset per calibration;
mounting/body would give the same one.

### All ten Subject 7 takes with video (overnight 2026-10-30/31)

Batch driver, logs and per-take outputs: scratchpad `video_gt/` (`run_all.sh`, `session_table.py` →
`session_table.txt`, `tpose_proximity.py`, `start_pose.py`). Video ↔ Shadow pairing by length, confirmed by
sync correlation (takes 4/5 were ambiguous by length: 4 = sarukkal 0.86 vs 0.44, 5 = pachaladavu 0.88 vs 0.42).
Video takes 1–10 = thatabavu, naatadavu, kuthitha metadavu, sarukkal, pachaladavu, kuthadavu, theyatheiyi,
mandiadavu, padam, jathiswaram. Sync 0.57–0.88. Take 8 (mandiadavu, deep squats) is unreliable (spine 34°).
Take 1 (thatabavu) is hands-on-hips nearly throughout, so a constant fits it trivially.

Right upper arm own-frame offset, session order: 36.7, 28.3, 28.6, 36.1, 30.6, 32.2, 23.7, (37.8), 30.6, 32.1°
— **the same ~31° (±4°) in every take**, always mostly the turn about the T-pose vertical (rotvec y +20..+37,
x −4..−6, z +3..+24), steady within each take (halves 2–7° apart in the long takes). The left upper arm in the
same takes scatters 2–19°, so take-to-take differences of ±4° on the right are within the measuring noise.
Constant correction brings the right arm from 27–39° to 9–13° (left arm's own level 8–16°).

Error vs how far the TRUE arm is from T-pose (chest frame, pooled, take 8 out): right arm 31° at 0–15° from
T-pose, 31–39° everywhere out to 90°. **The error is already there at T-pose**, so it is not something that
grows as the arm leaves the calibration pose (soft tissue, cable pull). Shadow draws the right arm behind and
above the truth. No take starts in a T-pose (calibration is not in the recordings), so the calibration posture
itself cannot be checked on this video.

Reading: the right upper arm's sensor-to-segment offset is wrong by a near-identical amount at every
calibration of the day. A random calibration error would differ take to take, and a field that varies as she
moves would not be constant in the sensor frame. What stays: something that repeats identically at each
calibration (her habitual right-arm T-pose; the right upper-arm sensor's heading consistently disturbed at the
calibration spot or by the sensor itself — this session predates the 2026-06-17 recal), or the sensor settling
into the same shifted position right after each calibration. Practical consequence regardless of cause: for
this session, one fixed own-frame rotation of the right upper arm (~31°, rotvec ≈ [−5, +27, +14]° in the rest
frame) is supported by video; symmetric / hang corrections are not.

**Correction (David, 2026-10-31): in this session the T-pose was done facing ~90–100° counterclockwise (compass)
from the dance direction**, to register the Vive tracker. So "constant in the sensor frame" no longer rules out a
heading-dependent (magnetic) error: one locked in at the calibration facing would look constant while she dances
facing one way. Test (`facing_offset.py`, `calib_facing_tpose.py`, takes 2–7, 9, 10; facing + = turned
counterclockwise seen from above, 0 = camera ≈ dance front):
- Right-arm own-frame offset by facing: −90° 36.8, −60° 32.3, −30° 31.7, 0° 30.4, +30° 29.8, +60° 28.6,
  +90° 30.2 (179 frames, 5 takes), +120° 33.6, +150° 14.0 (take 9 only). Mostly flat, with a mild trend
  (~8° from −90° to +60°) in the direction of the calibration facing.
- Near-T-pose arm at each facing: −90° 45°, −30° 33°, +30° 28°, +60° 28°, +120° 16–31°, +150° 15–19°
  (mostly take 9); almost no near-T-pose frames at +90°.
- A heading error locked in at calibration predicts the offset nearly vanishes back at the calibration facing
  (+90..+100). At +90 it is still ~30° (thin data), so that is not supported; but coverage of that facing is too
  thin to rule it out, and the drop at +120..+150 is one take. **Undecided.** Decisive studio test: T-poses (and
  relaxed hanging arms) at the calibration facing and at the dance facing, filmed, right after calibrating.

David (2026-10-31): the calibration is filmed (side-on, unclear) and shows no consistent arm error; a 30°
posture error would have been caught; a blade sensor pulled by its wires stays rotated rather than returning,
so it gets fixed at every calibration. **His conclusion: the rotated calibration facing is the source** (a
heading error of the right upper-arm sensor locked in at that facing). The blade is not implicated by the
video either: an error constant in the collar frame fits worse (14°) than one constant in the arm's own frame
(10–12°), and the shoulder line is within 6°.

Is the offset the same in every pose? No (`offset_by_pose.py`, pooled takes 2–7, 9, 10, fitted separately per
pose group). Right upper arm, heading part (rotation about the T-pose vertical) by true arm elevation:
hanging 29.7, −60..−35° 28.2, −35..−10° 27.9, level 24.9, +15..+40° 22.7, overhead 22.2 — ~8° smaller with the
arm raised. The tilt part moves more: +7.7, +12.1, +20.0, +4.8, −2.9, +0.3. Side vs front at the same height:
identical (28.3 vs 27.8). Total offset 31 hanging → 22 overhead. A posture or mounting offset would be the same
in every pose; a magnetic heading error depends on how the sensor sits in the field, so a pose-dependent size
fits the calibration-facing explanation. Left upper arm: 0–6° up to −10°, 11–15° at and above level (partly the
video's depth weakness for horizontal arms).

### Hard-iron fit (2026-10-31, scratchpad `video_gt/hard_iron_fit.py`, `hard_iron_R.txt`, `hard_iron_L.txt`)

Model: bias b fixed on the upper-arm sensor (3), earth field heading (1; 54 µT, dip 70°), calibration facing
(1); heading error from the horizontal part of H + S·b; error at the calibration T-pose locked into the offset.
Held-out (fit one half of every take, test the other), right upper arm, median direction error:
none 32.5/33.0, constant 13.9/13.2, hard-iron 12.5/11.1, hard-iron + constant 11.8/10.4. Hard-iron handles
raised arms better (40..91°: 9.4/10.5 vs 11.7/14.2).
- **Robust part:** a locked heading error E ≈ +31° on the right in both halves; the left gets E ≈ 0.
- **Not robust:** the needed bias is ~10 µT (17–19% of field); bias sign and field heading flip between halves;
  calibration facing is unidentifiable (+45..+115° fit equally). The LEFT control arm also gets a 6–7 µT bias
  for a 0.1–0.8° gain, so the model is flexible enough to invent ~6 µT from noise.
- **Session date correction:** the videos' creation time (2026-05-28) is the DaVinci export date and the file
  mtimes are copy dates. Raw Shadow files are dated 2025-11-07 and the hang-bank ledger lists S1–S7 as
  2025-06-19 … 11-07 (old fusion), so the session is ~2025-11-07 — about 7 months BEFORE the 2026-06-01 sweep,
  not 4 days. The sweep's clean right upper-arm sensors (0.81 / 0.52 / 0.23 µT on suits 1–3) therefore do NOT
  rule out a magnetized sensor in Nov 2025; the same sweep found 5 sensors at 9–18 µT, so ~10 µT is within what
  happens. Source of the locked error (sensor bias vs a field near the right arm at the calibration spot)
  stays OPEN; the fit's pose-dependence evidence for a bias is weak (left-arm control).
- Shirt-pocket lens (`shoulder_pocket.py`): chest→shoulder error grows with that arm raised (L 8→12→16°,
  R 9–10→15° overhead), shoulder line flat 5–8°, spine flat. Fits pockets loosening with raised arms, but the
  video's shoulder keypoint also weakens with raised arms; and the upper arm's world direction comes from its own
  strapped sensor, so this does not feed the arm numbers.

### Second performer, same calibration, different suit (2026-10-31) — the offset is the CALIBRATION, not a sensor

`second/` holds the other suited performer (`_b`, the taller woman standing centre in video takes 1–8; David:
calibrated together with Subject 7, different suits). Most `_b` files are raw `quaternions` (no positions) —
same layout as `quats`, converted with zero positions (scratchpad `video_gt/second_shadow` → `second_smpl`).
GVHMR on the centre figure (`run_b.sh`, outputs `S8_takeN`). She barely moves, so motion sync is weak
(0.39–0.55); her own sync agrees with Subject 7's within 0.01–0.09 s, and Subject 7's was used. Tracking
confirmed by spine 2–6° and shoulder line ~4°. Table: `session_table_b.txt`.

| | Subject 7 (suit A), 9 takes | Second performer (suit B), takes 1–8 |
|---|---|---|
| R upper arm offset | 24–37° (≈31), rotvec y +20..+37 | **29–38° (≈35), rotvec y +28..+37** |
| L upper arm | 3–11° (mostly near noise) | 10–17°, rotvec y +6..+12 (same sign) |
| R forearm | 3–24°, inconsistent | 27–36°, rotvec y +21..+35 |
| L forearm | 4–26° | 13–30°, mostly rotvec z (different axis) |

Two different suits, calibrated together at the rotated facing, both show the right upper arm turned ~30–35°
about the T-pose vertical in the SAME direction. A magnetized sensor on one suit cannot do that; the cause is
common to the calibration. Error is largest on the right arm (and the second performer's right forearm), smaller
on her left upper arm with the same sign. Candidates: a local field at the calibration spot on the side the right
arms pointed to (nothing visible — building steel/wiring/floor would not be), or something in the procedure at
that facing (but a procedural heading error would hit all sensors alike, not one arm). Caveat: the second
performer holds nearly one pose throughout (few gated right-arm frames, 83–550), so for her a constant fits
trivially and own-frame vs world cannot be separated — the SIZE of her error against the video is what counts.

---

## 2026-10-31: fixed rotations applied, video comparison re-run

Correction: each affected segment's world orientation becomes Seg . C^-1, with C one rotation in the segment's
own rest frame fitted on all takes of that performer; every child keeps its world orientation (child local
becomes C . local). C is restricted to axes perpendicular to the bone ("swing"): a turn about the bone itself
does not move the bone direction, so the video cannot see it (a free 3-axis fit returns arbitrary 50-240 deg
turns about the bone). Swing beat the T-pose-vertical-only rotation on held-out halves (S7 R upper arm
13.9/13.2 vs 17.4/15.7), so it was used.

Corrections (rotation vector in the rest frame, deg):
- Subject 7, R upper arm [+4.7 -27.9 -9.9]  (left arm and forearms not changed: fits gain little)
- second performer, R upper arm [+4.1 -34.7 +5.7]
- second performer, L upper arm [-4.0 -8.7 -9.4]
- second performer, R forearm [-0.2 -28.8 +11.1]  (L forearm not changed: halves disagree, 15 vs 18)

Re-run (median error vs video, raw -> corrected, every take improves):
- Subject 7 R upper arm: 34.0 -> 13.1 overall; per take 27-39 -> 11-16.
  By true elevation: hanging 31.9 -> 9.4, low 37.2 -> 14.5, level 33.2 -> 14.1, raised 23.8 -> 15.0.
  The raised arm is left ~6 deg worse than hanging: the fixed rotation slightly over-corrects there,
  as the smaller raised-arm offset predicted.
- second performer R upper arm 36.2 -> 7.1; L upper arm 14.1 -> 4.5; R forearm 32.2 -> 12.0.
- Spine, shoulder line and the uncorrected segments are untouched, as they should be.

Candy-wrapper shoulders (twist about the bone, swing-twist of the joint's local rotation; video-trusted frames):
- Subject 7 shoulder twist L +59 / R -48 raw, video +12 / -5. After correction: L +59 / R -52, unchanged.
  The jathiswaram opening hold (frames 0-800) is +104 / -103 raw and +104 / -105 corrected.
- Second performer shoulder twist L +33 / R -40 raw, video +16 / -15; corrected +33 / -30.
- So the correction does not fix the candy-wrapper shoulders, and cannot: it is twist-free at the shoulder by
  construction, and the shoulder twist is mirror-symmetric (left +, right -), unlike the one-sided heading
  error. It belongs with the symmetric T-pose elbow-turn offset (`elbow_turn`, MAGNETIZATION_FINDINGS.md).
- Elbow twist does change, because the forearm keeps its world orientation under a turned upper arm:
  Subject 7 R +42 -> +8 (video -3), opening hold R +74 -> +17; second performer R +11 -> -10 (video 0),
  L -2 -> -19 (video -18). The right elbow / forearm wrap should look better.
- The elbow stays far from a pure hinge (rotation left after the best turn about rest vertical): 47-66 deg in
  Shadow, raw or corrected, vs 12-14 in the video. That is the upper-arm roll error, still uncorrected.
- Caveat: the video's twist about the bone comes from GVHMR's learned prior and appearance, not from the
  keypoints, so it is a plausible reference, not a measurement.

Files (scratchpad video_gt/fixed/): fit_fixed.py, apply_fixed.py, run_fixed.sh, compare.py, candy.py,
fit_S7.json, fit_S8.json, corrected npz in smpl_a/ and smpl_b/, session_table_C7.txt, session_table_C8.txt,
candy_S7.txt, candy_S8.txt.

## 2026-10-31: elbow_turn checked against the video

The turn correction rolls the upper arm about its own bone with the forearm held, so it moves no joint and the
keypoints cannot see it. What the video can check is the reading it rests on. Same measure as elbow_turn
(forearm azimuth around the upper-arm bone, zero = elbow bending forward), three ways on the same frames:
shadow = Shadow forearm in Shadow's upper-arm frame (what elbow_turn sees); video = the video's forearm in
Shadow's upper-arm frame swung (no roll) onto the video's upper-arm direction, which is the upper-arm sensor's
roll error if the real elbow is a forward hinge; gvhmr = GVHMR's own arm (its prior, near 0 expected).
The SMPL-file version of the measure reproduces the node on raw quats in sign and size (jathiswaram 36/46 vs
node 43/42; padam 38/54 vs 45/53; differences come from frame selection).

Subject 7, pooled over takes, by video elbow bend (deg; L then R):
- bend 20-45: shadow +41 / +43, video +34 / +54, gvhmr -6 / +5
- bend 45-80 (elbow_turn's band): shadow +40 / +50, video +38 / +48, gvhmr -5 / +1
- bend 80-120: shadow +16 / +58, video +31 / +34, gvhmr -24 / -19
- bend 120-180: shadow +6 / +72, video +22 / +40, gvhmr -25 / -10
Second performer, bend 45-80: shadow +23 / +39, video +31 / +32 (her suit: same turn, both arms).

Reading:
- CONFIRMED: the turn is real. In elbow_turn's band the video-implied roll error matches the Shadow reading
  within 2-3 deg for Subject 7, and the second performer has a similar turn (~30 both arms).
- The true roll error stays roughly level with bend (video ~30-50); Shadow's own reading drifts with deep bend
  (R climbs to +72, L falls to +6). The bend-growth noted earlier (R 39 -> 76) is in the forearm sensor's
  direction, not in the upper-arm roll, so the deep-bend readings and a bend-dependent roll term should not
  be trusted.
- Per take, Shadow and video-implied values do not track each other (n 20-500 frames per take; e.g. L take 6
  shadow +51 vs video +20). The per-take steps elbow_turn reports are not confirmed; a session-constant turn
  is as well supported.
- Shoulder twist after undoing each take's measured turn (median; video in brackets):
  Subject 7 bent elbows L +63 -> +20 (+14), R -53 -> 0 (-10); straight arms L -2 -> -45 (-14, only 133
  frames), R -33 -> +21 (+16). Second performer bent L +33 -> +18 (+17), R -38 -> +6 (-15); straight R
  -43 -> -7 (-15). Undoing the turn is most of the candy-wrapper shoulder fix. Straight arms mostly favour
  applying it there too (no fade), but GVHMR's straight-arm twist is prior, not measurement.
- Possible cause for both performers: a T-pose held palms forward rather than palms down turns the inside of
  the elbow up, which is the sign seen. Untested.

Files (scratchpad video_gt/fixed/): turn_video.py, turn_video_S7.txt, turn_video_C7.txt (R upper arm
corrected; the video-implied numbers barely change), turn_video_S8.txt.

## 2026-10-31: one turn for the whole session vs one per take

Per take = elbow_turn's reading of that take (Shadow only). Session = the mean of the OTHER takes' readings,
each take counted once (pooling frames let take 1, hands on hips with thousands of frames, set the value).
Score: bent-elbow (>= 45) shoulder twist after undoing the turn, median |difference| from the video's twist.

| arm | no turn | per take | session (Shadow) | session (video-implied) |
|---|---|---|---|---|
| Subject 7 L | 48.5 | 13.1 | 15.6 | 13.0 |
| Subject 7 L, without take 1 | 48.4 | 13.0 | 12.8 | 14.6 |
| Subject 7 R | 45.3 | 15.1 | 13.2 | 10.9 |
| Subject 7 R, without take 1 | 42.7 | 12.8 | 11.3 | 9.9 |
| second performer L | 17.9 | 5.2 | 8.7 | 7.9 |
| second performer R | 20.6 | 19.5 | 16.4 | 9.8 |

- A session-wide turn from Shadow alone does as well as per take or better on three of four arms, by 1-3 deg
  (inside the noise). The exception is the second performer's left arm (5.2 vs 8.7), whose per-take readings
  are small and spread 5-20.
- The per-take reading fails when a take has few frames in the band: Subject 7 take 1 left reads +8.5 against a
  video-implied +56; the session value avoids that.
- Session values (all takes, each once): Subject 7 L +37 (+40 without take 1), R +49; second performer L +13,
  R +37. Video-implied: Subject 7 L +40, R +43 (+46 without take 1); second performer L +20, R +31.
- Straight arms (Subject 7 R, 2900 frames): no turn 42, per take 23, session 17. Left straight arms too few.
- So the recalibration steps between takes are not visible in the video; one turn per arm per session is enough.

Files: scratchpad video_gt/fixed/session_turn.py, session_turn_S7.txt, session_turn_S7_no1.txt, session_turn_S8.txt.

## 2026-10-31: finished corrected files and the process

Files: `Subject7_Bharathanatyam/video_corrected/subject7/*_vc.npz` and `video_corrected/second/*_vc.npz`, one per
SMPL take, same layout as the originals (play with smpl_take). Each carries `_video_correction_*` keys
recording what was applied. Scripts and text results copied to `~/Projects/GVHMR/tools/shadow_compare/video_fix/`
(they expect the scratchpad layout for the per-take comparison npz files, which were not copied for lack of disk).

To see the correction: `gvhmr_video_referee/shadow_video_correction_compare.json` (copy of
`patches/shadow_video_correction_compare.json`) plays jathiswaram before the arm correction (left: the
`_rhipflexstat_stomp` file, hip fix and stomp smoothing) beside the `_vc` file (right), so the arms are the
only difference, from one frame slider; mgl_rotate's y turns both. Look at the opening hands-on-waist hold (frames 0-800).
For another take, change both smpl_take paths and the slider and counter maximums to its frame count.

Process, per session:
1. GVHMR on each take's video, then sensor_error_landscape.py (as in run_all.sh / run_b.sh).
2. fit_fixed.py: one swing per arm segment per performer (no turn about the bone), held-out halves decide
   which segments to correct.
3. build_final.py: apply the swing (children keep their world orientation), then one session turn per arm,
   undone as an upper-arm roll about its bone with the forearm held.
   - Turn = mean of per-take elbow_turn readings, each take once, measured on the swung files.
   - Exception: where a swing moved the upper arm a lot (Subject 7 R, 28 deg), Shadow's own reading after the
     swing (+68) mixes in the forearm sensor's direction error. The video-implied value (+46) scored better
     (bent-elbow shoulder twist vs video: 17.6 at +68, 13.4 at +56, 11.4 at +46) and was used.
   - Used: Subject 7 L +44.8, R +46; second performer L +18.1, R +29.2.
4. Check: the finished files read back near 0 on elbow_turn (most takes within 10). Bent-elbow shoulder twist
   vs video: Subject 7 L 10.3, R 11.4 (raw 48.5 / 45.3); second performer L 5.9, R 17.3 (raw 17.9 / 20.6).
   Her right arm keeps a 17 deg disagreement between the hinge-implied roll and GVHMR's twist.
Without video, step 2 is unavailable; step 3's session turn can still come from Shadow alone.
Not yet done: shadow_mag_correct's learned weights were trained on uncorrected takes (its 'hang' absorbed the
right-arm offset); they would need retraining on the _vc files, without the elbow-bend roll feature.

## 2026-10-31: root-position jolt at right stomps (Subject 7)

The root position comes from the position sensor on the right thigh. Stomps found from the pose alone (571 right,
532 left over the 10 takes): after a right stomp the fast (> ~15 Hz) part of the root position is 2.7x its
pre-stomp level (2.76 vs 1.05 mm RMS); after a left stomp it is unchanged. Each thigh's own rotation shakes equally
on its own stomp (2.2x / 2.1x), so this is the sensor bouncing on its strap, not the leg. Small (~3 mm; the slower
quarter-second swing 2.0-2.3 cm vs 1.5 before). The hip fix (correct_thigh_offset_flexion.py, a slow
angle-following mount error smoothed over 0.5 s) does not touch it.

Fix: `smpl_utilities/smooth_stomp_jolt.py` (after the hip fix; root position only). A soft weight, 1 from 20 ms
before to 180 ms after each sensor-leg strike with 40 ms fades, blends the root position toward a 30 ms Gaussian
smoothing. Result: fast shake after right stomps 2.76 -> 0.66 mm, quarter-second swing 2.0/2.0/2.3 -> 1.3/1.2/1.5 cm;
the planted-foot slide (4-5 cm on both sides, a separate problem) is unchanged. Files: `*_rhipflexstat_stomp.npz`
beside the hip-fixed ones; the `video_corrected/subject7/*_vc.npz` files now carry the same root position
(`_stomp_smooth_*` keys). Pipeline per take: hip fix -> stomp smoothing -> video arm correction (poses only).

## 2026-11-01: which body model draws Shadow's rotations

The converter copies Shadow's joint rotations straight into SMPL, so the drawn bone directions depend on the model's
rest pose. Shadow's own skeleton has the upper arms 3 deg below level, pointing straight out; SMPL-H (what
dpg_system draws with) is close to that. GVHMR's neutral SMPL-X is lopsided: left upper arm 15.6 deg below level and
9 deg back, right 7.6 deg below. All the video comparisons above drew Shadow on the neutral SMPL-X model.

Re-measured (Subject 7, 8 takes, upper-arm direction error vs video, median):
- right arm, raw: 32.8 (neutral) / 29.7 (Shadow skeleton) / 33.6 (SMPL-H); corrected: 13.6 / 15.8 / 14.2.
  The right-arm offset and its correction stand whatever the model.
- left arm, raw: 13.2 / 15.9 / 14.7. Small either way; the left-arm conclusions stand.
- But the turn had been rolled about the NEUTRAL bone, so drawn with SMPL-H it moved the left upper arm
  (corrected left 17.1 vs raw 14.7). Fixed 2026-11-01: the turn is re-applied about the SMPL-H bone (file gender and
  betas); the drawn left upper arm is now exactly the raw one, and corrected left = raw left = 14.7.
  Files carry `_video_correction_turn_axis`. build_final.py now does this itself; reaxis_turn.py and axis_check.py
  are in video_fix/.
- Not redone: the second performer's left upper-arm swing (10 deg) was fitted with Shadow on the lopsided
  neutral left arm and may be partly that model's rest pose. For later sessions, the fitting step should draw
  Shadow with SMPL-H (or Shadow's skeleton) and the video with its own neutral SMPL-X.

## 2026-11-01: Shadow-only screening of the old-fusion sessions (S1-S7)

`video_fix/screen_sessions.py` on the hip-fixed files (results `screen_old_fusion.txt`). Measures, per session:

| session | stomp shake L/R | elbow turn L / R (sd over takes) | hang bank | right reads higher at level |
|---|---|---|---|---|
| S1 | 0.90 / 1.33 | +19 (4) / +42 (11) | -4.6 | +0.4 |
| S2 | 1.00 / 1.07 | +43 (6) / +32 (9) | -7.9 | +2.1 |
| S3 | 0.92 / 1.54 | +33 (5) / +41 (12) | -3.2 | +1.0 |
| S4 | 1.34 / 2.91 | +11 (7) / +26 (6) | -2.1 | -2.2 |
| S5 | 0.99 / 1.26 | +5 (9) / +21 (5) | +0.8 | -2.4 |
| S6 | 1.17 / 1.70 | +28 (8) / +29 (7) | -2.4 | +4.5 |
| S7 | 1.27 / 2.42 | +47 (10) / +56 (7) | -16.3 | +10.1 |

- Stomp shake (fast root shake after / before each leg's stomps): the right thigh carries the position sensor in
  every session; S2 shows no shake on either side.
- Elbow turn from Shadow alone: positive on both arms in every session, steady over takes. For S7 this reading
  matched the video within 2-3 deg on arms that needed no big swing, so it is a usable provisional correction.
- One-sided right upper-arm offset like S7's: the lens that moves with the S7 correction is how much higher the
  right arm reads with both arms near level (S7 +10.1 raw, -2.5 corrected). No other session comes close; S6 is
  next (+4.5; takes 7, 11, 12 at +7..+9). S2 has the next-largest bank but reads level, so its bank looks like the
  shared old-fusion lean, not a one-sided offset. Video is most likely to add something for S6, then S2.
- Calibration facing cannot be read from the files: S7's takes start facing the dance direction, not the
  calibration facing (~95 deg away), and rarely in a T-pose.
- S6 take7 and take8 are byte-identical, raw files too.

Stomp smoothing applied to S1, S3, S4, S5, S6, S7 (`*_stomp.npz` beside the hip-fixed files; S2 left out, no
shake). smooth_stomp_jolt.py now fades the smoothing where it would move the root more than ~25 mm: dropout jumps of
25-71 cm in one frame sometimes fall near a stomp, and the first run smeared them (up to 373 mm moves). After the
fix the largest move is 14 mm; fast shake after right stomps, hip fix -> stomp: S1 1.33 -> 0.52, S3 1.54 -> 0.72,
S4 2.80 -> 0.92, S5 1.31 -> 0.49, S6 1.62 -> 0.66, S7 2.60 -> 0.82. S7 `_vc` files carry the new root position.

## 2026-11-01: the elbow turn under the new fusion (all sessions)

Question: is the elbow turn (the upper arm's roll about its own bone, the candy-wrapper source) a property of
Shadow's old fusion, like the big hang bank? Measured with `elbow_turn`'s own measure (`ElbowTurnNode.measure`)
on the raw Shadow rotations of every session, since the 2026 files are in a layout the SMPL converter does not
read. Strict band (bend 45-80, upper arm more than 30 below level), no widening; session value = circular mean
over takes with at least 100 band frames, each take once. The old sessions reproduce the SMPL-file screen
(S1 +18/+43 vs +19/+42; S7 +45/+50 vs +47/+56).

| Session | Fusion | Left | Right |
|---|---|---|---|
| S1 | old | +18 | +43 |
| S2 | old | +41 | +30 |
| S3 | old | +37 | +47 |
| S4 | old | +21 | +36 |
| S5 | old | -2 | +23 |
| S6 | old | +27 | +28 |
| S7 | old | +45 | +50 |
| S10 | new | +32 | +48 |
| S11 | new | +29 | +29 |
| S12 | new | +24 | +17 |
| S13 | new | +27 | +28 (take-to-take spread 29) |
| S14 | new | +27 | +29 |
| S15 | new | +35 | +12 |
| S16 | new | -6 | +11 |
| S17 | new | +42 | +17 |
| S18 | new | +27 | +29 |
| S19 | new | +28 | +35 |
| S20 | new | +8 | +20 |
| S21 | new | +22 | +40 |
| S22 | new | +9 | -4 |
| S9 retakes | new | +31 | +33 |
| Gaia suits | new | +17 | +24 |

Reading:
- Not the fusion: averages are about +27 / +37 under the old fusion and +23 / +25 under the new one.
- Nearly always positive (inside of the elbow turned up) on both arms, a few calibrations near zero (S22 both
  arms, S16 and S5 left), take-to-take spread 4-17. A sensor or fusion cause would not come out clean in some
  sessions; the dancer's T-pose posture would. Consistent with a palms-forward or elbow-turned T-pose, but
  untested: see test 8 in `HANG_BANK_STUDIO_TESTS.md` (rolled T-poses).
- Arm alignment is checked before each calibration, but a roll of the upper arm does not change where the arm
  points, so it is easy to miss by eye.

Byte-identical takes found (skipped): S6 take7 = take8, S13 take3 = take4 and take12 = take13, S16 take2 =
take3, S18 `Mime` = `mime_1_baton_pole_rope`, and the S9 retakes `temp_take_*` files repeat the named takes
(except `temp_take_20261009_122939`, which has no named copy and reads +58 / +61).

Files: `video_fix/turn_all_sessions.py`, `video_fix/turn_all_sessions.txt` (per take).

## 2026-11-01: the hang bank after the video correction (Subject 7)

The same screen (`screen_sessions.py`) run on the `_vc` files beside the files before the arm correction (hip fix,
stomp smoothing). Session values, median over takes:

| lens | before | `_vc` |
|---|---|---|
| hang bank (+ = toward the dancer's left) | -16.3 | -3.5 |
| arm splayed out, L / R | +9.2 / +35.3 | +8.9 / +15.0 |
| arm forward, L / R | +5.6 / +10.4 | +6.1 / +3.0 |
| right reads higher at level | +10.1 | +1.3 |
| frames with both arms hanging, jathiswaram | 4698 | 14987 |

Per take the bank is now -1.7 (jathiswaram) to -10.9 (sarukkal, 181 hanging frames); the takes with the most
hanging frames read -1.7 to -3.4.

Reading:
- Nearly all of Subject 7's hang bank was the one-sided right upper-arm offset, not a lean of both arms, as the
  2026-10-30 measurement suggested (left arm hanging within 6 deg of the video). Swinging the right arm back
  removes it; the left arm's lens values barely move.
- The remaining -3.5 is inside the range of the new-fusion sessions (about +-5), so `_vc` files need no fixed bank.
  The old `shadow_mag_correct` setting ('fixed bank' -17.9) would now push both arms about 14 deg the other way.
- Three times as many jathiswaram frames now qualify as both arms hanging: the screen needs the two arms at
  matching heights, and the right arm no longer reads higher.
- The right arm still splays out about 6 deg more than the left (+15 vs +9). Could be the dancer, or what is left
  of the offset; not separated.
- The elbow turn reads back L +2 / R +25 on the `_vc` files. The right is expected: the video-implied +46 was used
  instead of Shadow's post-swing +68 (see "finished corrected files").
- For the other old-fusion sessions: a bank alone cannot tell a lean of both arms from a one-sided offset. The
  "right reads higher at level" lens can; S7 was the only session high on it (screen table above).

Files: `video_fix/screen_S7_vc.txt`.

## 2026-11-02: Subject 11 (new fusion) — file layout, frame rate and the Vive tracker

Videos: `Subject11_Shaolin_Tai_Chi_60/Subject11_take1..9.mov`, beside the Shadow files, one per take, 1080p at
59.94 fps, one camera position (the new space). A second person walks in front of the lens near the end of some takes.

Several takes are in unusual positions: crouching and hops (take 3), seated and floor stretches (take 6),
low stances. The back is upright (pelvis and chest within 15 deg of vertical) in only 32-64% of each take's frames.

**The files run at 100 frames per second, though they say 60.** Every video is 0.6 x its Shadow take's length (all 9
takes), and the hops take syncs to its video with correlation 0.51 at 100 vs 0.27 at 60 (offset +0.08 s at 100, as
S7's +0.10; at 60 the best offset sits at the search edge).

**Layout:** 20-joint `quats` (local, w first), `trans` and `tracker_ori`, with no Shadow positions. `trans` and
`tracker_ori` come from a Vive tracker worn about mid lower back: a rough root position, a little higher than the
SMPL root, in the Vive's world frame, not Shadow's. As stored, the horizontal axes do not match Shadow's (hops take:
Shadow's x and y travel swapped against the video).

**Coordinating the two worlds:** fit tracker_ori = W * pelvis * M, where W is a constant turn between the Vive's world
and Shadow's magnetic world and M is the tracker's mount on the pelvis. M comes from relative motions
(T_i^-1 T_j = M^-1 (S_i^-1 S_j) M, aligning rotation axes), then W = mean(T M^-1 S^-1).
- `tracker_ori` is w first (x first leaves about 30 deg unexplained). The mount is about 174 deg (the tracker is turned
  round on the back). The two worlds agree on which way is up within 1-6 deg; W is a turn about vertical.
- **Use upright-back frames only.** From experience (David), the Vive-to-suit relationship changes with the angle of
  the back, so only frames with the back fairly straight and perpendicular to the floor are used. The data agree:
  with one session mount, the leftover is 4.5 deg median with the pelvis within 15 deg of vertical, 5.7 at 15-30, 8.1
  at 30-45, 11.1 at 45-60. Mount fitted from upright frames pooled over all takes (short takes alone give unstable
  mounts: take 4 alone moves W by 26 deg).
- W per take (pooled mount, upright frames), leftover median / 90th:

| take | upright | W turn | leftover | per minute (upright frames) |
|---|---|---|---|---|
| 1 rouquan | 45% | +39.4 | 5.8 / 18.5 | +35.5 +46.9 |
| 2 rouqyan kungfu | 53% | +69.2 | 3.6 / 11.9 | +71.2 +70.1 +56.4 +69.8 |
| 3 hops | 43% | +70.6 | 2.6 / 7.5 | +69.2 |
| 4 short centipede | 32% | +57.5 | 3.6 / 6.6 | (too short) |
| 5 spiraling fist | 55% | +66.1 | 7.0 / 13.5 | +72.0 +69.8 +60.4 |
| 6 stretches misaligned | 46% | +77.1 | 4.0 / 6.1 | +78.1 +79.0 +78.3 (3 min on the floor) +76.9 +76.0 |
| 7 slow misaligned | 50% | +68.5 | 2.2 / 4.7 | +68.5 +68.4 +68.5 +69.4 +68.9 +68.7 +68.5 |
| 8 slow 2 eventually misaligned | 61% | +71.4 | 3.2 / 7.9 | +71.8 +71.5 +69.8 +71.4 |
| 9 final combo | 64% | +40.2 | 5.2 / 13.3 | +40.2 +39.8 +41.9 |

Two things change between takes (David): the tracker was probably adjusted on the back several times, and the suit
is usually recalibrated between takes. Recalibration between takes is not absolute, but it is the starting
assumption for every session. On upright frames a tracker turned on the back (mount) and a turn of the worlds (W)
are the same rotation about vertical, so the orientation fit alone cannot separate them.

**W from travel, independent of the mount** (`video_fix/vive_from_video.py`): GVHMR's root travel, put in Shadow's
world by the constant GVHMR-to-Shadow rotation of the root orientations, against the Vive's horizontal travel
(2 s displacements over 0.2 m, reliable frames). Signs as the table above (Shadow to Vive):

| take | W from travel | direction leftover median / 90th | video / Vive travel length | W from orientation (pooled mount) |
|---|---|---|---|---|
| 1 | +59.4 | 10.8 / 35.1 | 1.08 | +39.4 |
| 2 | +76.5 | 9.5 / 29.9 | 1.06 | +69.2 |
| 3 | +71.1 | 5.0 / 21.5 | 0.98 | +70.6 |
| 4 | (too little travel) | | | +57.5 |
| 5 | +73.1 | 8.6 / 27.6 | 1.07 | +66.1 |
| 9 | +71.6 | 8.5 / 27.5 | 1.03 | +40.2 |

Reading:
- Take 9's 30 deg jump in the orientation fit is the tracker re-seated on the back, not a heading reset: by travel
  it sits with takes 2-5. Take 1 differs by 20 deg the same way. Takes 2, 3 and 5 agree within 1-7 deg, so the pooled
  mount fits them.
- Across recalibrations, Shadow's heading against the Vive stays within about 59-77 deg (take 1 lowest).
- Within a slow take on upright frames the orientation turn holds within about 1 deg for 7 minutes (take 7): no
  steady drift of Shadow's pelvis heading against the Vive inside a take. The vigorous takes swing up to 15 deg
  minute to minute (take 2's +56, take 5's +60); not separated from the tracker moving on the back.
- Whatever 'misaligned' in the take names refers to, it is not a drift of the pelvis heading against the Vive.
- So: with video, take W from travel; without video, the orientation fit needs a mount per stretch between strap
  adjustments, which the files do not mark.
- An all-frames fit (mount per take, every posture) gave different values (takes 1-9: +55 to +69). It is superseded.

Tools (`video_fix/`):
- `convert_active20.py OUT_DIR [--fps F] [--vive] TAKE.npz ...`: 20-joint files to SMPL (joints placed in the 37
  layout, then `convert_shadow_to_smpl`). `--vive` turns `trans` into Shadow's world by W^-1. The lower-back offset
  is not removed. Caution: it still uses the all-frames, per-take fit; switch it to upright frames and the pooled
  mount before using positions for anything finer than the room bins.
- `vive_fit.py` (all frames, superseded), `vive_upright.py` (upright frames, pooled mount, tilt bins),
  `vive_from_video.py GVHMR_OUT RAW_TAKE SHADOW_SMPL` (W from travel).
- `demo_mac.py` new options: `--stride 2` (59.94 fps video; keeps the comparison's 29.97 timing) and `--h_tol` (height
  tolerance of the person follower; 0.5 for a performer who crouches and sits — at 0.3 the hops take lost 30% of
  frames, at 0.5 3%). `arm_2d_referee.py` reads the stride for its picture.

(Written before the leg fix of 2026-11-03: see the re-measured per-segment section. Travel figures use only the root and stand.)

Hops take, first reading (with the Vive-turned trans): Shadow's travel x 1.1 / y 4.7 m vs the video's 1.3 / 5.5 (the
axes now match). Both upper arms carry about 40 deg of twist (the elbow turn); the left upper arm's direction is 23 deg
off against the right's 9.5. One take only. All nine takes are running through the S7 pipeline (scratchpad
`s11/run_s11.sh`, `s11/progress.txt`).

## 2026-11-03: Subject 11 per-segment error picture (all takes, re-measured)

This replaces the 2026-11-02 version of this section, which had two errors:
- `convert_active20.py` used `as_pose_tensor`, which only carries the upper body, so hips, knees, ankles and head were
  at rest in every Subject 11 SMPL file. A claim that Shadow loses the seated pose in take 6 came from this and is
  withdrawn: the raw file shows the dancer seated (knees about 104-123 deg). Now fixed, checked joint by joint against the raw.
- The comparison posed Shadow on the neutral SMPL-X body. Its lopsided rest arms make a hanging left arm read about 12 deg
  inward and the right about 4 (see 'which body model draws Shadow's rotations'), which faked much of a left upper-arm
  offset. Shadow is now posed on the SMPL-H body the viewer draws, with the dancer's shape; the video stays on its own body
  (`SHADOW_BODY=drawn` in `sensor_error_landscape.py`, `video_fix/own_frame_drawn.py`, `video_fix/s11_table2.py`).

Root-travel readings in the Vive section above stand. Its orientation fits used only the torso and also stand.

Tracking, sync, and direction against the video (median deg, gated frames):

| take | frames lost | sync corr | spine | L thigh | R thigh | L upper arm | R upper arm | L forearm | R forearm |
|---|---|---|---|---|---|---|---|---|---|
| 1 rouquan | 9% | 0.86 | 5.7 | 7.6 | 9.7 | 17.5 | 19.7 | 15.6 | 14.1 |
| 2 rouqyan kungfu | 5% | 0.83 | 6.9 | 10.0 | 5.1 | 16.3 | 13.1 | 13.4 | 20.0 |
| 3 hops | 3% | 0.56 | 10.1 | 6.3 | 6.0 | 11.3 | 12.8 | 10.9 | 16.9 |
| 4 short centipede | 6% | -- | unusable | | | | | | |
| 5 spiraling fist | 13% | 0.82 | 9.6 | 10.3 | 7.3 | 19.6 | 12.0 | 18.6 | 16.4 |
| 6 stretches misaligned | 3% | 0.66 | 31.6 | 24.1 | 25.4 | 38.0 | 32.6 | 28.1 | 19.2 |
| 7 slow misaligned | 0% | 0.73 | 11.5 | 12.3 | 8.5 | 12.6 | 12.9 | 13.0 | 13.3 |
| 8 slow 2 eventually misaligned | 2% | 0.68 | 10.5 | 8.6 | 7.1 | 18.2 | 14.2 | 15.3 | 16.9 |
| 9 final combo | 2% | 0.84 | 6.2 | 5.5 | 5.4 | 12.4 | 13.4 | 12.8 | 11.7 |

- The spine and thighs are the method's floor: about 5-12 deg.
- Take 6 is off everywhere, spine included, and about as much standing as seated (thigh about 21-29 vs 22-23), so
  sitting is not the reason. The cause is not known. Its arm numbers should not be read as arm errors.

**Constant arm offset**, in each segment's own frame (deg; in brackets the part about the rest vertical):

| take | L upper arm | R upper arm | L forearm | R forearm |
|---|---|---|---|---|
| 1 | 14.5 (-13.3) | 18.2 (-18.2) | 12.5 (-12.2) | 10.6 (-10.5) |
| 2 | 15.1 (+14.5) | 8.5 (+4.9) | 5.3 (+4.6) | 16.7 (+16.1) |
| 3 | 10.1 (+9.7) | 10.2 (+9.8) | 8.5 (+5.7) | 15.2 (+14.4) |
| 5 | 16.2 (+15.8) | 6.5 (+5.5) | 9.0 (+8.9) | 10.3 (+9.8) |
| 7 | 8.5 (+1.7) | 7.3 (-4.7) | 4.1 (+3.2) | 4.0 (+2.8) |
| 8 | 15.8 (+14.4) | 6.6 (-1.3) | 9.6 (+9.3) | 12.5 (+12.5) |
| 9 | 9.0 (+7.9) | 8.3 (-6.2) | 6.5 (+5.2) | 4.0 (+3.7) |

- The left upper arm is still larger than the right in takes 2, 5 and 8 (15-16 vs 6-9), with no inward bias behind it now:
  the offset is mostly about the rest vertical, which on a hanging arm is forward/back rather than sideways.
- Take 1 has the opposite sign on all four segments (-10 to -18). Four sensors sharing one offset points at the body's
  heading rather than the arms. The arm swings fitted for it improved agreement in the room's frame but worsened it
  relative to the torso, so `_vc` leaves them out. Not yet tested directly.
- Offsets change from take to take, as expected with a calibration per take.

**Hanging upper arms** (video elevation below -50, upright; Shadow minus video, deg; + = dancer's left, forward):

| take | L left | L fwd | R left | R fwd | bank per video | Shadow's own bank |
|---|---|---|---|---|---|---|
| 1 | +15.5 | -4.9 | +8.5 | +16.0 | +11.8 | +4.3 |
| 2 | -7.9 | +8.7 | -8.0 | +2.0 | -7.6 | -11.0 |
| 3 | -8.4 | +2.9 | -8.7 | -2.3 | -7.9 | -12.2 |
| 5 | -2.0 | +14.0 | -7.8 | +0.6 | -5.1 | -4.5 |
| 7 | +6.2 | +7.7 | +3.1 | +10.0 | +4.9 | -8.0 |
| 8 | +1.0 | +13.4 | -2.4 | +10.6 | -0.7 | -9.9 |
| 9 | +0.5 | +7.0 | +0.2 | +7.8 | +0.2 | -6.0 |

- Bank: the reading survives the re-measure. In takes 2, 3 and 5 most of Shadow's bank is error. In slow takes 7, 8 and 9
  it is mostly the dancer's real arm position, so read the bank only from relaxed, still hanging arms. Take 1's has the
  opposite sign: per calibration.
- Forward: hanging arms read forward of the video in most takes, up to +16. Facing the camera, forward is GVHMR's weak
  depth axis. Side-on frames are too few to settle it (229 frames: left +8.1, right +2.1; `hang_forward_by_facing.py`).

**Elbow turn**: twist against the video, L / R (deg; magnitude, `compare_shadow.py`):
1: 47.6 / 13.4; 2: 27.8 / 34.3; 3: 39.5 / 40.5; 5: 43.1 / 32.9; 7: 38.2 / 41.7; 8: 30.2 / 23.7; 9: 34.2 / 33.0.
Still the main arm error. Take 1's split matches a shared heading offset added to one arm and taken from the other.

**Penetration**, forearm or wrist more than 3 cm inside the torso, % of synced frames, on the drawn body (L / R).
Hands are not measured.

| take | Shadow | video |
|---|---|---|
| 1 | 5.6 / 3.8 | 1.4 / 0.3 |
| 2 | 9.7 / 0.8 | 1.2 / 0.1 |
| 3 | 13.7 / 0.0 | 5.6 / 11.1 |
| 5 | 18.6 / 3.3 | 0.6 / 0.9 |
| 6 | 12.7 / 1.5 | 0.3 / 0.0 |
| 7 | 0.3 / 0.4 | 1.6 / 0.0 |
| 8 | 6.9 / 0.0 | 0.1 / 0.1 |
| 9 | 3.8 / 0.0 | 1.1 / 0.0 |

- The left arm dominates, well below what the neutral body showed (take 2: 32 to 9.7; take 5: 30 to 18.6). GVHMR's body prior favours clear hands, so its
  low numbers are partly habit.
- No single segment causes it. Taking one direction from the video (`arm_penetration_swap.py`):
  - left upper arm helps most in takes 2 and 3 (9.7 to 4.2, 13.7 to 0.2);
  - left forearm helps most in takes 1, 5 and 9 (5.6 to 1.2, 18.6 to 6.4, 3.8 to 1.1);
  - both together reach 0-1.2 in every take except 3 (2.4) and 6 (9.2);
  - take 1's right side is cured by its upper arm alone (3.8 to 0).
- The frames that go in have the elbow bent about 76-107 deg with the wrist near the midline, with the forearm further back
  and further in than the video. These frames were picked for being inside, so that direction is partly built in.
- One fixed turn of the whole arm per take does not do it (fit on one half, scored on the other): good for take 1 left
  (9.6 to 1.9), some for take 5, nothing elsewhere, and worse in take 8 left (0.2 to 4.2), take 3 right (0 to 16), take 6 left.
- Turning the forearm outward about the upper arm by 20 deg clears much of it, and brings the left forearm closer to the
  video in takes 1, 7, 8 and 9, but moves the right forearm further away in most takes.

**Could a magnetized sensor produce this?** Sweep in `video_fix/arm_mag_sweep.py`: a fixed field in the sensor, the same with
bleed into tilt, and a magnet at 18 places on the torso, each on top of a constant own-frame offset, fitted on one half and
scored on the other. Takes 1, 2, 5, 7, 8, 9.
- Fixed field vs constant alone: usually within 2 deg. Largest gains are left forearm in takes 7 and 8 (about 3) and right
  upper arm in take 5 (3); left forearm in take 5 is 2 worse.
- Fitted offsets differ in direction between takes, although remanent magnetism should survive recalibration.
- One shared offset, leave one take out: most changes are within 2 deg and go both ways.
- Bleed into tilt: gain from -16 to +42, not physical. Torso magnet: no better, and the best place changes per take.
- Fusion recovering slowly: no fade in the seconds after the arm leaves the pose (`arm_error_after_exit.py`).
- Penetration does not improve overall over the constant.

So no magnetic model tested earns its place as the main cause. A small effect is not excluded: 12-25 deg of disagreement
remains after every fit. The June arm-sensor offsets (`magnetometer_assessment/`) allow at most about 9 deg of heading
on shoulders and elbows; wrist outliers allow 36-59 deg on the hand (tested under Hands). A field that
varies across the room is untested.

**Hands** (`video_fix/hand_measure.py`, `hand_drift.py`, `hand_inject.py`; uncorrected poses). Deep inside (> 3 cm), % of synced
frames, L / R, on the drawn body's hand surface:

| take | forearm-wrist | hand | forearm or hand | video hand |
|---|---|---|---|---|
| 1 | 5.6 / 3.8 | 8.4 / 0.0 | 10.1 / 3.8 | 0.2 / 0.0 |
| 2 | 9.7 / 0.8 | 3.5 / 3.4 | 11.2 / 3.8 | 0.1 / 1.6 |
| 3 | 13.7 / 0.0 | 1.1 / 0.9 | 14.5 / 0.9 | 9.6 / 15.7 |
| 5 | 18.6 / 3.2 | 20.1 / 6.2 | 24.1 / 6.7 | 0.5 / 5.1 |
| 6 | 12.7 / 1.5 | 3.2 / 0.5 | 15.6 / 1.6 | 0.2 / 0.0 |
| 7 | 0.3 / 0.4 | 0.1 / 1.0 | 0.4 / 1.3 | 0.0 / 0.0 |
| 8 | 6.9 / 0.0 | 8.2 / 0.7 | 10.6 / 0.7 | 3.4 / 4.2 |
| 9 | 3.8 / 0.0 | 3.5 / 0.0 | 4.1 / 0.0 | 4.3 / 2.9 |

- The files carry no finger rotation, so hands are drawn open. A fist would reach less far, so these are upper figures for
  closed hands.
- The video cannot referee the hand: no hand keypoints, and GVHMR's wrist differs from Shadow's by a median 22-45 deg.
- Magnetized hand sensor, no video: a fixed field in the hand sensor should make the hand-to-forearm rotation steadier
  once corrected. On held-out frames it does no better than a constant, and no better than a placebo using the hand's orientation
  from 30 s away. Fitted fields point different ways between takes; one shared field does not help on a left-out take.
- Sensitivity, injected into the left hand of takes 2, 5, 7, 9: size 0.2 (units of horizontal field) cannot be found; 0.59 (June's suit 2
  left wrist) gains 1-4 deg with about the right size; 0.86 (suit 1 right wrist) gains 5-10. With nothing injected the fit still
  returns about 0.3, so fitted sizes up to that mean nothing. Reading: no left-hand field of 0.59 or more in these takes. The right
  hand was not injected but behaves alike. Either the suit was not one with those outliers or the offsets came after March.
- Take 1's left hand-to-forearm centre looked like a 40 deg step at 40-60 s in 20 s windows. In 2 s windows both hands swing together, and David sees no change in how the hand is held: choreography.

**Root from the Vive** (`video_fix/vive_root.py`, `build_s11.py`):
- Repeated samples (3-11% of frames: the tracker is slower than the frames), jumps and spikes over 2.5 cm off a smooth path are filled by
  interpolation, then an 8 Hz zero-phase low-pass. David saw the root jerk in take 1. It has the most missing samples and had spikes of 3-7 cm;
  across takes the jitter also grows with speed (about 0.04 cm still to 1.6 cm fast), which points at sample timing. Take 1's top
  speed goes from 5.5 to 2.3 m/s; planted-foot jitter improves only about 15% (90th percentile 0.45 to 0.38 m/s), so most of what remains is
  not this noise (`trans_jerk.py`, `trans_filter_test.py`).
- World turn and tracker offset per take from planted feet, offset limited to the lower back. Unlimited, the offset grows
  to the length of the leg and glues the foot to the tracker.
- Turn -62 to -71 deg (halves within about 2-6); video travel gives a median of -71 and is noisier per take; `raw/` take 2
  `_trans` is the main trans turned by -77.41, about 11 from this take's -66.2.
- Offset 2-6 cm above and 8.5-14 behind the pelvis joint at rest, at most about 5 sideways; take 9 pins at the 8 cm
  sideways limit, which may be the re-seated tracker.
- One room floor at Vive height -3.47.
- Feet within 10 cm of the floor in 94-100% of frames, slip down on every take. Seated take 6: soles about 18 cm low.

**Corrected files** in `~/Projects/GVHMR/outputs/s11_work/`:
- `tr/`: uncorrected poses with the new root, all nine takes.
- `tr_nofilter/`, `vc_nofilter/`: the same without the spike filter and low-pass.
- `vc/`: new root, elbow turn per take, and a swing only where it beat no swing by 1.5 deg in both held-out halves, improved
  the direction relative to the torso, and did not raise penetration. Takes 1, 2, 5, 7, 8, 9. Swings kept: take 2 left upper
  arm and right forearm; take 5 left upper arm; take 8 left upper arm and right forearm.

Left arm deep penetration raw to `vc`: take 2 9.7 to 4.6, take 5 18.6 to 7.5, take 8 6.9 to 4.9; takes 1, 7, 9 unchanged.

Open:
- S7 and S8 swings were fitted with the neutral rest bones; the S8 left upper-arm swing is the most suspect.
- Take 1's body heading; take 6's overall disagreement.

## 2026-11-04: Subject 2 (old fusion, Contemporary, all three takes)

Measured as Subject 11: Shadow on the drawn SMPL-H body (female, file betas), the video on its own body. Input is the
`*_rhipflexstat_100hz.npz` files; S2 has no stomp smoothing because it shows no stomp shake. Video is 59.94 fps, so `--stride 2`;
`--target_h 600 --target_x 700 --h_tol 0.5`. Scripts `video_fix/s2_table.py`, `pelvis_split.py`, `pelvis_belt.py`, `build_s2.py`,
`verify_s2.py`, `turn_video_dir.py` (`turn_video.py` with the results folder from `LANDSCAPE_DIR`).

Tracking, sync, and direction against the video (median deg, gated frames):

| take | frames lost | sync corr | spine | L thigh | R thigh | L upper arm | R upper arm | L forearm | R forearm |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3% | 0.75 | 12.7 | 12.5 | 14.9 | 17.6 | 22.7 | 14.9 | 17.9 |
| 2 batteryIssue | 0% | 0.92 | 5.5 | 8.0 | 11.0 | 13.0 | 17.6 | 10.3 | 12.2 |
| 3 | 17% | 0.85 | 9.4 | 8.4 | 10.1 | 13.0 | 17.8 | 12.9 | 14.9 |

- Take 2 agrees best despite its label.
- Take 1 has long floor stretches (0-120 s and 360-420 s, lower spine difference 47-52 deg against 10-25 elsewhere). In the sampled
  frames a seated person's legs are at the right edge, which is probably why two or more people are detected in nearly every frame; the tracker
  was on the dancer in all six referee frames. That is a spot check, not a full one.
- Sync drifts by about 0.2 s across take 1. Following the drift (`video_fix/sync_warp.py`) lifts sync from 0.749 to 0.768
  and changes almost nothing else: all joints 21.4 to 21.2 deg, upper arm offsets within 0.1. Its weak sync stretches are mostly
  slow movement, which gives speed correlation little to lock onto.

**Pelvis tilt in forward bends** (from vertical, Shadow / video, by whole-torso lean):

| torso lean | take 1 | take 2 | take 3 |
|---|---|---|---|
| 0-10 | 12 / 13 | 4 / 8.5 | 6.5 / 8.5 |
| 10-20 | 15 / 14 | 7 / 11 | 12 / 11 |
| 20-30 | 22 / 15 | 17 / 9 | 22 / 14.5 |
| 30-45 | 32 / 21 | 27 / 9 | 31 / 18 |
| 45-60 | 51 / 28 | 44 / 16 | 48 / 21 |
| 60-90 | 69 / 44 | 51 / 19 | 65 / 34 |

- The torso lean itself agrees; the two split it differently between pelvis and spine.
- The belt correction is not the cause: tilt before and after it (`pre/smpl` against the final file) differs by at most 0.5 deg in any bin.
- Probably mostly the video's error, but this rests on the eye and a few small frames: in take 3 at 959 s and 232 s the dancer is low on hands and
  knees with a fairly level back, where Shadow's 73 and 63 look plausible and the video's 33 and 26 do not. Most camera angles are from the front, where
  this tilt is hard to see. If so, lower spine and pelvis differences in bends should not be read as Shadow errors, take 1's included.

**Constant arm offset**, in each segment's own frame (deg; in brackets the part about the rest vertical):

| take | L upper arm | R upper arm | L forearm | R forearm |
|---|---|---|---|---|
| 1 | 12.2 (+4.6) | 19.0 (+13.5) | 4.5 (+3.7) | 11.2 (+9.2) |
| 2 | 8.8 (-7.8) | 14.8 (+14.8) | 5.6 (+3.1) | 7.0 (+7.0) |
| 3 | 6.9 (-3.2) | 13.4 (+12.0) | 2.8 (-2.7) | 8.1 (+6.7) |

The right arm is the stable finding: about +12 to +15 on the upper arm and +7 to +9 on the forearm, in every take. The left is smaller and changes direction.

**Hanging upper arms** (video elevation below -50, upright; Shadow minus video, deg; + = dancer's left, forward):

| take | L left | L fwd | R left | R fwd | bank per video | both out by |
|---|---|---|---|---|---|---|
| 1 | +0.6 | +11.0 | -12.4 | +8.4 | -5.9 | 13.1 |
| 2 | +13.5 | +5.9 | -11.3 | +3.3 | +0.8 | 24.7 |
| 3 | +4.7 | +6.5 | -9.9 | +0.9 | -2.3 | 14.6 |

Shadow's own session bank is -7.9, closest to take 1. Both arms sit out from the body in every take, the right steadily.

**Elbow turn, checked the 2026-10-31 way** (`turn_video.py`; elbow bent 45-80, arm low; Shadow's own reading / video-implied, frames in brackets):

| take | L, raw | R, raw | applied in `vc`, L / R |
|---|---|---|---|
| 1 | 26.7 / 19.8 (3561) | 22.5 / 24.9 (1591) | 28.0 / 34.4 |
| 2 | 38.8 / 34.8 (602) | 36.5 / 57.7 (86) | 41.1 / 45.2 |
| 3 | 33.5 / 28.0 (4111) | 25.3 / 29.7 (1748) | 39.3 / 32.2 |

- The turn is real: Shadow and the video agree within about 7 deg except take 2's thin right arm. Pooled, the video-implied value holds about 20-34 deg at every bend range.
- Undoing it on bent arms brings shoulder twist to the video: L +34.7 to +1.3 (video +5.6), R -31.4 to -6.7 (video -4.5). On straight arms the video reads L -15, R +15
  and undoing gives -35, +31, but GVHMR's straight-arm twist is prior, not measurement.
- Shadow's reading after the right upper-arm swing climbs (take 1 22.5 to 36.3, take 3 25.3 to 34.8) while the video-implied value barely moves, the same pattern
  as Subject 7. So the applied turns run high by up to about 10 deg in places, mostly takes 1 and 3. Not yet rebuilt with video-implied values.
- `compare_shadow.py`'s elbow twist gets much worse after the turn (take 2 from 8 and 6 to 42 and 45). This is not evidence against it: GVHMR's own elbow stays near a
  hinge by its prior, so a real turn shows up as disagreement there.

**Corrected files** in `~/Projects/GVHMR/outputs/s2_work/` (trans unchanged: no Vive in this session):
- `sw/`: swings only.
- `vc/`: swings and the per-take elbow turn, turn read after the swing.
- `build.txt`, `verify.txt`, `turn_video_raw.txt`, `turn_video_sw.txt`: logs.

Same checks as Subject 11. Swings kept: right upper arm in all three takes (18.5, 15.9, 13.4 deg), left upper arm in take 3 (6.5). Dropped by the torso check: take 1
left upper arm (world gain, torso-relative 14.5 to 15.5), take 2 right forearm (7.5 to 8.5). No other forearm passed.

| error against the video (deg) | take 1 | take 2 | take 3 |
|---|---|---|---|
| R upper arm, world | 22.7 to 14.0 | 17.6 to 11.0 | 17.8 to 12.5 |
| R upper arm, relative to the torso | 19.2 to 14.9 | 16.8 to 10.3 | 16.5 to 15.2 |
| L upper arm, world | -- | -- | 13.0 to 11.3 |
| L upper arm, relative to the torso | -- | -- | 15.3 to 12.9 |

Deep penetration is near zero and rises only in take 3: L 0.1 to 1.1%, R 0.0 to 1.8% (video 1.1, 0.2).

**`armfix` (take 3, the earlier manual fix)** against the video, without / with: L upper arm 13.0 / 20.6, R upper arm 17.8 / 15.7, L forearm 12.9 / 15.2, R forearm 14.9 / 15.5.
It helps the right upper arm only, and worsens the left.

Open:
- Pelvis split: no independent check yet, only the frames above.
- Whether to rebuild `vc` with video-implied turns. The corrected motion has not been viewed yet.

## 2026-11-05 to 11-08: Subject 11 corrections, from take 8 to all nine takes

**How to repeat it:** `SHADOW_CORRECTION_PROCESS.md`, beside this file, covers the method, the guards and the traps. This entry covers what was found, what was rejected, and what is still open.

### What was built, in order

1. **pd (pose-dependent arm swings)**, `video_fix/build_posedep.py`.
   - **What it fits:** per segment, a rigid swing that is a smooth function of heading and/or pose.
   - **Evidence:** the raw self-contacts (palm to nose, palm to palm, forearm to torso) and the elbow's anatomical limit are kept in the fit.
   - **Choosing a model:**
     - A model passes if its held-out video error is within 1.5° of the best.
     - Among those, the one with the fewest constraint violations wins.
     - A richer model must win by 0.5°.
   - **Files:** `s11_work/pd/`, with logs in `s11_work/pd_logs/`.
   - The pd fit's elbow turn is used later, in the hinge step.
2. **Take 8, worked by hand with you** (final2 to final5). The steps are listed below.
3. **Generalised** into `video_fix/correct_take.sh N`, then run on every take.

### Take 8: what each look found

**Lower-back sensor roll drift.** After about 90 s the pelvis rolls 6–10° against the video.
- **The Vive tracker can't referee it:** it shares the belt and rolls even further, 11–14°.
- **Referee:** the video hip line. **Check:** the planted feet.
- **Applied graded:** pelvis 1, spine1 2/3, spine2 1/3.
- **Effect:** lean against the video 4.9° → 2.1°.
- **It also explains the "high left shoulder":** the correlation is +0.88. So the collar fix (`build_collar.py`) is superseded.

**Constant upper-arm offsets** from mirror pairs plus the video:
- Fitted: left 12.8°, right 4.8°.
- Rejected: a heading (magnetic) model and a hinge-deformation model.

**Left elbow off its hinge** (29% of bent frames).
- **Cause:** the left upper arm's roll error varies over time.
- **Fix:** roll the upper arm about its bone, per frame.
- **First version failed:** it caused the "candy wrapper" at 12356. On a nearly straight arm, a small sideways forearm error looked like a 90° roll.
- **Fixed with the 'map' mode, which balances three things:**
  - a roll prior of 40°
  - the sideways forearm spread (6.7°) and carrying angle (−4.4°), both from the uncorrected right arm
  - a soft wall at the take's own raw twist range
- **Rejected:**
  - swinging the upper arm instead (it added 460°/s of joint speed)
  - turning the forearm (the video got worse)

**Removed on 7 November: the torso-clearance step (`build_mc.py`) and the back-contact step (`build_backcontact.py`).**
- Both engaged when the pose crossed a threshold, and both invented movement.
- The case that showed it: take 7, 23750–24250. A hand hanging by the hip was pulled 10–13 cm inward and back out.
- **Rule from now on:** corrections come from a sensor-error model. Contact and penetration may only enter the fit as soft evidence.

**What the models alone leave on take 8:**
- At 5848 the left forearm is 8 cm inside the chest (raw 10 cm).
- In the holds, the hands are off the back.

**Other take 8 observations not yet modelled:**
- A left upper-arm error tied to the arm crossing the body. In face-on video it's 32° with the wrist past the midline, against 9–13° on its own side. It comes and goes with the gesture. Room direction isn't the cause.
- The right collar sits about 5° lower from 140 s on.

### All nine takes (`s11_work/logs/summary_all_takes.txt`, made by `video_fix/summarize_takes.py`)

**Trunk drift** (median/max correction; held-out hip line, raw > corrected, in two halves):

| Take | Correction (median / max) | Held-out hip line | Verdict |
|---|---|---|---|
| 8 | 8.1° / 12.5° | 5.2 > 2.1, 4.5 > 3.2 | supported; feet 4 windows better, 2 worse |
| 1, 3, 5, 9 | 0.1–1.4° / 4.0° or less | mixed | negligible either way |
| 2 | 3.1° / 4.5° | mixed | feet warning |
| 7 | 3.9° / 15.9° | mixed | feet warning; the video and the feet disagree |
| 6 | 8.7° / 15.3° | 11.6 > 12.2, 4.5 > 8.4 | worse |
| 4 | 0 | none | take shorter than the 60 s reference |

Take 6 is the stretches take with floor work. The hip-line referee probably fails once the dancer is off her feet, and the raw twist range is ±174°, so the twist wall is meaningless there.

**Constant arm offsets:** mostly 3–16°, and they vary per take, as expected if the suit is recalibrated between takes.
- **Unsupported (held-out video worse in both halves):**
  - right arm, takes 6, 7, 8
  - left arm, take 7
- **Take 4:** unverifiable (see below).

**Carrying angle:** −2.7° to −9.1° across takes, consistent with a property of the dancer's body.

**Left elbow off-hinge:** reduced by the map mode in every take.

**Arm sources chosen by the face-on video:**
- **pd:** 13 of the 18 arms.
- **Hinge file:** left arm in takes 4, 5, 7, 8, and right arm in take 1.

**Unexplained: forearm more than 3 cm inside the torso** (% of frames; hand-behind-back frames excluded; raw / pd / corrected):
- take 2: left 14.8 / 11.8 / 12.4
- take 3: left 18.3 / 11.0 / 10.9
- take 5: left 22.4 / 21.3 / 11.4; right 14.1 / 15.3 / 15.3
- take 6: left 16.5 / 16.5 / 16.3
- take 1: left 8.0 / 7.5 / 7.3; right 6.0 / 6.0 / 2.6
- takes 4, 7, 8, 9: 3.3% or less on both arms

The left side dominates in every take except 5. That points to the same left-arm error tied to crossing the body seen on take 8. It is the main lead for the next model.

### Take 4 (short_centipede, 7.26 s): why it had no pd, and what fixed it

- **The sync had failed.** Its video file had never been synced. `compare_shadow.py` matches total joint speed between video and Shadow. On a take shorter than its video, every candidate lag puts some video frames outside the Shadow take, where the interpolation holds the end value. The correlation came out nan, and the search fell back to its −10 s edge. The pd build then saw a torso-relative error of 61° and left take 4 out.
- **First fix:** `compare_shadow.py` now correlates only where video and Shadow overlap. Long takes keep their lags (take 3: 0.151 s).
- **The speed match then found a false peak** at −1.57 s, which is easy to hit on 7 s of slow movement.
- **Root height gives the true lag:** +1.28 s, correlation 0.95. The pose-agreement scan agrees (+1.2 s). On take 3, root height finds that take's known lag (+0.10 against 0.151).
- **Second fix:** a new `SYNC_OFFSET` environment variable in `compare_shadow.py` fixes the lag. `sensor_error_landscape.py` reads it from the `_vs_shadow.npz`.
- **Take 4 at +1.28 s:** upper arms 8–17°, forearms 17–22°, spine 19°. That's in line with the other takes.
- **Two copies of each video file:**
  - `s11_work/S11_takeN.npz` (2 November, default body) is what the correction stages read.
  - `scratch_results/s11/S11_takeN.npz` (3 November, drawn SMPL-H body).
  - They differ only in the Shadow-joint fields. The stages read only the video joints, times, lag, facing and reliability masks, and those agree (facing within 0.01°, the masks identical).
  - The rebuilt take 4 file was copied into `s11_work/`. The broken one is kept as `s11_work/S11_take4.bad_sync_minus10s.npz`; the older `scratch_results` copy is `S11_take4.pre_remeasure.npz`.
- **pd build:** take 4 is back in `build_posedep.py`. Run alone (`TAKES=4`), it had no takes to average the session elbow turn over (nan, then a crash). It now uses the turns recorded in the pd files already written. Take 4's turns: left +33.6° (its own measurement), right +28.2° (the session value).
- **A scipy 1.15.2 bug.** `median_filter` with a window wider than the array returns zeros and corrupted the frame indices next to it in memory. That surfaced in `build_root_tilt.py`, whose window is now capped at the data length. **Any other `median_filter` call on short data has the same exposure.**
- **Robustness only, no change to results:** a take shorter than two 20 s blocks gets no held-out trunk test, and an empty delayed-pair set no longer crashes the mirror search.
- **Take 4 results:**
  - trunk 0
  - no delayed mirror pairs, only 4 synchronous ones
  - constant offset left 13.5°, right 1.8°, **unverified** (no held-out test is possible)
  - left arm from the hinge file (in-sample video 11.3 against pd's 13.8), right arm from pd
  - no penetration
  - `ARM_L=pd correct_take.sh 4` takes the left arm from pd instead, if the unverified offset shouldn't stand

### Open questions and what to try next

1. **Per-take trunk decision.** Apply the trunk roll only where the held-out hip line and the feet support it: take 8 yes, take 6 no, take 7 undecided. This is a choice of model for the whole take, not a pose trigger. `correct_take.sh` currently applies it everywhere.
2. **Is the left forearm penetration real?** Look at takes 5 and 6 in their patches. The oval torso model counts a hand resting at the hip as "inside" (seen on take 7), so some of the 11–16% may be grazing, not error.
3. **A left-arm error tied to crossing the body.** Fit it across takes, with penetration and contact as soft evidence (residuals weighted by how likely the contact is), and test it held-out. Candidate mechanism: the strap sits near the elbow, held tight against the body.
4. **Hands on the back in the take 8 holds.** These are a test for item 3, not something to enforce.
5. **The right collar on take 8,** about 5° low from 140 s on. Not modelled.
6. **Take 4's left offset.** Keep it, or take pd? Is a session-level left offset defensible, given that the per-take values are 3–16°?
7. **Other subjects.** Only Subject 11 is wired into `correct_take.sh`.

### Files

Scripts, all in `~/Projects/GVHMR/tools/shadow_compare/video_fix/`:
- `correct_take.sh` (the driver; `FROM=k` reruns from stage k)
- `build_root_tilt.py`
- `build_lowspine.py`
- `build_mirror_upper.py`
- `build_hinge.py`
- `build_elbow_hinge.py`
- `combine_arms.py`
- `make_three_body_patch_final.py`
- `summarize_takes.py`
- `build_posedep.py`
- `build_mc.py` and `build_backcontact.py` (removed from the pipeline; kept for the record)

Outputs, in `~/Projects/GVHMR/outputs/s11_work/`:
- `corr_out/*_corrected.npz` (all nine takes)
- `logs/corr_takeN_<stage>.txt`
- `logs/summary_all_takes.txt`
- `final5_out/` (take 8 as reviewed, still with the triggered steps)

Patches: `~/dpg_system/patches/shadow_raw_pd_corrected_s11_takeN.json`, one per take, 1 to 9.

## Where we left off (2026-11-01): picking up the old-fusion sessions

Decision: no arm corrections on S1-S6 until their video has been analysed. S2's is done (2026-11-04 section above). S6's video arrived 2026-11-04, except take 8, whose
data is byte-identical to take 7's; its batch writes to `~/Projects/GVHMR/outputs/s6_work/`.

Done for every old-fusion session:
- hip fix (`*_rhipflexstat.npz`, right thigh everywhere)
- stomp smoothing (`*_rhipflexstat_stomp.npz`, S1 and S3-S7; S2 has no stomp shake and was left out)
- Shadow-only screen (`video_fix/screen_old_fusion.txt`, table above)

Use the `_stomp` files as the input to any further correction (S2: its `_rhipflexstat` files).

When the video is in, per session (S6 first, then S2; the others only if video turns up):
1. Run GVHMR on the takes (see Setup), draw Shadow with SMPL-H (file gender and betas), not GVHMR's neutral
   model (2026-11-01 section).
2. Fit the fixed swing per segment (`fit_fixed.py`), as for S7.
3. Elbow turn: measure it AFTER the swing (`build_final.py` does swing, then turn, about the SMPL-H bone). Check
   it against the video with `turn_video.py`; in S7 the video-implied turn (L 44.8 / R 46) replaced Shadow's
   post-swing right reading (68), which over-corrected.
4. Check `_vc` files with `compare.py` and `make_compare_patch.py`.

If a session never gets video, its Shadow-only session turn (screen table) can be applied as a provisional value
(`build_final.py` with no swing segments; untested that way, and it measures its own turn unless given `L=..,R=..`).

Open:
- the second performer's left-arm swing may be partly the neutral-model artefact (not redone; not needed)
- calibration facing cannot be read from the files; S7-style offsets need filmed T-poses to tell the cause
- the elbow turn's cause (T-pose upper-arm roll?): test 8 in `HANG_BANK_STUDIO_TESTS.md`; if it confirms, the fix
  is a calibration instruction and archive takes keep the per-session turn correction
- S6 take7 and take8 are byte-identical (raw too): ask whether one was saved over the other
- the swing and turn scripts live outside the repo, in `~/Projects/GVHMR/tools/shadow_compare/video_fix/`
- recalibration between takes is the starting assumption for every session (2026-11-02): fit corrections per take
  where the data allow, and treat a session value as the case to be shown. S7 was already read this way (right
  upper-arm offset 24-38 deg across takes, the same error made at each calibration of the day). For the session-wide
  elbow turn (2026-10-31 'one turn for the whole session vs one per take') the per-take values are the default.

## Scripts (`~/Projects/GVHMR/tools/shadow_compare/`, run in env `gvhmr`)

- `compare_shadow.py GVHMR_NPZ SHADOW_SMPL_NPZ` — time sync, per-joint rotation differences,
  joint positions, heading and trajectory; writes `*_vs_shadow.npz` (needed by the others).
  The sync correlates only where video and Shadow overlap. `SYNC_OFFSET=<s>` fixes the lag when the
  speed match is unreliable (short or slow takes); check it against root height (S11 take 4).
- `arm_tests.py GVHMR_NPZ SHADOW_SMPL_NPZ` — left/right mirror test and constant-bias split per joint
- `arm_2d_referee.py OUT_DIR SHADOW_SMPL_NPZ VIDEO PICTURE_DIR` — the 2D referee for one Shadow
  file, plus a picture of the worst right-forearm frames
- `arm_referee_multi.py OUT_DIR SHADOW_SMPL_NPZ [...]` — the 2D referee table for several Shadow
  versions side by side

Shadow files in Shadow's own format (`quats`, `positions`, e.g. `beta/*.npz`) must first go through
`shadow_to_smpl.py` (`--output-dir` somewhere scratch).
