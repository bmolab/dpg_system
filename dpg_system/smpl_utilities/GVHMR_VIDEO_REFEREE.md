# GVHMR video referee for Shadow arms

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
`patches/shadow_video_correction_compare.json`) plays jathiswaram as recorded (left) beside the corrected file
(right) from one frame slider; mgl_rotate's y turns both. Look at the opening hands-on-waist hold (frames 0-800).
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

## Where we left off (2026-11-01): picking up the old-fusion sessions

Decision: no arm corrections on S1-S6 until their video has been analysed.

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
- S6 take7 and take8 are byte-identical (raw too): ask whether one was saved over the other
- the swing and turn scripts live outside the repo, in `~/Projects/GVHMR/tools/shadow_compare/video_fix/`

## Scripts (`~/Projects/GVHMR/tools/shadow_compare/`, run in env `gvhmr`)

- `compare_shadow.py GVHMR_NPZ SHADOW_SMPL_NPZ` — time sync, per-joint rotation differences,
  joint positions, heading and trajectory; writes `*_vs_shadow.npz` (needed by the others)
- `arm_tests.py GVHMR_NPZ SHADOW_SMPL_NPZ` — left/right mirror test and constant-bias split per joint
- `arm_2d_referee.py OUT_DIR SHADOW_SMPL_NPZ VIDEO PICTURE_DIR` — the 2D referee for one Shadow
  file, plus a picture of the worst right-forearm frames
- `arm_referee_multi.py OUT_DIR SHADOW_SMPL_NPZ [...]` — the 2D referee table for several Shadow
  versions side by side

Shadow files in Shadow's own format (`quats`, `positions`, e.g. `beta/*.npz`) must first go through
`shadow_to_smpl.py` (`--output-dir` somewhere scratch).
