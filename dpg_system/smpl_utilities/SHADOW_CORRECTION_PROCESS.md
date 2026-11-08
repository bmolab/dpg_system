# Correcting a Shadow take: process and palette

Distilled from Subject 11 take 8 ("slow_2_eventually_misaligned", 2026-11-05 to 11-07), building on the
Subject 7 / Subject 2 work in `GVHMR_VIDEO_REFEREE.md` (setup, running GVHMR, the session-wide fixed
corrections). This file is the how-to; that one is the lab notebook.

Scripts: `~/Projects/GVHMR/tools/shadow_compare/video_fix/`, run with the `dpg_system_2025` python
(`/Users/drokeby/miniforge3/envs/dpg_system_2025/bin/python`, not `conda run`, which swallows output).
Work files: `~/Projects/GVHMR/outputs/s11_work/`.
**One take, end to end:** `correct_take.sh N [FRAME ...]` runs steps 1–4, assembles the arms and makes the patch:
- **Writes:** `corr_out/*_corrected.npz` and `patches/shadow_raw_pd_corrected_s11_takeN.json`.
- **Logs:** full stage logs in `s11_work/logs/corr_takeN_*.txt`.
- **Needs:** `tr/`, `pd/` and `S11_takeN.npz` for the take.
- **Stage scripts:** each takes `TAKE=N` (default 8); take 8's own check frames print only for take 8.
- **Combine step:** `combine_arms.py` chooses each arm (elbow-hinge file or pd) by the face-on video.
- **Patch frames:** without frames on the command line, the patch gets the six moments where the corrected arms move most from raw.

---

## The principle

Each correction needs a **referee**: an independent observation that the error must answer to.
- Prefer referees that come from the body itself: mirror symmetry, the elbow hinge, the torso surface, planted feet.
- Next best is the video, used face-on only.
- A correction that no referee can check is a guess. Keep it smaller than its alternatives, or leave it out.

Order matters, because errors upstream contaminate referees downstream. Fix the trunk first, then the
constant arm offsets, then the time-varying arm errors, and contact last.

Always fit against **raw** (or the file with only upstream fixes applied), never against an earlier correction. pd's
pose-dependent correction moved blame between segments and confounded the mirror tests until the tests were
redone on raw.

---

## The process for one take

### 0. Look first

Play the take in the three-body patch (raw | previous | candidate), and note:
- what looks wrong
- where the dancer is symmetrical (mirror pairs)
- where hands touch the body
- where the dancer faces the camera

David's eye finds the problems; the measurements say which model explains them.

Patch generator: `make_three_body_patch_final.py TAKE [FRAME ...]` (raw, pd, corrected). It needs one or two frame buttons per problem; without frames it picks the six biggest arm changes. `make_three_body_patch_take8_final.py` is the take 8 historical one.

### 1. Trunk: lower-back roll drift → `build_root_tilt.py`, then `build_lowspine.py`

| | |
|---|---|
| Symptom | Whole spine leans to one side, slowly growing; one shoulder looks high across the file. |
| Cause | The lower-back sensor's roll drifts. The Vive tracker shares its belt, so the Vive drifts with it and **cannot referee it**. |
| Referee | The GVHMR hip line, face-on, as drift since the first 60 s, 10 s Gaussian (`DRIFT=1`). |
| Independent check | The difference in height between the left and right soles when both feet are planted should not get worse. |
| Apply | The world roll is spread down the trunk: pelvis 1, spine1 2/3, spine2 1/3. The chest (it has its own sensor) and the thighs keep their world orientation. Trans is moved so that spine1 stays put. |
| Evidence for the graded version | Lower-trunk lean against the video 4.9° → 2.1°; shoulder L−R along the trunk 3.1 → 2.0 cm; feet unchanged. Pelvis-only left the "high shoulder"; turning the whole lower spine over-corrected (−8° at 120 s). |

Don't fix the high shoulder at the collar (`build_collar.py`, superseded): it matched the video but explained nothing.

### 2. Constant upper-arm offset → `build_mirror_upper.py`, `build_hinge.py write const`

| | |
|---|---|
| Symptom | Arms consistently off in one direction (calibration, or a constant mounting offset). |
| Referee | **Mirror pairs** + face-on video together. The mirror alone has a mirror-symmetric null space (both arms wrong in mirror image satisfies it); the video splits it. |
| Pairs | Found from the **legs only** (the arms carry the error being fitted), both delayed mirrors (the same move later on the other side, at a steady lag) and synchronous symmetry. In slow material the automatic pairing is ambiguous: check the pairs against David's anchors (here 5848 ↔ ~43.2 s, not the automatic 45.6 s). |
| Model test | Fit constant, heading-dependent (magnetic) and hinge variants and compare them on held-out pairs. Take 8: constant won (L 12.8°, R 4.8°); the heading term and the hinge model (−2.1°) weren't supported. |

### 3. Elbow turn → session or take value (see `GVHMR_VIDEO_REFEREE.md`, elbow_turn)

| | |
|---|---|
| What | A constant roll of the upper arm about its bone, forearm held, so the elbow's hinge axis is right. |
| Source | The pd fit's per-take values (take 8: L 23.2, R 32.0), plus a correction of the constant part from step 4 (L +13.7). |

### 4. Elbow as hinge: time-varying upper-arm roll → `build_elbow_hinge.py` (mode `map`, `SIG_ROLL=40 RANGE=1`)

| | |
|---|---|
| Symptom | The elbow looks hyperextended or bends sideways (e.g. leading up to 5848). |
| Referee | In the upper arm's own frame, the forearm must lie in the flexion plane, bent forward. Rolling the upper arm about its bone brings it back without moving any joint, so only the skin changes (the elbow crease, and the twist at the shoulder). |
| **Guard 1: leverage** | At a shallow bend, a small sideways forearm error explains an off-plane forearm as well as a large roll does. A 20° bend with the forearm 20° to the side reads as "90° off-plane". Per frame, pick the roll that minimises roll²/40² + sideways residual²/6.7². The sideways spread (6.7°) and the carrying angle (−4.4°) come from the **uncorrected other arm**. |
| **Guard 2: anatomical range** | Keep the upper-arm twist (local, about the rest bone) inside the take's own raw range, both arms pooled and mirrored, 1st–99th percentile (take 8: −63..+120), as a soft wall. |
| What happened without the guards | Rolls up to 110° on nearly straight arms, twist out of range 28% of frames, and a "candy wrapper" shoulder at 12356. |
| Rejected | Swinging the upper arm instead (it moved the elbow, 460°/s added speed). Turning the forearm into the plane: the video rejected it (17.4 → 19.1°). |

### 5–6. Contact and penetration: evidence, not corrections (the triggered steps were removed 2026-11-07)

The torso-clearance step (`build_mc.py`) and the hand-on-back step (`build_backcontact.py`) are **out of the pipeline**. Each engaged when the pose crossed a threshold ("forearm inside the torso oval", "hand behind the pelvis and near the back") and pulled the arm locally, easing in and out. That invents movement wherever the trigger misfires.
- **What happened on take 7:** a left hand hanging beside the hip (23750–24250) qualified as "on the back". It was pulled 10–13 cm inward and back out, a transit toward the leg that never happened.
- **A tighter threshold is no fix:** it is the same mistake again.

The rule:
- **Corrections** are the output of a model of a sensor error, applied smoothly.
- **Contact and non-penetration** enter only as soft evidence when fitting those models, weighted by how likely the contact is.
- **What the models can't explain is information:** a missing error source to model, not a constraint to enforce.

What the error models alone leave on take 8:
- **5848:** the left forearm is 8 cm inside the chest (raw 10, pd 11).
- **Whole take:** forearm deep inside, left 3.6% of frames (raw 9.0%).
- **Holds:** the hands are off the back again.

A known candidate for what's missing is the gesture-linked error when the left arm crosses the body: the strap sits near the elbow, held tight against the body.

Next, to be built: add contact and penetration residuals to the arm-error fit (`build_hinge.py` / the mirror fit), then fit a gesture-linked term and test it on held-out data.

The two scripts stay on disk for the record (take 8's final2–final5 used them).

### 7. Check, then show

The checks:
- Face-on video, per segment, against the previous version.
- Mirror pairs.
- Forearm deep inside the torso (> 3 cm), excluding behind-the-back frames.
- Hand-to-back gaps in the holds.
- Elbow off-hinge at well-bent frames.
- **Upper-arm twist distribution against the raw range.**
- Added joint speed.
- The feet.

Then the three-body patch for David. To launch the app:
1. Point `dpg_app_default_patcher.json` at the patch.
2. Launch with `nohup … python -u dpg_system_main.py`.
3. Restore `{"path": ""}`.

Never start a second instance while his app is open.

---

## The palette at a glance

| Technique | Fixes | Referee | Script |
|---|---|---|---|
| Video hip-line roll, graded down the spine | lower-back sensor drift, lean, "high shoulder" | video hip line; feet check | build_root_tilt, build_lowspine |
| Mirror + video constant offset | per-arm calibration offset | mirror pairs (legs), face-on video | build_mirror_upper, build_hinge |
| Elbow turn | hinge-axis roll of the upper arm | Shadow's own elbow bends, video | (pd values; GVHMR_VIDEO_REFEREE) |
| Hinge roll with leverage and range guards | time-varying upper-arm roll (strap near the elbow) | elbow hinge plane, other arm, twist range | build_elbow_hinge `map` |
| (Torso clearance, back contact) | removed: posture-triggered; to return only as soft evidence in the arm-error fit | torso surface, skinned mesh | build_mc, build_backcontact (not run) |
| Three-body patch | David's eye | — | make_three_body_patch_final |

Tried and set aside (don't repeat without a new reason):
- Vive-referenced root roll (the tracker drifts with the sensor).
- Whole-take mirror forearm fit (asymmetry floor, hyperextension).
- Heading or magnetic upper-arm model on this take.
- Upper-arm hinge-deformation model.
- Collar lowering.
- Hinge roll without guards.
- Hinge swing.
- Forearm turn.

---

## Verdict lines (printed by the driver)

- `CHECK feet`: per 30 s window, whether the planted feet's left-minus-right height gets better or worse with the trunk roll. More worse than better is a WARNING: the video and the feet disagree, and the trunk roll is unconfirmed. Take 7: the video shows a 13° hip drift late in the take, while the feet get worse there.
- `CHECK const offset`: held-out video change for each arm's constant offset. Worse in both halves is a WARNING. Normally `combine_arms.py` then takes that arm from pd anyway; check that it did.
- `combine_arms`: which source each arm came from, and both scores.

## Traps

- **Corrections confound referees.** Fit each step on raw plus upstream fixes only.
- **A shared mount can't referee itself** (Vive on the lower-back belt).
- **GVHMR**, use it with care:
  - its depth is poor side-on (use face-on, |facing| < 40°)
  - it can't see hands behind the back
  - it has a level, symmetric body prior
  - in some stretches there's no face-on footage at all (5600, 12356), so the body referees must carry those moments.
- **Mirror symmetry has a null space**: the whole pattern mirrored also satisfies it. It needs the video, or a fixed side, to break it.
- **A roll about a bone moves no joint.** Joint-position checks can't see it, so check the skin: the twist at the shoulder and the elbow crease.
- **Shallow bends hide the hinge**: below about 30–40° a sideways forearm says little about the upper arm.
- **Caps interact with skip rules**: decide on uncapped quantities.
- **Window padding** creates large pulls at the edges; cap per sample.
- **Short takes** (take 4, 7 s):
  - The joint-speed sync can lock onto a false peak, so check the lag against root height. Set it with `SYNC_OFFSET` in `compare_shadow.py`.
  - There's no 60 s drift reference, so the trunk step returns zero.
  - There are no delayed mirror pairs and no held-out blocks, so the constant offset is fitted but can't be verified.

## For a new take

1. Run GVHMR on the video; build the take's `S11_takeN.npz` video file: frame times, offset, joints, facing (`run_s11.sh`; all nine Subject 11 takes have one).
2. `correct_take.sh N`.
3. Read each stage's summary before trusting the result. Look for:
   - held-out hip line improving
   - the feet not getting worse
   - the mirror pairs being plentiful
   - the twist staying inside range
   - which arm source won
   - for a short take, whether the sync lag agrees with root height (see Traps)
4. Reuse session-level numbers where they should hold (constant offsets, carrying angle, twist range, elbow turn) and **re-measure** them per take: check whether take 8's L 12.8 / R 4.8 hold across the session before relying on them.
5. Three-body patch (made by the driver), David's review, iterate. Other subjects (old fusion, other folders) need the paths in the scripts generalised too; only Subject 11 is wired.
6. `summarize_takes.py 1 2 ... 9` puts every take in one table: trunk support, offsets and their verdicts, arm sources, unexplained penetration.

## Status and open questions

The results for all nine Subject 11 takes, the take 4 sync story and the list of open questions and next steps are in `GVHMR_VIDEO_REFEREE.md`, section "2026-11-05 to 11-08: Subject 11 corrections".

The short version:
- the trunk roll is supported only on take 8 and harms take 6, so it should become a per-take decision
- pd supplies most arms
- left-forearm penetration that no model explains, in takes 2, 3, 5 and 6, is the main lead: an error tied to the arm crossing the body, to be fitted with contact as soft evidence
