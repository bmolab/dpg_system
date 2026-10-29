# Arm-hang bank: studio tests

Tests to run the next time the suit is on, to find what makes lowered arms hang shifted
to one side together. The evidence so far is in `MAGNETIZATION_FINDINGS.md` (the "Asymmetric
hang" and "Hang bank timeline" entries, 2026-10-27 to 10-29). The analysis scripts are in
`hang_bank_analysis/`.

## What we know

- **What it looks like:** a good T-pose, then once the arms are lowered both arms hang shifted
  to one side together, as if the pair has turned about the chest's forward axis. It reads 0 in
  a T-pose and arrives with the lowering itself: it is complete about half a second after the
  arms come down, and doesn't grow while they hold still.
- **It is mostly a property of Shadow's old fusion algorithm.** Shadow changed its fusion on
  2025-12-18. Every session before that leans toward the dancer's right, by 3 to 18° (Subject 7,
  16 to 20°). Sessions since then range from about −8 to +9°, with both signs, and most are
  within ±5°.
- **What remains drifts during a session.** In 13 of 18 sessions the bank moves toward the
  dancer's left as the session goes on, typically 4 to 6° per hour. Subject 13 drifted from
  −10 to +4 over an hour, straight through a recalibration. The cause is unknown: sensor warming,
  battery discharge, the filter adapting, or warm-up choreography early in the session.
- **It isn't the sensors' own magnetization, at least lately.** On 2026-08-13 the arm sensors were
  measured clean (about 1 µT, worth about 1° of tilt). The next day's session still read +3°.
  The sign then changed between August (+3 to +6) and October (−2 to −3), so new magnetization
  since August is possible and worth checking (test 2).
- **Not the calibration posture, not the root sensor.**
  - A simulation shows that how a sensor sits on the body cancels out in calibration. Tilt of
    the root or pelvis is shared by chest and arms, and the bank is measured against the chest.
  - Only the arms being turned against the torso during the T-pose would produce it. Arm
    alignment is checked carefully before every calibration, and the Subject 7 video T-poses
    are clean.
- **No step at the studio move (2026-07-06).** Position on the floor within a session is untested.
- **Forward/back:** in some sessions (Subjects 12, 18, 19, 20) the right arm hangs 5 to 20°
  further forward than the left in nearly every take.
- **The forearms are a separate effect.** They hang angled forward, so a forearm heading error
  shows as a sideways shift. Don't compare forearm and upper-arm numbers.
- **Sign convention:** in the measurements, + means toward the dancer's left.

## Recording protocol (use for every test take)

Record one short take per condition, and name it after the condition (e.g. `hang_start`):

1. T-pose, still, 5 s.
2. Arms down, relaxed and still, 10 s. Holding still keeps the reading clean of choreography.
3. T-pose, still, 5 s.
4. Arms down again, 10 s.
5. Optional: with the arms down, turn slowly through a full circle (about 20 s).
6. T-pose, 5 s.

**Log with every take:**
- the suit number;
- the time since the suit was powered on;
- the battery level;
- where on the floor it was recorded;
- any recalibration or sensor re-seating since the last take.

Without the suit number, sessions can't be compared.

## Tests, in order of value

### 1. Start, middle, end (cheap; do it every session)
Record the protocol take right after calibration, again in the middle of the session, and at
the end, standing in the same spot and facing the same way.
- **Bank changes steadily with time:** the drift is real, not choreography. Next, check whether
  it follows warming (time since power-on) or battery level. A short session after a long
  warm-up with the suit on, without recording, separates the two.
- **Bank steady:** the session drift seen in the archive is a choreography effect (warm-ups
  first).

### 2. Fresh magnetometer sweep (checks for new magnetization)
Run `shadow_sensor` and `mag_offset` on the upper-arm, forearm, shoulder-blade and chest sensors
of all three suits, as on 2026-06-18 and 08-13. Compare with those sweeps.
- **New offsets on the arm or shoulder-blade sensors:** new magnetization. Recalibrate those
  sensors, then repeat test 1 to see whether the bank shrinks.
- **All clean:** the remaining bank isn't from the sensors themselves.

### 3. Floor position
In one session, record the protocol take (steps 1–4) at several marked spots: the middle of the
floor, near each of the areas known to have interference, and one well away from them. Keep
the facing the same.
- **Bank changes with position:** the floor field is involved. The hanging arms sit lower, closer
  to the floor, than the T-pose arms.
- **Bank the same everywhere:** rule out the floor.

### 4. Turn with the arms down
This is protocol step 5, so it's already in each take. It checks whether the bank depends on
facing.
- **Bank changes as the dancer turns (once per turn):** the earth's field is involved directly,
  through magnetization.
- **Bank stays steady:** the cause turns with the body.

### 5. Belt reversal
Wear the belt turned round, with the battery on the right and the controller on the left.
Recalibrate, then run the protocol. Lower priority now: the old fusion algorithm explains the
large 2025 bank, and the remaining one is small and changes sign.
- **Bank flips:** the belt contributes.
- **Bank unchanged:** not the belt.

If the cables won't allow a reversal, move the battery off the body (an assistant holds it about
1 m away, or it sits on a long lead) instead.

### 6. Swap the upper-arm sensors (only if the harness allows)
Exchange the left and right upper-arm sensor units, recalibrate, and run the protocol.
- **Bank changes or flips:** the units themselves matter.
- **Bank unchanged:** the units are not the cause.

### 7. Different suit, same performer, same day
If a second suit is available, run the protocol in it. With suit numbers logged, sizes and signs
across sessions will show whether the bank follows the suit.

### Dropped
- **Deliberate root tilt.** The simulation shows that a rotated root sensor cancels out in
  calibration and can't produce the bank.
- **Field probe at the hanging sensor positions.** Superseded by test 2's direct sweep and
  test 3's floor positions.

## Reading the results

- **In the patch:** the `hang_symmetry` node reads the bank from any take. Its `shift` outlet
  gives degrees, + toward the dancer's left. Give it the arms-down stretches as `ranges`, e.g.
  `1200-1800, 2400-3000`, with `every 1` for a full take.
- **Across a session:** `hang_bank_analysis/session_drift.py` fits the bank against recording
  time, per session. Add the new session's folder to the list at its top.
- **For the timing and the T-pose return:** ask Claude to measure the bank against time after
  each lowering. Point to the take and the step frames.
- **Size that matters:** take-to-take scatter in clean sessions is about ±3°. Compare session
  medians, or protocol takes in the same spot, not single choreography takes.
