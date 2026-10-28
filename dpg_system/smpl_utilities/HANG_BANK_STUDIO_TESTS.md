# Arm-hang bank: studio tests

Tests to run the next time the suit is on, to find what makes lowered arms hang shifted
to one side together. The evidence so far is in `MAGNETIZATION_FINDINGS.md` (the "Asymmetric
hang" entries, 2026-10-27).

## What we know

- **What it looks like:** a good T-pose, then once the arms are lowered both arms hang shifted
  to one side together, as if the pair has turned about the chest's forward axis.
- **It reads 0 in a T-pose.** Subject 2 take 3 has three pretend T-poses at very different
  facings, and all three read level within 2°. Going back to the T-pose makes it disappear.
- **It's always toward the dancer's right.** Across 48 takes and 6 subjects it is never clearly
  the other way. Its size depends on the subject or session:

  | Subject | Bank with arms down |
  |---|---|
  | 7 | −16 to −20° |
  | 2 | −8 to −11° |
  | 1, 3, 6 | about −1 to −11° |
  | 4 | about −7 to +1° |

  If each sensor's own magnetization were the cause, the direction would vary from suit to suit.
  So the cause is fixed to the body and lopsided the same way every time.
- **It doesn't depend on facing.** The bank is the same whichever way the dancer faces.
- **Part of it creeps.** Most appears while the arms come down, but with the arms then held still
  it grows by about 5° over about 2 seconds. Back in the T-pose it takes about a second to fade.
  A drift with the body still points to the sensor-fusion filter being pulled toward a wrong
  reference, possibly a disturbed magnetic field.
- **It isn't a baked-in compass error.** A heading error baked in at the T-pose would swing the
  hanging arms forward and back, one each way. That is not seen.
- **The upper arms bank more than the forearms in the big cases.** In Subjects 7 and 2, the
  upper arms bank about twice as much as the forearms, which hang beside the belt. That argues
  against the belt for the large effect. In Subjects 1, 4 and 6, where the bank is small, the
  forearms bank more.
- **Field notes:**
  - careful recalibration doesn't fix it;
  - readjusting the upper-arm or shoulder-blade sensors doesn't fix it;
  - changing suits sometimes fixes it;
  - squaring the root sensor occasionally helps;
  - it may be worse on muscular performers.
- **Belt layout:** the battery is on the left and heavier, the controller / wifi unit is on the
  right, and the root sensor is in the middle of the lower back, on the belt.
- **Sign convention:** in the measurements, + means toward the dancer's left. The bank seen so
  far is negative.

## Recording protocol (use for every test take)

Record one short take per condition, and name it after the condition (e.g. `hang_belt_reversed`):

1. T-pose, still, 5 s.
2. Arms down, relaxed and still, 10 s. Hold still: the creep shows over the first 2–3 s.
3. T-pose, still, 5 s.
4. Arms down again, 10 s.
5. Optional: with the arms down, turn slowly through a full circle (about 20 s).
6. T-pose, 5 s.

Write down the frame or time where each step starts, or call them out on the video. Note the
suit, the performer, and anything moved or changed.

Always record a **baseline** take first, with the suit exactly as normally worn. Each test is
judged against it, in the same session.

## Tests, in order of value

### 1. Reverse the belt (decisive for the belt)
Wear the belt turned round, with the battery on the right and the controller on the left. Recalibrate, then run the protocol.
- **Bank flips toward the dancer's left:** the belt (battery or controller) causes it.
- **Bank shrinks but keeps its side:** the belt contributes, but something else does too.
- **Bank unchanged:** not the belt. Look at arm cable routing, the torso sensors, or Shadow's
  processing.

If the cables won't allow a reversal, use **1b**: move the battery off the body (an assistant
holds it about 1 m away, or it sits on a long lead), recalibrate, and run the protocol. Then
do the same with the controller, if that's possible.

### 2. Field probe (measures the disturbance directly)
Use a phone magnetometer app (one that shows total field in µT), or a spare Shadow sensor with
the `shadow_sensor` and `mag_offset` tools. With the performer standing, arms hanging, hold the
probe still at:
- where the left upper-arm sensor sits, and the right;
- where the left forearm sensor sits, and the right;
- the root sensor position;
- the chest sensor position.

Read each spot with the belt worn and with the belt taken off and moved away. Keep the probe in
the same orientation each time, and use the same spots.

Clean field here is about **54.6 µT**. A 10–15° bank needs roughly **10–15 µT** of sideways
disturbance at the hanging sensor, which is easy to see. Hold the probe still at each spot;
sweeping it misleads, because of the floor's field gradient (see the new-space field survey
notes).

### 3. Root tilt, deliberately
To find out whether the root's tilt drives the bank:
1. Calibrate with the root sensor carefully square, then run the protocol.
2. Tilt the root sensor about 10° sideways (rotated in the plane of the lower back), recalibrate,
   and run it again.
3. Tilt it 10° the other way, recalibrate, and run it a third time.

- **Bank follows the tilt:** the root is the reference the arms inherit.
- **Bank ignores the tilt:** rule the root out.

### 4. Turn with the arms down
This is protocol step 5, so it's already in each take. It checks that the bank doesn't depend on
facing.
- **Bank changes as the dancer turns (once per turn):** the earth's field is involved directly,
  through ordinary magnetization.
- **Bank stays steady:** the cause is fixed to the body.

### 5. Swap the upper-arm sensors (only if the harness allows)
Exchange the left and right upper-arm sensor units, recalibrate, and run the protocol.
- **Bank changes or flips:** the units themselves matter. Run `mag_offset` on both.
- **Bank unchanged:** the units are not the cause.

### 6. Different suit, same performer, same day
If a second suit is available, run the protocol in it. Sizes and signs collected over sessions
will test the "changes suits", "muscular build" and "root square" impressions.

## Reading the results

- **In the patch:** the `hang_symmetry` node reads the bank from any take. Its `shift` outlet
  gives degrees, + toward the dancer's left. Give it the arms-down stretches as `ranges`, e.g.
  `1200-1800, 2400-3000`, with `every 1` for a full take.
- **For the timing and the T-pose return:** ask Claude to measure the bank against time after
  each lowering. Point to the take and the step frames; the method is in the findings ledger.
- **What would settle it:** a bank that flips with the belt reversed (test 1), plus a field
  disturbance of the right size at the hanging upper-arm sensors (test 2).
