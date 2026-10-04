# Per-take arm-offset tuning procedure

The Shadow arm-sensor offsets **evolve across a session** (progressive magnetization),
so every take needs its own pass. We know the structure now, so each take is a quick
sequence: one automatic symmetry fit, then dial each joint against the dancer's video,
using `--sweep` to emit a few candidates at a time.

Tool: `correct_upper_arm_offset.py`. Originals are never modified; each run writes a new
`<take>_armfix<tags>.npz`. Render the variants, pick the best value, move to the next dial.


## Step 0 — anchor frames (once per take)
Find a few frames whose **upper body should be bilaterally symmetric** and (ideally) one
**relaxed arms-down** window. Pass them as `--sym a:b,a:b,...` and `--relax a:b`.
If a take has no usable symmetric frames, use `--no-fit` and recover the swing manually
with the abduction dials instead.

## Step 1 — symmetry fit (automatic)
```
correct_upper_arm_offset.py TAKE --sym 0:1721,3767:4000 --relax 10430:11242 --twist 0
```
Render; confirm the two arms are mirror-symmetric. The printed `asym_before/after` should drop.
The fit produces:
- **`cl`, `cr`** — per-arm constant upper-arm world correction (an average over the L/R
  symmetric anchors plus a verticality term on the relaxed window).
- **`hy_l`, `hy_r`** — per-arm heading-dependent yaw curve `δ(ψ) = b·sin ψ + c·cos ψ` (a
  per-frame world-vertical yaw vs upper-arm azimuth — the once-per-rev hard-iron signature).
  Applied to the humerus only (forearm/elbow untouched). Catches pose-varying L/R asymmetry that
  a constant `cl`/`cr` can only average over; halves the per-pose mirror residual on heading-
  heavy takes. Print line: `heading-yaw amp |L|=… |R|=…`.

The forearm/elbow are not touched by the fit; they're tuned with the dials below.

## Step 2 — dial the joints, in this order (sweep, render, pick)
Each command holds the already-chosen dials and sweeps one. Carry the chosen `--sym/--relax`
(or `--no-fit`) on every command.

**The right magnitudes differ per take** (the offsets grow over the session). So scan COARSE
first with the range form `--sweep twist:-80:20:20` (start:stop:step, stop inclusive), find
the neighbourhood, then refine with explicit values `--sweep twist:-55,-50,-45`. Coarse->fine
keeps it to two render passes per dial.

1. **Twist** — shoulder/deltoid roll about the upper-arm long axis (+X, forearm-fixed —
   recomputes only Shoulder + Elbow locals): `--sweep twist:-40,-60,-80,-90`.
2. **Elbow** — bend vs video (hands on hips?): `--twist <T> --sweep elbow:12,18,24,28`
3. **Wrist** — over-flexion: `--twist <T> --elbow <E> --sweep wrist:10,15,20`
4. **Wrist-twist** — thumb forward/back: `... --sweep wtwist:-15,-10,10,15`
5. **Abduction (per arm)** — shoulders out + L/R balance:
   `... --sweep abl:8,12,16` then `... --abduct-l <L> --sweep abr:4,8,12`
   (positive = outward; use the L/R difference to kill any residual asymmetry)

## Step 3 — final render + record
Run once with all chosen values (no `--sweep`) to write the final file, e.g.:
```
correct_upper_arm_offset.py TAKE --sym ... --relax ... \
  --twist -55 --elbow 22 --wrist 12 --wtwist -12 --abduct-l 10 --abduct-r 6 \
  -o TAKE_armfix.npz
```
Record the per-take values (a row per take) so the set is reproducible.

## What each dial fixes (when something looks wrong)
- arms not mirror-symmetric -> redo Step 1 / better anchor frames
- shoulders pinched/deltoid twisted -> `--twist`
- elbows too bent / hands too high -> `--elbow` (open) ; per-frame hinge, safe
- wrist over/under-bent -> `--wrist`
- palm/thumb rolled wrong -> `--wtwist`
- arms tight to body, or one arm further out -> `--abduct-l` / `--abduct-r`

All dials are constant per take but use world/cross axes (not body-local), so they stay
true across poses and don't distort joints. Reference (Subject7 jathiswaram, end of session):
`--twist -60 --abduct-l 12 --abduct-r 5 --elbow 28 --wrist 15 --wtwist -15`.
