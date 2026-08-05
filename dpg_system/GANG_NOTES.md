# Torque gangs — working notes

**Status (2026-08-05):** core + nodes built and passing. Preset weight signs
are UNVALIDATED — that is the next thing to do, and it needs live data.

Files:

    dpg_system/gang_core.py    formalism, preset table, compiler, registry
    dpg_system/gang_nodes.py   torque_gang / gang / torque_residual nodes
    dpg_system/dpg_app.py      'gang_nodes' added to optional_import
    dpg_system_config.json     "gang_nodes": true

---

## The idea

Some torque is only legible as a group. Forward/back bending of the spine is
not any single joint — it is distributed across spine1, spine2 and spine3, and
no one of them *is* the movement. Same for triple extension in a leg, and for
the collar/shoulder split that SMPL models but no listener could perceive
separately. So the useful sonic parameter is often the group, not its members.

A **gang** is a named linear functional on the torque field:

    s = sum over j of  w_j . tau_j

The weight vector carries three things at once: axis selection, anatomical
sign, and relative contribution. Triple extension is the clearest case — hip
extends, knee extends, ankle plantarflexes, and those don't share a sign in
local-frame torque. The weights absorb the flips so the whole leg push reads
as one rising scalar.

The muscle atlas already contained the degenerate case of this: a muscle is
`(joint, flex_axis)` → scalar, i.e. a gang with one term.

## Three values, not one

Ganging forces a choice single joints don't have: what happens when
contributions oppose each other. Both answers are useful, so compute both,
plus the ratio between them:

    net       = sum  w.tau        signed; cancellation allowed
    total     = sum |w.tau|       magnitude; no cancellation
    coherence = |net| / total     in [0, 1]

**Coherence is the thing ganging gives you that per-joint torque cannot.** A
spine hinging as one unit and a spine curling at the waist while extending at
the chest have the *same total* and completely different coherence. 1.0 = the
group acting as a single unit; 0.0 = pure internal counter-effort, work being
done against itself.

Verified through the node path:

    hinge (100, 100, 100):   net=1.0800  total=1.0800  coh=1.0000
    curl  (100, 100, -100):  net=0.5200  total=1.0800  coh=0.4815

Suggested mapping: `net` → pitch/direction, `total` → amplitude, `coherence`
→ consonance/noisiness/detune spread. They're perceptually independent.

## What a gang must declare

Weights alone under-specify it. Four more fields, all folded in at compile
time so none of them costs anything per frame:

- **stream** — gravity / dynamic / passive / total. The biggest lever on how a
  gang *sounds*. Spine flex on gravity torque is postural load: slow,
  continuous, drone material. The same weights on dynamic torque are bending
  *effort*: transient, percussive. Same gang, different instrument.
- **frame** — local (parent-relative) is nearly always right. "Spine flex"
  means each vertebra bending relative to the one below. World frame sums
  posture, not action, and would drift as the performer turns.
- **normalization** — divide by the joint's entry in `max_torque_array` before
  summing, or the lumbar joint (250 N·m) swamps everything and the gang is a
  single-joint signal wearing a group's name. Normalized, each term is
  "fraction of this joint's capacity" and the weights become purely aesthetic.
- **reduction** — `net` is linear and commutes with filtering; `total` and
  `coherence` do not. Order is fixed: smooth per-joint first, then reduce.

## Which joints gang, and why

**Spine — the strongest case, all three axes.** Sagittal flexion, lateral bend
and axial twist are all genuinely distributed across spine1/2/3. Twist is the
most distributed (thoracic does most axial rotation, lumbar almost none), so
summing there is the most physically honest. Pelvis deliberately excluded —
treat it separately, it's the hinge to the root.

**Neck + head, kept separate from the spine.** Head motion is gaze-carrying
and reads as a different order of gesture. Folded into spine flex it just adds
a small noisy term.

**Collar + shoulder — gang almost unconditionally.** SMPL's scapular /
glenohumeral split is model-dependent and unstable, and nobody could hear the
two separately. Ganging removes a modelling artefact rather than adding an
abstraction.

**Leg support chain (hip/knee/ankle sagittal).** Triple extension. Tightly
coupled in gait, jumping, rising from the floor; probably the most legible
whole-body gesture available.

**Arm reach (shoulder + elbow).** Wrist deliberately left out — its torques
are small in SMPL and on Shadow it carries the forearm/hand yaw magnetization
error, so ganging it in imports that noise into an otherwise clean signal.
Available as its own preset when that's what you want.

**Bilateral pairs → mid/side.** For any left/right pair form both:

    common       = (L + R) / 2     support: rising, landing, both legs pushing
    differential = (L - R) / 2     alternation: gait, weight shift, asymmetry

This is exactly mid/side. Differential is where gait lives.

**Contralateral / diagonal.** Left shoulder with right hip and its mirror —
the X of gait, the spiral. Invisible to per-joint *and* bilateral views alike;
it only exists as a relation between opposite limbs.

## What's built

15 presets, 9 of them bilateral → 42 gangs across all preset/side
combinations, 111 terms.

    spine_flex / spine_bend / spine_twist
    head_flex / head_turn
    leg_push (triple extension) / hip_flex / leg_abduct / leg_twist
    arm_elevate (collar+shoulder) / arm_reach / wrist_flex / shoulder_girdle
    contralateral_swing / counter_rotation

Nodes:

    torque_gang <preset> [side] [stream]      alias: gang
        inlets  : torque, gravity, dynamic, passive + gang/side/stream combos
        options : normalize, gender, invert
        outlets : net, total, coherence

    torque_residual [stream]
        outlets : residual (24 per-joint magnitudes), magnitude (scalar)

Examples:

    torque_gang spine_flex
    torque_gang leg_push differential dynamic
    torque_gang arm_reach left

## Design decisions — settled, don't re-litigate

**Term-level compilation.** The compiled artifact is a `(terms, 288)` matrix
whose rows are grouped contiguously by gang, with `reduceat` offsets at the
group boundaries. 288 = 4 streams × 24 joints × 3 axes, so stream selection is
a choice of column, not a branch. Evaluation is five numpy calls regardless of
gang count: matmul, reduceat, abs+reduceat, divide.

**One row per joint, not per axis.** `total` takes `|w·tau|` per *joint*.
Splitting a joint's two axes into two rows would take two absolute values
where the definition wants one, so the compiler merges them. This is easy to
get quietly wrong and is explicitly tested.

**Symbolic axes, never raw vectors.** The local frame is not oriented the same
way all over the body (`smpl_processor.py:3574`):

    Legs  (hip/knee/ankle):       bone Y.  X=flex/ext, Y=twist, Z=abd/add
    Arms  (shoulder/elbow/wrist): bone X.  X=twist,    Y=flex/ext, Z=abd/add
    Spine (pelvis/spine/neck):    bone Y.  X=flex/ext, Y=twist, Z=lat bend

**Arms carry flexion on Y where everything else carries it on X.** A preset
written with raw axis vectors would silently mean "flexion" on the spine and
"twist" on the arm. Presets name the anatomical role (`flex`/`twist`/`abduct`)
and the compiler resolves the axis per joint family.

**One node = one gang.** Flexibility comes from patching several; efficiency
comes from the shared bank. Mirrors synth: one node = one unit.

**Trigger retargeting.** `smpl_torque` sends `torque_vectors` FIRST, then
dynamic/gravity/passive (`smpl_nodes.py:2475-2489`). A node triggering on the
`torque` inlet would evaluate this frame's total against **last frame's**
dynamic — a one-frame skew that would be nearly invisible. `triggers_execution`
is read when data is received (`node.py:876`), so each node points execution
at only the inlet its stream names and leaves the others passive.

**Cache keyed on input identity, not first-wins.** All gang nodes fed from one
`smpl_torque` see the same array objects, hit one cache entry, and the bank is
computed once. A patch with genuinely different sources stays correct (it just
computes twice) instead of having whichever node ran first decide what
everyone else reads.

**Registry mirrors `synth_core.SynthGraph`.** Cheap per-frame signature
comparison, recompile only on change — catches widget edits, patch load,
paste, undo and deletion through one path.

## Verified

`gang_core`: 81 checks. `gang_nodes`: 38 checks. Both suites pass. Scripts
live in the session scratchpad (`test_gang_core.py`, `test_gang_nodes.py`) —
NOT checked in, since the repo has no Python test convention (tests here are
patches). Worth re-creating or checking in if this work resumes.

Covered: net and total against a brute-force reading of the definition across
all 42 gangs; per-joint row merge; batched `(F, 288)` ≡ frame-by-frame;
mid/side algebra; streams not bleeding; capacity normalization; residual
orthogonality under parallel claims; empty-gang rejection (reduceat with equal
offsets returns a stray element, not zero); the stale-stream trap; the load
guard; side-list coercion across a bilateral → non-bilateral round trip; and
five nodes sharing exactly one evaluation per frame, instrumented and counted.

Performance: **5.1 µs/frame** for 42 gangs / 111 terms, flat in gang count.
Compile 0.95 ms — comfortably inside a frame, so live weight editing needs no
special path.

## Open

**1. Preset signs are unvalidated.** ← start here

Which rotation direction is positive in each joint's local frame was never
checked against real data, and a wrong sign turns a coherent gang into a
cancelling one. The test is built in: drive a gesture that plainly reads as
unified and watch `coherence`. A gang sitting near zero during unified
movement has a flipped weight. Use the `invert` option to try the flip before
editing the table. `leg_push` (three joints, alternating signs) is the most
likely to be wrong and the most worth getting right.

**2. `MAX_TORQUE_NM` duplicates `SMPLProcessor._compute_max_torque_profile`.**
Importing the real one pulls in torch and the SMPL model to read a small dict.
Two copies that must be kept in step; commented on both sides.

**3. Designed but NOT built** — discussed, decided, not implemented:

- *Free-form terms* as a property string on the node, e.g.
  `spine1:flex 1.0, spine2:flex 1.0, spine3:flex 0.7`, parsed at compile.
  One node per gang keeps patches readable; a node per term would explode them.
- *Gang-of-gangs by patch cord.* Composition is linear, so a gang consuming
  another gang's `net` can be **flattened into a single flat term list at
  compile time** — hierarchy in the patch, zero cost at runtime. This is the
  part that would most earn the compiler's keep.
- *Nonlinear staging.* Flattening only works through `net`. A gang consuming
  another's `total` or `coherence` can't be folded and needs a real topological
  sort into stages, one matmul per stage. The algorithm to copy is already in
  `synth_core.py:3300`. Most patches would compile to a single stage.

**4. Not yet driven by real data at all.** Everything so far is synthetic
torque. Nothing is known about the actual dynamic range of `net`/`total` in
performance, which is what the scaling into audio parameters depends on.
