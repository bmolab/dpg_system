"""
Hybrid mocap / physics drive for an SMPL body.

Two physics cores.  The default is pybullet (`engine` = bullet, see
smpl_bullet.py): the body is a real articulated multibody with full inertia
tensors, impulse-based contact and friction, self-collision and a coupled
solve -- driven joints are motors, the driven root is a constraint retargeted
through each frame, and release is switching those off, so the momentum a
release inherits is the engine's own.  The native core below (`engine` =
native) is the decoupled per-joint model this file was written around; it
remains as the fallback when pybullet is not installed, and its notes follow.

The smpl_torque pipeline is an *inverse* dynamics analyzer: pose in, torques
out.  This module runs the same body model *forward* for a chosen subset of
joints, so that some joints follow the motion capture stream while the rest
fall under gravity, joint limits and the inertial forces of the body they are
attached to.

Everything about the body itself is borrowed from SMPLProcessor -- segment
masses and lengths derived from the betas, bone offsets, the joint limit
tables, and the per-joint actuator ceilings.  What is added here is the part
the analyzer has no need for: a forward integration step, and a model of the
inertial environment a free limb experiences.

Model notes and honest limits
-----------------------------
*   Like the analyzer, each joint is integrated with a single scalar moment of
    inertia for the subtree hanging off it (parallel-axis sum of the segments).
    There is no articulated-body mass matrix, so a free limb does not push back
    on its parent.  For hybrid drive that omission is deliberate rather than
    accidental -- the driven joints are kinematically prescribed, which is
    exactly the assumption the decoupled model makes.

*   The free limb does feel its parent, though, via the transport (D'Alembert)
    forces below.  This is the term that makes a released limb read as attached
    to a moving body rather than merely hung from a hook.

*   A free chain is simulated in the accelerating, rotating reference frame of
    its pivot -- the joint where it meets the driven body.  A point at offset r
    from the pivot sees an effective gravity of

        g_effective = g_world - (a_pivot + angular_acc x r + W x (W x r))

    where a_pivot is the pivot's linear acceleration, W the pivot frame's
    angular velocity and angular_acc its angular acceleration -- all determined
    entirely by the motion capture stream.  The three correction terms are, in
    order: the body accelerating out from under the limb, the whip when the
    body starts or stops rotating, and the outward fling of a spin.

    The Coriolis term, -2 W x v_relative, is omitted.  It depends on the limb's
    own velocity within the pivot frame and is the smallest of the four; the
    transport model is therefore exact for a limb at rest in the pivot frame
    and approximate for one that is swinging fast while the body also rotates.

*   Joint limits are this module's own (RAGDOLL_JOINT_LIMITS), a box of
    min/max per local axis rather than the analyzer's cone-or-hinge.  The
    analyzer's table is deliberately left alone -- it feeds smpl_torque, whose
    output has been validated at length -- but it is far too loose to run
    forward: anything it does not name falls through to a 180 degree cone with
    almost no stiffness, so ankles, wrists, feet and the head come out as free
    balls and a knee modelled as a cone bends forwards.  A per-axis box can say
    what a joint actually does: bends a long way one way, not at all the other,
    twists a little.

*   Self-collision is modelled, as capsules against capsules -- see
    SELF_CAPSULES and _self_contacts.  It is coarse: eleven capsules, forty-odd
    pairs, and radii chosen to be the largest that leave the body clear of
    itself in ordinary poses rather than anatomically honest ones.  Without it
    a forearm sank as much as 18 cm into the torso; with it the worst overlap
    over the same falls is about 1.5 cm, which is the penalty spring's
    compliance and reads as flesh.  Contact between parts is internal, so it
    contributes no net force or torque on the centre of mass -- a body cannot
    push itself anywhere by folding up -- and only the joints feel it.

*   A body on the ground settles.  It did not, for a long while: a landed
    ragdoll hopped indefinitely and rested on one point where it should lie on
    several, and this was written up as a structural limit of the decoupled
    model.  It was not.  The joints were being driven by a contact force
    frozen across the frame -- a spring held constant over a light limb's
    whole oscillation -- and re-evaluating it at each joint substep removed
    the injection at source.  A limp body now lies down on hips, spine, knees
    and ankles, bounce under a centimetre.

The root
--------
The root's six degrees of freedom are not a joint and are not driven by a joint
torque.  In flight the governing statements are conservation laws, and they are
used directly:

*   Gravity acts at the centre of mass, so the centre of mass is a parabola.
    It is integrated exactly (the half-a-t-squared term is included), and the
    root is then placed each frame so that the body's actual centre of mass,
    recomputed from the real kinematics, lands on it.  Limb motion therefore
    cannot make the trajectory drift, and the root correctly wanders around the
    centre of mass as the body turns -- only the centre of mass is ballistic.

*   Gravity exerts no torque about the centre of mass, so the total angular
    momentum is constant.  The root's spin is solved from it each frame:

        total = full inertia tensor x root spin + momentum of the limbs' motion

    which is what makes a released somersault keep somersaulting, and speed up
    when the limbs tuck.  This needs a real 3x3 inertia tensor, built here from
    the same segment table (see _body_dynamics); the analyzer's scalar inertia
    could not express it, and could not tumble.

The velocity and spin that a release inherits come from differencing the
capture, which amplifies its noise by the frame rate -- and translation is the
noisy channel.  Both are therefore carried as smoothed running estimates with
the averaging lag cancelled, so smoothing costs noise rejection only and not
trajectory accuracy.

Ground contact
--------------
Contact is a penalty model at a set of points with a radius standing for the
flesh around each joint centre (see CONTACT_POINTS).  Normal force is
Hunt-Crossley -- damping scaled by penetration, so it rises continuously from
zero at touchdown and can never pull -- with Coulomb friction regularised below
a small slip speed.  Stiffness is set from how far the body's own weight should
sink.  It enters the root as the external force and torque in the momentum
equations, and reaches the joints in full -- pin joints transmit force, and a
hand's load has to be able to fold the elbow above it -- re-evaluated at each
joint substep from the chain's current position, since a spring force frozen
across a frame is unstable at the rates a loaded hand kicks its wrist.

Friction is a spring to where the point touched down, capped at the cone, not a
brake on velocity: a planted hand reads several centimetres a second of
estimation noise, and Coulomb friction regularised at a slip speed handed it
two thousand newtons for that -- two planted hands on a rotating body fought
each other and stripped a released cartwheel of its turn.

Five separate bounds, each of which a body found a way to need:

*   a point's force is capped by the mass it is actually decelerating -- the
    subtree below the first free joint above it -- so a toe on a limp leg
    develops a foot's worth of force, not a body's, and a limb swinging into
    the floor cannot deliver a body-turning impulse on its own (a toe at the
    old flat cap took most of a cartwheel's rotation in one frame),
*   the spring responds to penetration only up to a limit,
*   the damping term saturates with approach speed,
*   one point cannot claim more than its share of the total force, and the
    total is capped,
*   the spring fades out as a point separates, so contact can push a body out
    of the ground but never faster than the recovery speed.

The last is what makes a body released below the floor -- which real capture
does, whenever it sits under the estimated floor -- rise out rather than be
fired into the air by a spring that started loaded.  It also makes landings
inelastic, which is what a body does.

The whole-body ragdoll, and why it needed an energy budget
----------------------------------------------------------
A ragdoll of every joint at once first refused to settle: it landed and then
pinballed indefinitely.  Contact looked like the culprit and was not.  Measured
against a frozen-joint body in flight -- which the root integrator carries to
within 0.06 percent over five seconds -- a free body with no contact anywhere
near it gains energy too, and the gain grows with the length of the free chain:

    joints frozen        +0.06 %      worst frame  +0.09 J
    one free arm         +0.23 %      worst frame  +204 J
    all 21 joints free  +21.6  %      worst frame +3236 J

which is the decoupled model itself.  Every joint integrates as though the rest
of the body were rigid for the frame while the rest of the body does the same
back, and that mutual assumption is not energy-conserving over a long chain.
It is why a single limb behaved from the start and a whole body did not, and
contact never caused it -- it only made it violent enough to notice.

The model cannot be made conservative without solving the whole chain together,
but it does not have to be.  A free body is passive: nothing does work, gravity
is conservative, damping and friction only dissipate, and the springs give back
less than they store.  So any rise in total mechanical energy is fabricated, and
therefore measurable and removable -- see _enforce_passivity.  What that took:

*   summing kinetic energy over segments, not over joints, because a per-joint
    sum counts each distal segment once for every joint above it;

*   never taking it from the angular momentum, which in flight is exactly
    conserved and correctly integrated -- the fabrication is in the joints, and
    scaling the body's global spin to pay for it strips a released cartwheel of
    the rotation it was thrown with.  Taken from the joints instead, the spin
    then follows from conservation: the limbs slow, so the body turns faster;
*   accumulating the change from its parts rather than differencing a total,
    since gravitational potential dwarfs the motion by two orders of magnitude;
*   counting the joint limit springs, or every release out of a joint stop
    reads as energy from nowhere;
*   judging the excess over time rather than per frame, because internal energy
    genuinely swings frame to frame and taking on every up-swing without
    returning on the down-swing strangles all the motion -- that mistake froze
    the ragdoll solid in mid-air before it was found;
*   taking it from the body's motion about its centre of mass, never from the
    centre of mass itself except while the ground is pushing, so the ballistic
    trajectory stays exact;
*   standing down entirely unless the whole body is free, since a joint the
    capture drives does work that cannot be told from fabricated energy -- with
    the observer wrongly active, a tuck no longer speeds up a spin.

Frame rate
----------
`framerate` (from `config`) is the amount of *captured time* each call carries,
not how fast the patch runs.  Feed every frame of a 120 Hz take and it is 120;
feed every second frame and it is 60.  Every velocity in the simulation is
differenced against it and every integration step is sized by it, so a mismatch
is not a small error: setting 60 while feeding all 120 frames advances the
simulation twice per frame of data, and the body gains momentum it never had
and bounces metres into the air.  Set it too low by a factor of eight and the
body leaves the ground entirely.

Nothing in a pose stream reveals its own rate, and the damaging direction reads
as *slower* motion rather than faster, so this cannot be detected reliably from
the inside.  The resolved body is printed once when it is built -- check it.

Substeps are derived from the rate rather than fixed: `substep_rate` says what
the integration should effectively run at, and `substeps` is the ceiling.  Four
substeps at 120 Hz do the work of eight at 60 and cost twice as much for
nothing -- and at 120 Hz this node was over a frame's budget on its own, which
stalls a patch outright.

Pose formats
------------
Both of this system's pose layouts are accepted, told apart by shape alone, so
neither source needs a converter placed in front of the node:

    20 joints, quaternions   the active joints derived from the Shadow suit
                             (a flat 80, or 20x4)
    22 or 24 joints          SMPL, axis-angle or quaternions
    52 joints                SMPL-H, cropped to the body

Twenty is unambiguous -- no SMPL layout has twenty joints -- so detection is by
size, not by a setting that can be wrong.

There are two pose outputs, emitted together every frame:

    pose        the layout that came in, so the node drops into an existing
                chain without disturbing it
    smpl_pose   always SMPL, 24 joints, axis-angle -- what mgl_smpl_mesh,
                smpl_body and smpl_torque want

so neither consumer needs a converter and the choice does not have to be made
in advance.  A node with nothing set free still emits both, which makes it a
working format converter rather than a dead end.

Losing support
--------------
A driven root goes where the capture went, whatever the simulation is doing
underneath it -- so releasing the arms during a cartwheel leaves the body
sailing along on hands that are no longer touching anything.  With
`auto_release_unsupported` on, the upward force the captured motion needs is
compared each frame against what the simulated contacts actually supply, and
the root is let go in proportion to what has gone missing (the `support`
output reports the fraction).  A ballistic capture asks for no support and is
exempt, so ordinary flight never triggers it.

It is off by default, and it wants an accurate `floor_height`: contact force is
present or absent with nothing in between, so a capture sitting a few
centimetres high reads as unsupported from the first frame.  Three conditions
keep that from running away -- it will not act until something has actually
been released (with the whole body driven, a shortfall is a floor set wrong,
not support lost), nor while any weight is ramping, and the loss must persist
for `auto_release_delay`.  A `catch` restarts that timer, so taking hold again
is never undone on the next frame; if the support really is still missing it
will let go again, but it has to earn it.

Free joints may be named in either convention: SMPL's (`head`, `spine3`,
`left_collar`) or the active one (`base_of_skull`, `mid_vertebrae`,
`left_shoulder_blade`).  The field names the joints handed to physics; every
joint not named stays driven by the capture.

Blend weights
-------------
Every joint carries a weight in [0, 1], and index 0 is the root.

    1.0   the motion capture stream prescribes the joint outright; no
          simulation, no controller tuning needed for the driven body.
    0.0   fully free -- gravity, transport, limits, damping.
    else  a proportional-derivative controller pulls toward the captured pose,
          clipped at the joint's actuator ceiling scaled by the weight.  This
          is the tired / weak / injured limb: it tries, and sags.  Controller
          authority is weight / (1 - weight), so the range joins the prescribed
          case continuously at 1 rather than stepping into it.

`blend_weight` is the weight the named joints hold when *not* released, and it
is what `catch` returns them to -- so it defaults to 1, leaving the node armed
rather than already limp.  Set it to 0 for a limb that should simply hang, in
which case release and catch have nothing left to do.

Weights are also settable per joint, which is how a body is let go in stages.
Messages, sent to any input as a string or a list:

    weight <joints...> <value>      "weight arms 0", "weight left_leg 0.3"
    release <joints...>             "release left_arm"   (no names: all)
    catch <joints...>               "catch base_of_skull"

with joints named as groups or in either joint-name convention, all ramped over
`ramp_ms`.  The `weights` input takes a whole per-joint array instead: 22 in
SMPL order, root first, or 20 in the active order.  A weight only means
something for a joint in the free set, and changing that set resets the
simulation -- so `free_joints` defaults to `all`, everything is
simulation-ready, and the weights decide.  A message naming a joint that is
not free says so and leaves it alone.  The `blend_weight` slider and a bare
`catch` still act on the whole set.

Catching a body that has fallen means dragging it back onto the captured pose,
so give `ramp_ms` several hundred milliseconds for that; the short ramps suit
releasing, not recovering.

While a joint is prescribed its simulated state is continuously slammed to the
captured pose *and its captured angular velocity*, so the moment a weight drops
below 1 the limb carries the momentum it had -- which is what makes a release
in the middle of a leap continue rather than start from rest.  For the root the
weight is a crossfade rather than a controller, since nothing actuates it.

Note that this shadowing has to keep running while everything is merely
prescribed and waiting for a release; skipping those frames as a fast path
would leave the body letting go from rest, with no momentum at all.

"""

import math
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R

# Joint limits are read as Euler angles with the tightest axis in the middle,
# so its singularity lies outside the joint's range; scipy still warns on the
# rare exact hit, and the fallback there is harmless.
warnings.filterwarnings('ignore', message='Gimbal lock detected')


def _cross(a, b):
    """Cross product of two 3-vectors.

    numpy's cross() routes through generic axis-handling machinery that costs
    an order of magnitude more than the arithmetic itself, and this is the
    innermost operation in the torque sum -- over a thousand calls per frame
    for a whole-body ragdoll.  Writing it out directly is the same identity.
    """
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


def _cross_one_many(a, B):
    """Cross product of one 3-vector with an (n, 3) array of them."""
    return np.stack((a[1] * B[:, 2] - a[2] * B[:, 1],
                     a[2] * B[:, 0] - a[0] * B[:, 2],
                     a[0] * B[:, 1] - a[1] * B[:, 0]), axis=1)


def _cross_many(A, B):
    """Row-wise cross product of two (n, 3) arrays."""
    return np.stack((A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1],
                     A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2],
                     A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]), axis=1)


# Points that can touch the ground, as (joint index, radius in metres).
# The radius stands for the flesh around the joint centre: a knee is not a
# point at the bone's pivot, it is a surface some centimetres outside it.
# Without it a body would come to rest with its joint centres on the floor,
# sunk to the waist.  Indices 24 and above are the virtual tips the processor's
# forward kinematics already produces (toes, finger tips, heels).
#
# A hand needs all three of wrist, palm and finger tip.  With the palm missing,
# an inverted body came down on a single finger tip carrying three times its
# weight, and that one point's friction wrenched the rotation out of a released
# cartwheel within a second.  Feet already had three points each.
#
# These are for a body of roughly average size and are not scaled by the betas.
CONTACT_POINTS = [
    # The torso needs width and length, not a midline.  With only the pelvis
    # and the chest, the trunk is two points on a centre line: a body can
    # balance along it, roll on it, and arch clean over the gap between the two
    # -- which is how ragdolls end up resting on the head and one toe with the
    # back bridged in the air.  The hips carry the pelvis's real width (their
    # joint centres sit about eight centimetres either side), and the two
    # lumbar points fill the span between pelvis and chest.
    (0,  0.11),                      # pelvis
    (1,  0.10), (2,  0.10),          # hips -- the pelvis's width
    (3,  0.10),                      # lumbar
    (6,  0.11),                      # mid back
    (9,  0.11),                      # chest
    (15, 0.10),                      # head
    (16, 0.07), (17, 0.07),          # shoulders
    (18, 0.05), (19, 0.05),          # elbows
    (20, 0.035), (21, 0.035),        # wrists
    (22, 0.028), (23, 0.028),        # palms -- what a hand actually stands on
    (26, 0.02), (27, 0.02),          # finger tips
    (4,  0.06), (5,  0.06),          # knees
    (7,  0.045), (8,  0.045),        # ankles
    (24, 0.025), (25, 0.025),        # toe tips
    (28, 0.03), (29, 0.03),          # heels
]


# Anatomical joint limits, as (min, max) per local axis, in radians.
#
# SMPLProcessor's own table is left untouched: it feeds the passive-limit term
# in smpl_torque, whose output has been validated at length, and it is shaped
# for that job -- isotropic cones for most joints, with a hinge and "locked
# axes" only where the analyzer needed structural support.  Run forward that is
# far too loose: anything it does not name falls through to a 180 degree cone
# with almost no stiffness, so ankles, wrists, feet and the head are
# effectively free balls, and a knee modelled as a cone will bend forwards.
#
# The axis meanings below were measured off the model rather than assumed, by
# rotating each joint about each local axis at the rest pose and watching where
# the distal joint went (see the conventions noted per joint).  The body's
# internal frame is +Y up, +Z forward, +X to the subject's left, and the rest
# pose is a T-pose, so arm angles are measured from horizontal -- a hanging arm
# sits near -1.65 on the shoulder's Z, not at zero.
#
# Ranges are ordinary living ranges, not extremes of flexibility.
_L, _R = 'left_', 'right_'
RAGDOLL_JOINT_LIMITS = {
    # X flex(-)/extend(+), Y twist, Z ab/adduct.  Z mirrors between legs.
    'left_hip':   ((-2.10, -0.70, -0.35), (0.52, 0.70, 0.79)),
    'right_hip':  ((-2.10, -0.70, -0.79), (0.52, 0.70, 0.35)),
    # X flexion only -- a knee does not bend forwards, so the minimum is zero.
    # Secondary axes wider than a textbook knee: a walk-and-hop capture put
    # the fitted knee 35 degrees in twist and 20 in varus, and a limit the
    # performer exceeds is the model's error.
    'left_knee':  ((0.0, -0.60, -0.35), (2.60, 0.60, 0.35)),
    'right_knee': ((0.0, -0.60, -0.35), (2.60, 0.60, 0.35)),
    # X dorsiflex(-)/plantarflex(+), Y inversion/eversion.
    'left_ankle':  ((-0.60, -0.45, -0.45), (0.87, 0.45, 0.45)),
    'right_ankle': ((-0.60, -0.45, -0.45), (0.87, 0.45, 0.45)),
    'left_foot':   ((-0.35, -0.12, -0.12), (0.87, 0.12, 0.12)),
    'right_foot':  ((-0.35, -0.12, -0.12), (0.87, 0.12, 0.12)),
    # Three spine segments, each carrying a third of the trunk's range: 120
    # degrees of flexion in all, which is what a limp trunk curls to.  At 30
    # per segment the lowest one sat on its flexion stop 95 percent of the
    # time through a released cartwheel -- a torso braced against a spring.
    'spine1': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    'spine2': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    'spine3': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    # Neck and head together give about 65 degrees each way; the head takes
    # the larger share because, being light, it is the one that lolls.
    'neck':   ((-0.61, -0.70, -0.61), (0.61, 0.70, 0.61)),
    'head':   ((-0.52, -0.52, -0.52), (0.52, 0.52, 0.52)),
    # The collars barely move; they carry shoulder-blade shrug and protraction.
    'left_collar':  ((-0.26, -0.70, -0.55), (0.26, 0.70, 0.55)),
    'right_collar': ((-0.26, -0.70, -0.55), (0.26, 0.70, 0.55)),
    # X humeral twist, Y flex(-)/extend(+) for the left, Z elevation.
    # Both Y and Z mirror between arms.
    'left_shoulder':  ((-1.20, -2.00, -1.80), (1.20, 1.20, 1.80)),
    'right_shoulder': ((-1.20, -1.20, -1.80), (1.20, 2.00, 1.80)),
    # Y flexion, one way only; X is forearm pronation, which is real.
    'left_elbow':  ((-1.40, -2.60, -0.35), (1.40, 0.0, 0.35)),
    'right_elbow': ((-1.40, 0.0, -0.35), (1.40, 2.60, 0.35)),
    'left_wrist':  ((-0.35, -1.20, -0.50), (0.35, 1.20, 0.50)),
    'right_wrist': ((-0.35, -1.20, -0.50), (0.35, 1.20, 0.50)),
}


# Self-collision colliders: capsules spanning a pair of joints, with a radius.
#
# The radii are not anatomical, they are the largest that leave every modelled
# pair clear of itself in ordinary poses -- measured off the model rather than
# guessed, because a collider that already overlaps at rest pushes the body
# apart permanently.  The binding constraints are tight: the thighs pass within
# 0.119 m of each other standing, and the spine axis runs 0.103 m from the
# thigh line, so a torso of realistic girth would swallow the hips.  Thighs are
# therefore slimmer than real ones (real thighs touch, and a model that keeps
# them 0.17 m apart would splay the legs), and the pelvis-to-thigh pair is not
# modelled at all -- the hip attaches there, and the hip's own limits are what
# keep a leg out of the pelvis.
SELF_CAPSULES = [
    ('torso_low', 0, 6,  0.130),
    ('torso_up',  6, 12, 0.120),
    ('head',      12, 15, 0.100),
    ('L_uarm',    16, 18, 0.055),
    ('R_uarm',    17, 19, 0.055),
    ('L_farm',    18, 20, 0.050),
    ('R_farm',    19, 21, 0.050),
    ('L_thigh',   1, 4,  0.055),
    ('R_thigh',   2, 5,  0.055),
    ('L_shin',    4, 7,  0.060),
    ('R_shin',    5, 8,  0.060),
]

# Pairs never tested: those sharing a joint (adjacent links, always touching),
# and the pelvis against either thigh (the hip joins them).
SELF_PAIR_EXCLUDE = {('torso_low', 'L_thigh'), ('torso_low', 'R_thigh')}


def _closest_points_on_segments(p1, q1, p2, q2):
    """Closest points on two line segments, and the distance between them."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return p1, p2, float(np.linalg.norm(r))
    if a <= 1e-12:
        t = float(np.clip(f / e, 0.0, 1.0))
        s = 0.0
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return c1, c2, float(np.linalg.norm(c1 - c2))


# Named joint groups, by SMPL index.  Arm groups start at the shoulder so the
# collar stays driven -- the collar's global rotation is the free chain's pivot
# frame, and it needs to come from the capture.
JOINT_GROUPS = {
    'left_arm':   [16, 18, 20],
    'right_arm':  [17, 19, 21],
    'arms':       [16, 18, 20, 17, 19, 21],
    'neck':       [12],
    'head':       [12, 15],
    'spine':      [3, 6, 9],
    'left_leg':   [1, 4, 7, 10],
    'right_leg':  [2, 5, 8, 11],
    'legs':       [1, 4, 7, 10, 2, 5, 8, 11],
    'upper_body': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    # Index 0 is the root's six degrees of freedom, not an ordinary joint --
    # it is simulated by conservation rather than by a joint torque.  It is in
    # 'all', because a set called "all" that quietly leaves out the one degree
    # of freedom that lets a body fall is a trap: every joint goes limp, the
    # root keeps following the capture, and the body sails on through the air
    # with the translation output passing straight through unchanged.
    'root':       [0],
    'joints':     list(range(1, 22)),      # every joint, root still driven
    'all':        list(range(0, 22)),
    'everything': list(range(0, 22)),      # synonym for 'all'
}


class RagdollParams:
    """Plain holder for the per-frame simulation knobs."""

    def __init__(self):
        self.dt = 1.0 / 60.0
        self.engine = 'bullet'        # 'bullet' (pybullet) or 'native'
        self.motor_strength = 1.0     # multiplier on a partially driven joint's force limit
        self.motor_kp = 0.6           # pybullet POSITION_CONTROL gains for those motors
        self.motor_kd = 0.3
        self.blend_soft = 180.0       # degrees a partial joint gives under its typical load at weight 0
        self.blend_firm = 1.0         # ... and at weight 1; log-spaced between (root: half a metre per radian)
        self.partial_damping = 0.5    # damping ratio of a partially driven joint's spring (1 = critical)
        self.spring_rate = 60.0       # 1/s: a spring closes at most this fraction of its error per second
        self.gravity_comp = 1.0       # share of the weight below a partial joint its 'muscle' carries, times the weight
        self.root_spring = 40.0       # (native core) 1/s^2, the pull on a partially driven root
        self.root_hold_force = 1.0e5  # N: the constraint holding a driven root
        self.root_tether = 1.0        # multiplier on the sag a partial root is allowed (bullet)
        self.root_erp = 1.0           # how much of the root's tracking error the constraint closes per step
        self.ramp_s = 0.12            # the node's ramp, for a catch that completes over it
        self.root_catch_speed = 2.5   # m/s floor on the speed a catch reels the root in at
        self.root_catch_rate = 6.0    # rad/s
        self.drive_kp = 0.9           # bullet motor gains for driven joints on a free root
        self.drive_kd = 0.9
        self.drive_force = 3000.0     # N m: a driven joint tracks the capture, whatever it takes
        self.limit_entry_s = 0.5      # s over which a joint released outside its box is eased in
        self.spike_ratio = 3.0        # feed-forward rate clipped at this multiple of the joint's running speed
        self.spike_floor = 6.0        # rad/s, never clipped below this
        self.limit_gain = 0.3         # bullet motor gains for the joint-limit motors
        self.limit_damping_gain = 0.1
        self.joint_damping_gain = 0.05    # limp-joint damping through the motor
        self.joint_damping_fraction = 0.05  # ceiling on that damping as a fraction of the joint's
                                          # torque scale (hip ~26 N m, knee ~22, ankle ~9, foot ~3.5):
                                          # one flat 2 N m cap held a foot but let a knee whip at
                                          # fifteen radians a second and pass it down to the foot
        self.substeps = 4             # ceiling; the rate below sets the need
        self.substep_rate = 240.0     # Hz the integration should effectively run at
        self.solver_iterations = 40   # bullet constraint solver iterations per substep

        self.gravity = 1.0            # scale on the true gravity field
        self.transport = 1.0          # scale on the pivot-frame pseudo-forces
        self.damping = 1.5            # joint viscous damping, 1/s
        self.drag = 0.15              # joint drag growing with speed, 1/rad
        # Was 8: heavy extra damping on a ground-loaded joint, added against
        # pinball settling.  Per-substep joint contact solved that at source,
        # and the boost was then measured holding a landing knee to half its
        # fold rate.  A loaded limp knee folds; it does not stiffen.
        self.contact_damp_boost = 0.0  # extra joint damping while ground-loaded
        self.passivity = True         # remove energy the coupling fabricates
        self.passivity_bleed = 1.0    # fraction of the excess removed per frame
        self.passivity_deadband = 0.01  # tolerance band, as a share of motion
        # The body's own rotation is exact at any setting, since the angular
        # momentum is no longer bled; what this trades is how freely the limbs
        # swing in flight against how well the body settles once it lands.
        # Lowering it to 0.10 does leave the limbs livelier, but the energy it
        # spares in the air arrives with the landing -- measured at five times
        # the settled wander and a third of the contact points bearing load --
        # so it stays here.
        # In flight the budget can only stiffen: a limb gains energy through
        # the transport term as the body flings it, the root cannot pay for it
        # (it is the whole-body centre of mass and angular momentum, which
        # internal forces do not touch), and the only place the budget can
        # take the excess from is the limb -- the one place it belongs.  It
        # cost 45 percent of limb motion in flight, measured.  The drift it
        # guarded against is slow (a fifth over five seconds of tumbling), so
        # it is off in flight by default and left as a knob for long throws.
        self.passivity_rate = 0.0     # share of the excess per frame, in flight
        self.passivity_rate_contact = 0.25  # ... and while the ground is pushing
        self.limit_stiffness = 1.0    # multiplier on the analyzer's limit springs
        self.stop_softness = 0.087    # rad of soft engagement before a stop holds
        self.locked_stiffness = 1.0   # multiplier on the locked-axis springs

        self.kp = 120.0               # controller stiffness, 1/s^2
        self.kd = 12.0                # controller damping, 1/s

        self.pivot_smoothing = 0.25   # EMA alpha on pivot acceleration terms
        self.root_seed_smoothing = 0.3    # EMA alpha on the released velocity/spin
        self.max_ang_vel = 40.0       # rad/s safety clamp
        self.max_pivot_acc = 60.0     # m/s^2 clamp on differenced pivot accel

        # Ground contact.  Stiffness is expressed as the penetration the body's
        # own weight produces at rest, which is the number that actually says
        # how the contact will look; friction is a Coulomb coefficient.
        self.self_collision = True
        self.self_depth = 0.04        # m of overlap under body weight
        self.self_max_g = 2.0         # cap on one pair's force, in body weights
        self.floor_enable = True
        self.floor_height = 0.0
        self.floor_auto = True        # (bullet) the floor follows the driven capture's lowest point
        self.floor_rise = 0.5         # 1/s, how fast that estimate may rise
        self.support_tolerance = 0.05 # m: a driven link this close to the floor is standing on it
        self.contact_depth = 0.01     # m of sink under body weight
        self.contact_damping = 4.0    # s/m, Hunt-Crossley style
        self.friction = 0.8
        self.max_penetration = 0.05   # m of penetration the spring responds to
        self.max_contact_g = 50.0     # cap on total ground force, in body weights
        self.max_point_g = 10.0       # cap on one point's force, in body weights
        self.max_point_accel = 300.0  # m/s^2 a point may decelerate its own mass at
        self.recovery_speed = 0.1     # m/s ceiling on how fast contact pushes out
        self.damping_speed = 2.0      # m/s at which the damping term saturates


class _PivotState:
    """Differenced kinematics of one free chain's attachment point.

    Both the position and the frame rotation come from joints the capture
    drives, so this is a measurement of the body, not of the simulation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_pos = None
        self.prev_vel = np.zeros(3)
        self.prev_rot = None
        self.prev_ang_vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.ang_acc = np.zeros(3)

    def update(self, pos, rot_mat, dt, p):
        alpha = float(np.clip(p.pivot_smoothing, 0.0, 1.0))

        if self.prev_pos is None:
            self.prev_pos = pos.copy()
            self.prev_rot = rot_mat.copy()
            return

        # A teleport in the capture must not become a thousand-g kick.
        if np.linalg.norm(pos - self.prev_pos) > 2.0:
            self.reset()
            self.prev_pos = pos.copy()
            self.prev_rot = rot_mat.copy()
            return

        vel = (pos - self.prev_pos) / dt
        acc = (vel - self.prev_vel) / dt
        acc = np.clip(acc, -p.max_pivot_acc, p.max_pivot_acc)
        self.acc = self.acc + alpha * (acc - self.acc)
        self.prev_pos = pos.copy()
        self.prev_vel = vel

        # Angular velocity of the pivot frame, world axes.  The log map by
        # hand: per-frame pivot rotations are small, and this ran through
        # scipy once per free joint per frame -- a millisecond on a whole body.
        d = rot_mat @ self.prev_rot.T
        c = 0.5 * (d[0, 0] + d[1, 1] + d[2, 2] - 1.0)
        c = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)
        theta = math.acos(c)
        sn = math.sin(theta)
        if theta < 1e-7:
            ang_vel = np.zeros(3)
        elif sn < 1e-6:
            ang_vel = R.from_matrix(d).as_rotvec() / dt
        else:
            f = theta / (2.0 * sn * dt)
            ang_vel = np.array(((d[2, 1] - d[1, 2]) * f,
                                (d[0, 2] - d[2, 0]) * f,
                                (d[1, 0] - d[0, 1]) * f))
        ang_vel = np.clip(ang_vel, -p.max_ang_vel, p.max_ang_vel)
        ang_acc = (ang_vel - self.prev_ang_vel) / dt
        ang_acc = np.clip(ang_acc, -p.max_pivot_acc, p.max_pivot_acc)

        self.ang_vel = self.ang_vel + alpha * (ang_vel - self.ang_vel)
        self.ang_acc = self.ang_acc + alpha * (ang_acc - self.ang_acc)
        self.prev_rot = rot_mat.copy()
        self.prev_ang_vel = ang_vel


class SMPLRagdollSim:
    """Forward integration of a chosen subset of an SMPL body's joints."""

    def __init__(self, processor):
        self.processor = processor
        self.free_indices = []
        self.root_free = False
        self._configure_tables()
        self.reset()

    def set_root_free(self, free):
        """Whether the root's six degrees of freedom are simulated."""
        free = bool(free)
        if free != self.root_free:
            self.root_free = free
            self.reset()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_free_joints(self, indices):
        """Set which joints are simulated.  Joint 0 (pelvis) is never free
        here -- the root carries the whole body's six degrees of freedom, which
        this milestone does not simulate."""
        idx = sorted({int(i) for i in indices if 1 <= int(i) < 22})
        if idx != self.free_indices:
            self.free_indices = idx
            self._configure_tables()
            self.reset()

    def _configure_tables(self):
        proc = self.processor
        self.parents = list(proc._get_hierarchy())
        if not hasattr(proc, '_passive_limits_precomputed'):
            proc._precompute_passive_limit_tables()

        # Per-axis limits, and a stiffness taken from each joint's own
        # strength: the full actuator torque a quarter radian past the stop.
        joint_names = proc.joint_names
        self._lim_min = np.zeros((22, 3))
        self._lim_max = np.zeros((22, 3))
        self._lim_k = np.zeros(22)
        self._lim_active = np.zeros(22, dtype=bool)
        self._lim_order = [(0, 1, 2)] * 22
        for idx in range(22):
            name = joint_names[idx] if idx < len(joint_names) else ''
            limits = RAGDOLL_JOINT_LIMITS.get(name)
            if limits is None:
                continue
            self._lim_min[idx] = limits[0]
            self._lim_max[idx] = limits[1]
            # Full actuator torque a quarter radian past the stop, with a floor
            # so a joint the strength table rates as weak (a foot) still has a
            # stop that holds -- measured sagging nine degrees through it.
            self._lim_k[idx] = max(4.0 * float(np.mean(proc.max_torque_array[idx])), 30.0)
            self._lim_active[idx] = True
            # Euler order: widest axis first, tightest in the middle, so the
            # middle-axis singularity sits outside the range the joint uses.
            span = np.asarray(limits[1]) - np.asarray(limits[0])
            first, mid = int(np.argmax(span)), int(np.argmin(span))
            if first == mid:
                first, mid = 0, 1
            last = 3 - first - mid
            self._lim_order[idx] = (first, mid, last)

        free = set(self.free_indices)

        # Chain roots: free joints whose parent is driven.  Each root's parent
        # supplies the pivot frame.
        self.roots = [j for j in self.free_indices
                      if self.parents[j] not in free]

        # Every joint whose world placement depends on a simulated rotation.
        # Closed under descent, and includes the virtual tip joints so that
        # segment centres of mass match the analyzer's convention.
        children = getattr(proc, '_hierarchy_children', None)
        if children is None:
            children = {i: [] for i in range(30)}
            for i in range(30):
                pa = self.parents[i]
                if pa >= 0:
                    children[pa].append(i)
        self.children = children

        sim_nodes = []
        for root in self.roots:
            stack = [root]
            while stack:
                cur = stack.pop()
                sim_nodes.append(cur)
                stack.extend(children.get(cur, []))
        # Ascending index order is a valid topological order in SMPL: every
        # joint, virtual ones included, has a lower-indexed parent.
        self.sim_nodes = sorted(set(sim_nodes))

        # Which chain each free joint belongs to, so it uses the right pivot.
        self.root_of = {}
        for root in self.roots:
            stack = [root]
            while stack:
                cur = stack.pop()
                if cur in free:
                    self.root_of[cur] = root
                stack.extend([c for c in children.get(cur, []) if c < 24])

        # One pivot per free joint, not per chain -- each joint's transport
        # forces belong to its own parent.  Sharing the chain root's pivot puts
        # the root frame's centrifugal term across the whole limb, so a foot
        # nearly a metre from the hip is whipped as though it were rigidly
        # carried by the hip frame, which a free knee and ankle mean it is not.
        # A joint's own position is moved only by its ancestors, never by its
        # own rotation, so measuring it from the simulated kinematics is
        # feedback-free.
        self.pivots = {j: _PivotState() for j in self.free_indices}

        # Contact points, and for each free joint the contact points that hang
        # off it -- a force under the hand torques the elbow and the shoulder,
        # so each free joint needs the list of points it is an ancestor of.
        self.contact_idx = np.array([c[0] for c in CONTACT_POINTS], dtype=int)
        self.contact_radius = np.array([c[1] for c in CONTACT_POINTS], dtype=float)
        self.contact_of_joint = {}
        for j in self.free_indices:
            owned = []
            for n, k in enumerate(self.contact_idx):
                cur = int(k)
                while cur != -1:
                    if cur == j:
                        owned.append(n)
                        break
                    cur = self.parents[cur] if cur < len(self.parents) else -1
            self.contact_of_joint[j] = owned

        # Self-collision pairs, and for each capsule the free joints a force on
        # it would torque.  A point anywhere along a capsule is rigidly carried
        # by the joint the capsule starts at, so the joints affected are that
        # one and its ancestors.
        self._self_pairs = []
        names = [c[0] for c in SELF_CAPSULES]
        for i in range(len(SELF_CAPSULES)):
            for k in range(i + 1, len(SELF_CAPSULES)):
                _, ai, bi, _ = SELF_CAPSULES[i]
                _, ak, bk, _ = SELF_CAPSULES[k]
                if {ai, bi} & {ak, bk}:
                    continue
                if (names[i], names[k]) in SELF_PAIR_EXCLUDE \
                        or (names[k], names[i]) in SELF_PAIR_EXCLUDE:
                    continue
                self._self_pairs.append((i, k))
        self._self_cap_joints = []
        for _, a, _b, _r in SELF_CAPSULES:
            owners = []
            cur = a
            while cur != -1:
                if cur in free:
                    owners.append(cur)
                cur = self.parents[cur] if cur < len(self.parents) else -1
            self._self_cap_joints.append(owners)

        # The mass each contact point is actually decelerating when it hits.
        #
        # Pin joints transmit force, so a toe striking the floor does torque
        # the whole body's angular momentum -- but only with whatever force the
        # toe can develop, and a toe on a limp leg cannot develop much: the leg
        # folds, and the toe is stopping a one-kilogram foot.  The penalty
        # spring does not know that.  Its stiffness is set so the *whole body*
        # sinks a centimetre, and a toe arriving alone at five metres a second
        # saturates at the same cap as a body standing on it -- ten body
        # weights, at a metre of lever, for a frame -- which was measured to
        # strip a released cartwheel of most of its rotation in one step.
        #
        # So each point's force is bounded by the mass below the first free
        # joint above it, times a deceleration.  Under a free ankle that is a
        # foot; under a fully driven leg it is the whole body, and the point
        # transmits like the rigid body it then belongs to.
        self.point_mass_eff = np.zeros(len(CONTACT_POINTS))
        total_mass = float(proc._seg_mass.sum())
        for n_c, (k, _r) in enumerate(CONTACT_POINTS):
            cur = int(k)
            m_eff = total_mass
            while cur != -1:
                if cur in free:
                    members = proc._subtree_members.get(cur, [cur])
                    m_eff = float(sum(proc._seg_mass[s] for s in members
                                      if s < 24 and proc._seg_mass[s] > 0.0))
                    break
                if cur == 0:
                    m_eff = total_mass if self.root_free else total_mass
                    break
                cur = self.parents[cur] if cur < len(self.parents) else -1
            self.point_mass_eff[n_c] = max(m_eff, 0.05)

        # Segments that contribute mass to each free joint's subtree, and that
        # subtree's share of the body's mass.
        #
        # (A contact force used to reach a joint scaled by this share of the
        # mass, to keep a 600 N ground reaction from flinging a one-kilogram
        # foot.  That was treating the symptom: the force should never have
        # been 600 N at a foot.  Now that each point's force is bounded by the
        # mass it decelerates, the joints take the full force -- they must, or
        # a straight limp arm under load stands as a column.)
        self.subtree_of = {}
        for j in self.free_indices:
            members = proc._subtree_members.get(j)
            if members is None:
                members = [j]
            self.subtree_of[j] = [s for s in members
                                  if s < 24 and proc._seg_mass[s] > 0.0]

    def reset(self):
        """Drop all simulation state.  Joints re-seed from the capture on the
        next frame."""
        self.local_rot = {}      # joint -> Rotation, local to parent
        self.ang_vel = {}        # joint -> (3,) angular velocity, parent frame
        self.prev_mocap = {}     # joint -> Rotation, previous captured pose
        self.mocap_ang_vel = {}  # joint -> (3,) captured angular velocity
        self.last_torque = np.zeros((22, 3))
        self.last_inertia = np.zeros(22)
        for st in getattr(self, 'pivots', {}).values():
            st.reset()

        # Root six-degree-of-freedom state.  Position is carried as the whole
        # body's centre of mass, because that is the quantity with the simple
        # equation of motion in flight -- the root itself wanders relative to
        # it as the limbs move.
        self.root_rot = None          # world rotation of the pelvis
        self.trans = None             # root translation, internal frame
        self.com = None               # world centre of mass
        self.com_vel = np.zeros(3)
        self.ang_momentum = np.zeros(3)   # about the centre of mass, conserved
        self.root_ang_vel = np.zeros(3)
        self._prev_com = None
        self._prev_root_rot = None
        self._prev_mocap_all = None
        self._omega_seed = np.zeros(3)
        self._vel_seed = np.zeros(3)
        self._seed_acc = np.zeros(3)
        self._seed_ang_acc = np.zeros(3)
        self._prev_raw_vel = None
        self._prev_raw_omega = None
        self._was_prescribed = True
        self._warned_rate = False
        self._friction_anchor = None
        self._anchor_valid = None
        self._frame_kfac = None
        self._frame_cap = None
        self._frame_vtan = None
        self._frame_kt = 0.0
        self._frame_ct = 0.0
        self._frame_contact = None
        self._frame_penetration = None
        self._frame_self = []
        self._frame_self_potential = 0.0
        self._prev_energy = None
        self._energy_excess = 0.0
        self._active_joints = []
        self._controller_work = 0.0
        self._contact_work = 0.0
        self._contact_work_root = 0.0
        self.last_energy_injected = 0.0
        self.last_support = 1.0
        self._body_weight = float(self.processor._seg_mass.sum()) * 9.81
        self.last_contact_force = np.zeros((len(CONTACT_POINTS), 3))

    # ------------------------------------------------------------------
    # Forward kinematics of the simulated chains only
    # ------------------------------------------------------------------

    @staticmethod
    def _substeps(p):
        """How many substeps this frame rate actually needs."""
        want = int(round(p.substep_rate * p.dt)) if p.dt > 0 else int(p.substeps)
        return int(max(1, min(int(p.substeps), max(1, want))))

    def _chain_fk(self, root, pivot_pos, frame_rot, local_mats):
        """Place the simulated subtree given its pivot.

        The driven body is fixed for the duration of a frame, so only this
        short chain is re-solved between substeps.

        Returns dicts of world position and world rotation, keyed by joint.
        """
        pos = {root: pivot_pos}
        rot = {}
        offsets = self.processor.skeleton_offsets

        parent_rot = {root: frame_rot}
        stack = [root]
        order = []
        while stack:
            cur = stack.pop()
            order.append(cur)
            stack.extend(self.children.get(cur, []))
        order.sort()

        for j in order:
            pr = parent_rot[j]
            rot[j] = pr @ local_mats[j] if j in local_mats else pr
            for c in self.children.get(j, []):
                pos[c] = pos[j] + rot[j] @ offsets[c]
                parent_rot[c] = rot[j]
        return pos, rot

    def _segment_com(self, j, pos):
        """Segment centre of mass, matching the analyzer's convention."""
        kids = [c for c in self.children.get(j, []) if c in pos]
        if kids:
            end = pos[kids[0]] if len(kids) == 1 else sum(pos[c] for c in kids) / len(kids)
            return 0.5 * (pos[j] + end)
        # Leaf: project half a segment length along the bone direction.
        pa = self.parents[j]
        if pa in pos:
            d = pos[j] - pos[pa]
            n = np.linalg.norm(d)
            if n > 1e-6:
                return pos[j] + (d / n) * (0.5 * self.processor._seg_length[j])
        return pos[j].copy()

    # ------------------------------------------------------------------
    # Torque terms
    # ------------------------------------------------------------------

    _EVEN = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}

    def _limit_angles(self, j, mat):
        """The joint's rotation as three angles, one per local axis, read in
        its Euler order, from its rotation matrix.

        These are not the components of the axis-angle vector.  Those scale
        with the total rotation: a knee flexed 120 degrees about a hinge tilted
        a harmless five degrees carries a "twist" component past its stop, and
        a head lolling thirty degrees on a diagonal reads twenty on two axes
        at once and is caught in the corner of its box.  Every large rotation
        was fighting its secondary stops, which is a joint that looks braced --
        measured as a quarter of all joint axes sitting on a limit throughout a
        released cartwheel.  Euler angles read the primary rotation first and
        the secondaries in the rotated frame, which is what a limit means.

        Five matrix elements and three inverse trig calls, no allocation: this
        runs for every free joint at every substep.  The joint's order is
        handled as a permutation into the XYZ case; an odd permutation is a
        reflection, and conjugating by a reflection reverses the sense of every
        rotation, so its angles are negated.
        """
        i, k, m = self._lim_order[j]
        sb = mat[i, m]
        if sb > 1.0:
            sb = 1.0
        elif sb < -1.0:
            sb = -1.0
        beta = math.asin(sb)
        alpha = math.atan2(-mat[k, m], mat[m, m])
        gamma = math.atan2(-mat[i, k], mat[i, i])
        if (i, k, m) not in self._EVEN:
            alpha, beta, gamma = -alpha, -beta, -gamma
        out = np.empty(3)
        out[i] = alpha
        out[k] = beta
        out[m] = gamma
        return out

    def _limit_axes(self, j, angles):
        """Directions, in the parent frame, of the three Euler rotations for
        the angles above -- the axes a limit torque acts about.  Closed form:
        an elementary rotation of a basis vector is a cosine of itself plus a
        sine of the third axis, with the sign of the cyclic order."""
        i, k, m = self._lim_order[j]
        sigma = 1.0 if (i, k, m) in self._EVEN else -1.0
        ca, sa = math.cos(angles[i]), math.sin(angles[i])
        cb, sb = math.cos(angles[k]), math.sin(angles[k])
        axes = np.zeros((3, 3))
        axes[i, i] = 1.0
        axes[k, k] = ca
        axes[k, m] = sigma * sa
        axes[m, m] = cb * ca
        axes[m, k] = -sigma * cb * sa
        axes[m, i] = sigma * sb
        return axes

    def _limit_torque(self, j, mat, w, inertia, p):
        """Passive joint limit torque, in the parent's frame.

        A box limit per axis on the joint's Euler angles (see _limit_angles):
        outside the allowed range on any axis, a linear spring pushes back
        about that axis, damped near critical so the stop does not ring.
        Stiffness is set from the joint's own strength -- the full actuator
        torque a quarter radian past the limit -- with a floor.
        """
        if not self._lim_active[j]:
            return np.zeros(3)
        k = self._lim_k[j] * p.limit_stiffness
        if k <= 0.0:
            return np.zeros(3)
        angles = self._limit_angles(j, mat)
        lo, hi = self._lim_min[j], self._lim_max[j]
        if not (angles[0] < lo[0] or angles[0] > hi[0] or angles[1] < lo[1]
                or angles[1] > hi[1] or angles[2] < lo[2] or angles[2] > hi[2]):
            return np.zeros(3)
        axes = self._limit_axes(j, angles)
        torque = np.zeros(3)
        damping = None
        soft = max(p.stop_softness, 1e-4)
        for ax in range(3):
            over = 0.0
            if angles[ax] < lo[ax]:
                over = lo[ax] - angles[ax]
            elif angles[ax] > hi[ax]:
                over = -(angles[ax] - hi[ax])
            if over != 0.0:
                # Progressive: a ligament gives softly for the first few
                # degrees and then holds.  Quadratic within `stop_softness`,
                # linear beyond, continuous in both value and slope -- a
                # hard-edged stop at full stiffness from the first degree,
                # sat on 80 percent of the time, reads as a braced joint.
                mag = abs(over)
                if mag < soft:
                    spring = k * mag * mag / (2.0 * soft)
                    engage = mag / soft
                else:
                    spring = k * (mag - 0.5 * soft)
                    engage = 1.0
                if damping is None:
                    damping = 2.0 * math.sqrt(max(k * inertia, 1e-12))
                rate = float(w @ axes[ax])
                torque += (np.sign(over) * spring - engage * damping * rate) * axes[ax]
        return torque

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, mocap_aa, world_pos, global_rot_mats, weights, p):
        """Advance one captured frame.

        Args:
            mocap_aa:        (24, 3) captured local rotations, internal frame.
            world_pos:       (30, 3) forward kinematics of the captured pose.
            global_rot_mats: (30, 3, 3) world rotations of the captured pose.
            weights:         (22,) blend weight per joint.
            p:               RagdollParams.

        Returns:
            dict of joint index -> (3,) simulated local rotation vector.
        """
        proc = self.processor
        if not self.free_indices:
            return {}

        dt = p.dt
        self.last_torque[:] = 0.0
        self.last_inertia[:] = 0.0

        # --- Captured angular velocity, and re-seeding of prescribed joints ---
        for j in self.free_indices:
            q_m = R.from_rotvec(mocap_aa[j])
            prev = self.prev_mocap.get(j)
            if prev is None:
                self.mocap_ang_vel[j] = np.zeros(3)
            else:
                delta = q_m * prev.inv()
                wv = delta.as_rotvec() / dt
                self.mocap_ang_vel[j] = np.clip(wv, -p.max_ang_vel, p.max_ang_vel)
            self.prev_mocap[j] = q_m

            # While prescribed, the simulation shadows the capture -- pose and
            # momentum both -- so a release starts from the real motion.
            if weights[j] >= 1.0 - 1e-6 or j not in self.local_rot:
                self.local_rot[j] = q_m
                self.ang_vel[j] = self.mocap_ang_vel[j].copy()

        gravity = np.zeros(3)
        gravity[getattr(proc, 'internal_y_dim', 1)] = -9.81 * p.gravity
        g0, g1, g2 = gravity[0], gravity[1], gravity[2]
        tr_ = p.transport

        # --- Pivot kinematics, once per frame, one per free joint ---
        for j in self.free_indices:
            pa = self.parents[j]
            self.pivots[j].update(world_pos[j], global_rot_mats[pa], dt, p)

        # Substeps exist to bound the integration step, so a fast capture
        # already provides part of what they were for.  Running four of them at
        # 120 Hz does the work of eight at 60 and costs twice as much for
        # nothing -- and at 120 Hz this node was over a frame's budget on its
        # own, which stalls the patch outright.  The count is therefore derived
        # from the frame rate, with `substeps` as the ceiling.
        n_sub = self._substeps(p)
        sub_dt = dt / n_sub
        active = [j for j in self.free_indices if weights[j] < 1.0 - 1e-6]
        self._controller_work = 0.0
        self._contact_work = 0.0
        if not active:
            return {j: self.local_rot[j].as_rotvec() for j in self.free_indices}

        max_torque = proc.max_torque_array

        # Driven descendants of a free joint keep their captured rotation, which
        # is constant across the substeps of one frame -- build those once.
        free_set = set(self.free_indices)
        driven_mats = {j: R.from_rotvec(mocap_aa[j]).as_matrix()
                       for j in self.sim_nodes if j < 24 and j not in free_set}

        for _ in range(n_sub):
            # Rotation matrices for the current simulated configuration.
            local_mats = dict(driven_mats)
            for j in self.free_indices:
                if j in self.local_rot:
                    local_mats[j] = self.local_rot[j].as_matrix()
                else:
                    local_mats[j] = R.from_rotvec(mocap_aa[j]).as_matrix()

            torques = {}
            inertias = {}

            for root in self.roots:
                frame_rot = global_rot_mats[self.parents[root]]
                pos, rot = self._chain_fk(root, world_pos[root], frame_rot,
                                          local_mats)

                coms = {}
                for j in self.sim_nodes:
                    if j < 24 and proc._seg_mass[j] > 0.0 and j in pos:
                        coms[j] = self._segment_com(j, pos)

                for j in active:
                    if self.root_of.get(j) != root:
                        continue

                    st = self.pivots[j]
                    st_w, st_a, st_acc = st.ang_vel, st.ang_acc, st.acc
                    inertia = 0.0
                    t_world = np.zeros(3)
                    pj = pos[j]

                    for s in self.subtree_of[j]:
                        com = coms.get(s)
                        if com is None:
                            continue
                        m = proc._seg_mass[s]
                        r = com - pj
                        rx, ry, rz = r[0], r[1], r[2]
                        inertia += proc._seg_local_inertia[s] + m * (rx * rx + ry * ry + rz * rz)

                        # Effective gravity in the pivot's accelerating,
                        # rotating frame (see module docstring).  Each joint
                        # pivots about itself, so this lever and the one the
                        # torque is taken about are the same vector.  Cross
                        # products written out: this is the innermost loop.
                        wx, wy, wz = st_w[0], st_w[1], st_w[2]
                        ax_, ay_, az_ = st_a[0], st_a[1], st_a[2]
                        # w x r
                        c1x = wy * rz - wz * ry; c1y = wz * rx - wx * rz; c1z = wx * ry - wy * rx
                        # a_frame = acc + ang_acc x r + w x (w x r)
                        fx = st_acc[0] + (ay_ * rz - az_ * ry) + (wy * c1z - wz * c1y)
                        fy = st_acc[1] + (az_ * rx - ax_ * rz) + (wz * c1x - wx * c1z)
                        fz = st_acc[2] + (ax_ * ry - ay_ * rx) + (wx * c1y - wy * c1x)
                        gx = m * (g0 - tr_ * fx); gy = m * (g1 - tr_ * fy); gz = m * (g2 - tr_ * fz)
                        t_world[0] += ry * gz - rz * gy
                        t_world[1] += rz * gx - rx * gz
                        t_world[2] += rx * gy - ry * gx

                    # Ground reaction under any point hanging off this joint --
                    # a hand pressing on the floor torques the elbow and the
                    # shoulder above it.
                    # The full force, not a mass-fraction share of it.  Pin
                    # joints transmit force; what stops a hand's load from
                    # reaching the elbow is nothing at all, and with it scaled
                    # to a hand's three percent of the body a straight limp arm
                    # under load stood as a column and held the body up --
                    # measured: a released handstand sinking seven centimetres
                    # in a third of a second on arms that should have folded.
                    # The force itself is already bounded by the mass the point
                    # is decelerating, which is what made the share necessary.
                    #
                    # And re-evaluated here, at this substep's position, rather
                    # than held for the frame.  A hand carrying its cap kicks
                    # its wrist at fifteen hundred radians per second squared;
                    # a spring force frozen across a frame at that rate is
                    # unstable, and showed up as friction alternating sign on
                    # every frame.  Depth is refreshed from the chain the
                    # substep just solved; the velocity factors stay the
                    # frame's.
                    load = 0.0
                    tc = np.zeros(3)
                    if self._frame_contact is not None:
                        c_pos, c_force = self._frame_contact
                        up = getattr(proc, 'internal_y_dim', 1)
                        kfac = self._frame_kfac
                        for n_c in self.contact_of_joint.get(j, ()):
                            k_c = int(self.contact_idx[n_c])
                            p_now = pos.get(k_c)
                            if p_now is None or kfac is None:
                                fv = c_force[n_c]
                                p_use = c_pos[n_c]
                            else:
                                pen = (p.floor_height + self.contact_radius[n_c]) - p_now[up]
                                if pen <= 0.0:
                                    continue
                                normal = min(kfac[n_c] * min(pen, p.max_penetration),
                                             self._frame_cap[n_c])
                                if self._anchor_valid[n_c]:
                                    disp = p_now - self._friction_anchor[n_c]
                                    disp[up] = 0.0
                                else:
                                    disp = np.zeros(3)
                                tangent = -self._frame_kt * disp - self._frame_ct * self._frame_vtan[n_c]
                                mag = float(np.linalg.norm(tangent))
                                cone = p.friction * normal
                                if mag > cone and mag > 1e-12:
                                    tangent = tangent * (cone / mag)
                                fv = tangent.copy()
                                fv[up] += normal
                                p_use = p_now
                            if fv[0] or fv[1] or fv[2]:
                                tc += _cross(p_use - pj, fv)
                                load += math.sqrt(float(fv @ fv))
                        load = min(load / max(self._body_weight, 1e-6), 4.0)
                    t_world += tc

                    # Body against body.  Both sides of a pair are pushed, each
                    # about whichever joints carry it.
                    for ci, ck, p1, p2, force in self._frame_self:
                        if j in self._self_cap_joints[ci]:
                            t_world += _cross(p1 - pj, force)
                        if j in self._self_cap_joints[ck]:
                            t_world += _cross(p2 - pj, -force)

                    inertia = max(inertia, 1e-6)
                    inertias[j] = inertia

                    # Into the parent's frame -- the frame the analyzer uses
                    # for local torque, and the frame the pose vector lives in.
                    pa = self.parents[j]
                    parent_rot = rot[pa] if pa in rot else frame_rot
                    t = parent_rot.T @ t_world

                    aa_j = self.local_rot[j].as_rotvec()
                    w_j = self.ang_vel[j]
                    # The ground's work on this joint, for the energy budget:
                    # what the floor puts into a folding knee is real, and was
                    # being confiscated as fabricated (see _enforce_passivity).
                    self._contact_work += float((parent_rot.T @ tc) @ w_j) * sub_dt

                    t += self._limit_torque(j, local_mats[j], w_j, inertia, p)
                    # Viscous damping, plus a drag term that grows with speed.
                    # Soft tissue and joint capsules really do resist harder
                    # the faster a limb moves, and without it a whole-body
                    # ragdoll being slammed by the ground pins its distal
                    # joints at the velocity clamp and never recovers -- this
                    # model has no Coriolis or gyroscopic coupling to bleed
                    # that energy back out through the chain.
                    # Optional extra damping while the joint is carrying ground
                    # load (`contact_damp_boost`, off by default).  It was
                    # added against pinball settling, which per-substep joint
                    # contact then solved at source -- and left on, it held a
                    # landing knee to half its fold rate.  A loaded limp knee
                    # folds; it does not stiffen.  Kept as a knob for a body
                    # that should read as heavy or exhausted on the ground.
                    speed = math.sqrt(float(w_j @ w_j))
                    damp = (p.damping * (1.0 + p.contact_damp_boost * load)
                            + p.drag * (1.0 + load) * speed)
                    t += -inertia * damp * w_j

                    wgt = weights[j]
                    if wgt > 1e-6:
                        # Authority diverges as the weight approaches 1 so that
                        # the controlled branch meets the prescribed branch
                        # instead of stepping into it.  At a fixed gain the
                        # limb droops under gravity by an amount that does not
                        # vanish, and a release ramping down from 1.0 would
                        # jump on its very first frame.  The damping gain
                        # follows the square root, holding the damping ratio
                        # constant across the range.
                        authority = min(wgt / max(1.0 - wgt, 1e-4), 1.0e4)
                        err = (self.prev_mocap[j] * self.local_rot[j].inv()).as_rotvec()
                        t_pd = inertia * (
                            p.kp * authority * err
                            + p.kd * np.sqrt(authority) * (self.mocap_ang_vel[j] - w_j))
                        # The actuator ceiling still scales with the weight, so
                        # a weak joint cannot chase the target however high the
                        # gain goes -- it simply sags within its strength.
                        ceiling = wgt * max_torque[j]
                        t_pd = np.clip(t_pd, -ceiling, ceiling)
                        t += t_pd
                        # A controller is an energy source, unlike everything
                        # else here.  Its work is tracked so the passivity
                        # observer below can tell real work from fabricated.
                        self._controller_work += float(t_pd @ w_j) * sub_dt

                    torques[j] = t

            # Integrate every active joint from the same configuration.
            for j in active:
                t = torques.get(j)
                if t is None:
                    continue
                inertia = inertias[j]
                ang_acc = t / inertia
                w_new = self.ang_vel[j] + ang_acc * sub_dt

                if not np.all(np.isfinite(w_new)):
                    # Blow-up guard: fall back to the captured pose rather than
                    # emit garbage into the renderer.
                    self.local_rot[j] = self.prev_mocap[j]
                    self.ang_vel[j] = np.zeros(3)
                    continue

                n = np.linalg.norm(w_new)
                if n > p.max_ang_vel:
                    w_new = w_new * (p.max_ang_vel / n)
                self.ang_vel[j] = w_new
                self.local_rot[j] = R.from_rotvec(w_new * sub_dt) * self.local_rot[j]

                self.last_torque[j] = t
                self.last_inertia[j] = inertia

        return {j: self.local_rot[j].as_rotvec() for j in self.free_indices}

    # ------------------------------------------------------------------
    # Root: six degrees of freedom by conservation
    # ------------------------------------------------------------------

    def _full_fk(self, aa24, trans):
        """Forward kinematics of the whole body, internal frame."""
        quats = R.from_rotvec(np.asarray(aa24).reshape(24, 3)).as_quat().reshape(1, 24, 4)
        wp, gr, _tips = self.processor._compute_forward_kinematics(
            np.asarray(trans, dtype=float).reshape(1, 3), quats)
        rot = np.zeros((30, 3, 3))
        for i in range(30):
            rot[i] = np.asarray(gr[i].as_matrix()).reshape(-1, 3, 3)[0]
        return wp[0], rot

    def _body_dynamics(self, world_pos, rot_mats, local_ang_vel):
        """Whole-body mass properties and the momentum of the limbs' own motion.

        Returns (mass, centre of mass, inertia tensor about the centre of mass,
        relative angular momentum).

        The inertia tensor is built from the same segment table the analyzer
        uses, treating each segment as a thin rod: a rod of mass m and length L
        along unit direction d has inertia (m L^2 / 12)(Identity - d d^T) about
        its own centre, which reduces to the analyzer's scalar on any axis
        perpendicular to the bone.  The torso segments are short, so the body's
        inertia here is a little lower than a real one -- a released body will
        spin slightly fast for its angular momentum.

        The relative angular momentum is what the body's own limbs carry while
        moving in the root's frame.  Subtracting it from the conserved total is
        what lets a tuck speed up a spin.
        """
        proc = self.processor
        parents = self.parents
        seg_mass = proc._seg_mass
        seg_len = proc._seg_length
        eye = np.eye(3)

        coms = np.zeros((24, 3))
        dirs = np.zeros((24, 3))
        for s in range(24):
            if seg_mass[s] <= 0.0:
                continue
            kids = self.children.get(s, [])
            if kids:
                end = np.mean([world_pos[c] for c in kids], axis=0)
            else:
                pa = parents[s]
                d = world_pos[s] - world_pos[pa] if pa >= 0 else np.array([0.0, 1.0, 0.0])
                n = np.linalg.norm(d)
                d = d / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
                end = world_pos[s] + d * seg_len[s]
            coms[s] = 0.5 * (world_pos[s] + end)
            dv = end - world_pos[s]
            n = np.linalg.norm(dv)
            dirs[s] = dv / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])

        mass = float(seg_mass.sum())
        com = (seg_mass[:, None] * coms).sum(axis=0) / max(mass, 1e-9)

        # Angular velocity of each segment, and the velocity of each joint
        # origin, both measured in the root's frame -- so the root itself is
        # stationary by construction and only the joints contribute.
        omega_rel = np.zeros((24, 3))
        u_origin = np.zeros((24, 3))
        for j in range(1, 24):
            pa = parents[j]
            omega_rel[j] = omega_rel[pa] + rot_mats[pa] @ local_ang_vel[j]
            u_origin[j] = u_origin[pa] + _cross(omega_rel[pa],
                                                world_pos[j] - world_pos[pa])

        # Kept for the contact solver, which needs each contact point's
        # velocity in the root's frame, and for the exact kinetic energy.
        self._omega_rel = omega_rel
        self._u_origin = u_origin
        self._seg_coms = coms
        self._seg_dirs = dirs
        self._world_pos_cache = world_pos

        inertia = np.zeros((3, 3))
        rel_momentum = np.zeros(3)
        for s in range(24):
            m = seg_mass[s]
            if m <= 0.0:
                continue
            d = dirs[s]
            i_seg = (m * seg_len[s] ** 2 / 12.0) * (eye - np.outer(d, d))
            r = coms[s] - com
            inertia += i_seg + m * (float(r @ r) * eye - np.outer(r, r))
            u = u_origin[s] + _cross(omega_rel[s], coms[s] - world_pos[s])
            rel_momentum += i_seg @ omega_rel[s] + m * _cross(r, u)

        return mass, com, inertia, rel_momentum

    @staticmethod
    def _rodrigues(v):
        """Rotation matrix for a rotation vector, without a Rotation object."""
        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            return np.eye(3)
        k = v / theta
        kx = np.array([[0.0, -k[2], k[1]],
                       [k[2], 0.0, -k[0]],
                       [-k[1], k[0], 0.0]])
        return (np.eye(3) + np.sin(theta) * kx
                + (1.0 - np.cos(theta)) * (kx @ kx))

    def _integrate_root(self, inertia, rel_momentum, offset_body, mass,
                        contact, dt, p):
        """Integrate the root's six degrees of freedom across one frame.

        The joints do not move within a frame, so the inertia tensor, the
        limbs' relative momentum, and every contact point's place in the body
        are constant in the *body's* frame -- only their world orientation
        changes.  That lets the whole thing substep on plain matrix algebra:
        no repeated kinematics, and the single matrix inverse taken once per
        frame rather than once per substep.

        Written in the body frame, the spin relation is simply

            body spin = inverse body inertia @ (body-frame momentum - relative)

        The equations of motion are the two conservation statements with the
        ground's reaction added as the only external force:

            mass x centre-of-mass acceleration = weight + total contact force
            rate of change of angular momentum = contact torque about the
                                                 centre of mass

        With no contact the second is zero and the first is gravity alone,
        which recovers the free-flight case exactly.  A single first-order step
        per frame is too coarse for a fast tumble -- it loses energy visibly
        over a long flight -- so the rotation uses a midpoint evaluation,
        second order in the step size.

        Updates the centre of mass, its velocity, the angular momentum and the
        root's rotation.  Returns the final world angular velocity.
        """
        up = getattr(self.processor, 'internal_y_dim', 1)
        gravity = np.zeros(3)
        gravity[up] = -9.81 * p.gravity

        rot = self.root_rot.as_matrix()
        i_body = rot.T @ inertia @ rot
        rel_body = rot.T @ rel_momentum
        try:
            i_body_inv = np.linalg.inv(i_body + 1e-9 * np.eye(3))
        except np.linalg.LinAlgError:
            return self.root_ang_vel.copy()

        def world_spin(q):
            om = q @ (i_body_inv @ (q.T @ self.ang_momentum - rel_body))
            n = np.linalg.norm(om)
            if n > p.max_ang_vel:
                om = om * (p.max_ang_vel / n)
            return om

        # The root is cheap -- matrix algebra, no kinematics -- so it keeps a
        # floor of eight regardless.
        n_sub = max(self._substeps(p), 8)
        h = dt / n_sub
        omega = self.root_ang_vel.copy()
        inv_mass = 1.0 / max(mass, 1e-9)
        force_sum = np.zeros((len(self.contact_idx), 3))
        elapsed = 0.0
        self._contact_work_root = 0.0

        for _ in range(n_sub):
            force = np.zeros(3)
            torque = np.zeros(3)
            if contact is not None:
                p_body, v_rel_body = contact
                root_pos = self.com - rot @ offset_body
                # The joints are frozen for the frame, so a contact point's
                # place in the body would be too -- and a limb travelling ten
                # metres a second would appear to jump several centimetres deep
                # between evaluations, which arrives as one enormous impulse.
                # Carrying the point forward at its own relative velocity keeps
                # the penetration growing smoothly across the substeps.
                p_world = root_pos + (p_body + v_rel_body * elapsed) @ rot.T
                lever = p_world - self.com
                om = world_spin(rot)
                v_world = (self.com_vel + _cross_one_many(om, lever)
                           + v_rel_body @ rot.T)
                f, touching = self._contact_eval(p_world, v_world, mass, p)
                if touching.any():
                    force = f.sum(axis=0)
                    torque = _cross_many(lever, f).sum(axis=0)
                    force_sum += f
                    self._contact_work_root += (float(force @ self.com_vel)
                                                + float(torque @ om)) * h

            # Exact for constant acceleration over the substep.
            acc = gravity + force * inv_mass
            self.com = self.com + self.com_vel * h + 0.5 * acc * h * h
            self.com_vel = self.com_vel + acc * h
            self.ang_momentum = self.ang_momentum + torque * h

            om1 = world_spin(rot)
            if not np.all(np.isfinite(om1)):
                return np.zeros(3)
            mid = self._rodrigues(om1 * (0.5 * h)) @ rot
            omega = world_spin(mid)
            rot = self._rodrigues(omega * h) @ rot
            elapsed += h

        if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(self.com)):
            return np.zeros(3)
        # from_matrix re-orthonormalises, keeping the product of many small
        # rotations from creeping away from a valid rotation.
        self.root_rot = R.from_matrix(rot)
        self.last_contact_force = force_sum / n_sub
        return omega

    # ------------------------------------------------------------------
    # Ground contact
    # ------------------------------------------------------------------

    def _contact_state(self, world_pos, root_rot_mat, root_pos):
        """Contact points, expressed in the root's own frame.

        Within a frame the joints do not move, so a point's place and velocity
        relative to the root are fixed -- only the root's position and
        orientation change across the substeps.  Taking them into the body
        frame once therefore lets the contact solve substep on plain matrix
        algebra, with no repeated kinematics.
        """
        idx = self.contact_idx
        pw = world_pos[idx]
        vw = np.zeros((len(idx), 3))
        omega_rel, u_origin = self._omega_rel, self._u_origin
        for i, k in enumerate(idx):
            k = int(k)
            if k < 24:
                vw[i] = u_origin[k]
            else:
                # A virtual tip turns with its parent joint.
                pa = self.parents[k]
                vw[i] = u_origin[pa] + _cross(omega_rel[pa],
                                              world_pos[k] - world_pos[pa])
        rt = root_rot_mat.T
        return (pw - root_pos) @ rt.T, vw @ rt.T

    def _contact_eval(self, p_world, v_world, mass, p, commit=False):
        """Ground reaction at every contact point.

        Normal force is a Hunt-Crossley penalty: the damping term scales with
        the penetration, so the force rises continuously from zero at
        touchdown instead of stepping, and it can never pull the body down.
        Stiffness is set from how far the body's own weight should sink.

        Friction is a spring to an anchor.  When a point touches down, where it
        touched is remembered; the tangential force is stiffness times how far
        it has since moved from that spot, capped at the friction cone, and
        when the cap is reached the anchor is dragged along behind the point --
        which is sliding, and is Coulomb.

        It is not velocity-based, and that is the point.  Coulomb friction
        regularised at a small slip speed hands the full sliding force to any
        point moving faster than that speed, and a hand this simulation holds
        "planted" reads several centimetres a second: its velocity is the small
        difference of three large estimated terms.  So a planted hand was being
        braked with two thousand newtons for the crime of a millimetre of
        estimation noise, and two planted hands on a rotating body were fighting
        each other -- measured as the friction impulse that stripped a released
        cartwheel of its rotation while the normal forces were correctly
        turning it over.  Displacement is what a planted hand actually resists,
        and noise in velocity makes almost none.

        `commit` says this is the once-per-frame evaluation, at which anchors
        are placed for new touches and dragged for sliding ones.  The root's
        substepped solve evaluates the same springs without moving anchors.
        """
        up = getattr(self.processor, 'internal_y_dim', 1)
        forces = np.zeros_like(p_world)
        penetration = (p.floor_height + self.contact_radius) - p_world[:, up]
        touching = penetration > 0.0

        n_pts = p_world.shape[0]
        if self._friction_anchor is None or self._friction_anchor.shape[0] != n_pts:
            self._friction_anchor = p_world.copy()
            self._anchor_valid = np.zeros(n_pts, dtype=bool)
        if commit:
            newly = touching & ~self._anchor_valid
            self._friction_anchor[newly] = p_world[newly]
            self._anchor_valid[:] = touching

        if not touching.any():
            return forces, touching

        weight = mass * 9.81
        stiffness = weight / max(p.contact_depth, 1e-4)
        v = v_world[touching]
        v_up = v[:, up]

        # Two bounds, both needed.  The spring only responds to penetration up
        # to a limit, and the total ground force is capped -- without them, a
        # body that is already below the floor when it is released (which real
        # capture does, whenever it sits under the estimated floor) sees a
        # force proportional to how deep it is and gets fired into the air.
        # The cap scales the whole set down together rather than clipping each
        # point, so the distribution of load across the points is preserved.
        depth = np.minimum(penetration[touching], p.max_penetration)

        # The spring is faded out as the point separates, so contact can push a
        # body out of the ground but never faster than the recovery speed.
        # Without this a body released while already below the floor starts
        # with the spring loaded, and that stored energy launches it -- correct
        # spring behaviour, entirely wrong bodies.  It also makes landings
        # inelastic, which is what a body does: it lands, it does not bounce.
        taper = np.clip(1.0 - np.maximum(v_up, 0.0) / max(p.recovery_speed, 1e-4),
                        0.0, 1.0)

        # The damping term saturates with approach speed.  Left unbounded it
        # grows without limit, so a single light limb whipping into the floor
        # at ten metres a second generates tens of kilonewtons, swallows the
        # whole body's force budget and throws the body into the air.  Real
        # contact damping saturates; so does this.
        approach = np.minimum(np.maximum(0.0, -v_up), p.damping_speed)
        normal = stiffness * depth * taper * (1.0 + p.contact_damping * approach)
        normal = np.maximum(normal, 0.0)
        # One point may not claim the whole body's budget: a light limb
        # whipping into the floor would otherwise deliver a body-launching
        # impulse on its own.  The total is bounded as well, and scaled as a
        # set so the load stays distributed the way the geometry says.
        point_cap = np.minimum(p.max_point_g * weight,
                               self.point_mass_eff[touching] * p.max_point_accel)
        normal = np.minimum(normal, point_cap)
        total = normal.sum()
        ceiling = p.max_contact_g * weight
        if total > ceiling:
            normal = normal * (ceiling / total)

        idx = np.where(touching)[0]
        if commit:
            # Kept for the joints' substepped re-evaluation: the velocity
            # factors are held for the frame, the depth is not.
            self._frame_kfac = np.zeros(n_pts)
            self._frame_kfac[idx] = stiffness * taper * (1.0 + p.contact_damping * approach)
            self._frame_cap = np.minimum(p.max_point_g * weight,
                                         self.point_mass_eff * p.max_point_accel)
            self._frame_vtan = np.zeros((n_pts, 3))
            self._frame_kt = stiffness
            self._frame_ct = float(np.sqrt(stiffness * mass * 0.1))
        anchor = np.where(self._anchor_valid[idx, None],
                          self._friction_anchor[idx], p_world[idx])
        disp = p_world[idx] - anchor
        disp[:, up] = 0.0
        v_tan = v.copy()
        v_tan[:, up] = 0.0
        if commit:
            self._frame_vtan[idx] = v_tan

        # The ground is as stiff sideways as it is downward, and lightly
        # damped -- a tenth of the body as the effective mass at a limb, half
        # critical -- so a planted point settles rather than rings.
        k_t = stiffness
        c_t = float(np.sqrt(k_t * mass * 0.1))
        tangent = -k_t * disp - c_t * v_tan
        mag = np.linalg.norm(tangent, axis=1)
        cone = p.friction * normal
        over = mag > cone
        if over.any():
            tangent[over] *= (cone[over] / np.maximum(mag[over], 1e-12))[:, None]
            if commit:
                # Sliding: the anchor trails the point at the edge of the cone.
                slid = idx[over]
                dragged = p_world[slid] + tangent[over] / k_t
                dragged[:, up] = self._friction_anchor[slid, up]
                self._friction_anchor[slid] = dragged

        out = tangent
        out[:, up] += normal
        forces[touching] = out
        return forces, touching

    # ------------------------------------------------------------------
    # Passivity: removing the energy the coupling fabricates
    # ------------------------------------------------------------------

    def _mechanical_energy(self, mass, p):
        """Total mechanical energy of the simulated body.

        Kinetic (the centre of mass, the root's spin, and each free joint's own
        rotation), gravitational potential, and the energy stored in the
        contact springs.
        """
        up = getattr(self.processor, 'internal_y_dim', 1)
        energy = self._kinetic_energy(mass)
        if self.com is not None:
            energy += mass * 9.81 * float(self.com[up])
        energy += self._limit_potential(p) + self._frame_self_potential
        if self._frame_penetration is not None:
            stiffness = (mass * 9.81) / max(p.contact_depth, 1e-4)
            pen = self._frame_penetration
            energy += 0.5 * stiffness * float(pen @ pen)
        return energy

    def _limit_potential(self, p):
        """Energy stored in the joint limit springs.

        A limb that swings into a joint stop loads that spring and gets the
        energy back on the way out.  Left out of the accounting, every such
        release reads as energy appearing from nowhere and the observer
        confiscates it -- which quietly strangles all the internal motion, so
        a ragdoll released in flight stops moving its limbs altogether.

        This is the integral of the restoring torque in _limit_torque, so it
        must be kept in step with it: a linear spring outside the box gives a
        quadratic potential in how far past the limit each axis is.
        """
        energy = 0.0
        for j in self._active_joints:
            if not self._lim_active[j]:
                continue
            rot = self.local_rot.get(j)
            if rot is None:
                continue
            k = self._lim_k[j] * p.limit_stiffness
            if k <= 0.0:
                continue
            angles = self._limit_angles(j, rot.as_matrix())
            over = (np.maximum(self._lim_min[j] - angles, 0.0)
                    + np.maximum(angles - self._lim_max[j], 0.0))
            # Integral of the progressive stop in _limit_torque.
            soft = max(p.stop_softness, 1e-4)
            for mag in over:
                if mag <= 0.0:
                    continue
                if mag < soft:
                    energy += k * mag ** 3 / (6.0 * soft)
                else:
                    energy += k * soft * soft / 6.0 + 0.5 * k * ((mag - 0.5 * soft) ** 2
                                                                  - 0.25 * soft * soft)
        return energy

    def _kinetic_energy(self, mass, spin=None):
        return self._kinetic_energies(mass)[1 if spin is not None else 0]

    def _kinetic_energies(self, mass):
        """Kinetic energy summed over the segments, which is the only way to
        get it right.

        Adding up half-inertia-omega-squared per joint counts every distal
        segment once for each joint above it -- a hand lands in the shoulder's
        subtree, the elbow's and the wrist's -- so it over-reports, and worse,
        it can appear to rise when a distal joint speeds up while a proximal
        one slows.  An energy observer fed that measure sees injection that
        never happened and bleeds real motion away.

        Summing over segments instead:

            total = centre-of-mass translation
                  + each segment's motion relative to the centre of mass
                  + each segment's spin

        The cross term vanishes because the relative momenta sum to zero about
        the centre of mass.
        """
        proc = self.processor
        seg_mass = proc._seg_mass
        seg_len = proc._seg_length
        coms = self._seg_coms
        dirs = self._seg_dirs
        world_pos = self._world_pos_cache
        omega = self.root_ang_vel
        omega_rel, u_origin = self._omega_rel, self._u_origin
        eye = np.eye(3)

        # Segment centre-of-mass velocities within the root's frame.
        u = np.zeros((24, 3))
        for seg in range(24):
            if seg_mass[seg] <= 0.0:
                continue
            u[seg] = u_origin[seg] + _cross(omega_rel[seg],
                                            coms[seg] - world_pos[seg])
        drift = (seg_mass[:, None] * u).sum(axis=0) / max(mass, 1e-9)

        translation = 0.5 * mass * float(self.com_vel @ self.com_vel)
        energy = translation
        no_spin = translation
        for seg in range(24):
            m = seg_mass[seg]
            if m <= 0.0:
                continue
            lever = coms[seg] - self.com
            rel0 = u[seg] - drift
            rel = _cross(omega, lever) + rel0
            energy += 0.5 * m * float(rel @ rel)
            no_spin += 0.5 * m * float(rel0 @ rel0)
            d = dirs[seg]
            i_seg = (m * seg_len[seg] ** 2 / 12.0) * (eye - np.outer(d, d))
            spin = omega + omega_rel[seg]
            energy += 0.5 * float(spin @ i_seg @ spin)
            wr = omega_rel[seg]
            no_spin += 0.5 * float(wr @ i_seg @ wr)
        return energy, no_spin

    def _enforce_passivity(self, mass, p):
        """Measure the energy the integration invents, and take it back out.

        This body is passive: with the joints free nothing does work, gravity
        is conservative, damping and friction only ever dissipate, and the
        contact spring returns less than it stored because it fades out as a
        point separates.  So the total mechanical energy cannot rise.  If it
        does, the increase did not come from the physics -- it came from every
        joint integrating as though the rest of the body were rigid for the
        frame, when the rest of the body is doing the same thing back.

        That makes the fault directly measurable rather than something to
        detect by symptom and damp against.  Whatever the energy rose by is
        removed, by scaling the momenta -- exactly the excess, no more.

        Measured against a frozen-joint body in flight, which the root
        integrator carries to within 0.06 percent over five seconds, the
        fabricated energy grows with the length of the free chain: a single
        free arm drifts 0.2 percent and needs almost nothing taken back, while
        a whole free body drifts 22 percent.  That is why one limb behaved from
        the start and a whole ragdoll did not, and it is why this runs in
        flight as well as on the ground -- contact never caused the injection,
        it only made it violent enough to see.

        A controller at an intermediate blend weight is a genuine energy
        source, so the work it did is subtracted first; only what is left over
        counts as fabricated.
        """
        self.last_energy_injected = 0.0

        # The balance only holds for a closed body.  A joint the capture is
        # driving does work that never passes through a torque computed here,
        # so its energy cannot be distinguished from fabricated energy -- and
        # confiscating it stops, for instance, a tuck from speeding up a spin,
        # which is real physics arriving from outside the simulation.  A
        # controlled joint's work is subtracted explicitly; a prescribed one's
        # cannot be, so the observer stands down unless the whole body is free.
        closed = (self.root_free
                  and len(self._active_joints) == len(self.free_indices)
                  and all(j in self.free_indices for j in range(1, 22)))
        if not p.passivity or self.com is None or not closed:
            self._prev_energy = None
            self._energy_excess = 0.0
            return

        # The change is accumulated from its parts rather than by differencing
        # a total.  Gravitational potential dwarfs everything else -- tens of
        # kilojoules against a few hundred of motion -- so subtracting one
        # frame's total from the next is a small difference of large numbers,
        # and the rounding noise alone is bigger than the effect being looked
        # for.  Each part changes by a small amount per frame, so differencing
        # the parts is exact where differencing the total is not.
        up = getattr(self.processor, 'internal_y_dim', 1)
        kinetic_now, kinetic_no_spin = self._kinetic_energies(mass)
        height_now = float(self.com[up])
        # The joint-limit and self-collision springs are part of the body;
        # the ground is not.  The ground's spring used to be counted here from
        # the root-side penetration, while the joints were being driven by the
        # same springs evaluated at their own substep positions -- so the work
        # the floor did on a folding knee arrived as kinetic energy with no
        # matching potential the budget could see, read as fabricated, and
        # was removed as fast as the ground supplied it.  Measured: a landing
        # knee held at half the fold rate of one with the budget off.  The
        # ground is now external, its work subtracted below, and what remains
        # is the integration inconsistency alone -- which is all this exists
        # to correct.
        spring_now = self._limit_potential(p) + self._frame_self_potential

        previous = self._prev_energy
        self._prev_energy = (kinetic_now, height_now, spring_now)
        if previous is None:
            return
        k_prev, h_prev, s_prev = previous
        delta = ((kinetic_now - k_prev)
                 + mass * 9.81 * (height_now - h_prev)
                 + (spring_now - s_prev))

        # Only kinetic energy can be taken back out by scaling momenta, so a
        # frame that fabricates more than the body currently has -- an impact
        # that lifts it, say -- cannot be settled on the spot.  The remainder
        # is carried as a debt and collected once there is motion to take it
        # from, which makes the energy constraint hold over time rather than
        # only frame by frame.  Without it the observer runs permanently one
        # frame behind and the body never comes to rest.
        # Judged over time, not frame by frame.  The body's internal kinetic
        # energy genuinely swings from one frame to the next -- the same joint
        # rates give different segment velocities as the pose turns over -- so
        # taking energy on every up-swing while never returning it on a
        # down-swing ratchets all the motion away, whatever dead zone is used.
        # Accumulated instead, an oscillation cancels itself and only a
        # sustained gain survives, which is exactly the thing worth removing.
        # The floor's work is the same force counted at the root and at each
        # ancestor joint, and that is not double counting: F . v_foot expands
        # exactly into the root's F . v_com + torque . spin plus every joint's
        # torque . rate.  The decoupled model's two applications of the force
        # are the two halves of the true work.
        self._energy_excess += (delta - self._controller_work
                                - self._contact_work - self._contact_work_root)

        # The fabricated energy is internal -- it comes from the joints, not
        # from the centre of mass, whose parabola is exact.  So both the scale
        # the excess is judged against and the motion it is taken from are the
        # body's motion *about* its centre of mass.  In flight the trajectory
        # is left strictly alone; only while the ground is pushing can contact
        # have put fabricated energy into the centre of mass itself, and only
        # then is that velocity part of the correction.
        translation = 0.5 * mass * float(self.com_vel @ self.com_vel)

        # What may be taken, and from where.  Not the angular momentum: in
        # flight it is exactly conserved and integrated correctly, and the
        # fabricated energy is in the joints, not in the body's global spin.
        # Scaling it robs a released cartwheel of the rotation it was thrown
        # with -- measured at 86 percent gone within two seconds -- which is
        # the whole point of releasing mid-move.  Taking it from the joints
        # instead still corrects the error, and the spin then follows from
        # conservation: as the limbs slow, the body turns faster, which is what
        # a body does.
        relative = max(kinetic_no_spin - translation, 0.0)
        internal = relative

        # A dead zone, because the observer can only ever take energy out and
        # never give it back.  Left to react to every positive wobble it
        # rectifies its own noise into a steady drain -- measured at thirty
        # percent of a flight's energy over five seconds, which reads as a
        # ragdoll going strangely limp in mid-air.
        # Real dissipation must not build up credit that would later mask an
        # injection, so the accumulator has a floor.
        band = p.passivity_deadband * max(internal, 1.0)
        if self._energy_excess < -band:
            self._energy_excess = -band
        if self._energy_excess <= band:
            return

        in_contact = self._frame_contact is not None
        kinetic = relative + (translation if in_contact else 0.0)
        if kinetic <= 1e-9:
            return

        # A separate rate while the ground is pushing.  Contact fabricates more
        # per frame than free flight does, and this once had to run near 1 to
        # keep a landed body from hopping; with the ground's work accounted
        # exactly and the joints re-evaluating contact per substep, 0.25
        # settles the body as well as 0.9 did and no longer strangles a
        # folding knee.  In the air a low rate keeps the tumble alive.
        rate = p.passivity_rate_contact if in_contact else p.passivity_rate
        removable = min(self._energy_excess * float(np.clip(rate, 0.0, 1.0)),
                        kinetic * float(np.clip(p.passivity_bleed, 0.0, 1.0)))
        if removable <= 0.0:
            return
        self._energy_excess -= removable
        scale = float(np.sqrt(max(0.0, 1.0 - removable / kinetic)))
        self.last_energy_injected = removable

        # Scaling the momenta scales every kinetic term by the square of the
        # factor, so the total drops by exactly the amount removed.
        if in_contact:
            self.com_vel = self.com_vel * scale
        for j in self._active_joints:
            if j in self.ang_vel:
                self.ang_vel[j] = self.ang_vel[j] * scale
        self._prev_energy = (kinetic_now - removable, height_now, spring_now)

    # ------------------------------------------------------------------
    # Self-collision
    # ------------------------------------------------------------------

    def _self_contacts(self, world_pos, mass, p):
        """Capsule-against-capsule contacts between the body's own parts.

        These forces are internal, which has a useful consequence: each pair is
        equal and opposite, so the whole set contributes no net force on the
        centre of mass and no net torque about it.  The root's ballistic
        trajectory and its angular momentum are therefore untouched -- a body
        cannot push itself anywhere by folding up -- and only the joints feel
        it, which is exactly right.

        Returns a list of (capsule, capsule, point, point, force on the first).
        """
        if not p.self_collision or not self._self_pairs:
            return []
        self._self_half = [0.5 * float(np.linalg.norm(world_pos[b] - world_pos[a]))
                           for _n, a, b, _r in SELF_CAPSULES]

        weight = mass * 9.81
        stiffness = weight / max(p.self_depth, 1e-4)
        ceiling = p.self_max_g * weight
        omega_rel, u_origin = self._omega_rel, self._u_origin
        out = []
        potential = 0.0

        for i, k in self._self_pairs:
            _, ai, bi, ri = SELF_CAPSULES[i]
            _, ak, bk, rk = SELF_CAPSULES[k]
            reach = ri + rk

            # Cheap rejection first.  Most of these pairs are nowhere near each
            # other most of the time, and comparing bounding spheres costs one
            # subtraction against a full closest-point solve on two segments.
            m1 = 0.5 * (world_pos[ai] + world_pos[bi])
            m2 = 0.5 * (world_pos[ak] + world_pos[bk])
            gap = m2 - m1
            span = (self._self_half[i] + self._self_half[k] + reach)
            if float(gap @ gap) > span * span:
                continue

            c1, c2, dist = _closest_points_on_segments(
                world_pos[ai], world_pos[bi], world_pos[ak], world_pos[bk])
            if dist >= reach:
                continue
            overlap = min(reach - dist, p.max_penetration)
            potential += 0.5 * stiffness * overlap * overlap

            if dist > 1e-9:
                normal = (c1 - c2) / dist
            else:
                # Exactly coincident: push apart along the body's up axis
                # rather than divide by zero.
                normal = np.zeros(3)
                normal[getattr(self.processor, 'internal_y_dim', 1)] = 1.0
            depth = min(reach - dist, p.max_penetration)

            # Relative velocity of the two touching points, measured in the
            # root's frame -- the root's own motion is common to both and
            # cancels, so it does not enter.
            v1 = u_origin[ai] + _cross(omega_rel[ai], c1 - world_pos[ai])
            v2 = u_origin[ak] + _cross(omega_rel[ak], c2 - world_pos[ak])
            closing = float((v2 - v1) @ normal)

            # Same shape as the ground: damping saturating with approach speed,
            # and the spring fading out as the pair separates so it can push
            # apart but never fling.
            approach = min(max(closing, 0.0), p.damping_speed)
            taper = float(np.clip(1.0 + min(closing, 0.0) / max(p.recovery_speed, 1e-4),
                                  0.0, 1.0))
            magnitude = stiffness * depth * taper * (1.0 + p.contact_damping * approach)
            magnitude = float(np.clip(magnitude, 0.0, ceiling))
            if magnitude > 0.0:
                out.append((i, k, c1, c2, normal * magnitude))
        self._frame_self_potential = potential
        return out

    @staticmethod
    def _blend_rot(sim_rot, mocap_rot, weight):
        """Geodesic blend: weight 0 gives the simulation, 1 gives the capture."""
        delta = (mocap_rot * sim_rot.inv()).as_rotvec()
        return R.from_rotvec(delta * weight) * sim_rot

    def advance(self, mocap_aa, mocap_trans, weights, p):
        """Advance one captured frame, root included.

        Args:
            mocap_aa:    (24, 3) captured local rotations, internal frame.
            mocap_trans: (3,) captured root translation, internal frame.
            weights:     (22,) blend weight per joint; index 0 is the root.
            p:           RagdollParams.

        Returns:
            (joint rotation dict, root world rotation, root translation),
            the last two already blended toward the capture by the root weight.
        """
        proc = self.processor
        dt = p.dt
        mocap_aa = np.asarray(mocap_aa, dtype=float).reshape(-1, 3)[:24]
        mocap_trans = np.asarray(mocap_trans, dtype=float).reshape(3)
        mocap_root_rot = R.from_rotvec(mocap_aa[0])

        root_w = float(np.clip(weights[0], 0.0, 1.0)) if self.root_free else 1.0
        seed = (self.root_rot is None or self.com is None)
        prescribed = (not self.root_free) or root_w >= 1.0 - 1e-6

        # While prescribed, the simulated root shadows the capture, so a
        # release starts from the real trajectory and the real spin.
        if prescribed or seed:
            self.root_rot = mocap_root_rot
            self.trans = mocap_trans.copy()

        # Captured local angular velocity for every joint.  The driven joints
        # contribute to the body's relative momentum just as the free ones do.
        cur = mocap_aa.copy()
        if self._prev_mocap_all is None:
            mocap_lav = np.zeros((24, 3))
        else:
            delta = R.from_rotvec(cur) * R.from_rotvec(self._prev_mocap_all).inv()
            mocap_lav = np.clip(delta.as_rotvec() / dt, -p.max_ang_vel, p.max_ang_vel)
        self._prev_mocap_all = cur

        # Kinematics of the simulated configuration.
        aa = mocap_aa.copy()
        aa[0] = self.root_rot.as_rotvec()
        for j in self.free_indices:
            # Only a joint that is actually free contributes its simulated
            # rotation.  A prescribed one is slammed to the capture at the end
            # of the frame, so using last frame's simulated value here would
            # put the limbs a frame behind the root -- and on the frame a catch
            # completes that lag becomes a second discontinuity, arriving one
            # frame after the root's and slipping past the guard for it.
            if j in self.local_rot and weights[j] < 1.0 - 1e-6:
                aa[j] = self.local_rot[j].as_rotvec()
        world_pos, rot_mats = self._full_fk(aa, self.trans)

        free_set = set(self.free_indices)
        lav = mocap_lav.copy()
        for j in free_set:
            if j in self.ang_vel:
                lav[j] = self.ang_vel[j]
        lav[0] = 0.0

        mass, com_fk, inertia, rel_momentum = self._body_dynamics(
            world_pos, rot_mats, lav)

        # Contact is evaluated once at this frame's configuration, which is
        # what the joint torques use, and carried into the root's substepped
        # solve in body-frame form.
        # Self-collision, from the same kinematics.  Internal, so it never
        # reaches the root -- only the joints.
        self._frame_self_potential = 0.0
        self._frame_self = self._self_contacts(world_pos, mass, p)

        contact_body = None
        self._frame_contact = None
        self._frame_penetration = None
        if p.floor_enable and len(self.contact_idx):
            root_mat = self.root_rot.as_matrix()
            contact_body = self._contact_state(world_pos, root_mat, world_pos[0])
            p_world = world_pos[self.contact_idx]
            lever = p_world - com_fk
            v_world = (self.com_vel
                       + _cross_one_many(self.root_ang_vel, lever)
                       + contact_body[1] @ root_mat.T)
            f, touching = self._contact_eval(p_world, v_world, mass, p, commit=True)
            self._body_weight = mass * 9.81
            self.last_contact_force = f
            up = getattr(proc, 'internal_y_dim', 1)
            self._frame_penetration = np.clip(
                (p.floor_height + self.contact_radius) - p_world[:, up],
                0.0, p.max_penetration)
            if touching.any():
                self._frame_contact = (p_world, f)

        # The energy check runs here, before anything moves, so each frame is
        # measured at the same point in the cycle and against the relative
        # motion this frame's kinematics actually describe.  It applies only
        # when the root is free: with a driven root the body is not a closed
        # system -- the capture is free to do work on it -- and no energy
        # balance holds.
        if not prescribed and not seed:
            self._active_joints = [j for j in self.free_indices
                                   if weights[j] < 1.0 - 1e-6]
            self._enforce_passivity(mass, p)

        if prescribed or seed:
            # The velocity and spin held at the instant of release determine
            # the entire flight, and both come from differencing the capture
            # once -- which amplifies its noise by the frame rate, and the
            # translation channel is the noisy one (a body-mounted sensor with
            # mass, which flops).  So they are carried as smoothed running
            # estimates rather than taken raw from the last pair of frames.
            a = float(np.clip(p.root_seed_smoothing, 1e-3, 1.0))

            # Re-engaging after a release is a discontinuity, not motion.  The
            # body has been falling on its own while the capture carried on
            # somewhere else, so the frame the root is taken back the centre of
            # mass jumps from where it fell to where the capture is -- and
            # differencing that reads as tens of metres a second, which lands
            # in the stored velocity and angular momentum.  Release again
            # immediately afterwards and the body is fired off along that jump,
            # which looks like it remembers where it was let go before.
            # The frame is used to re-anchor rather than to measure.
            reengaged = prescribed and not self._was_prescribed
            if not reengaged and self._prev_com is not None:
                jump = float(np.linalg.norm(com_fk - self._prev_com)) / dt
                # A capture teleport is a discontinuity too, and no body
                # travels at fifteen metres a second.
                reengaged = jump > 15.0
            if reengaged:
                self._prev_com = com_fk.copy()
                self._prev_root_rot = self.root_rot
                self._prev_raw_vel = None
                self._prev_raw_omega = None

            if self._prev_com is not None and self._prev_root_rot is not None \
                    and not reengaged:
                raw_vel = (com_fk - self._prev_com) / dt
                raw_omega = (self.root_rot * self._prev_root_rot.inv()).as_rotvec() / dt
                raw_omega = np.clip(raw_omega, -p.max_ang_vel, p.max_ang_vel)

                # The frame rate is the amount of captured time each call
                # carries, not how fast the patch happens to run.  Feed every
                # frame of a 120 Hz take while this is set to 60 and the body
                # is handed twice the motion per step: momentum comes out
                # halved, gravity accumulates twice per frame of data, and the
                # thing bounces metres into the air.  Nothing in the pose
                # stream reveals the rate, so the only check available is
                # whether the speed it implies is one a body could have.
                speed = float(np.linalg.norm(raw_vel))
                if speed > 12.0 and not self._warned_rate:
                    self._warned_rate = True
                    print('smpl_ragdoll: capture implies %.0f m/s of body motion at '
                          '%g Hz -- no body moves that fast, so the framerate is '
                          'probably wrong.  It must be the captured time between '
                          'the frames you feed: every frame of a 120 Hz take is '
                          '120, every second frame is 60.' % (speed, 1.0 / max(dt, 1e-9)))

                self._vel_seed = self._vel_seed + a * (raw_vel - self._vel_seed)
                self._omega_seed = self._omega_seed + a * (raw_omega - self._omega_seed)

                # An exponential average of a steadily accelerating signal sits
                # a known distance behind it -- (1-a)/a frames.  Left alone,
                # that bias goes straight into the release velocity and bends
                # the whole flight.  Tracking the acceleration as well lets the
                # lag be cancelled outright, so smoothing costs noise rejection
                # only, not trajectory accuracy.
                if self._prev_raw_vel is not None:
                    self._seed_acc = self._seed_acc + a * (
                        (raw_vel - self._prev_raw_vel) / dt - self._seed_acc)
                    self._seed_ang_acc = self._seed_ang_acc + a * (
                        (raw_omega - self._prev_raw_omega) / dt - self._seed_ang_acc)
                self._prev_raw_vel = raw_vel
                self._prev_raw_omega = raw_omega

                lag = (1.0 - a) / a * dt
                self.com_vel = self._vel_seed + self._seed_acc * lag
                omega = np.clip(self._omega_seed + self._seed_ang_acc * lag,
                                -p.max_ang_vel, p.max_ang_vel)
            elif reengaged:
                # Keep the estimates from before the release as a prior; the
                # next frame differences cleanly and they carry on converging.
                self.com_vel = self._vel_seed + self._seed_acc * (
                    (1.0 - a) / a * dt)
                omega = np.clip(self._omega_seed, -p.max_ang_vel, p.max_ang_vel)
            else:
                self._vel_seed = np.zeros(3)
                self._omega_seed = np.zeros(3)
                self.com_vel = np.zeros(3)
                omega = np.zeros(3)
            self.com = com_fk.copy()
            # Seed the conserved quantity from the motion actually captured.
            self.ang_momentum = inertia @ omega + rel_momentum

            # How much of the captured motion the simulated body can actually
            # account for.  The capture holding a body up says nothing about
            # whether anything is still holding it: in a cartwheel with the
            # arms let go, the hands stop touching the floor and the ground
            # force vanishes, but the captured root sails on regardless and the
            # body hangs in the air on nothing.  Comparing the upward force the
            # captured motion needs against what contact is really supplying
            # measures exactly that, as a fraction rather than a threshold.
            up = getattr(proc, 'internal_y_dim', 1)
            needed = mass * (self._seed_acc[up] + 9.81 * p.gravity)
            if needed > 0.05 * mass * 9.81:
                supplied = float(np.sum(np.maximum(self.last_contact_force[:, up], 0.0)))
                self.last_support = float(np.clip(supplied / needed, 0.0, 1.0))
            else:
                # Ballistic, or falling: the capture is asking for no support,
                # so there is nothing it could be leaning on that is missing.
                self.last_support = 1.0
        else:
            # Flight.  Gravity acts at the centre of mass, so it exerts no
            # torque about it: the total angular momentum is constant and the
            # centre of mass is a parabola.
            offset_body = self.root_rot.inv().apply(com_fk - self.trans)
            omega = self._integrate_root(inertia, rel_momentum, offset_body,
                                         mass, contact_body, dt, p)

            # Place the root so the body's centre of mass lands where the
            # ballistic trajectory says it should.  The offset from root to
            # centre of mass is taken into the body's own frame first, so it
            # is re-applied under the *new* rotation rather than lagging a
            # frame behind it.  Re-derived every frame from the actual
            # kinematics, so limb motion cannot make it drift.
            self.trans = self.com - self.root_rot.apply(offset_body)

        self._was_prescribed = prescribed
        self.root_ang_vel = omega
        self._prev_com = self.com.copy()
        self._prev_root_rot = self.root_rot

        joint_aa = self.step(mocap_aa, world_pos, rot_mats, weights, p)

        if prescribed:
            return joint_aa, mocap_root_rot, mocap_trans
        out_rot = self._blend_rot(self.root_rot, mocap_root_rot, root_w)
        out_trans = (1.0 - root_w) * self.trans + root_w * mocap_trans
        return joint_aa, out_rot, out_trans


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

import os
from dpg_system.node import Node
from dpg_system.interface_nodes import SliderBankNode
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.body_defs import JointTranslator


# This system's other pose format: the twenty "active" joints derived from the
# Shadow suit, carried as quaternions.  Twenty is unambiguous -- no SMPL layout
# has twenty joints -- so the two can be told apart by shape alone and neither
# source needs a converter bolted on in front of the node.
ACTIVE_JOINT_COUNT = 20

# Active joint names, as aliases for naming free joints.  Most match SMPL, but
# the spine and collars are named differently in the two conventions.
_SMPL_NAME_TO_INDEX = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3, 'left_knee': 4,
    'right_knee': 5, 'spine2': 6, 'left_ankle': 7, 'right_ankle': 8,
    'spine3': 9, 'left_foot': 10, 'right_foot': 11, 'neck': 12,
    'left_collar': 13, 'right_collar': 14, 'head': 15, 'left_shoulder': 16,
    'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
}
ACTIVE_NAME_TO_SMPL_INDEX = {
    active: _SMPL_NAME_TO_INDEX[smpl]
    for smpl, active in JointTranslator.smpl_from_bmolab_active_joint_map.items()
    if smpl in _SMPL_NAME_TO_INDEX and active != 'empty'
}
# Active slot -> SMPL index, for per-joint scalars like weights.  (The pose
# translator is not used for these: it pads with an identity rotation, which
# has no meaning for a scalar and the wrong width.)
ACTIVE_INDEX_TO_SMPL_INDEX = {
    JointTranslator.bmolab_active_joints[active]: smpl_idx
    for active, smpl_idx in ACTIVE_NAME_TO_SMPL_INDEX.items()
    if active in JointTranslator.bmolab_active_joints
}


class SMPLRagdollNode(Node):
    """Drive an SMPL body from motion capture while a chosen set of joints
    falls under physics instead.

    Pose in, pose out, in whatever format arrived -- so this drops between a
    capture source and smpl_body, mgl_smpl_mesh or smpl_torque unchanged.
    """

    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]

    @staticmethod
    def factory(name, data, args=None):
        return SMPLRagdollNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # State that property callbacks may touch must exist before the
        # widgets do -- patch load fires those callbacks during restore.
        self.processor = None
        self.sim = None
        self.framerate = 60.0
        self.gender = 'neutral'
        self.betas = np.zeros(10)
        self.total_mass = 75.0
        self.free_indices = []
        self.root_free = False
        self._unsupported_time = 0.0
        self.weights = np.zeros(22)
        self.weight_targets = np.zeros(22)
        self.params = RagdollParams()

        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.config_input = self.add_input('config')
        self.weights_input = self.add_input('weights')

        self.pose_output = self.add_output('pose')
        self.smpl_pose_output = self.add_output('smpl_pose')
        self.trans_output = self.add_output('trans')
        self.weights_output = self.add_output('blend_weights')
        self.contact_force_output = self.add_output('contact_forces')
        self.energy_output = self.add_output('energy_removed')
        self.support_output = self.add_output('support')
        self.torque_output = self.add_output('free_torque_vectors')

        # Everything free by default: weights decide what physics gets, and
        # changing the free set resets the simulation, so it should not need
        # changing mid-performance.
        self.free_joints_prop = self.add_property(
            'free_joints', widget_type='text_input', default_value='all',
            callback=self._on_free_joints_changed)
        # Defaults to 1: the node arrives armed rather than already limp.
        # `catch` returns the weights to this value, so a default of 0 made the
        # catch button silently do nothing -- released and held were the same
        # state.  Set it to 0 for a limb that should just hang, in which case
        # release and catch have nothing to do and that is self-consistent.
        self.weight_prop = self.add_property(
            'blend_weight', widget_type='drag_float', default_value=1.0,
            callback=self._on_weight_changed)
        self.release_input = self.add_input(
            'release', widget_type='button', callback=self._release)
        self.catch_input = self.add_input(
            'catch', widget_type='button', callback=self._catch)
        self.ramp_prop = self.add_property(
            'ramp_ms', widget_type='drag_float', default_value=120.0)
        self.reset_input = self.add_input(
            'reset', widget_type='button', callback=self._reset_sim)

        self.gravity_prop = self.add_property(
            'gravity', widget_type='drag_float', default_value=1.0)
        self.transport_prop = self.add_property(
            'transport', widget_type='drag_float', default_value=1.0)
        self.damping_prop = self.add_property(
            'damping', widget_type='drag_float', default_value=1.5)
        self.drag_prop = self.add_property(
            'drag', widget_type='drag_float', default_value=0.15)
        self.contact_damp_boost_prop = self.add_property(
            'contact_damp_boost', widget_type='drag_float', default_value=0.0)
        self.passivity_prop = self.add_property(
            'passivity', widget_type='checkbox', default_value=True)
        self.auto_release_prop = self.add_property(
            'auto_release_unsupported', widget_type='checkbox', default_value=True)
        self.auto_release_delay_prop = self.add_option(
            'auto_release_delay', widget_type='drag_float', default_value=0.15)
        self.passivity_rate_prop = self.add_option(
            'passivity_rate', widget_type='drag_float', default_value=0.0)
        self.passivity_rate_contact_prop = self.add_option(
            'passivity_rate_contact', widget_type='drag_float', default_value=0.25)
        self.passivity_deadband_prop = self.add_option(
            'passivity_deadband', widget_type='drag_float', default_value=0.01)
        self.limit_stiffness_prop = self.add_property(
            'limit_stiffness', widget_type='drag_float', default_value=1.0)
        self.stop_softness_prop = self.add_option(
            'stop_softness', widget_type='drag_float', default_value=0.087)
        self.kp_prop = self.add_property('kp', widget_type='drag_float', default_value=120.0)
        self.kd_prop = self.add_property('kd', widget_type='drag_float', default_value=12.0)

        self.self_collision_prop = self.add_property(
            'self_collision', widget_type='checkbox', default_value=True)
        self.self_depth_prop = self.add_option(
            'self_depth', widget_type='drag_float', default_value=0.04)
        self.self_max_g_prop = self.add_option(
            'self_max_g', widget_type='drag_float', default_value=2.0)
        self.floor_enable_prop = self.add_property(
            'floor_enable', widget_type='checkbox', default_value=True)
        self.floor_height_prop = self.add_property(
            'floor_height', widget_type='drag_float', default_value=0.0)
        self.floor_auto_prop = self.add_property(
            'floor_auto', widget_type='checkbox', default_value=True)
        self.friction_prop = self.add_property(
            'friction', widget_type='drag_float', default_value=0.8)
        self.contact_depth_prop = self.add_option(
            'contact_depth', widget_type='drag_float', default_value=0.01)
        self.contact_damping_prop = self.add_option(
            'contact_damping', widget_type='drag_float', default_value=4.0)

        self.engine_prop = self.add_property(
            'engine', widget_type='combo', default_value='bullet',
            callback=self._on_engine_changed)
        self.engine_prop.widget.combo_items = ['bullet', 'native']
        self.motor_strength_prop = self.add_option(
            'motor_strength', widget_type='drag_float', default_value=1.0)
        self.motor_kp_prop = self.add_option('motor_kp', widget_type='drag_float', default_value=0.6)
        self.motor_kd_prop = self.add_option('motor_kd', widget_type='drag_float', default_value=0.3)
        self.partial_damping_prop = self.add_option('partial_damping', widget_type='drag_float', default_value=0.5)
        self.spring_rate_prop = self.add_option('spring_rate', widget_type='drag_float', default_value=60.0)
        self.gravity_comp_prop = self.add_option('gravity_comp', widget_type='drag_float', default_value=1.0)
        self.blend_soft_prop = self.add_option('blend_soft', widget_type='drag_float', default_value=180.0)
        self.blend_firm_prop = self.add_option('blend_firm', widget_type='drag_float', default_value=1.0)
        self.root_spring_prop = self.add_option(
            'root_spring', widget_type='drag_float', default_value=40.0)
        self.substeps_prop = self.add_option(
            'substeps', widget_type='drag_int', default_value=4)
        self.substep_rate_prop = self.add_option(
            'substep_rate', widget_type='drag_float', default_value=240.0)
        self.locked_stiffness_prop = self.add_option(
            'locked_stiffness', widget_type='drag_float', default_value=1.0)
        self.pivot_smoothing_prop = self.add_option(
            'pivot_smoothing', widget_type='drag_float', default_value=0.25)
        self.root_seed_smoothing_prop = self.add_option(
            'root_seed_smoothing', widget_type='drag_float', default_value=0.3)
        self.max_ang_vel_prop = self.add_option(
            'max_ang_vel', widget_type='drag_float', default_value=40.0)
        self.max_penetration_prop = self.add_option(
            'max_penetration', widget_type='drag_float', default_value=0.05)
        self.max_contact_g_prop = self.add_option(
            'max_contact_g', widget_type='drag_float', default_value=50.0)
        self.max_point_g_prop = self.add_option(
            'max_point_g', widget_type='drag_float', default_value=10.0)
        self.max_point_accel_prop = self.add_option(
            'max_point_accel', widget_type='drag_float', default_value=300.0)
        self.recovery_speed_prop = self.add_option(
            'recovery_speed', widget_type='drag_float', default_value=0.1)
        self.damping_speed_prop = self.add_option(
            'damping_speed', widget_type='drag_float', default_value=2.0)
        self.total_mass_prop = self.add_option(
            'total_mass', widget_type='drag_float', default_value=75.0)

        self.up_axis_prop = self.add_option('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']
        self.axis_perm_prop = self.add_option(
            'axis_permutation', widget_type='text_input', default_value='x, z, -y')
        self.quat_format_prop = self.add_option('quat_format', widget_type='combo', default_value='wxyz')
        self.quat_format_prop.widget.combo_items = ['xyzw', 'wxyz']

        self._parse_free_joints('all')

        # Take over the button names as a superset: with no arguments they do
        # what the button does.
        self.message_handlers['weight'] = self._weight_message
        self.message_handlers['release'] = self._release_message
        self.message_handlers['catch'] = self._catch_message

    # ------------------------------------------------------------------
    # Free set and weights
    # ------------------------------------------------------------------

    def _resolve_joints(self, tokens):
        """Joint indices for a list of names: groups ('left_arm'), SMPL joint
        names ('left_elbow'), active-convention names ('left_shoulder_blade')
        or indices.  Returns (indices, unknown tokens)."""
        indices = []
        unknown = []
        for token in tokens:
            tok = str(token).strip().lower()
            if not tok:
                continue
            if tok in JOINT_GROUPS:
                indices.extend(JOINT_GROUPS[tok])
            elif tok in self.joint_names:
                indices.append(self.joint_names.index(tok))
            elif tok in ACTIVE_NAME_TO_SMPL_INDEX:
                indices.append(ACTIVE_NAME_TO_SMPL_INDEX[tok])
            elif tok.isdigit():
                indices.append(int(tok))
            else:
                unknown.append(tok)
        return sorted({i for i in indices if 0 <= i < 22}), unknown

    def _parse_free_joints(self, text):
        """Accept group names ('left_arm'), joint names ('left_elbow') or
        indices, comma separated."""
        indices, unknown = self._resolve_joints(
            str(text).replace(';', ',').split(','))
        if unknown:
            print(f'smpl_ragdoll: unknown joint or group {unknown}; '
                  f'known groups: {sorted(JOINT_GROUPS)}')

        self.free_indices = sorted({i for i in indices if 0 <= i < 22})
        self.root_free = 0 in self.free_indices
        # Reported once per change, because whether the root is in the set is
        # not visible anywhere else and decides whether the body can fall at
        # all -- with it driven, the translation output passes the capture
        # straight through however limp the joints go.
        joints = [self.joint_names[i] for i in self.free_indices if i > 0]
        listed = ', '.join(joints) if len(joints) <= 6 else '%d joints' % len(joints)
        print('smpl_ragdoll: free = %s; root %s'
              % (listed or 'nothing',
                 'FREE (the body can fall)' if self.root_free
                 else 'DRIVEN (translation follows the capture)'))
        if self.sim is not None:
            self.sim.set_free_joints([j for j in self.free_indices if j > 0])
            self.sim.set_root_free(self.root_free)
        self._apply_weight_immediately()

    def _base_weight(self):
        try:
            return float(np.clip(self.weight_prop(), 0.0, 1.0))
        except Exception:
            return 1.0        # matches the property's default

    def _apply_weight_immediately(self):
        w = self._base_weight()
        self.weights[:] = 1.0
        self.weight_targets[:] = 1.0
        for j in self.free_indices:
            self.weights[j] = w
            self.weight_targets[j] = w

    def _on_free_joints_changed(self):
        try:
            self._parse_free_joints(self.free_joints_prop())
        except Exception as e:
            print(f'smpl_ragdoll: free_joints parse failed: {e}')

    def _on_weight_changed(self):
        self._apply_weight_immediately()

    # ------------------------------------------------------------------
    # Per-joint weights
    # ------------------------------------------------------------------
    #
    # The simulation has always carried a weight per joint; this is the
    # control surface for it.  Messages, sent to any input as a string or a
    # list:
    #
    #     weight <joints...> <value>     ramp the named joints to a weight
    #     release <joints...>            ramp them to 0   (no names: all)
    #     catch <joints...>              ramp them back to blend_weight
    #
    # with joints named as groups or in either joint-name convention, e.g.
    # "weight arms 0", "release left_leg", "catch base_of_skull".  The
    # `weights` input takes a whole array instead -- 22 in SMPL order (root
    # first), or 20 in the active order.  All of it goes through the ramp.
    #
    # A weight means something only for a joint in the free set, and adding a
    # joint to that set resets the simulation -- so for live work name
    # everything free up front (the default) and let the weights decide.  A
    # message naming a joint that is not free says so and leaves it alone.
    #
    # The `blend_weight` slider and a bare `catch` still act on the whole set.

    def _targets_for(self, tokens, what):
        """Free joints named by tokens, warning about the rest."""
        indices, unknown = self._resolve_joints(tokens)
        if unknown:
            print(f'smpl_ragdoll: {what}: unknown joint or group {unknown}')
        free = set(self.free_indices)
        not_free = [self.joint_names[i] for i in indices if i not in free]
        if not_free:
            print(f'smpl_ragdoll: {what}: {not_free} not in free_joints -- '
                  f'ignored (changing free_joints resets the simulation; '
                  f'set it to "all" and use weights instead)')
        return [i for i in indices if i in free]

    def _weight_message(self, message='', args=None):
        args = list(args or [])
        if len(args) < 2:
            print('smpl_ragdoll: usage: weight <joints...> <value>')
            return
        try:
            value = float(np.clip(float(args[-1]), 0.0, 1.0))
        except (TypeError, ValueError):
            print(f'smpl_ragdoll: weight: last argument must be a number, got {args[-1]!r}')
            return
        for j in self._targets_for(args[:-1], 'weight'):
            self.weight_targets[j] = value

    def _release_message(self, message='', args=None):
        args = list(args or [])
        if not args:
            self._release()
            return
        for j in self._targets_for(args, 'release'):
            self.weight_targets[j] = 0.0

    def _catch_message(self, message='', args=None):
        args = list(args or [])
        if not args:
            self._catch()
            return
        self._unsupported_time = 0.0
        w = self._base_weight()
        for j in self._targets_for(args, 'catch'):
            self.weight_targets[j] = w

    def _apply_weights_array(self, data):
        """A whole per-joint array of weight targets: 22 (SMPL, root first) or
        20 (active order)."""
        a = np.asarray(data, dtype=float).reshape(-1)
        if a.size == ACTIVE_JOINT_COUNT:
            # Joints with no active counterpart (the feet) stay driven.
            mapped = np.ones(22)
            for act_i, smpl_i in ACTIVE_INDEX_TO_SMPL_INDEX.items():
                if act_i < ACTIVE_JOINT_COUNT and smpl_i < 22:
                    mapped[smpl_i] = a[act_i]
            a = mapped
        elif a.size >= 22:
            a = a[:22]
        else:
            print(f'smpl_ragdoll: weights input needs 22 (SMPL) or 20 (active) '
                  f'values, got {a.size}')
            return
        a = np.clip(a, 0.0, 1.0)
        for j in self.free_indices:
            self.weight_targets[j] = a[j]

    def _release(self):
        """Ramp every free joint's weight to zero -- let go."""
        for j in self.free_indices:
            self.weight_targets[j] = 0.0

    def _catch(self):
        """Ramp back to the set blend weight -- take hold again."""
        # An explicit catch outranks the automatic release: the timer restarts
        # so the ramp can finish before support is judged again.
        self._unsupported_time = 0.0
        w = self._base_weight()
        if w <= 1e-6:
            # Never leave the button silently inert.
            print('smpl_ragdoll: catch returns the joints to blend_weight, '
                  'which is 0 -- raise it (1 = fully driven by the capture) '
                  'for catch to take hold.')
            return
        for j in self.free_indices:
            self.weight_targets[j] = w

    def _reset_sim(self):
        if self.sim is not None:
            self.sim.reset()
        self._unsupported_time = 0.0
        self._apply_weight_immediately()

    def _advance_weights(self, dt):
        # A driven root that the simulated body can no longer be holding up is
        # let go, in proportion to how much of its support has actually gone --
        # what a cartwheel with the arms released needs, since the capture
        # keeps the root travelling over hands that are no longer touching.
        #
        # Four conditions before it may act, each learned from a way this went
        # wrong:
        #
        #   something must already be released -- with every joint driven the
        #     simulated body *is* the captured body, so a shortfall is not lost
        #     support but a floor set wrong, and acting on it drops the body
        #     the instant the patch starts;
        #   no weight may be ramping up, and a catch restarts the timer, or
        #     taking hold again is undone on the very next frame and the root
        #     can never be recovered at all;
        #   the loss must persist, because contact force is present or absent
        #     with nothing in between, and a capture a few centimetres high
        #     reads as unsupported the moment it is not quite touching.
        if not bool(self.auto_release_prop()):
            self._unsupported_time = 0.0
        else:
            # released means meaningfully so: a joint at 0.999 is driven, and
            # counting it let a body at 0.999 be dropped for lack of support
            released = bool(np.any(self.weights[1:] < 0.5))
            # only a rising weight (a catch) blocks it: a release in progress
            # is exactly when the loss it is measuring begins
            ramping = bool(np.any(self.weight_targets - self.weights > 1e-6))
            if self.root_free and self.sim is not None and released and not ramping:
                support = float(getattr(self.sim, 'last_support', 1.0))
                if support < 0.5:
                    self._unsupported_time += dt
                else:
                    self._unsupported_time = 0.0
                if (self._unsupported_time >= float(self.auto_release_delay_prop())
                        and support < self.weight_targets[0]):
                    self.weight_targets[0] = support
            else:
                self._unsupported_time = 0.0

        ramp_s = max(float(self.ramp_prop()), 1e-3) / 1000.0
        rate = dt / ramp_s
        delta = self.weight_targets - self.weights
        step = np.clip(delta, -rate, rate)
        self.weights += step

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------

    def _to_array(self, d):
        return np.asarray(d, dtype=float) if not isinstance(d, np.ndarray) else d.astype(float)

    def _ensure_processor(self, rebuild=False):
        if self.processor is None or rebuild:
            self.processor = SMPLProcessor(
                framerate=self.framerate,
                betas=self.betas,
                gender=self.gender,
                total_mass_kg=float(self.total_mass_prop()),
                model_path=os.path.dirname(os.path.abspath(__file__)))
            self.sim = self._make_sim()
            self.sim.set_free_joints([j for j in self.free_indices if j > 0])
            self.sim.set_root_free(self.root_free)
            # Reported because nothing else shows it and the frame rate is not
            # guessable from the pose stream.  Every velocity in the simulation
            # is differenced against it, so a body captured at 120 Hz and run
            # as 60 is handed half its real speed and the physics degenerates
            # from there.  Connect `config` (smpl_beta_editor emits it, or
            # build it from the take) to set it.
            print('smpl_ragdoll: body = %s, %g Hz, betas %s'
                  % (self.gender, self.framerate,
                     'supplied' if np.any(self.betas) else 'ZERO (no config?)'))

    def _make_sim(self):
        """The physics core.  Bullet -- a real articulated solver -- when it is
        installed and selected; the native decoupled core otherwise."""
        want = 'bullet'
        try:
            want = str(self.engine_prop())
        except Exception:
            pass
        if want == 'bullet':
            try:
                from dpg_system.smpl_bullet import BulletRagdollSim, PYBULLET_AVAILABLE
                if PYBULLET_AVAILABLE:
                    print('smpl_ragdoll: engine = bullet')
                    return BulletRagdollSim(self.processor)
                print('smpl_ragdoll: pybullet not installed -- using the native core')
            except Exception as e:
                print(f'smpl_ragdoll: bullet engine unavailable ({e}) -- using the native core')
        else:
            print('smpl_ragdoll: engine = native')
        return SMPLRagdollSim(self.processor)

    def _on_engine_changed(self):
        # Rebuilt on the next frame with the chosen core.
        if self.sim is not None and hasattr(self.sim, 'close'):
            try:
                self.sim.close()
            except Exception:
                pass
        self.sim = None

    def _handle_config(self):
        if not self.config_input.fresh_input:
            return
        cfg = self.config_input()
        if not isinstance(cfg, dict):
            return
        changed = False
        for k in ('motioncapture_framerate', 'mocap_framerate', 'framerate'):
            if k in cfg:
                fr = float(cfg[k])
                if fr != self.framerate:
                    self.framerate = fr
                    changed = True
                break
        if 'gender' in cfg and str(cfg['gender']) != self.gender:
            self.gender = str(cfg['gender'])
            changed = True
        if 'betas' in cfg:
            b = self._to_array(cfg['betas'])
            if self.betas is None or not np.array_equal(self.betas, b):
                self.betas = b
                changed = True
        if changed:
            self._ensure_processor(rebuild=True)

    # ------------------------------------------------------------------
    # Pose shape handling
    # ------------------------------------------------------------------

    @staticmethod
    def _split_pose(orig):
        """Normalize any accepted pose layout to (F, n_joints, C).

        Returns (view, F, n_joints, C) or None if the layout is unrecognised.
        The returned array is a fresh copy -- the caller's buffer is not
        touched.
        """
        a = np.array(orig, dtype=float)
        if a.ndim == 1:
            sizes = {60: (20, 3), 80: (20, 4),          # active joints
                     66: (22, 3), 72: (24, 3), 88: (22, 4),
                     96: (24, 4), 156: (52, 3), 208: (52, 4)}
            if a.size not in sizes:
                return None
            n, c = sizes[a.size]
            return a.reshape(1, n, c), 1, n, c
        if a.ndim == 2:
            if a.shape[1] in (3, 4) and (a.shape[0] >= 22
                                         or a.shape[0] == ACTIVE_JOINT_COUNT):
                return a.reshape(1, a.shape[0], a.shape[1]), 1, a.shape[0], a.shape[1]
            return None
        if a.ndim == 3 and a.shape[2] in (3, 4):
            return a, a.shape[0], a.shape[1], a.shape[2]
        return None

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self):
        self._handle_config()
        self._ensure_processor()

        if not self.pose_input.fresh_input:
            return

        raw = self.pose_input()
        split = self._split_pose(raw)
        trans = self.trans_input()
        trans_raw = trans
        trans = np.zeros(3) if trans is None else self._to_array(trans)
        trans = np.asarray(trans, dtype=float).reshape(-1)
        root_trans = np.zeros(3)
        root_trans[:min(3, trans.size)] = trans[:3]

        if split is None:
            print('smpl_ragdoll: unrecognised pose layout, passing through')
            self.pose_output.send(raw)
            self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
            return
        out_pose, F, n_joints, C = split

        # An active-joint stream is lifted into SMPL here and put back on the
        # way out, so the simulation only ever deals in one layout and the
        # patch does not need converters wrapped around the node.
        is_active = (n_joints == ACTIVE_JOINT_COUNT)
        if is_active:
            work = np.zeros((F, 24, C))
            if C == 4:
                work[:, :, 0 if self.quat_format_prop() == 'wxyz' else 3] = 1.0
            for f in range(F):
                work[f, :22] = JointTranslator.translate_from_bmolab_active_to_smpl(
                    out_pose[f])
            work_joints = 24
        else:
            work = out_pose
            work_joints = n_joints

        if self.weights_input.fresh_input:
            self._apply_weights_array(self.weights_input())

        dt = 1.0 / max(self.framerate, 1.0)
        self._advance_weights(dt)

        # Nothing configured as free: the node is inert, pass through.
        #
        # Note this deliberately does NOT skip out when everything is merely
        # prescribed (weight 1) and waiting for a release.  While prescribed,
        # the simulation shadows the capture and accumulates the velocity and
        # spin that a release inherits; skipping those frames would make the
        # body let go from rest, with no momentum at all.
        if not self.free_indices:
            self.weights_output.send(self.weights.copy())
            self.pose_output.send(raw)
            # Still emitted, so an inert node is a working format converter
            # rather than a dead end in the chain.
            self.smpl_pose_output.send(self._as_smpl_axis_angle(work, F, C))
            self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
            return

        options = SMPLProcessingOptions(
            input_type='quat' if C == 4 else 'axis_angle',
            input_up_axis=self.up_axis_prop(),
            axis_permutation=self.axis_perm_prop(),
            quat_format=self.quat_format_prop(),
            dt=dt)

        p = self.params
        p.dt = dt
        p.engine = str(self.engine_prop())
        p.ramp_s = max(float(self.ramp_prop()), 1e-3) / 1000.0
        p.motor_strength = float(self.motor_strength_prop())
        p.motor_kp = float(self.motor_kp_prop())
        p.motor_kd = float(self.motor_kd_prop())
        p.partial_damping = float(self.partial_damping_prop())
        p.spring_rate = float(self.spring_rate_prop())
        p.gravity_comp = float(self.gravity_comp_prop())
        p.blend_soft = float(self.blend_soft_prop())
        p.blend_firm = float(self.blend_firm_prop())
        p.root_spring = float(self.root_spring_prop())
        p.substeps = max(1, int(self.substeps_prop()))
        p.substep_rate = float(self.substep_rate_prop())
        p.gravity = float(self.gravity_prop())
        p.transport = float(self.transport_prop())
        p.damping = float(self.damping_prop())
        p.drag = float(self.drag_prop())
        p.contact_damp_boost = float(self.contact_damp_boost_prop())
        p.passivity = bool(self.passivity_prop())
        p.passivity_rate = float(self.passivity_rate_prop())
        p.passivity_rate_contact = float(self.passivity_rate_contact_prop())
        p.passivity_deadband = float(self.passivity_deadband_prop())
        p.limit_stiffness = float(self.limit_stiffness_prop())
        p.stop_softness = float(self.stop_softness_prop())
        p.locked_stiffness = float(self.locked_stiffness_prop())
        p.kp = float(self.kp_prop())
        p.kd = float(self.kd_prop())
        p.pivot_smoothing = float(self.pivot_smoothing_prop())
        p.root_seed_smoothing = float(self.root_seed_smoothing_prop())
        p.self_collision = bool(self.self_collision_prop())
        p.self_depth = float(self.self_depth_prop())
        p.self_max_g = float(self.self_max_g_prop())
        p.floor_enable = bool(self.floor_enable_prop())
        p.floor_height = float(self.floor_height_prop())
        p.floor_auto = bool(self.floor_auto_prop())
        p.friction = float(self.friction_prop())
        p.contact_depth = float(self.contact_depth_prop())
        p.contact_damping = float(self.contact_damping_prop())
        p.max_penetration = float(self.max_penetration_prop())
        p.max_contact_g = float(self.max_contact_g_prop())
        p.max_point_g = float(self.max_point_g_prop())
        p.max_point_accel = float(self.max_point_accel_prop())
        p.recovery_speed = float(self.recovery_speed_prop())
        p.damping_speed = float(self.damping_speed_prop())
        p.max_ang_vel = float(self.max_ang_vel_prop())

        proc = self.processor
        out_trans = root_trans
        for f in range(F):
            # Crop to the SMPL body joints. An AMASS / SMPL-H stream carries
            # 52, and the processor's quaternion reshape assumes 24.
            frame = work[f:f + 1, :24].copy()
            try:
                t_int, aa_int, _quats = proc._prepare_trans_and_pose(
                    frame, root_trans.reshape(1, 3), options)
                result, root_rot, trans_int = self.sim.advance(
                    aa_int[0], t_int[0], self.weights, p)
            except Exception as e:
                print(f'smpl_ragdoll: simulation failed ({e}); passing through')
                self.pose_output.send(raw)
                self.trans_output.send(root_trans)
                return

            # A non-root joint's local rotation is the same in the incoming
            # frame and the internal one -- the axis permutation and the
            # up-axis conversion touch only the root -- so it can be written
            # straight back.  The root cannot, and is converted below.
            for j, aa in result.items():
                self._write_joint(work, f, j, work_joints, C, aa)

            if self.root_free:
                in_rot, out_trans = self._to_input_frame(root_rot, trans_int)
                self._write_joint(work, f, 0, work_joints, C, in_rot.as_rotvec())

        if is_active:
            for f in range(F):
                back = JointTranslator.translate_from_smpl_to_bmolab_active(work[f])
                out_pose[f] = back[:ACTIVE_JOINT_COUNT]
        shaped = out_pose.reshape(np.shape(raw)) if np.ndim(raw) != 3 else out_pose
        self.pose_output.send(shaped)
        self.smpl_pose_output.send(self._as_smpl_axis_angle(work, F, C))
        self.trans_output.send(out_trans)
        self.weights_output.send(self.weights.copy())
        self.torque_output.send(self.sim.last_torque.copy())
        self.contact_force_output.send(self.sim.last_contact_force.copy())
        self.energy_output.send(self.sim.last_energy_injected)
        self.support_output.send(self.sim.last_support)

    def _as_smpl_axis_angle(self, work, F, C):
        """The pose as SMPL axis-angle, whatever came in.

        The `pose` output matches the incoming layout so the node drops into an
        existing chain unchanged; this one is always SMPL, which is what
        mgl_smpl_mesh, smpl_body and smpl_torque want.  Both are emitted every
        frame, so neither consumer needs a converter and neither has to be
        chosen in advance.
        """
        out = np.zeros((F, 24, 3))
        # The working array may carry 22 joints (a 66-float SMPL stream), 24,
        # or 52 (SMPL-H); pad or crop to the 24 the SMPL body expects.
        n = min(24, work.shape[1])
        for f in range(F):
            if C == 3:
                out[f, :n] = work[f, :n, :3]
            else:
                q = np.array(work[f, :n, :4], dtype=float)
                if self.quat_format_prop() == 'wxyz':
                    q = np.roll(q, -1, axis=-1)          # -> xyzw for scipy
                norm = np.linalg.norm(q, axis=-1, keepdims=True)
                q = np.where(norm > 1e-9, q / np.maximum(norm, 1e-9),
                             np.array([0.0, 0.0, 0.0, 1.0]))
                out[f, :n] = R.from_quat(q).as_rotvec()
        return out[0] if F == 1 else out

    def _write_joint(self, out_pose, f, j, n_joints, C, aa):
        """Write one local rotation back in the layout that arrived."""
        if j >= n_joints:
            return
        if C == 3:
            out_pose[f, j] = aa
        else:
            q = R.from_rotvec(aa).as_quat()              # xyzw
            if self.quat_format_prop() == 'wxyz':
                q = np.roll(q, 1)
            out_pose[f, j] = q

    def _to_input_frame(self, root_rot, trans):
        """Invert the processor's frame conversion for the root.

        Mirrors _prepare_trans_and_pose, which applies the axis permutation and
        then, for Z-up input, a further -90 degree rotation about X.  Only the
        root's rotation and the translation are affected, which is why every
        other joint needs no inverse.
        """
        proc = self.processor
        t = np.asarray(trans, dtype=float).reshape(3)
        rot = root_rot

        if self.up_axis_prop() == 'Z':
            # Forward was  internal = (x, z, -y)  on the permuted translation.
            t = np.array([t[0], -t[2], t[1]])
            rot = R.from_euler('x', -90, degrees=True).inv() * rot

        basis = getattr(proc, 'perm_basis', None)
        if basis is not None:
            # Forward was  permuted = t @ basis.T, and a signed permutation is
            # orthogonal, so the inverse is a right-multiply by the basis.
            t = t @ basis
            rot = R.from_matrix(proc.perm_basis_rot).inv() * rot
        return rot, t


class RagdollBlendUINode(SliderBankNode):
    """ragdoll_blend_ui: one slider per body region, each sending
    'weight <region> <value>' -- connect its output to any input of an
    smpl_ragdoll.  Names and the message template are the slider_bank
    options, so a slider can be renamed to a single joint ('left_elbow') or
    the template changed to drive something else."""
    default_names = ['root', 'spine', 'head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    default_template = 'weight {name} {value}'
    default_min = 0.0
    default_max = 1.0
    default_value = 1.0

    @staticmethod
    def factory(name, data, args=None):
        return RagdollBlendUINode(name, data, args)
