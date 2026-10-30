"""Seed the help pages for the Shadow magnetization learning nodes (2026-10-26).

    python3 dpg_system/help/tools/seed_shadow_mag_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

shadow_mag_correct, elbow_hinge, elbow_turn, hang_symmetry. The elbow pages read the synthetic take
help/data/elbow_hinge_example.npz (make_elbow_hinge_example.py writes it). The full
learning setup on a real take is examples/shadow_mag_learning.json. Not a make_*
script: rebuild.py never runs it, and it will not write over an existing page
without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, C, N, see_also
from seed_torch_batch4_help import held, W


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


# 20 degrees of roll on the left upper arm at rest: the first feature (world X seen in
# the upper arm's frame) is 1 when the arm is at rest, so its weight acts as the angle.
ROLL = '0.35'

page('shadow_mag_correct', 'shadow_mag_correct - a learnable magnetization fix for the arms',
"""
Corrects the Shadow arm sensors. First, if it is given one, the take's T-pose turn
(elbow_turn measures it): how far each elbow was turned when the suit was calibrated.
Then four corrections per arm, each an angle worked out afresh every frame:

  roll   the upper arm turned about its own bone
  yaw    the upper arm turned about vertical
  fyaw   the forearm turned about vertical
  hang   the whole arm turned sideways about the chest's forward axis, coming in as
         the arm drops from 25 to 55 degrees below level (judged where
         the correction puts the arm, so the error can't hide an arm)

Each angle is a weighted sum of numbers that say how the sensor sits. Six are the
world's two horizontal axes, seen from inside the sensor: the shape a magnetization
(hard-iron) error takes, which unlike a heading stays defined when the arm hangs
straight down. The roll also uses two about the body: how high the arm is, and how
far the elbow is bent past 62 degrees (the middle of elbow_turn's band, where the turn
sets the roll) - for a strap, cable or muscle that pulls the sensor round as the arm
moves. The hang is one number per arm: lowered arms that sit to one side together
(a good T-pose, then an asymmetric hang) is a common Shadow error. The corrected upper
arm and forearm are written back into the shoulder, elbow and wrist; the hand keeps
where it was in the world, except under 'hang', which carries it with the arm.

The 42 weights are meant to be learned: t.param holds them and t.optimize tunes them
to keep the elbows on their hinges (elbow_hinge) and the lowered arms centred
(hang_symmetry). Give it one pose and it corrects
that pose; give it a whole take and it corrects every frame at once, keeping the
trail for t.optimize.

SYNTAX:
shadow_mag_correct

INPUTS and PARAMETERS:

pose in:
Shadow quaternions, scalar first: one pose (37 or 20 joints, 4 each) or a take
(frames, joints, 4). Numpy in gives numpy out; a tensor stays a tensor.

weights:
42 numbers, in radians, 21 per arm, left arm then right: roll (6 field numbers, then
arm height and elbow bend), yaw (6), fyaw (6), hang (1; positive moves a hanging arm
toward the dancer's left). A shorter list fills the rest with 0; the older 40- and
36-number layouts are still accepted. New weights
re-correct the pose already held, so a paused body follows the learning.

turn:
Two numbers in degrees, left then right: the take's T-pose turn, from elbow_turn.
It is undone before the learned corrections. Also re-corrects the held pose.

fixed bank (float box, also an inlet):
Degrees, as hang_symmetry's shift reads it (+ toward the dancer's left). Used only when
'hang from' is 'fixed bank': both arms then undo this bank instead of taking the hang
from the weights. Also re-corrects the held pose.

correct (checkbox, also an inlet):
Off passes the pose through untouched - for comparing.

amount (float box, also an inlet):
Scales the correction, turn included: 0 none, 1 all of it.

roll, yaw, fyaw, hang:
Options, under the node: switch each correction on or off.

turn fade:
An option, in degrees of elbow bend (0 by default). With the arm straight nothing
shows the turn; above 0 the turn comes in smoothly from none with the arm straight
to all of it at this bend. 0 undoes all of it everywhere.

hang from:
An option: 'weights' (the default) learns the hang with the rest; 'fixed bank' takes it
from the 'fixed bank' inlet instead. Use 'fixed bank' for takes recorded with Shadow's
fusion before 2025-12-18, where a session's bank is steady: measure it once over all of
the session's takes (np.robust_mean of hang_symmetry's shift per take, or of the banks of
every arms-down frame) and apply that one number to every take of the session. The other
corrections can still be learned alongside; the hang weights are then ignored.

OUTPUTS:

pose out:
The corrected pose or take, in the layout it came in.

READING THE DEMO:
At rest every sensor is level, so the first feature is 1, and 0.35 - the first of the
42 weights, the rest 0 - turns the left upper arm 0.35 radians (20 degrees) about its
bone. Click the button: the left_shoulder row of the angles shows the roll, and the
left_elbow row the opposite turn, so the forearm stays where it was in the world.

RELATED:
elbow_hinge and hang_symmetry measure the errors the weights are learned against;
t.param and t.optimize learn them. elbow_turn measures the turn. shadow_arm_correct is the hand-dialled arm fix; mag_yaw_correct
sets constant yaw offsets. The full learning setup on a real take is
examples/shadow_mag_learning.json.
""",
[
    C('c1', 'click to send the pose', 28, 53),
    N('bt', 'button', 28, 92, 45, 42),
    C('c2', 'every joint at rest', 28, 160),
    N('ap', 'active_pose', 28, 195, 321, 350),
    C('c3', '20 degrees of roll, left upper arm', 400, 53),
    N('mw', 'message', 400, 92, 100, 42, MSG(ROLL)),
    N('lb', 'load_bang', 400, 150, 88, 46),
    C('c4', 'corrected', 400, 230),
    N('mc', 'shadow_mag_correct', 400, 265, 200, 160),
    C('c5', 'each joint as angles', 640, 230),
    N('qe', 'quaternion_to_euler', 640, 265, 125, 42, {'degrees': True}),
    N('vc', 'vector', 640, 330, 200, 126),
],
[('lb', 'out', 'mw', ''), ('mw', 'message out', 'mc', 'weights'), ('bt', '', 'ap', 'in'),
 ('ap', 'out', 'mc', 'pose in'), ('mc', 'pose out', 'qe', 'quaternion'), ('qe', 'euler angles', 'vc', 'in')],
['elbow_turn', 'elbow_hinge', 'hang_symmetry', 't.param', 't.optimize', 'shadow_arm_correct', 'mag_yaw_correct'], 600, 1400)


EXAMPLE = 'dpg_system/help/data/elbow_hinge_example.npz'

page('elbow_hinge', 'elbow_hinge - how far the forearms leave their elbow hinges',
"""
Measures an impossibility in a take of Shadow data. The elbow is a hinge: seen from
the upper arm, the forearm can only point somewhere around one fixed circle - bending
moves it around the circle, and forearm twist does not move it at all. However the
dancer moves, a real arm stays on that circle. A frame where the forearm points off
it cannot have happened, so how far off it is measures an error in the sensors, not
how natural the pose looks.

By default the circle is the one a real elbow bends on: forward, as SMPL's does from
its rest pose. That pins the upper arm's roll at every bend. Or the circle can be
fitted to the take, assuming nothing about the dancer's elbow; then a constant twist
of the upper arm (which turns the whole circle) cannot be seen, and a correction
learned against it can drift in roll where nothing shows it. Either way elbow_hinge
wants a take (many frames), not one pose. The loss is ready for t.optimize, to learn
a correction such as shadow_mag_correct's.

SYNTAX:
elbow_hinge

INPUTS and PARAMETERS:

take in:
Shadow quaternions, (frames, 37 or 20 joints, 4), scalar first. A tensor carrying
gradients keeps them.

robust width:
An option, in degrees (4 by default). Misses smaller than this count by their
square, larger ones only in proportion, so a few wild frames cannot dominate.

arms:
An option: both (the default), left or right - which arms make the loss.

circle:
An option: forward (the default) or fitted - see above.

OUTPUTS:

loss:
One number for t.optimize; smaller is better.

off hinge:
For each frame, how far each forearm is off its circle, in degrees: (frames, 2),
left then right.

rms:
The typical miss over the take, in degrees, left then right.

READING THE DEMO:
A made-up take, 600 frames of both elbows bending to 100 degrees and back. In the
clean version nothing else happens, and both forearms stay exactly on their circles:
0. In the other, each upper-arm sensor misreads its roll by up to 6 degrees as time
goes by, and the forearms are pushed off by about 3 degrees.

RELATED:
shadow_mag_correct is the correction it is meant to judge; t.optimize learns it.
elbow_turn measures the T-pose turn as a single number for the take. The full learning setup on a
real take is examples/shadow_mag_learning.json.
""",
held(53) + [
    C('c1', 'a made-up take', 180, 53),
    N('nl', 'np.load', 180, 92, 170, 110, {'path': EXAMPLE}),
    C('c2', 'clean, then misread', 180, 230),
    N('de', 'dict_extract clean quats', 180, 265, 190, 90),
    C('c3', 'off the hinge, left right', 420, 53),
    N('h1', 'elbow_hinge', 420, 92, 170, 110),
    N('l1', 'list', 640, 92, 220, 42, LIST),
    N('h2', 'elbow_hinge', 420, 265, 170, 110),
    N('l2', 'list', 640, 265, 220, 42, LIST),
],
[('lb', 'out', 'nl', 'send'), ('nl', 'dict out', 'de', 'dict in'), ('de', 'clean', 'h1', 'take in'),
 ('de', 'quats', 'h2', 'take in'), ('h1', 'rms', 'l1', ''), ('h2', 'rms', 'l2', '')],
['elbow_turn', 'shadow_mag_correct', 't.optimize', 't.param'], 430, 1000)


page('elbow_turn', 'elbow_turn - how far each elbow was turned in the calibration T-pose',
"""
Measures a constant error in a take of Shadow data that comes from calibration. In
a T-pose the shoulder and wrist can hold still while the elbow turns a long way -
further still for a trained, mobile dancer - and people settle into it differently,
calibration to calibration. The upper-arm sensor takes whatever turn the elbow had
as its zero, so for the whole take every elbow bend shows up turned by it, and the
body renders with the twist piled into the shoulder.

A real elbow bends forward (as SMPL's does, from its palms-down rest). So the
direction a bent forearm points around the upper arm gives the turn. Only bent
frames say anything, so elbow_turn needs a take, not one pose. Feed it to
shadow_mag_correct's 'turn' inlet; its 'turn fade' option decides how much of it
reaches a straight arm, where nothing shows it. Measure it for each take - it changes
whenever the suit is calibrated.

SYNTAX:
elbow_turn

INPUTS and PARAMETERS:

take in:
Shadow quaternions, (frames, 37 or 20 joints, 4), scalar first.

bend from, bend to:
Options, in degrees (45 and 80): only frames with the elbow bent this much count.
Moderate bends keep out errors that grow as the elbow closes.

arm low only:
An option (on): only frames with the upper arm more than 30 degrees below level,
keeping out errors that grow as the arm rises. With fewer than 100 frames to go
on, it drops this, then takes any bend past 'bend from' ('frames used' tells).

OUTPUTS:

turn:
Degrees, left then right. Positive: the inside of the elbow was turned up toward
the ceiling.

spread:
How much the frames used disagree, in degrees: small for a clean take.

frames used:
How many frames each arm's turn came from.

READING THE DEMO:
A made-up take of both elbows bending: clean, then as it reads after a T-pose held
with the elbows turned 30 degrees (left) and 15 (right).

RELATED:
shadow_mag_correct undoes the turn; elbow_hinge measures the error that changes
frame to frame. The full setup on a real take is examples/shadow_mag_learning.json.
""",
held(53) + [
    C('c1', 'a made-up take', 180, 53),
    N('nl', 'np.load', 180, 92, 170, 110, {'path': EXAMPLE}),
    C('c2', 'clean, then turned', 180, 230),
    N('de', 'dict_extract clean turned', 180, 265, 190, 90),
    C('c3', 'T-pose turn, left right', 420, 53),
    N('t1', 'elbow_turn', 420, 92, 170, 110),
    N('l1', 'list', 640, 92, 220, 42, LIST),
    N('t2', 'elbow_turn', 420, 265, 170, 110),
    N('l2', 'list', 640, 265, 220, 42, LIST),
],
[('lb', 'out', 'nl', 'send'), ('nl', 'dict out', 'de', 'dict in'), ('de', 'clean', 't1', 'take in'),
 ('de', 'turned', 't2', 'take in'), ('t1', 'turn', 'l1', ''), ('t2', 'turn', 'l2', '')],
['shadow_mag_correct', 'elbow_hinge'], 430, 1060)


page('hang_symmetry', 'hang_symmetry - lowered arms that sit to one side together',
"""
Measures a common error in Shadow takes: a good T-pose, but once the arms come down
they hang to one side - both arms, together. A body hangs its arms as mirror images,
so however far each one splays out, the pair is centred on the chest. hang_symmetry
measures how far the pair has shifted to one side, the same way for both: not how
vertical the arms are, and not each arm's splay.

It reads the frames where both upper arms are well below level and at about the same
height - all of them, or only those in the ranges you give. Its loss is ready for
t.optimize, to learn shadow_mag_correct's 'hang' weights; add it to elbow_hinge's
with + when learning both.

SYNTAX:
hang_symmetry

INPUTS and PARAMETERS:

take in:
Shadow quaternions, (frames, 37 or 20 joints, 4), scalar first. A tensor carrying
gradients keeps them.

ranges:
An option: frame ranges to use, like 10430-11242, 3767-4000 - stretches where the
dancer's arms hang symmetrically (from the video). Empty uses every frame where both
arms hang.

every:
An option: if the take was thinned (t[] ::6), the step, so the ranges still name
frames of the full take.

below level:
An option, in degrees (50): how far below level both upper arms must hang.

robust width:
An option, in degrees (4). Misses smaller than this count by their square, larger
ones only in proportion - frames where the choreography itself is asymmetric pull
little.

OUTPUTS:

loss:
One number for t.optimize; smaller is better.

shift:
The typical shared shift, in degrees: positive toward the dancer's left.

frames used:
How many frames it read.

READING THE DEMO:
A made-up take of both arms hanging, swaying in and out together: as mirror images
(0), then with both shifted 15 degrees toward the dancer's right (-15).

RELATED:
shadow_mag_correct's 'hang' term is the correction; elbow_hinge is the loss for the
elbows. The full setup on a real take is examples/shadow_mag_learning.json.
""",
held(53) + [
    C('c1', 'a made-up take', 180, 53),
    N('nl', 'np.load', 180, 92, 170, 110, {'path': EXAMPLE}),
    C('c2', 'mirrored, then shifted', 180, 230),
    N('de', 'dict_extract hanging shifted', 180, 265, 210, 90),
    C('c3', 'shared shift, degrees', 440, 53),
    N('h1', 'hang_symmetry', 440, 92, 170, 110),
    N('l1', 'list', 660, 92, 160, 42, LIST),
    N('h2', 'hang_symmetry', 440, 265, 170, 110),
    N('l2', 'list', 660, 265, 160, 42, LIST),
],
[('lb', 'out', 'nl', 'send'), ('nl', 'dict out', 'de', 'dict in'), ('de', 'hanging', 'h1', 'take in'),
 ('de', 'shifted', 'h2', 'take in'), ('h1', 'shift', 'l1', ''), ('h2', 'shift', 'l2', '')],
['shadow_mag_correct', 'elbow_hinge', 't.optimize'], 430, 1150)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height'], subpatch=p['subpatch']))


if __name__ == '__main__':
    main()
