"""Learnable magnetization correction for Shadow arm sensors, and the elbow-hinge
loss that drives it. Built for t.param + t.optimize: everything stays in torch,
so a loss worked out from a corrected take traces back to the weights.

Correction model (per arm, session-shared). Each correction angle is a weighted sum
of features worked out from the uncorrected reading:
  roll  : upper arm turned about its own bone axis - 6 heading features of the upper-arm
          sensor, plus 2 body features: arm height (the upper arm's up component in the
          chest frame) and the cosine of the elbow bend
  yaw   : upper arm turned about world vertical - 6 heading features of the upper arm
  fyaw  : forearm turned about world vertical - 6 heading features of the forearm
  hang  : the whole arm (upper arm, forearm, hand) turned sideways about the chest's
          forward axis - 1 weight, faded in as the upper arm drops from 25 to 55 degrees
          below level. Subject 7: with both arms lowered they sit 16-20 degrees toward
          the dancer's right together, in every take and every facing; none at T-pose
          height. hang_symmetry is the loss that sees it.
The heading features are the world X and Z axes seen in that sensor's frame: the
first-harmonic hard-iron shape of magnetization, with no heading singularity when the
arm hangs. The body features carry error tied to the pose rather than to the room - a
strap or cable pulling the sensor round, or muscle sliding under it. Subject 7 showed a
right upper-arm roll error growing with arm height and elbow bend, the same whichever
way the dancer faced. Body features on all three terms (or the full arm direction)
overfit a take; these two on the roll alone held up on a held-out take.
Weights: 21 per arm (roll 8, yaw 6, fyaw 6, hang 1), left then right: 42.
Before them, an optional per-take 'turn' (elbow_turn measures it): how far each
elbow was turned in that take's calibration T-pose. It is constant for a take and
changes at each calibration, so it is measured, not learned - the hinge loss
cannot see a constant roll.
The corrected upper-arm and forearm world orientations are written back into the
shoulder, elbow and wrist local quaternions; the hand keeps its world orientation
(except under 'hang', which carries the hand with the arm).
"""
import math
from pathlib import Path

import numpy as np
import torch

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.quaternion_nodes import quaternion_multiply_wxyz_t, rotate_vector_wxyz_t
from dpg_system.torch_base_nodes import run_with_cpu_fallback


def register_shadow_mag_nodes():
    Node.app.register_node('shadow_mag_correct', ShadowMagCorrectNode.factory)
    Node.app.register_node('elbow_hinge', ElbowHingeNode.factory)
    Node.app.register_node('elbow_turn', ElbowTurnNode.factory)
    Node.app.register_node('hang_symmetry', HangSymmetryNode.factory)
    Node.app.register_node('shadow_heading_adjust', ShadowHeadingAdjustNode.factory)
    Node.app.register_node('pose_symmetry', PoseSymmetryNode.factory)
    Node.app.register_node('body_contact', BodyContactNode.factory)


# Shadow (37) joint indices of the arm chains
ARMS = {'L': (5, 9, 10), 'R': (19, 23, 24)}           # shoulder (upper arm), elbow (forearm), wrist (hand)
# active (20) index -> shadow (37) index, for the joints the arm math reads or writes
ACT2SHA = {1: 17, 2: 1, 3: 32, 4: 31, 5: 4, 12: 13, 13: 5, 14: 9, 15: 10, 16: 27, 17: 19, 18: 23, 19: 24}
TERMS = ['roll', 'yaw', 'fyaw', 'hang']
CHEST = 1                                             # parent of both shoulder blades
N_HEADING, N_BODY = 6, 2
BEND_REF = 62.5                                       # middle of elbow_turn's 45-80 bend band, degrees
N_ROLL = N_HEADING + N_BODY                            # roll: heading + arm height + elbow bend
N_HANG = 1
PER_ARM = N_ROLL + 2 * N_HEADING + N_HANG               # 21
N_WEIGHTS = 2 * PER_ARM                                 # 42
OLD_WEIGHTS = 2 * 3 * N_HEADING                         # 36: the heading-only layout
OLD_WEIGHTS_40 = 2 * (N_ROLL + 2 * N_HEADING)           # 40: before 'hang'
HANG_FROM, HANG_TO = 25.0, 55.0                         # degrees below level: the hang fades in

_skeleton = None


def skeleton():
    """Parent index, FK order and bone axis per joint, from definition.xml (read once)."""
    global _skeleton
    if _skeleton is None:
        from dpg_system.smpl_utilities.diag_magnetometer_deviation import load_skeleton
        parent, order, bax = load_skeleton(Path(__file__).parent / 'definition.xml')
        _skeleton = (parent, order, torch.tensor(bax))
    return _skeleton


def conj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def axis_angle_q(axis, angle):
    return torch.cat([torch.cos(angle / 2)[..., None], torch.sin(angle / 2)[..., None] * axis], dim=-1)


def as_pose_tensor(data):
    """A pose or a take as a float tensor (..., joints, 4), the 37-joint layout, plus
    how to give it back: (was numpy, was 20-joint, original dtype)."""
    was_numpy = not isinstance(data, torch.Tensor)
    q = torch.as_tensor(np.asarray(data, dtype=np.float64)) if was_numpy else data
    if not q.is_floating_point():
        q = q.double()
    if q.shape[-1] != 4 or q.ndim < 2 or q.shape[-2] not in (37, 20):
        return None, None
    active = q.shape[-2] == 20
    if active:
        q37 = q.new_zeros(q.shape[:-2] + (37, 4))
        q37[..., 0] = 1.0
        idx = list(ACT2SHA.items())
        q37 = q37.index_copy(-2, torch.tensor([s for a, s in idx], device=q.device),
                             q.index_select(-2, torch.tensor([a for a, s in idx], device=q.device)))
        q = q37
    return q, (was_numpy, active, data)


def give_back(q37, how):
    was_numpy, active, original = how
    if active:
        idx = list(ACT2SHA.items())
        src = original if isinstance(original, torch.Tensor) else torch.as_tensor(np.asarray(original, dtype=np.float64))
        q = src.to(q37.dtype).index_copy(-2, torch.tensor([a for a, s in idx], device=q37.device),
                                         q37.index_select(-2, torch.tensor([s for a, s in idx], device=q37.device)))
    else:
        q = q37
    if was_numpy:
        return q.detach().cpu().numpy()
    return q


def world_orientations(q, joints):
    """World orientation of each joint in joints, from local quats (..., 37, 4)."""
    parent, order, _ = skeleton()
    needed = set()
    for j in joints:
        while j >= 0 and j not in needed:
            needed.add(j)
            j = parent[j]
    G = {}
    for j in order:
        if j in needed:
            p = parent[j]
            G[j] = q[..., j, :] if p < 0 else quaternion_multiply_wxyz_t(G[p], q[..., j, :])
    return G


def features(G):
    """World X and Z axes seen in a sensor's frame: (..., 6)."""
    x = G.new_tensor([1.0, 0.0, 0.0])
    z = G.new_tensor([0.0, 0.0, 1.0])
    return torch.cat([rotate_vector_wxyz_t(conj(G), x), rotate_vector_wxyz_t(conj(G), z)], dim=-1)


def body_features(G, sh, el):
    """Arm height (the upper arm's up component in the chest frame) and how far the elbow is
    bent past the middle of elbow_turn's band: (..., 2). The hinge cannot see a constant roll,
    so the turn measured in that band sets it, and the bend feature is 0 up to there - a weight
    learned on deep bends cannot reach the straight arm, where the hinge sees no roll at all."""
    _, _, bax = skeleton()
    bax = bax.to(device=G[sh].device, dtype=G[sh].dtype)
    ua = rotate_vector_wxyz_t(G[sh], bax[sh])
    fa = rotate_vector_wxyz_t(G[el], bax[el])
    height = rotate_vector_wxyz_t(conj(G[CHEST]), ua)[..., 1:2]
    past = math.cos(math.radians(BEND_REF)) - (ua * fa).sum(-1, keepdim=True)
    deep = 0.1 * torch.nn.functional.softplus(past / 0.1)     # a smooth max(past, 0)
    return torch.cat([height, deep], dim=-1)


def hang_valve(G, sh, upper=None):
    """0 with the upper arm above HANG_FROM degrees below level, 1 past HANG_TO, smooth between.
    upper: the upper arm's world orientation to judge by, in place of G[sh]."""
    _, _, bax = skeleton()
    bax = bax.to(device=G[sh].device, dtype=G[sh].dtype)
    ua = rotate_vector_wxyz_t(G[sh] if upper is None else upper, bax[sh])
    below = -rotate_vector_wxyz_t(conj(G[CHEST]), ua)[..., 1]
    lo, hi = math.sin(math.radians(HANG_FROM)), math.sin(math.radians(HANG_TO))
    x = ((below - lo) / (hi - lo)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def correct_pose(q, weights, amount=1.0, terms=('roll', 'yaw', 'fyaw', 'hang'), turn=None, turn_fade=0.0):
    """q: (..., 37, 4) local quats; weights: (2, 21) radians, per arm L/R: roll 8, yaw 6, fyaw 6, hang 1;
    turn: None or (2,) radians, each arm's calibration T-pose turn (elbow_turn), undone first by
    rolling the upper arm about its local +X with the forearm held in the world. turn_fade > 0
    (degrees of elbow bend) brings the turn in smoothly from 0 with the arm straight to all of it
    at that bend: the hinge cannot see the turn on a straight arm."""
    parent, order, bax = skeleton()
    bax = bax.to(device=q.device, dtype=q.dtype)
    weights = weights.to(device=q.device, dtype=q.dtype) * amount
    if weights.shape[-1] < PER_ARM:      # an older layout without the hang weight
        weights = torch.cat([weights, weights.new_zeros(2, PER_ARM - weights.shape[-1])], dim=-1)
    up = q.new_tensor([0.0, 1.0, 0.0])
    x_axis = q.new_tensor([1.0, 0.0, 0.0])
    G = world_orientations(q, [w for chain in ARMS.values() for w in chain] + [CHEST])
    out = q.clone()
    for a, (side, (sh, el, wr)) in enumerate(ARMS.items()):
        Gs, Ge, Gw = G[sh], G[el], G[wr]
        fs, fe = features(Gs), features(Ge)
        fr = torch.cat([fs, body_features(G, sh, el)], dim=-1)
        W = weights[a]
        w_roll, w_yaw = W[:N_ROLL], W[N_ROLL:N_ROLL + N_HEADING]
        w_fyaw, w_hang = W[N_ROLL + N_HEADING:N_ROLL + 2 * N_HEADING], W[N_ROLL + 2 * N_HEADING:]
        Gs2, Ge2, Gw2 = Gs, Ge, Gw
        if turn is not None:
            angle = -turn[a].to(device=q.device, dtype=q.dtype) * amount
            angle = angle.expand(Gs2.shape[:-1])
            if turn_fade > 0:
                ua = rotate_vector_wxyz_t(Gs, bax[sh])
                fa = rotate_vector_wxyz_t(Ge, bax[el])
                x = ((1.0 - (ua * fa).sum(-1)) / (1.0 - np.cos(np.radians(turn_fade)))).clamp(0.0, 1.0)
                angle = angle * x * x * (3.0 - 2.0 * x)
            Gs2 = quaternion_multiply_wxyz_t(Gs2, axis_angle_q(x_axis, angle))
        if 'roll' in terms:
            Gs2 = quaternion_multiply_wxyz_t(Gs2, axis_angle_q(bax[sh], fr @ w_roll))
        if 'yaw' in terms:
            Gs2 = quaternion_multiply_wxyz_t(axis_angle_q(up, fs @ w_yaw), Gs2)
        if 'fyaw' in terms:
            Ge2 = quaternion_multiply_wxyz_t(axis_angle_q(up, fe @ w_fyaw), Ge2)
        if 'hang' in terms:
            # upper arm and forearm turned sideways together about the chest's forward axis:
            # positive moves a hanging arm toward the dancer's left. The elbow local is unchanged;
            # the hand stays where its own sensor puts it (the hands read symmetric when the arms
            # do not - turning them with the arm broke hands resting on the hips).
            forward = rotate_vector_wxyz_t(G[CHEST], q.new_tensor([0.0, 0.0, 1.0]))
            # judge the arm's height where the correction would put it: the bank being undone
            # lifts one arm and lowers the other, so the raw height would give the lifted arm
            # less of the correction than its partner
            corrected = quaternion_multiply_wxyz_t(axis_angle_q(forward, w_hang[0].expand(Gs2.shape[:-1])), Gs2)
            H = axis_angle_q(forward, hang_valve(G, sh, corrected) * w_hang[0])
            Gs2 = quaternion_multiply_wxyz_t(H, Gs2)
            Ge2 = quaternion_multiply_wxyz_t(H, Ge2)
        out[..., sh, :] = quaternion_multiply_wxyz_t(conj(G[parent[sh]]), Gs2)
        out[..., el, :] = quaternion_multiply_wxyz_t(conj(Gs2), Ge2)
        out[..., wr, :] = quaternion_multiply_wxyz_t(conj(Ge2), Gw2)
    return out


def as_weights(data, device, dtype):
    if data is None:
        return None
    w = data if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data, dtype=np.float64))
    w = w.reshape(-1)
    if w.numel() > N_WEIGHTS or w.numel() == 0:
        return None
    if w.numel() == OLD_WEIGHTS:
        # the heading-only layout (2 arms x 3 terms x 6): the body and hang weights get 0
        old = w.reshape(2, 3, N_HEADING)
        w = torch.cat([old[:, 0], w.new_zeros(2, N_BODY), old[:, 1], old[:, 2],
                       w.new_zeros(2, N_HANG)], dim=-1).reshape(-1)
    elif w.numel() == OLD_WEIGHTS_40:
        # before 'hang' (20 per arm): the hang weight gets 0
        w = torch.cat([w.reshape(2, -1), w.new_zeros(2, N_HANG)], dim=-1).reshape(-1)
    elif w.numel() < N_WEIGHTS:
        # a short list fills the rest with 0, handy for trying the first few by hand
        w = torch.cat([w, w.new_zeros(N_WEIGHTS - w.numel())])
    return w.reshape(2, PER_ARM).to(device=device, dtype=dtype)


def as_turn(data, device, dtype):
    """Two numbers in degrees (left, right) -> (2,) radians, or None."""
    if data is None:
        return None
    t = data if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data, dtype=np.float64))
    t = t.detach().reshape(-1)
    if t.numel() == 1:
        t = t.repeat(2)
    if t.numel() != 2:
        return None
    return torch.deg2rad(t.to(device=device, dtype=dtype))


class ShadowMagCorrectNode(Node):
    """Applies a learned magnetization correction to the Shadow arms: a live pose
    (37 or 20 joints) or a whole take. Weights come from t.param (42 numbers,
    radians: per arm, left then right, roll 8, yaw 6, fyaw 6, hang 1; the older 40- and
    36-number layouts are accepted too); the 'turn' inlet takes elbow_turn's per-take T-pose
    turn (degrees, left right). With 'hang from' set to 'fixed bank', the hang is not learned:
    both arms undo the 'fixed bank' inlet's value instead (degrees, as hang_symmetry's shift
    reads it), for sessions with a known, steady bank. New weights, turn or fixed bank
    re-correct the pose already held. Numpy in gives numpy out, so it sits
    in a render chain unchanged; a tensor in stays a tensor, so t.optimize can
    trace a loss back through it."""
    @staticmethod
    def factory(name, data, args=None):
        return ShadowMagCorrectNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.weights = None
        self.pose = None
        self.pose_input = self.add_input('pose in', triggers_execution=True)
        # new weights re-correct the pose already held, so t.optimize's 'again'
        # only needs to reach t.param, and a paused body follows the learning
        self.weights_input = self.add_input('weights', triggers_execution=True)
        self.turn = None
        self.turn_input = self.add_input('turn', triggers_execution=True)
        self.bank_input = self.add_input('fixed bank', widget_type='drag_float', default_value=0.0,
                                         triggers_execution=True)
        self.on_input = self.add_input('correct', widget_type='checkbox', default_value=True)
        self.amount_input = self.add_input('amount', widget_type='drag_float', default_value=1.0,
                                           min=0.0, max=1.0)
        self.term_options = {t: self.add_option(t, widget_type='checkbox', default_value=True) for t in TERMS}
        # where the hang angle comes from: the learned weights, or the 'fixed bank' inlet -
        # a session's measured bank, undone on both arms
        self.hang_from_option = self.add_option('hang from', widget_type='combo', default_value='weights')
        self.hang_from_option.widget.combo_items = ['weights', 'fixed bank']
        # degrees of elbow bend over which the turn comes in; 0 applies all of it, arm straight too
        self.turn_fade_option = self.add_option('turn fade', widget_type='drag_float', default_value=0.0,
                                                min=0.0, max=90.0)
        self.pose_output = self.add_output('pose out')

    def execute(self):
        if self.weights_input.fresh_input:
            self.weights = self.weights_input()
        if self.turn_input.fresh_input:
            self.turn = self.turn_input()
        if self.pose_input.fresh_input:
            self.pose = self.pose_input()
        data = self.pose
        if data is None:
            return
        on = self.on_input()
        fixed = self.hang_from_option() == 'fixed bank'
        if (on is not None and not any_to_bool(on)) or (self.weights is None and self.turn is None and not fixed):
            self.pose_output.send(data)
            return
        q, how = as_pose_tensor(data)
        if q is None:
            if self.app.verbose:
                print('shadow_mag_correct: expects Shadow quaternions, (37, 4) or (20, 4) per frame')
            self.pose_output.send(data)
            return
        if self.weights is None:
            w = q.new_zeros(2, PER_ARM)
        else:
            w = as_weights(self.weights, q.device, q.dtype)
            if w is None:
                print('shadow_mag_correct: weights must be up to 42 numbers (per arm: roll 8, yaw 6, fyaw 6, hang 1)')
                self.pose_output.send(data)
                return
        if fixed:
            bank = self.bank_input()
            bank = 0.0 if bank is None else any_to_float(bank)
            # the bank reads + toward the dancer's left; a positive hang weight moves the arm
            # that way, so undoing it takes the opposite angle, the same on both arms
            w = w.clone()
            w[:, PER_ARM - 1] = -math.radians(bank)
        turn = as_turn(self.turn, q.device, q.dtype)
        if self.turn is not None and turn is None:
            print('shadow_mag_correct: turn must be two numbers in degrees, left then right')
        terms = [t for t in TERMS if self.term_options[t]()]
        amount = self.amount_input()
        amount = 1.0 if amount is None else float(amount)
        fade = self.turn_fade_option()
        fade = 0.0 if fade is None else float(fade)
        self.pose_output.send(give_back(correct_pose(q, w, amount, terms, turn, fade), how))


_offsets = None


def bone_offsets():
    """Each joint's offset from its parent, in the parent's frame, in cm (definition.xml)."""
    global _offsets
    if _offsets is None:
        from xml.etree import ElementTree as ET
        from dpg_system.smpl_utilities.diag_magnetometer_deviation import LIMB_TO_IDX
        off = np.zeros((37, 3))

        def walk(node):
            idx = LIMB_TO_IDX.get(node.get('id'))
            t = node.get('translate')
            if idx is not None and t:
                off[idx] = [float(v) for v in t.split()]
            for child in node:
                walk(child)
        walk(ET.parse(Path(__file__).parent / 'definition.xml').getroot())
        _offsets = torch.tensor(off)
    return _offsets


PELVIS, UPPER_SPINE = 4, 17
FINGERS = {'L': 6, 'R': 20}


def joint_positions(q, joints, scale=1.0):
    """World positions (cm, relative to the root) of the joints, from local quats (..., 37, 4)."""
    parent, order, _ = skeleton()
    off = bone_offsets().to(device=q.device, dtype=q.dtype) * scale
    needed = set()
    for j in joints:
        while j >= 0 and j not in needed:
            needed.add(j)
            j = parent[j]
    G = world_orientations(q, list(needed))
    P = {}
    for j in order:
        if j in needed:
            p = parent[j]
            P[j] = q.new_zeros(q.shape[:-2] + (3,)) if p < 0 else P[p] + rotate_vector_wxyz_t(G[p], off[j])
    return P, G


def torso_frame(q, scale=1.0):
    """The torso's axis (pelvis to upper spine) and the chest's sideways and forward directions,
    per frame: (A, axis, length, u, x, z)."""
    P, G = joint_positions(q, [PELVIS, UPPER_SPINE, CHEST], scale)
    A, B = P[PELVIS], P[UPPER_SPINE]
    axis = B - A
    length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True) + 1e-9
    xc = rotate_vector_wxyz_t(G[CHEST], q.new_tensor([1.0, 0.0, 0.0]))
    zc = rotate_vector_wxyz_t(G[CHEST], q.new_tensor([0.0, 0.0, 1.0]))
    return A, axis, length, axis / length, xc, zc


def torso_gap(frame, p, width=14.0, depth=10.0):
    """Distance (cm) of points p (..., [k,] 3) outside the torso: an oval cylinder (half-width x
    half-depth, in the chest's frame) along the frame's axis, rounded past its ends. Negative
    inside. With an extra point axis k, the frame broadcasts over it."""
    A, axis, length, u, xc, zc = frame
    if p.dim() == A.dim() + 1:
        A, axis, length, u, xc, zc = (x[..., None, :] for x in (A, axis, length, u, xc, zc))
    t = (((p - A) * u).sum(-1, keepdim=True) / length).clamp(0.0, 1.0)
    h = p - (A + t * axis)
    hl, hd, hv = (h * xc).sum(-1), (h * zc).sum(-1), (h * u).sum(-1)
    radial = torch.sqrt(hl * hl + hd * hd + 1e-9)
    # the oval's radius in this point's direction
    r = width * depth * radial / torch.sqrt((depth * hl) ** 2 + (width * hd) ** 2 + 1e-9)
    return radial - r + hv.abs()


def arm_gaps(q, width=14.0, depth=10.0, arm=4.0, scale=1.0, samples=6):
    """Per arm, how close the forearm and hand come to the torso surface (cm; negative = inside),
    and how far the wrist itself is from it: two (..., 2) tensors, left then right."""
    joints = [j for side in 'LR' for j in (ARMS[side][1], ARMS[side][2], FINGERS[side])]
    P, _ = joint_positions(q, joints, scale)
    frame = torso_frame(q, scale)
    t = q.new_tensor(np.linspace(0.0, 1.0, samples))[:, None]
    w, d, r = width * scale, depth * scale, arm * scale
    closest, wrist = [], []
    for side in 'LR':
        el, wr, fi = ARMS[side][1], ARMS[side][2], FINGERS[side]
        fore = P[el][..., None, :] + t * (P[wr] - P[el])[..., None, :]
        hand = P[wr][..., None, :] + t[1:] * (P[fi] - P[wr])[..., None, :]
        g = torso_gap(frame, torch.cat([fore, hand], dim=-2), w, d)
        closest.append(g.min(-1).values - r)
        wrist.append(torso_gap(frame, P[wr], w, d) - r)
    return torch.stack(closest, dim=-1), torch.stack(wrist, dim=-1)


def heading_weights(swing, phase):
    """A once-per-turn heading error, swing * sin(heading - phase) (degrees), as the six yaw
    weights of correct_pose (radians): at a sensor turned by 'heading' about vertical from its
    rest, features() reads (cos h, 0, sin h, -sin h, 0, cos h), and these weights give exactly
    that wave; split over both world axes, so a tilted sensor is treated evenly."""
    a, p = math.radians(swing) / 2.0, math.radians(phase)
    return [-a * math.sin(p), 0.0, a * math.cos(p), -a * math.cos(p), 0.0, -a * math.sin(p)]


def adjust_heading(q, yaw, offset):
    """q: (..., 37, 4) local quats; yaw: (2, 6) correct_pose yaw weights (radians), offset: (2,)
    radians. Turns each upper arm about vertical by yaw . features + offset, the forearm held
    where it was in the world (so the hand does not move): the error of an upper-arm sensor whose
    heading is wrong. Returns (corrected quats, (..., 2) angles applied)."""
    parent, _, _ = skeleton()
    up = q.new_tensor([0.0, 1.0, 0.0])
    G = world_orientations(q, [j for chain in ARMS.values() for j in chain[:2]])
    out = q.clone()
    angles = []
    for a, (sh, el, wr) in enumerate(ARMS.values()):
        angle = features(G[sh]) @ yaw[a] + offset[a]
        Gs2 = quaternion_multiply_wxyz_t(axis_angle_q(up, angle), G[sh])
        out[..., sh, :] = quaternion_multiply_wxyz_t(conj(G[parent[sh]]), Gs2)
        out[..., el, :] = quaternion_multiply_wxyz_t(conj(Gs2), G[el])
        angles.append(angle)
    return out, torch.stack(angles, dim=-1)


class ShadowHeadingAdjustNode(Node):
    """Hand-dialled heading correction for the upper arms: each arm turned about vertical by a
    once-per-turn wave (swing, phase) plus a constant offset, all in degrees, the forearm held in
    the world. The shape of a magnetized upper-arm sensor's heading error - for exploring it by
    eye before learning it. Swing and phase are the same correction as shadow_mag_correct's yaw
    weights; the 'yaw weights' outlet gives them (42 numbers, yaw slots filled) to seed t.param.
    The offset has no counterpart there."""
    @staticmethod
    def factory(name, data, args=None):
        return ShadowHeadingAdjustNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose = None
        self.pose_input = self.add_input('pose in', triggers_execution=True)
        self.on_input = self.add_input('adjust', widget_type='checkbox', default_value=True, triggers_execution=True)
        self.knobs = {}
        for side in ('left', 'right'):
            for what, lo, hi in (('swing', -90.0, 90.0), ('phase', -180.0, 180.0), ('offset', -90.0, 90.0)):
                self.knobs[(side, what)] = self.add_input(f'{side} {what}', widget_type='drag_float',
                                                          default_value=0.0, min=lo, max=hi,
                                                          triggers_execution=True)
        self.pose_output = self.add_output('pose out')
        self.angle_output = self.add_output('angles')
        self.weights_output = self.add_output('yaw weights')

    def value(self, side, what):
        v = self.knobs[(side, what)]()
        return 0.0 if v is None else any_to_float(v)

    def execute(self):
        if self.pose_input.fresh_input:
            self.pose = self.pose_input()
        swing = [self.value(s, 'swing') for s in ('left', 'right')]
        phase = [self.value(s, 'phase') for s in ('left', 'right')]
        offset = [self.value(s, 'offset') for s in ('left', 'right')]
        w42 = [0.0] * N_WEIGHTS
        for a in range(2):
            w42[a * PER_ARM + N_ROLL:a * PER_ARM + N_ROLL + N_HEADING] = heading_weights(swing[a], phase[a])
        self.weights_output.send(np.array(w42))
        data = self.pose
        if data is None:
            return
        on = self.on_input()
        if on is not None and not any_to_bool(on):
            self.pose_output.send(data)
            return
        q, how = as_pose_tensor(data)
        if q is None:
            if self.app.verbose:
                print('shadow_heading_adjust: expects Shadow quaternions, (37, 4) or (20, 4) per frame')
            self.pose_output.send(data)
            return
        yaw = q.new_tensor([w42[a * PER_ARM + N_ROLL:a * PER_ARM + N_ROLL + N_HEADING] for a in range(2)])
        out, angles = adjust_heading(q, yaw, q.new_tensor([math.radians(o) for o in offset]))
        self.angle_output.send(np.degrees(angles.detach().cpu().numpy()))
        self.pose_output.send(give_back(out, how))


class ElbowHingeNode(Node):
    """How far each forearm leaves its elbow-hinge circle, over a whole take.

    The elbow is a hinge: seen from the upper arm, the forearm's direction can only
    move around one fixed circle (forearm twist does not move it at all). A frame off
    that circle is impossible, so it measures error, not plausibility.

    circle 'forward' (the default) is the circle of an elbow that bends forward, as
    SMPL's does from its rest pose: it pins the upper arm's roll at every bend, so
    learning cannot drift. 'fitted' fits the circle to the take inside the calculation,
    assuming nothing about the dancer's elbow - but a constant twist of the upper arm
    is invisible to it, and a correction learned against it can drift in roll."""
    @staticmethod
    def factory(name, data, args=None):
        return ElbowHingeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('take in', triggers_execution=True)
        self.width_option = self.add_option('robust width', widget_type='drag_float', default_value=4.0)
        self.arms_option = self.add_option('arms', widget_type='combo', default_value='both')
        self.arms_option.widget.combo_items = ['both', 'left', 'right']
        self.circle_option = self.add_option('circle', widget_type='combo', default_value='forward')
        self.circle_option.widget.combo_items = ['forward', 'fitted']
        self.loss_output = self.add_output('loss')
        self.off_output = self.add_output('off hinge')
        self.rms_output = self.add_output('rms')

    @staticmethod
    def off_hinge(q, circle='fitted'):
        """q: (F, 37, 4) -> (F, 2) sine of the angle off each arm's hinge plane: the take's best
        plane ('fitted'), or the plane of an elbow bending forward about local vertical
        ('forward'), which keeps the rest forearm's small vertical component."""
        parent, order, bax = skeleton()
        bax = bax.to(device=q.device, dtype=q.dtype)
        G = world_orientations(q, [sh for sh, el, wr in ARMS.values()] + [el for sh, el, wr in ARMS.values()])
        offs = []
        for sh, el, wr in ARMS.values():
            d = rotate_vector_wxyz_t(quaternion_multiply_wxyz_t(conj(G[sh]), G[el]), bax[el])
            d = d / (torch.linalg.vector_norm(d, dim=-1, keepdim=True) + 1e-12)
            if circle == 'forward':
                offs.append(d[:, 1] - bax[el, 1])
                continue
            c = d - d.mean(0)
            cov = c.T @ c / max(1, len(c) - 1)
            _, vectors = run_with_cpu_fallback(torch.linalg.eigh, cov)
            normal = vectors[:, 0].to(q.device)               # smallest spread: the hinge plane's normal
            offs.append(c @ normal)
        return torch.stack(offs, dim=-1)

    def execute(self):
        q, how = as_pose_tensor(self.pose_input())
        if q is None or q.ndim != 3 or q.shape[0] < 10:
            if self.app.verbose:
                print('elbow_hinge: expects a take of Shadow quaternions, (frames, 37 or 20, 4)')
            return
        off = self.off_hinge(q, 'fitted' if self.circle_option() == 'fitted' else 'forward')
        arms = self.arms_option()
        if arms == 'left':
            used = off[:, :1]
        elif arms == 'right':
            used = off[:, 1:]
        else:
            used = off
        width = self.width_option()
        sig = math.sin(math.radians(4.0 if width is None or width <= 0 else float(width)))
        # pseudo-Huber: square-like for small misses, straight-line for big ones, so a few
        # wild frames cannot dominate
        loss = torch.mean(sig ** 2 * (torch.sqrt(1 + (used / sig) ** 2) - 1)) / sig ** 2
        degrees = torch.rad2deg(torch.asin(off.detach().clamp(-1.0, 1.0)))
        self.rms_output.send(np.round(degrees.pow(2).mean(0).sqrt().cpu().numpy(), 3))   # a summary, in degrees
        self.off_output.send(degrees.abs())
        self.loss_output.send(loss)


class ElbowTurnNode(Node):
    """How far each elbow was turned in a take's calibration T-pose, in degrees, left
    then right. A dancer can turn the elbow a long way while the shoulder and wrist
    hold the T-pose; the upper-arm sensor takes that turn as its zero, so afterwards
    every elbow bend shows up turned by it. A real elbow bends forward (as SMPL's does
    from its palms-down rest), so the direction a bent forearm points around the
    upper-arm bone gives the turn. Constant for a take, changing at each calibration,
    and invisible to elbow_hinge's loss: measure it per take and feed it to
    shadow_mag_correct's 'turn' inlet. Positive: the inside of the elbow was turned up
    toward the ceiling."""
    @staticmethod
    def factory(name, data, args=None):
        return ElbowTurnNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('take in', triggers_execution=True)
        self.low_option = self.add_option('bend from', widget_type='drag_float', default_value=45.0)
        self.high_option = self.add_option('bend to', widget_type='drag_float', default_value=80.0)
        self.arm_low_option = self.add_option('arm low only', widget_type='checkbox', default_value=True)
        self.turn_output = self.add_output('turn')
        self.spread_output = self.add_output('spread')
        self.count_output = self.add_output('frames used')

    @staticmethod
    def measure(q, bend_from=45.0, bend_to=80.0, arm_low=True, min_frames=100):
        """q: (F, 37, 4). Returns turn (2,), spread (2,) and frames used (2,), degrees. The
        moderate-bend band keeps out effects that grow with deep bend; arm low keeps out
        effects that grow as the arm rises. Too few frames there widens the choice."""
        parent, order, bax = skeleton()
        q = q.detach().to('cpu', torch.float64)
        bax = bax.double()
        G = world_orientations(q, [CHEST] + [j for sh, el, wr in ARMS.values() for j in (sh, el)])
        turns, spreads, counts = [], [], []
        for sh, el, wr in ARMS.values():
            d = rotate_vector_wxyz_t(quaternion_multiply_wxyz_t(conj(G[sh]), G[el]), bax[el]).numpy()
            d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
            bend = np.degrees(np.arccos(np.clip(d @ bax[sh].numpy(), -1.0, 1.0)))
            # A turn is a roll about the upper arm's local +X, which keeps the forearm's X component.
            # With no turn, the elbow bends about local vertical, which keeps the rest forearm's
            # small Y component (the skeleton's forearm points a few degrees down): so this frame's
            # unturned forearm sits at the same X, at that Y, on the forward side.
            b = float(bax[el, 1])
            reference = np.arctan2(np.sqrt(np.clip(1.0 - d[:, 0] ** 2 - b ** 2, 0.0, None)), b)
            turn_each = reference - np.arctan2(d[:, 2], d[:, 1])     # about local +X, per frame
            ua = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[sh], bax[sh])).numpy()
            low = ua[:, 1] < -0.5                                     # more than 30 degrees below level
            band = (bend >= bend_from) & (bend < bend_to)
            for use in ([band & low] if arm_low else []) + [band, bend >= bend_from]:
                if use.sum() >= min_frames:
                    break
            if use.sum() == 0:
                turns.append(np.nan); spreads.append(np.nan); counts.append(0)
                continue
            centre = np.arctan2(np.sin(turn_each[use]).mean(), np.cos(turn_each[use]).mean())
            off = (turn_each[use] - centre + np.pi) % (2 * np.pi) - np.pi
            turns.append(np.degrees(centre))
            spreads.append(np.degrees(np.sqrt(np.mean(off ** 2))))
            counts.append(int(use.sum()))
        wrap = lambda a: (np.asarray(a) + 180.0) % 360.0 - 180.0
        return wrap(turns), np.asarray(spreads), np.asarray(counts)

    def execute(self):
        q, how = as_pose_tensor(self.pose_input())
        if q is None or q.ndim != 3 or q.shape[0] < 10:
            if self.app.verbose:
                print('elbow_turn: expects a take of Shadow quaternions, (frames, 37 or 20, 4)')
            return
        lo, hi = self.low_option(), self.high_option()
        lo = 45.0 if lo is None else float(lo)
        hi = 80.0 if hi is None or hi <= lo else float(hi)
        arm_low = self.arm_low_option()
        turn, spread, count = self.measure(q, lo, hi, True if arm_low is None else bool(arm_low))
        self.count_output.send(count)
        self.spread_output.send(np.round(spread, 1))
        self.turn_output.send(np.round(turn, 1))


def parse_ranges(text):
    """'10430-11242, 3767-4000' -> [(10430, 11242), (3767, 4000)]; anything unreadable is skipped."""
    ranges = []
    if text is None:
        return ranges
    for part in str(text).replace(';', ',').split(','):
        bits = part.replace(' ', '').split('-')
        if len(bits) == 2 and bits[0].isdigit() and bits[1].isdigit():
            ranges.append((int(bits[0]), int(bits[1])))
    return ranges


class HangSymmetryNode(Node):
    """How far both lowered arms sit to one side together, over a take.

    With the arms hanging, a symmetric body hangs them as mirror images: whatever the
    outward splay, the pair is centred. A shared sideways shift of the two (the same way
    for both) is an error, and the one this measures - not how vertical the arms are,
    and not each arm's splay. Seen in Shadow takes as a good T-pose with an asymmetric
    hang once the arms come down. A loss for t.optimize; shadow_mag_correct's 'hang'
    term is the correction it drives."""
    @staticmethod
    def factory(name, data, args=None):
        return HangSymmetryNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('take in', triggers_execution=True)
        self.ranges_option = self.add_option('ranges', widget_type='text_input', default_value='')
        self.every_option = self.add_option('every', widget_type='drag_int', default_value=1, min=1)
        self.low_option = self.add_option('below level', widget_type='drag_float', default_value=50.0,
                                          min=0.0, max=90.0)
        self.width_option = self.add_option('robust width', widget_type='drag_float', default_value=4.0)
        self.loss_output = self.add_output('loss')
        self.shift_output = self.add_output('shift')
        self.count_output = self.add_output('frames used')

    @staticmethod
    def shifts(q, below=50.0, ranges=(), every=1):
        """q: (F, 37, 4) -> per-frame shared sideways shift of the two upper arms (radians,
        positive toward the dancer's left) for the frames used, and their mask."""
        _, _, bax = skeleton()
        bax = bax.to(device=q.device, dtype=q.dtype)
        G = world_orientations(q, [CHEST, ARMS['L'][0], ARMS['R'][0]])
        v = [rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[sh], bax[sh]))
             for sh in (ARMS['L'][0], ARMS['R'][0])]
        # each arm's angle in the chest's front plane, from straight down toward the dancer's left
        side = [torch.atan2(a[:, 0], -a[:, 1]) for a in v]
        low = -math.sin(math.radians(below))
        use = (v[0][:, 1] < low) & (v[1][:, 1] < low) & ((v[0][:, 1] - v[1][:, 1]).abs() < 0.2)
        if len(ranges) > 0:
            frame = torch.arange(len(q), device=q.device) * every
            inside = torch.zeros_like(use)
            for a, b in ranges:
                inside |= (frame >= a) & (frame <= b)
            use &= inside
        return (side[0] + side[1])[use] / 2, use

    def execute(self):
        q, how = as_pose_tensor(self.pose_input())
        if q is None or q.ndim != 3 or q.shape[0] < 2:
            if self.app.verbose:
                print('hang_symmetry: expects a take of Shadow quaternions, (frames, 37 or 20, 4)')
            return
        every = self.every_option()
        below = self.low_option()
        shift, use = self.shifts(q, 50.0 if below is None else float(below),
                                 parse_ranges(self.ranges_option()), 1 if every is None else max(1, int(every)))
        self.count_output.send(int(use.sum()))
        if shift.numel() == 0:
            print('hang_symmetry: no frames with both arms lowered' +
                  (' in the ranges' if parse_ranges(self.ranges_option()) else ''))
            return
        width = self.width_option()
        sig = math.radians(4.0 if width is None or width <= 0 else float(width))
        # pseudo-Huber: close to the median's pull, so asymmetric choreography in the frames
        # used cannot drag the answer the way a mean would
        loss = torch.mean(sig ** 2 * (torch.sqrt(1 + (shift / sig) ** 2) - 1)) / sig ** 2
        self.shift_output.send(round(float(torch.rad2deg(shift.detach().median())), 2))
        self.loss_output.send(loss)


class BodyContactNode(Node):
    """Keeps the hands out of the body, and on it where they should touch, over a take.

    The torso is an oval cylinder from the pelvis to the upper spine, turned with the chest;
    the forearms and hands are points along them with a thickness, placed by the Shadow
    skeleton. A forearm or hand inside the torso is impossible in any frame, so every frame's
    penetration counts. In the frames flagged as 'contact ranges' (hands resting on the hips,
    say) the wrists should touch the surface, so a gap counts there too, either way. Absolute,
    so unlike pose_symmetry it can tell which arm is wrong. A loss for t.optimize."""
    @staticmethod
    def factory(name, data, args=None):
        return BodyContactNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('take in', triggers_execution=True)
        self.ranges_option = self.add_option('contact ranges', widget_type='text_input', default_value='')
        self.every_option = self.add_option('every', widget_type='drag_int', default_value=1, min=1)
        self.width_option = self.add_option('torso width', widget_type='drag_float', default_value=13.0, min=1.0)
        self.depth_option = self.add_option('torso depth', widget_type='drag_float', default_value=10.0, min=1.0)
        self.arm_option = self.add_option('arm radius', widget_type='drag_float', default_value=4.0, min=0.0)
        self.contact_weight_option = self.add_option('contact weight', widget_type='drag_float', default_value=1.0, min=0.0)
        self.robust_option = self.add_option('robust width', widget_type='drag_float', default_value=1.0)
        self.loss_output = self.add_output('loss')
        self.inside_output = self.add_output('inside')
        self.gap_output = self.add_output('contact gap')

    def number(self, option, default):
        v = option()
        return default if v is None else float(v)

    def execute(self):
        q, how = as_pose_tensor(self.pose_input())
        if q is None or q.ndim != 3 or q.shape[0] < 2:
            if self.app.verbose:
                print('body_contact: expects a take of Shadow quaternions, (frames, 37 or 20, 4)')
            return
        width, depth = self.number(self.width_option, 13.0), self.number(self.depth_option, 10.0)
        arm = self.number(self.arm_option, 4.0)
        closest, wrist = arm_gaps(q, width, depth, arm)
        sig = max(1e-3, self.number(self.robust_option, 1.0))
        huber = lambda x: torch.mean(sig ** 2 * (torch.sqrt(1 + (x / sig) ** 2) - 1)) / sig ** 2
        # inside anywhere: only how deep, nothing for a hand clear of the body
        loss = huber(torch.relu(-closest))
        ranges = parse_ranges(self.ranges_option())
        gap = None
        if len(ranges) > 0:
            every = self.every_option()
            frame = torch.arange(len(q), device=q.device) * (1 if every is None else max(1, int(every)))
            use = torch.zeros(len(q), dtype=torch.bool, device=q.device)
            for a, b in ranges:
                use |= (frame >= a) & (frame <= b)
            if use.any():
                loss = loss + self.number(self.contact_weight_option, 1.0) * huber(wrist[use])
                gap = wrist[use].detach().median(0).values.cpu().numpy()
        self.inside_output.send(np.round((closest.detach() < 0).double().mean(0).cpu().numpy() * 100, 1))
        if gap is not None:
            self.gap_output.send(np.round(gap, 2))
        self.loss_output.send(loss)


class PoseSymmetryNode(Node):
    """How far the arms depart from mirror images, over the frames of a take where the dancer
    holds a symmetric pose (flagged as 'ranges').

    In a pose held symmetrically - hands on hips, arms in a matching gesture - the left upper
    arm and forearm point as the mirrored right ones do. Every difference is error, in every
    direction: sideways, height and forward/back, for each segment - so unlike hang_symmetry,
    which sees only the shared sideways shift of hanging arms, it constrains each arm on its
    own, at whatever arm configuration the pose has. A loss for t.optimize, alongside
    elbow_hinge and hang_symmetry; shadow_mag_correct is the correction it drives."""
    @staticmethod
    def factory(name, data, args=None):
        return PoseSymmetryNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.pose_input = self.add_input('take in', triggers_execution=True)
        # the uncorrected take, for 'halfway': link it before the corrector so it is held when the
        # corrected take arrives
        self.raw_input = self.add_input('raw take')
        self.ranges_option = self.add_option('ranges', widget_type='text_input', default_value='')
        self.every_option = self.add_option('every', widget_type='drag_int', default_value=1, min=1)
        self.segments_option = self.add_option('segments', widget_type='combo', default_value='upper arms and forearms')
        self.segments_option.widget.combo_items = ['upper arms and forearms', 'upper arms']
        # symmetry alone cannot say which arm is wrong: two arms wrong the same way are mirror
        # images too. 'halfway' pulls both to the symmetric pose midway between the two as recorded
        # (each wrong by half the difference); a reference arm is taken as right, and only the other
        # is pulled to it; 'both arms' only asks for symmetry.
        self.reference_option = self.add_option('reference', widget_type='combo', default_value='both arms')
        self.reference_option.widget.combo_items = ['both arms', 'halfway', 'left arm', 'right arm']
        self.width_option = self.add_option('robust width', widget_type='drag_float', default_value=4.0)
        self.loss_output = self.add_output('loss')
        self.asymmetry_output = self.add_output('asymmetry')
        self.count_output = self.add_output('frames used')

    @staticmethod
    def differences(q, ranges, every=1, forearms=True, reference='both arms', raw=None):
        """q: (F, 37, 4) -> (n, segments, 3) left minus mirrored right segment directions in the
        chest frame (x, y, z; about radians for small differences) for the frames in the
        ranges, and the mask of frames used. reference 'left arm' or 'right arm' holds that arm
        out of the gradient, so only the other arm is moved toward it. 'halfway' (raw: the
        uncorrected take, same frames) instead gives each arm's difference from the midway pose,
        the mean of the recorded left and mirrored right: (n, 2 x segments, 3)."""
        _, _, bax = skeleton()
        bax = bax.to(device=q.device, dtype=q.dtype)
        joints = [ARMS['L'][0], ARMS['R'][0]] + ([ARMS['L'][1], ARMS['R'][1]] if forearms else [])
        G = world_orientations(q, [CHEST] + joints)
        C = conj(G[CHEST])
        mirror = q.new_tensor([-1.0, 1.0, 1.0])
        diffs = []
        if reference == 'halfway' and raw is not None:
            R = world_orientations(raw, [CHEST] + joints)
            CR = conj(R[CHEST])
        for l, r in zip(joints[0::2], joints[1::2]):
            dl = rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[l], bax[l]))
            dr = rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[r], bax[r])) * mirror
            if reference == 'halfway' and raw is not None:
                rl = rotate_vector_wxyz_t(CR, rotate_vector_wxyz_t(R[l], bax[l]))
                rr = rotate_vector_wxyz_t(CR, rotate_vector_wxyz_t(R[r], bax[r])) * mirror
                mid = rl + rr
                mid = (mid / (torch.linalg.vector_norm(mid, dim=-1, keepdim=True) + 1e-12)).detach()
                diffs.append(dl - mid)
                diffs.append(dr - mid)
                continue
            if reference == 'left arm':
                dl = dl.detach()
            elif reference == 'right arm':
                dr = dr.detach()
            diffs.append(dl - dr)
        use = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        frame = torch.arange(len(q), device=q.device) * every
        for a, b in ranges:
            use |= (frame >= a) & (frame <= b)
        return torch.stack(diffs, dim=1)[use], use

    @staticmethod
    def asymmetry(q, use, forearms=True):
        """Median left-minus-mirrored-right angles in degrees over the frames used: per segment
        sideways (+ = left further out), height (+ = left higher), forward (+ = left further
        forward)."""
        _, _, bax = skeleton()
        bax = bax.to(device=q.device, dtype=q.dtype)
        joints = [ARMS['L'][0], ARMS['R'][0]] + ([ARMS['L'][1], ARMS['R'][1]] if forearms else [])
        G = world_orientations(q[use], [CHEST] + joints)
        C = conj(G[CHEST])
        out = []
        for l, r in zip(joints[0::2], joints[1::2]):
            ang = []
            for j, sign in ((l, 1.0), (r, -1.0)):
                v = rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[j], bax[j]))
                x = v[:, 0] * sign
                ang.append(torch.stack([torch.atan2(x, -v[:, 1]), torch.asin(v[:, 1].clamp(-1.0, 1.0)),
                                        torch.atan2(v[:, 2], torch.hypot(x, v[:, 1]))], dim=-1))
            out.append(torch.rad2deg(ang[0] - ang[1]).median(0).values)
        return torch.cat(out).detach().cpu().numpy()

    def execute(self):
        q, how = as_pose_tensor(self.pose_input())
        if q is None or q.ndim != 3 or q.shape[0] < 2:
            if self.app.verbose:
                print('pose_symmetry: expects a take of Shadow quaternions, (frames, 37 or 20, 4)')
            return
        ranges = parse_ranges(self.ranges_option())
        if len(ranges) == 0:
            print('pose_symmetry: flag the frames of the symmetric pose in its ranges option')
            return
        every = self.every_option()
        every = 1 if every is None else max(1, int(every))
        forearms = self.segments_option() != 'upper arms'
        reference = self.reference_option()
        reference = 'both arms' if reference is None else reference
        raw = None
        if reference == 'halfway':
            raw, _ = as_pose_tensor(self.raw_input())
            if raw is None or raw.shape != q.shape:
                print("pose_symmetry: 'halfway' needs the uncorrected take, the same frames, in 'raw take'")
                return
            raw = raw.to(device=q.device, dtype=q.dtype)
        d, use = self.differences(q, ranges, every, forearms, reference, raw)
        self.count_output.send(int(use.sum()))
        if d.shape[0] == 0:
            print('pose_symmetry: no frames in the ranges')
            return
        width = self.width_option()
        sig = math.radians(4.0 if width is None or width <= 0 else float(width))
        # pseudo-Huber on every component: a moment where the dancer breaks the pose cannot
        # drag the answer the way a mean would
        loss = torch.mean(sig ** 2 * (torch.sqrt(1 + (d / sig) ** 2) - 1)) / sig ** 2
        self.asymmetry_output.send(np.round(self.asymmetry(q, use, forearms), 2))
        self.loss_output.send(loss)
