"""Train shadow_mag_correct's 42 weights on jathiswaram (every 6th frame) against
elbow_hinge + hang_symmetry (A) and + pose_symmetry on the hands-on-hips holds (B);
report what is left of each error. Turn from elbow_turn, fade 30, as the example patch."""
import sys, os, tempfile, math, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (correct_pose, ElbowHingeNode, HangSymmetryNode, PoseSymmetryNode,
                                         ElbowTurnNode, parse_ranges, N_WEIGHTS, PER_ARM)
F = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
EVERY = 6
full = torch.as_tensor(np.load(F)['quats'].astype(np.float64))
q = full[::EVERY]
turn = torch.deg2rad(torch.as_tensor(ElbowTurnNode.measure(full)[0]))
AKIMBO = parse_ranges('1-827, 1052-1125, 3720-3796, 4748-5028, 12516-12545, 15428-15741, 21344-21702, 35945-36247')
HANG_RAW = parse_ranges('0-1721, 3767-4000, 5512-5734, 10430-11242, 16968-17082, 21195-21697, 35762-36237')


def minus(ranges, cut):
    out = []
    for a, b in ranges:
        pieces = [(a, b)]
        for c, d in cut:
            nxt = []
            for x, y in pieces:
                if d < x or c > y:
                    nxt.append((x, y)); continue
                if c > x: nxt.append((x, c - 1))
                if d < y: nxt.append((d + 1, y))
            pieces = nxt
        out += pieces
    return out


HANG = minus(HANG_RAW, AKIMBO)
sig_h = math.sin(math.radians(4.0)); sig = math.radians(4.0)


def huber(x, s):
    return torch.mean(s ** 2 * (torch.sqrt(1 + (x / s) ** 2) - 1)) / s ** 2


def losses(qc, reference='both arms'):
    raw = q if reference == 'halfway' else None
    off = ElbowHingeNode.off_hinge(qc, 'forward')
    hinge = huber(off, sig_h)
    shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
    hang = huber(shift, sig)
    d, _ = PoseSymmetryNode.differences(qc, AKIMBO, EVERY, True, reference, raw)
    pose = huber(d, sig)
    return hinge, hang, pose


def report(label, w):
    with torch.no_grad():
        qc = correct_pose(q, w, 1.0, ('roll', 'yaw', 'fyaw', 'hang'), turn, 30.0)
        off = ElbowHingeNode.off_hinge(qc, 'forward')
        rms = torch.rad2deg(torch.asin(off.clamp(-1, 1))).pow(2).mean(0).sqrt().numpy()
        shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
        _, use = PoseSymmetryNode.differences(qc, AKIMBO, EVERY, True)
        asym = PoseSymmetryNode.asymmetry(qc, use, True)
    print(f'{label:34s} hinge rms L/R {rms[0]:4.1f}/{rms[1]:4.1f}  hang shift {np.degrees(np.median(shift.numpy())):+5.1f}  '
          f'hands-on-hips L-R  upper arm side/height/fwd {asym[0]:+5.1f} {asym[1]:+5.1f} {asym[2]:+5.1f}  '
          f'forearm {asym[3]:+5.1f} {asym[4]:+5.1f} {asym[5]:+5.1f}')
    W = w.detach().numpy()
    print(f'{"":34s} hang L/R {np.degrees(W[0, -1]):+5.1f}/{np.degrees(W[1, -1]):+5.1f} deg')


def train(use_pose, steps=300, reference='both arms'):
    w = torch.zeros(2, PER_ARM, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.01)
    for i in range(steps):
        opt.zero_grad()
        qc = correct_pose(q, w, 1.0, ('roll', 'yaw', 'fyaw', 'hang'), turn, 30.0)
        hinge, hang, pose = losses(qc, reference)
        loss = hinge + hang + (pose if use_pose else 0.0)
        loss.backward(); opt.step()
    return w


print('hang ranges without the holds:', ', '.join(f'{a}-{b}' for a, b in HANG))
report('no correction', torch.zeros(2, PER_ARM, dtype=torch.float64))
wa = train(False); report('A: hinge + hang', wa)
wb = train(True); report('B: hinge + hang + pose_symmetry', wb)
wc = train(True, reference='left arm'); report('C: same, left arm as reference', wc)
wd = train(True, reference='halfway'); report('D: same, halfway', wd)
np.save(os.path.join(tempfile.gettempdir(), 'joint_weights.npy'),
        np.stack([wa.detach().numpy(), wb.detach().numpy(), wc.detach().numpy(), wd.detach().numpy()]))
