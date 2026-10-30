"""Train shadow_mag_correct on jathiswaram (every 6th frame) with elbow_hinge + hang_symmetry +
pose_symmetry + body_contact, against variants: B (pose_symmetry, both arms free, no contact),
C (left arm as reference), E (both arms free + body_contact). Reports the wrists in the
hands-on-hips holds, how often each forearm/hand is inside the torso, and the other losses."""
import sys, os, tempfile, math, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (correct_pose, ElbowHingeNode, HangSymmetryNode, PoseSymmetryNode, ElbowTurnNode,
                                         parse_ranges, arm_gaps, PER_ARM, TERMS)
F = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
EVERY = 6
full = torch.as_tensor(np.load(F)['quats'].astype(np.float64))
q = full[::EVERY]
turn = torch.deg2rad(torch.as_tensor(ElbowTurnNode.measure(full)[0]))
HOLDS = parse_ranges('1-827, 1052-1125, 3720-3796, 4748-5028, 12516-12545, 15428-15741, 21344-21702, 35945-36247')
HANG = parse_ranges('828-1051, 1126-1721, 3797-4000, 5512-5734, 10430-11242, 16968-17082, 21195-21343, 35762-35944')
frame = torch.arange(len(q)) * EVERY
inhold = torch.zeros(len(q), dtype=torch.bool)
for a, b in HOLDS:
    inhold |= (frame >= a) & (frame <= b)
sig_h, sig = math.sin(math.radians(4.0)), math.radians(4.0)


def huber(x, s):
    return torch.mean(s ** 2 * (torch.sqrt(1 + (x / s) ** 2) - 1)) / s ** 2


def losses(qc, reference, contact):
    hinge = huber(ElbowHingeNode.off_hinge(qc, 'forward'), sig_h)
    shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
    d, _ = PoseSymmetryNode.differences(qc, HOLDS, EVERY, True, reference)
    total = hinge + huber(shift, sig) + huber(d, sig)
    if contact:
        closest, wrist = arm_gaps(qc)
        total = total + huber(torch.relu(-closest), 1.0) + huber(wrist[inhold], 1.0)
    return total


def train(reference, contact, steps=300):
    w = torch.zeros(2, PER_ARM, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        losses(correct_pose(q, w, 1.0, TERMS, turn, 30.0), reference, contact).backward()
        opt.step()
    return w.detach()


def report(label, w):
    with torch.no_grad():
        qc = q if w is None else correct_pose(q, w, 1.0, TERMS, turn, 30.0)
        off = ElbowHingeNode.off_hinge(qc, 'forward')
        rms = torch.rad2deg(torch.asin(off.clamp(-1, 1))).pow(2).mean(0).sqrt().numpy()
        shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
        _, use = PoseSymmetryNode.differences(qc, HOLDS, EVERY, True)
        asym = PoseSymmetryNode.asymmetry(qc, use, True)
        closest, wrist = arm_gaps(qc)
        inside = (closest < 0).double().mean(0).numpy() * 100
        deep = (closest < -3).double().mean(0).numpy() * 100
        gap = wrist[inhold].median(0).values.numpy()
    hang = '' if w is None else f'   hang L/R {np.degrees(w[0, -1]):+5.1f}/{np.degrees(w[1, -1]):+5.1f}'
    print(f'{label:26s} holds: wrist gap L/R {gap[0]:+5.1f}/{gap[1]:+5.1f} cm, upper arm L-R {asym[0]:+5.1f} {asym[1]:+5.1f} {asym[2]:+5.1f}, '
          f'forearm {asym[3]:+5.1f} {asym[4]:+5.1f} {asym[5]:+5.1f} | inside L/R {inside[0]:4.1f}/{inside[1]:4.1f}% (>3 cm {deep[0]:4.1f}/{deep[1]:4.1f}) | '
          f'hinge {rms[0]:4.1f}/{rms[1]:4.1f}  hang shift {np.degrees(np.median(shift.numpy())):+5.1f}{hang}')


report('raw', None)
for label, ref, contact in (('B: symmetry, both arms', 'both arms', False), ('C: symmetry, left reference', 'left arm', False),
                            ('E: symmetry + body_contact', 'both arms', True)):
    w = train(ref, contact)
    report(label, w)
    if label.startswith('E'):
        np.save(os.path.join(tempfile.gettempdir(), 'contact_weights.npy'), w.numpy())
