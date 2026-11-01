"""body_contact with the torso and forearms measured from Subject 7's SMPL body (female, her
betas) against the guessed oval cylinder: how each ruler reads the raw take, and what training
E (elbow_hinge + hang_symmetry + pose_symmetry + body_contact) does with each. Everything is
judged with the measured body."""
import sys, os, tempfile, math, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (correct_pose, ElbowHingeNode, HangSymmetryNode, PoseSymmetryNode, ElbowTurnNode,
                                         parse_ranges, arm_gaps, body_profile, PER_ARM, TERMS)
D = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/'
EVERY = 6
full = torch.as_tensor(np.load(D + 'beta/Subject7_take_jathiswaram_a_beta.npz')['quats'].astype(np.float64))
q = full[::EVERY]
turn = torch.deg2rad(torch.as_tensor(ElbowTurnNode.measure(full)[0]))
PROFILE = body_profile(np.load(D + 'Subject7_betas.npy', allow_pickle=True).item()['betas'], 'female')
HOLDS = parse_ranges('1-827, 1052-1125, 3720-3796, 4748-5028, 12516-12545, 15428-15741, 21344-21702, 35945-36247')
HANG = parse_ranges('828-1051, 1126-1721, 3797-4000, 5512-5734, 10430-11242, 16968-17082, 21195-21343, 35762-35944')
frame = torch.arange(len(q)) * EVERY
inhold = torch.zeros(len(q), dtype=torch.bool)
for a, b in HOLDS:
    inhold |= (frame >= a) & (frame <= b)
sig_h, sig = math.sin(math.radians(4.0)), math.radians(4.0)


def huber(x, s):
    return torch.mean(s ** 2 * (torch.sqrt(1 + (x / s) ** 2) - 1)) / s ** 2


def gaps(qc, profile):
    return arm_gaps(qc, 13.0, 10.0, 4.0, profile=profile)


def losses(qc, profile):
    hinge = huber(ElbowHingeNode.off_hinge(qc, 'forward'), sig_h)
    shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
    d, _ = PoseSymmetryNode.differences(qc, HOLDS, EVERY, True, 'both arms')
    closest, wrist = gaps(qc, profile)
    return hinge + huber(shift, sig) + huber(d, sig) + huber(torch.relu(-closest), 1.0) + huber(wrist[inhold], 1.0)


def train(profile, steps=300):
    w = torch.zeros(2, PER_ARM, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        losses(correct_pose(q, w, 1.0, TERMS, turn, 30.0), profile).backward()
        opt.step()
    return w.detach()


def report(label, w, profile):
    with torch.no_grad():
        qc = q if w is None else correct_pose(q, w, 1.0, TERMS, turn, 30.0)
        _, use = PoseSymmetryNode.differences(qc, HOLDS, EVERY, True)
        asym = PoseSymmetryNode.asymmetry(qc, use, True)
        closest, wrist = gaps(qc, profile)
        inside = (closest < 0).double().mean(0).numpy() * 100
        deep = (closest < -3).double().mean(0).numpy() * 100
        gap = wrist[inhold].median(0).values.numpy()
        shift, _ = HangSymmetryNode.shifts(qc, 50.0, HANG, EVERY)
    print(f'{label:34s} holds: wrist gap L/R {gap[0]:+5.1f}/{gap[1]:+5.1f} cm, upper arm L-R {asym[0]:+5.1f} {asym[1]:+5.1f} {asym[2]:+5.1f}'
          f' | inside L/R {inside[0]:4.1f}/{inside[1]:4.1f}% (>3 cm {deep[0]:4.1f}/{deep[1]:4.1f})'
          f' | hang shift {np.degrees(np.median(shift.numpy())):+5.1f}')


print('profile: half-width', PROFILE['width'].numpy().round(1))
print('         front     ', PROFILE['front'].numpy().round(1))
print('         back      ', PROFILE['back'].numpy().round(1))
print('         forearm   ', PROFILE['arm'].numpy().round(1))
report('raw, judged with the cylinder', None, None)
report('raw, judged with the body', None, PROFILE)
for label, prof in (('E trained with the cylinder', None), ('E trained with the body', PROFILE)):
    w = train(prof)
    report(label + ', judged with body', w, PROFILE)
    np.save(os.path.join(tempfile.gettempdir(), 'contact_weights_%s.npy' % ('body' if prof else 'cylinder')), w.numpy())
