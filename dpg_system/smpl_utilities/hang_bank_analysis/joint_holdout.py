"""Weights trained on jathiswaram (A: hinge + hang, B: + pose_symmetry), applied to the
other Subject 7 takes: hang shift, hinge rms, and the hands-on-hips asymmetry where the
automatic detector finds holds (rougher than hand-flagged ranges)."""
import sys, os, tempfile, glob, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (correct_pose, ElbowHingeNode, HangSymmetryNode, PoseSymmetryNode, ElbowTurnNode,
                                         skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST)
_, _, bax = skeleton(); bax = bax.double()
W = np.load(os.path.join(tempfile.gettempdir(), 'joint_weights.npy'))
D = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/'


def akimbo_ranges(q):
    G = world_orientations(q, [CHEST, 5, 9, 19, 23]); C = conj(G[CHEST])
    v = {j: rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 9, 19, 23)}
    bend = {s: np.degrees(np.arccos(np.clip((v[a] * v[b]).sum(-1), -1, 1))) for s, a, b in (('L', 5, 9), ('R', 19, 23))}
    pick = ((bend['L'] > 45) & (bend['L'] < 165) & (bend['R'] > 45) & (bend['R'] < 165)
            & (v[5][:, 1] < -0.1) & (v[5][:, 1] > -0.97) & (v[19][:, 1] < -0.1) & (v[19][:, 1] > -0.97)
            & (v[9][:, 0] < -0.1) & (v[23][:, 0] > 0.1))
    idx = np.where(pick)[0]
    runs = [r for r in np.split(idx, np.where(np.diff(idx) > 30)[0] + 1) if len(r) >= 60] if len(idx) else []
    return [(int(r[0]), int(r[-1])) for r in runs]


print(f'{"take":30s} {"":10s} {"hinge L/R":>11s} {"hang":>6s}   hands-on-hips upper arm L-R side/height/fwd   forearm')
for f in sorted(glob.glob(D + 'Subject7_take*_beta.npz')):
    full = torch.as_tensor(np.load(f)['quats'].astype(np.float64))
    if 'jathiswaram_a' in f:
        continue
    turn = torch.deg2rad(torch.as_tensor(ElbowTurnNode.measure(full)[0]))
    q = full[::3]
    ak = akimbo_ranges(full)
    for name, w in (('none', np.zeros_like(W[0])), ('A', W[0]), ('B', W[1])):
        with torch.no_grad():
            qc = correct_pose(q, torch.as_tensor(w), 1.0, ('roll', 'yaw', 'fyaw', 'hang'), turn, 30.0) if name != 'none' else q
            off = ElbowHingeNode.off_hinge(qc, 'forward')
            rms = torch.rad2deg(torch.asin(off.clamp(-1, 1))).pow(2).mean(0).sqrt().numpy()
            shift, _ = HangSymmetryNode.shifts(qc, 50.0, (), 1)
            hs = f'{np.degrees(np.median(shift.numpy())):+6.1f}' if shift.numel() > 50 else '     .'
            txt = ''
            if ak:
                _, use = PoseSymmetryNode.differences(qc, ak, 3, True)
                if use.sum() > 30:
                    a = PoseSymmetryNode.asymmetry(qc, use, True)
                    txt = f'{a[0]:+6.1f} {a[1]:+6.1f} {a[2]:+6.1f}    {a[3]:+6.1f} {a[4]:+6.1f} {a[5]:+6.1f}  ({int(use.sum()) * 3} frames)'
        print(f'{os.path.basename(f)[9:39]:30s} {name:10s} {rms[0]:5.1f}/{rms[1]:5.1f} {hs}   {txt}')
