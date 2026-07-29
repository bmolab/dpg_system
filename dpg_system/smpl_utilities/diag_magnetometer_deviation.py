"""Diagnostic: per-sensor magnetometer heading-deviation estimation (applies NO correction).

This generalizes the upper-arm heading-yaw model in correct_upper_arm_offset.head_yaw to
*every* Shadow segment and fits all sensors JOINTLY under a sparsity prior. The point is to
let the fit DISCOVER which sensors carry a heading-correlated yaw error, and how large -- a
different direction from correcting the arms by symmetry one metric at a time.

Model
-----
Each Shadow IMU measures its own world orientation independently; the stored parent-relative
quats reconstruct to those world orientations exactly via forward kinematics. A magnetometer
error shows up as a yaw about world-vertical that depends on the sensor's own heading psi --
the classical compass-deviation series:

    delta_i(psi) = A_i + B_i sin psi + C_i cos psi + D_i sin 2psi + E_i cos 2psi

A = baked-in calibration zero-point error; (B,C) = hard-iron (1st harmonic); (D,E) = soft-iron
(2nd harmonic). The corrected world orientation is Yaw(-delta_i) * world_i, applied per sensor
(no chain re-FK -- each sensor's world estimate is independent, so each is corrected on its own).
psi is read from the MEASURED data (held constant), so delta_i is linear in the parameters.

Joint objective (MAP), several guideposts at once so none has to be sufficient alone:
  1. bilateral symmetry of limb bone directions, robust (Geman-McClure) over all frames, so a
     dancer's genuinely asymmetric moments down-weight themselves instead of biasing the fit;
  2. knee hinge anatomy -- the shin cannot leave the thigh's sagittal plane (squared barrier on
     the out-of-plane component): the "physically impossible joint angle" guidepost;
  3. trunk axial consistency -- consecutive spine sensors should share a heading except during
     genuine twists (robust), which probes whether the trunk chain is magnetized too;
  4. L1 sparsity on per-sensor amplitudes -> un-magnetized sensors collapse toward 0.
Gauge: no Vive here, so absolute heading is unobservable; the pelvis (the mirror reference) and
Body are pinned to delta=0 and everything is estimated relative to them.

Output: sensors ranked by RMS deviation over the visited headings, split into constant / 1st- /
2nd-harmonic amplitude, plus symmetry, knee and trunk residuals before/after. Nothing is written.
"""
import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch

UP = np.array([0.0, 1.0, 0.0])

# --- Shadow ("37") joint indices we name explicitly (see body_defs.shadow_joint_index_to_name) ---
BODY, PELV = 0, 4
LSH, LEL, LWR = 5, 9, 10
RSH, REL, RWR = 19, 23, 24
LBLADE, RBLADE = 13, 27
LHIP, LKNEE, LANK = 14, 12, 8
RHIP, RKNEE, RANK = 28, 26, 22
LTOE, RTOE = 15, 29
SPINEP, LOWV, MIDV, UPPV, SKULL = 31, 32, 1, 17, 2

# bmolab limb id (definition.xml node id) -> shadow joint index
LIMB_TO_IDX = {
    'Body': 0, 'Chest': 1, 'Head': 2, 'HeadEnd': 3, 'Hips': 4,
    'LeftArm': 5, 'LeftFinger': 6, 'LeftFingerEnd': 7, 'LeftFoot': 8, 'LeftForearm': 9,
    'LeftHand': 10, 'LeftHeel': 11, 'LeftLeg': 12, 'LeftShoulder': 13, 'LeftThigh': 14,
    'LeftToe': 15, 'LeftToeEnd': 16, 'Neck': 17, 'Reference': 18,
    'RightArm': 19, 'RightFinger': 20, 'RightFingerEnd': 21, 'RightFoot': 22, 'RightForearm': 23,
    'RightHand': 24, 'RightHeel': 25, 'RightLeg': 26, 'RightShoulder': 27, 'RightThigh': 28,
    'RightToe': 29, 'RightToeEnd': 30, 'SpineLow': 31, 'SpineMid': 32,
    'Tracker0': 33, 'Tracker1': 34, 'Tracker2': 35, 'Tracker3': 36,
}
IDX_TO_NAME = {
    0: 'Body', 1: 'MidVertebrae', 2: 'BaseOfSkull', 3: 'TopOfHead', 4: 'PelvisAnchor',
    5: 'LeftShoulder', 9: 'LeftElbow', 10: 'LeftWrist', 8: 'LeftAnkle', 12: 'LeftKnee',
    13: 'LeftShoulderBladeBase', 14: 'LeftHip', 15: 'LeftBallOfFoot', 17: 'UpperVertebrae',
    18: 'Reference', 19: 'RightShoulder', 23: 'RightElbow', 24: 'RightWrist', 22: 'RightAnkle',
    26: 'RightKnee', 27: 'RightShoulderBladeBase', 28: 'RightHip', 29: 'RightBallOfFoot',
    31: 'SpinePelvis', 32: 'LowerVertebrae',
}

# Limb pairs for the bilateral-symmetry term: (segment index, child index used for the bone axis)
SYM_PAIRS = [
    ((LSH, LEL), (RSH, REL)),        # upper arm
    ((LEL, LWR), (REL, RWR)),        # forearm
    ((LBLADE, LSH), (RBLADE, RSH)),  # shoulder blade
    ((LHIP, LKNEE), (RHIP, RKNEE)),  # thigh
    ((LKNEE, LANK), (RKNEE, RANK)),  # shin
    ((LANK, LTOE), (RANK, RTOE)),    # foot
]
# Knee hinge: (shin segment, shin child, thigh segment) -- shin must stay out of thigh's lateral axis
KNEES = [((LKNEE, LANK), LHIP), ((RKNEE, RANK), RHIP)]
# Trunk chain (parent->child) for the axial-consistency term
TRUNK = [PELV, SPINEP, LOWV, MIDV, UPPV, SKULL]


def load_skeleton(def_path):
    """Parse definition.xml -> parent index per joint (or -1), FK order, and bone-axis-local
    (unit vector toward each joint's continuing child, in that joint's local frame)."""
    root = ET.parse(def_path).getroot()
    parent = -np.ones(37, dtype=int)
    translate = {}
    order = []

    def walk(node, parent_idx):
        nid = node.get('id')
        idx = LIMB_TO_IDX.get(nid)
        if idx is not None:
            if parent_idx >= 0:
                parent[idx] = parent_idx
            order.append(idx)
            t = node.get('translate')
            translate[idx] = np.array([float(v) for v in t.split()]) if t else np.zeros(3)
            cur = idx
        else:
            cur = parent_idx
        for ch in node:
            walk(ch, cur)

    walk(root, -1)
    # bone axis of joint J = direction to its child (child's translate, in J's local frame)
    bone_axis = np.zeros((37, 3))
    child_of = {}
    for j in range(37):
        if parent[j] >= 0:
            child_of.setdefault(parent[j], j)
    for j in range(37):
        c = child_of.get(j)
        if c is not None:
            v = translate[c]
            bone_axis[j] = v / (np.linalg.norm(v) + 1e-9)
    return parent, order, bone_axis


# --------- numpy quaternion helpers (w-first), used for the constant measured-world FK ---------
def qmul_np(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def fk_world(Q, parent, order):
    """Reconstruct each sensor's measured world orientation from parent-relative local quats."""
    G = Q.copy()
    for j in order:
        p = parent[j]
        if p >= 0:
            G[:, j] = qmul_np(G[:, p], Q[:, j])
    return G


def qrot_np(q, v):
    qv = np.concatenate([np.zeros(q.shape[:-1] + (1,)), np.broadcast_to(v, q.shape[:-1] + (3,))], -1)
    qc = q * np.array([1, -1, -1, -1.0])
    return qmul_np(qmul_np(q, qv), qc)[..., 1:]


# --------- torch quaternion helpers (w-first) ---------
def qmul_t(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def qrot_t(q, v):
    # v: (...,3) tensor
    zeros = torch.zeros(q.shape[:-1] + (1,), dtype=q.dtype)
    qv = torch.cat([zeros, v.expand(q.shape[:-1] + (3,))], -1)
    qc = q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
    return qmul_t(qmul_t(q, qv), qc)[..., 1:]


def gm(s, c2):
    """Geman-McClure robust loss on squared residual s (saturates at 1)."""
    return (s / (s + c2)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--stride", type=int, default=0, help="frame stride (0 = auto ~4000 frames)")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l1", type=float, default=2e-3, help="sparsity weight on per-sensor params")
    ap.add_argument("--harmonics", type=int, default=2, choices=[1, 2], help="1=hard-iron only")
    ap.add_argument("--w-knee", type=float, default=0.5)
    ap.add_argument("--w-trunk", type=float, default=0.3)
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, bone_axis = load_skeleton(def_path)

    d = np.load(args.infile, allow_pickle=True)
    Q = d['quats'].astype(np.float64)
    T = Q.shape[0]
    stride = args.stride or max(1, T // 4000)
    Q = Q[::stride]
    n = Q.shape[0]
    print(f"{Path(args.infile).name}: {T} frames, using {n} (stride {stride})")

    Gw = fk_world(Q, parent, order)                       # (n,37,4) measured world, constant

    # per-sensor heading psi from the measured world +X axis azimuth (constant)
    xw = qrot_np(Gw, np.array([1.0, 0.0, 0.0]))            # (n,37,3)
    psi = np.arctan2(xw[..., 2], xw[..., 0])               # (n,37)

    # heading coverage per sensor (how identifiable its harmonics are)
    def coverage(p):
        h, _ = np.histogram(p, bins=12, range=(-math.pi, math.pi))
        return (h > n / 200).sum()                         # bins visited with >0.5% of frames

    # pelvis lateral axis (mirror-plane normal), horizontal, from the (fixed) pelvis world
    latv = qrot_np(Gw[:, PELV], np.array([1.0, 0.0, 0.0]))
    latv = latv - UP * (latv @ UP)[:, None]
    latv = latv / (np.linalg.norm(latv, axis=-1, keepdims=True) + 1e-9)
    M = np.eye(3)[None] - 2.0 * latv[:, :, None] * latv[:, None, :]   # (n,3,3) mirror

    # torch tensors
    Gt = torch.tensor(Gw)
    psit = torch.tensor(psi)
    Mt = torch.tensor(M)
    axis_t = {j: torch.tensor(bone_axis[j]) for j in range(37)}

    H = args.harmonics
    nparam = 1 + 2 * H                                     # A,(B,C),(D,E)
    P = torch.zeros(37, nparam, dtype=torch.float64, requires_grad=True)
    fixed = torch.ones(37, 1, dtype=torch.float64)
    fixed[BODY] = 0.0
    fixed[PELV] = 0.0                                      # gauge: pelvis & body pinned to 0

    def delta(idx):
        """delta_i(psi) for one sensor index, shape (n,)."""
        p = P[idx] * fixed[idx]
        out = p[0] * torch.ones_like(psit[:, idx])
        out = out + p[1] * torch.sin(psit[:, idx]) + p[2] * torch.cos(psit[:, idx])
        if H == 2:
            out = out + p[3] * torch.sin(2 * psit[:, idx]) + p[4] * torch.cos(2 * psit[:, idx])
        return out

    def corrected_world(idx):
        """Yaw(-delta_i) * world_i about world Y."""
        half = -delta(idx) / 2.0
        z = torch.zeros_like(half)
        qy = torch.stack([torch.cos(half), z, torch.sin(half), z], -1)   # (n,4)
        return qmul_t(qy, Gt[:, idx])

    def bone_dir(seg):
        return qrot_t(corrected_world(seg), axis_t[seg])

    def heading_of(seg):
        x = qrot_t(corrected_world(seg), axis_t[BODY] * 0 + torch.tensor([1.0, 0.0, 0.0]))
        return torch.atan2(x[..., 2], x[..., 0])

    def losses():
        # 1. bilateral symmetry (robust)
        sym = 0.0
        for (Ls, _), (Rs, _) in SYM_PAIRS:
            dL = bone_dir(Ls)
            dR = bone_dir(Rs)
            mdL = torch.einsum('nij,nj->ni', Mt, dL)
            sym = sym + gm(((mdL - dR) ** 2).sum(-1), 0.2 ** 2)
        sym = sym / len(SYM_PAIRS)
        # 2. knee hinge anatomy: shin out-of-thigh-lateral-plane component
        knee = 0.0
        for (shin, _), thigh in KNEES:
            shin_d = bone_dir(shin)
            thigh_lat = qrot_t(corrected_world(thigh), torch.tensor([1.0, 0.0, 0.0]))
            knee = knee + ((shin_d * thigh_lat).sum(-1) ** 2).mean()
        knee = knee / len(KNEES)
        # 3. trunk axial consistency (robust): consecutive spine headings agree off-twist
        trunk = 0.0
        for a, b in zip(TRUNK[:-1], TRUNK[1:]):
            dd = heading_of(b) - heading_of(a)
            dd = torch.atan2(torch.sin(dd), torch.cos(dd))
            trunk = trunk + gm(dd ** 2, math.radians(15) ** 2)
        trunk = trunk / (len(TRUNK) - 1)
        return sym, knee, trunk

    sym0, knee0, trunk0 = (float(x) for x in losses())

    opt = torch.optim.Adam([P], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        sym, knee, trunk = losses()
        spars = (P * fixed).abs().sum()
        loss = sym + args.w_knee * knee + args.w_trunk * trunk + args.l1 * spars
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            print(f"  it{it:4d}  sym={float(sym):.4f} knee={float(knee):.4f} "
                  f"trunk={float(trunk):.4f} l1={float(spars):.3f}")

    sym1, knee1, trunk1 = (float(x) for x in losses())

    # ---- report ----
    Pn = (P * fixed).detach().numpy()
    rows = []
    for j in range(37):
        if not np.any(Pn[j]):
            continue
        A = Pn[j, 0]
        h1 = math.hypot(Pn[j, 1], Pn[j, 2])
        h2 = math.hypot(Pn[j, 3], Pn[j, 4]) if H == 2 else 0.0
        dser = delta(j).detach().numpy()
        rms = math.sqrt(np.mean(dser ** 2))
        rows.append((rms, j, A, h1, h2, coverage(psi[:, j])))
    rows.sort(reverse=True)

    print("\nPer-sensor heading-deviation (degrees), ranked.  cov = heading bins/12 visited")
    print(f"{'sensor':24s} {'RMS':>6} {'const':>7} {'1stH':>6} {'2ndH':>6} {'cov':>4}")
    for rms, j, A, h1, h2, cov in rows:
        nm = IDX_TO_NAME.get(j, f'#{j}')
        print(f"  {nm:22s} {math.degrees(rms):6.1f} {math.degrees(A):+7.1f} "
              f"{math.degrees(h1):6.1f} {math.degrees(h2):6.1f} {cov:4d}")

    print(f"\nresiduals (lower=better)   before -> after")
    print(f"  symmetry (robust) : {sym0:.4f} -> {sym1:.4f}")
    print(f"  knee hinge        : {knee0:.4f} -> {knee1:.4f}")
    print(f"  trunk axial       : {trunk0:.4f} -> {trunk1:.4f}")
    print("\nNOTE: diagnostic only -- no correction written. Pelvis & Body pinned (gauge).")


if __name__ == "__main__":
    main()
