"""Soft-fusion corrector for the LEFT ARM CHAIN (shoulder + forearm).

Not minimal-correction and not any single metric: every guidepost is SOFT evidence and the JOINT
optimum is better than any one alone, because their confounds are independent --
  * symmetry fails away from symmetric moments  -> motion fills the gaps;
  * motion's heading-locked signal is subtle     -> symmetry anchors the absolute level;
  * VPoser rewards plausible-but-wrong poses     -> motion penalizes the impossible trajectory;
  * a minimality prior keeps unevidenced frames from drifting.

Per corrected sensor (LeftShoulder, LeftElbow): a heading-yaw deviation curve about world-vertical
delta(psi)=A+B sinpsi+C cospsi+D sin2psi+E cos2psi (the magnetometer error) PLUS a constant tilt
(cx,cz about world X,Z) -- a mount/calibration error that is NOT heading-dependent and that a yaw
correction structurally cannot fix (this is why the hanging-arm window only moves once tilt exists).

Soft terms (each robust; normalized by its delta=0 baseline so trust weights are comparable):
  symmetry  : full-orientation L/R mirror at labelled symmetric moments (right arm = reference)
  hanging   : relaxed-window upper arm vertical -> drives the TILT dof
  motion    : heading-locked spurious motion (body-relative forearm yaw-rate vs body yaw-rate,
              per heading bin) -> the magnetization signature, style-independent, across all frames
  smoothness: low-order delta (magnetization is smooth in heading)
  minimality: soft pull of all params toward 0
  hinge     : soft elbow out-of-plane barrier (relative between the two corrected sensors)
Velocity-spike + swept-through-torso are REPORTED as diagnostics (a smooth delta can't remove a
glitch; the capsule is front/back-blind), not fit. --vposer adds a low-trust naturalness prior.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np, qmul_t,
                                         qrot_t, gm, PELV, LSH, LEL, LWR, RSH, REL,
                                         LBLADE, SYM_PAIRS)

UP = np.array([0.0, 1.0, 0.0])
DOWN_T = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
SEC = [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])]
ARM_PAIRS = SYM_PAIRS[:2]                       # (upper arm, forearm) L/R index pairs


def parse_ranges(s):
    return [tuple(int(v) for v in r.split(":")) for r in s.split(",")] if s else []


def qconj_t(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def rotvec_to_quat_t(v):
    a = v.norm(dim=-1, keepdim=True)
    axis = v / a.clamp_min(1e-9)
    return torch.cat([torch.cos(a / 2), torch.sin(a / 2) * axis], -1)


def discover_hinge_normal(Gmeas, prox, dist, bone_axis):
    bone = bone_axis[prox]
    cands = [e for e in np.eye(3) if abs(float(e @ bone)) < 0.9]
    dd = qrot_np(Gmeas[:, dist], bone_axis[dist])
    return min(cands, key=lambda c: np.abs((dd * qrot_np(Gmeas[:, prox], c)).sum(-1)).mean())


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def twist_angle_t(qloc, axis):
    """Swing-twist: signed twist angle (rad) of a local quat about a local axis."""
    proj = (qloc[..., 1:] * axis).sum(-1)
    return 2.0 * torch.atan2(proj, qloc[..., 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("-o", "--out")
    ap.add_argument("--sym", default="7880:8500,20660:20953,10560:10692")
    ap.add_argument("--relax", default="14869:15556")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--turn-rate", type=float, default=20.0, help="body turn threshold for motion term (deg/s)")
    ap.add_argument("--flare", type=float, default=12.0, help="relaxed arm abduction away from hip (deg)")
    ap.add_argument("--forward", type=float, default=8.0, help="relaxed arm forward bias (deg; +fwd, -back)")
    ap.add_argument("--trust-sym", type=float, default=1.0)
    ap.add_argument("--trust-relax", type=float, default=1.0)
    ap.add_argument("--trust-motion", type=float, default=1.0)
    ap.add_argument("--trust-hinge", type=float, default=1.0)
    ap.add_argument("--trust-twist", type=float, default=3.0, help="penalize introduced elbow axial twist")
    ap.add_argument("--twist-limit", type=float, default=15.0, help="allowed elbow axial twist (deg)")
    ap.add_argument("--l2-smooth", type=float, default=0.5)
    ap.add_argument("--l2-min", type=float, default=0.2)
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args()

    parent, order, bone_axis = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 100.0
    Gw = fk_world(quats, parent, order)
    hinge_n = discover_hinge_normal(Gw, LSH, LEL, bone_axis)

    sym_idx = np.concatenate([np.arange(a, b) for a, b in parse_ranges(args.sym)]).astype(int)
    relax_idx = np.concatenate([np.arange(a, b) for a, b in parse_ranges(args.relax)]).astype(int)

    # measured sensor headings (for the deviation-curve argument)
    def yaw_of(j):
        v = qrot_np(Gw[:, j], np.array([1.0, 0.0, 0.0]))
        return np.arctan2(v[:, 2], v[:, 0])
    psiL_sh, psiL_fa, psi_pel = yaw_of(LSH), yaw_of(LEL), yaw_of(PELV)

    # mirror plane (pelvis lateral, horizontal)
    lat = qrot_np(Gw[:, PELV], np.array([1.0, 0.0, 0.0]))      # pelvis +X = body-left
    lat = lat - UP * (lat @ UP)[:, None]; lat /= np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9
    Mfull = np.eye(3)[None] - 2.0 * lat[:, :, None] * lat[:, None, :]

    # relaxed LEFT-arm rest direction: mostly down, flared out (+lateral=body-left) and forward.
    # Arms rarely hang dead-vertical, so DOWN is the wrong target and over-tilts (breaks symmetry).
    fwd = qrot_np(Gw[:, PELV], np.array([0.0, 0.0, 1.0]))
    fwd = fwd - UP * (fwd @ UP)[:, None]; fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9
    rest_L = (np.array([0.0, -1.0, 0.0]) + math.tan(math.radians(args.flare)) * lat
              + math.tan(math.radians(args.forward)) * fwd)
    rest_L /= np.linalg.norm(rest_L, axis=-1, keepdims=True)

    # tensors
    G = {j: torch.tensor(Gw[:, j]) for j in (LSH, LEL, LWR, RSH, REL)}
    psit = {LSH: torch.tensor(psiL_sh), LEL: torch.tensor(psiL_fa)}
    M = torch.tensor(Mfull)
    ax = {j: torch.tensor(bone_axis[j]) for j in (LSH, LEL, RSH, REL)}
    hinge_t = torch.tensor(hinge_n)
    sym_m = torch.zeros(T, dtype=torch.float64); sym_m[sym_idx] = 1.0
    relax_m = torch.zeros(T, dtype=torch.float64); relax_m[relax_idx] = 1.0
    rest_L_t = torch.tensor(rest_L)

    # motion: body yaw rate (rad/frame) + turn mask + heading-bin ids on the forearm
    psidot = wrap(torch.tensor(np.roll(psi_pel, -1) - psi_pel))
    turn = (psidot.abs() > math.radians(args.turn_rate) / fps) & (torch.arange(T) < T - 1)
    NB = 12
    fa_bin = ((torch.tensor(psiL_fa) + math.pi) // (2 * math.pi / NB)).long() % NB

    P = torch.zeros(2, 7, dtype=torch.float64, requires_grad=True)   # rows: LSH, LEL
    SENS = [LSH, LEL]

    def corrected(j_row):
        j = SENS[j_row]; p = P[j_row]
        psi = psit[j]
        delta = p[0] + p[1]*torch.sin(psi) + p[2]*torch.cos(psi) + p[3]*torch.sin(2*psi) + p[4]*torch.cos(2*psi)
        half = -delta / 2.0; z = torch.zeros_like(half)
        qy = torch.stack([torch.cos(half), z, torch.sin(half), z], -1)
        # constant tilt as world-X(-cx) then world-Z(-cz) quats (differentiable at 0, unlike a rotvec)
        hx, hz = -p[5] / 2.0, -p[6] / 2.0; zc = torch.zeros_like(hx)
        qx = torch.stack([torch.cos(hx), torch.sin(hx), zc, zc])
        qz = torch.stack([torch.cos(hz), zc, zc, torch.sin(hz)])
        tilt = qmul_t(qx, qz).expand(T, 4)
        return qmul_t(tilt, qmul_t(qy, G[j]))

    def terms():
        upL, faL = corrected(0), corrected(1)
        # symmetry (full orientation, right arm = reference): per pair, bone axis uses the
        # mirrored L/R pair, the two secondary axes are shared (their X-component is 0)
        sym = 0.0
        for Ls, lc, Rs, rc in [(LSH, upL, RSH, G[RSH]), (LEL, faL, REL, G[REL])]:
            for la, ra in [(ax[Ls], ax[Rs]), (SEC[0], SEC[0]), (SEC[1], SEC[1])]:
                lv = qrot_t(lc, la); rv = qrot_t(rc, ra)
                r2 = ((torch.einsum('nij,nj->ni', M, lv) - rv) ** 2).sum(-1)
                sym = sym + gm(r2[sym_m > 0], 0.3 ** 2)
        sym = sym / 6.0
        # hanging-arm rest pose (drives tilt) on relax frames: flared+forward, not dead-vertical
        upL_dir = qrot_t(upL, ax[LSH])
        relax = (((upL_dir - rest_L_t) ** 2).sum(-1)[relax_m > 0]).mean()
        # heading-locked spurious motion: body-relative forearm yaw-rate vs body yaw-rate, per bin
        fa_dir = qrot_t(faL, torch.tensor([1.0, 0.0, 0.0]))
        psi_fa_c = torch.atan2(fa_dir[..., 2], fa_dir[..., 0])
        r = wrap(psi_fa_c - torch.tensor(psi_pel))
        dr = wrap(torch.roll(r, -1) - r)
        mot = 0.0
        for b in range(NB):
            sel = turn & (fa_bin == b)
            if sel.sum() > 30:
                x = psidot[sel]; y = dr[sel]
                mot = mot + (x @ y) ** 2 / ((x @ x) + 1e-9)      # heading-locked explained energy
        # elbow hinge out-of-plane (relative, both corrected)
        oop = (qrot_t(faL, ax[LEL]) * qrot_t(upL, hinge_t)).sum(-1).abs()
        hinge = (torch.relu(oop - math.sin(math.radians(15))) ** 2).mean()
        # elbow axial twist: the elbow has ~no pronation DOF, so a correction that injects twist
        # is wrong -- this barrier acts at EVERY heading, reining in unanchored-side extrapolation
        elbow_loc = qmul_t(qconj_t(upL), faL)
        etw = twist_angle_t(elbow_loc, ax[LEL])
        twist = (torch.relu(etw.abs() - math.radians(args.twist_limit)) ** 2).mean()
        return sym, relax, mot, hinge, twist

    with torch.no_grad():
        s0, r0, m0, h0, t0 = (float(x) for x in terms())
    base = dict(sym=s0 + 1e-9, relax=r0 + 1e-9, mot=m0 + 1e-9, hinge=h0 + 1e-9)
    tref = math.radians(args.twist_limit) ** 2                       # normalize twist barrier to O(1)
    print(f"baselines (delta=0):  sym {s0:.4f}  relax {r0:.4f}  motion {m0:.5f}  hinge {h0:.4f}  twist {t0:.4f}")

    opt = torch.optim.Adam([P], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        sym, relax, mot, hinge, twist = terms()
        loss = (args.trust_sym * sym / base['sym'] + args.trust_relax * relax / base['relax']
                + args.trust_motion * mot / base['mot'] + args.trust_hinge * hinge / base['hinge']
                + args.trust_twist * twist / tref
                + args.l2_smooth * (P[:, 1:5] ** 2).sum() + args.l2_min * (P ** 2).sum())
        loss.backward(); opt.step()
        if it % 150 == 0 or it == args.iters - 1:
            print(f"  it{it:4d}  sym {float(sym):.4f}  relax {float(relax):.4f}  "
                  f"motion {float(mot):.5f}  hinge {float(hinge):.4f}  twist {float(twist):.4f}")

    with torch.no_grad():
        s1, r1, m1, h1, t1 = (float(x) for x in terms())
    Pn = P.detach().numpy()
    print(f"\n              before -> after   (lower=better)")
    for nm, a0, a1 in [('symmetry', s0, s1), ('hanging(tilt)', r0, r1), ('motion-locked', m0, m1),
                       ('hinge', h0, h1), ('elbow-twist', t0, t1)]:
        print(f"  {nm:14s} {a0:.4f} -> {a1:.4f}")
    for ri, nm in [(0, 'LeftShoulder'), (1, 'LeftElbow')]:
        p = Pn[ri]
        print(f"  {nm:13s} yaw: const {math.degrees(p[0]):+5.1f} 1stH {math.degrees(math.hypot(p[1],p[2])):4.1f} "
              f"2ndH {math.degrees(math.hypot(p[3],p[4])):4.1f}   tilt {math.degrees(math.hypot(p[5],p[6])):4.1f} deg")

    if args.no_apply:
        return
    # apply over full take (numpy), recompute LeftShoulder/Elbow/Wrist local quats
    def corr_np(j, psi):
        p = Pn[0] if j == LSH else Pn[1]
        delta = p[0] + p[1]*np.sin(psi) + p[2]*np.cos(psi) + p[3]*np.sin(2*psi) + p[4]*np.cos(2*psi)
        half = -delta / 2.0
        qy = np.stack([np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half)], -1)
        hx, hz = -p[5] / 2.0, -p[6] / 2.0
        qx = np.array([math.cos(hx), math.sin(hx), 0.0, 0.0])
        qz = np.array([math.cos(hz), 0.0, 0.0, math.sin(hz)])
        qt = qmul_np(qx, qz)
        return qmul_np(np.broadcast_to(qt, Gw[:, j].shape), qmul_np(qy, Gw[:, j]))
    upL_c = corr_np(LSH, psiL_sh); faL_c = corr_np(LEL, psiL_fa)
    Qc = quats.copy()
    Qc[:, LSH] = qmul_np(Gw[:, LBLADE] * np.array([1, -1, -1, -1.0]), upL_c)
    Qc[:, LEL] = qmul_np(upL_c * np.array([1, -1, -1, -1.0]), faL_c)
    Qc[:, LWR] = qmul_np(faL_c * np.array([1, -1, -1, -1.0]), Gw[:, LWR])
    out = args.out or args.infile.replace(".npz", "_armfuse.npz")
    save = {k: d[k] for k in d.files}; save['quats'] = Qc
    np.savez(out, **save)
    print(f"\nwrote corrected take -> {out}")


if __name__ == "__main__":
    main()
