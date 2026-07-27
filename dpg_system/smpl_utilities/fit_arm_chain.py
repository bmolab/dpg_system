"""Lockstep down-the-chain arm corrector: shoulders first, then elbows given corrected shoulders.

Why down-the-chain: a joint's twist/angle is a function of BOTH its segments, so the distal
signal is only attributable once the parent is fixed. And both arms are magnetized, so neither
can be the symmetry reference -- we correct LEFT and RIGHT in lockstep and mirror corrected-vs-
corrected. Each level is pinned by the constraints appropriate to it:

  STAGE 1 shoulders (upper arm; ball joint but NOT unconstrained):
     * humeral axial-twist limit (generous; catches gross unnatural twist, e.g. right arm @4318)
     * hanging-arm rest pose in the relax window (flared+forward) -> drives the tilt dof
     * heading-locked spurious motion of the upper arm vs body turn rate
     * L/R full-orientation symmetry at labelled moments (both corrected)
  STAGE 2 elbows (forearm; hinge): given the corrected upper arms, the elbow twist/out-of-plane
     signal is finally clean and attributable to the forearm sensor
     * tight elbow twist ~0 + out-of-plane ~0
     * heading-locked motion of the forearm vs the corrected upper arm
     * L/R forearm symmetry

Per sensor: yaw curve delta(psi)=A+B sinpsi+C cospsi+D sin2psi+E cos2psi (about world-Y) + constant
tilt (cx,cz). Writes *_chainfix.npz. Originals untouched.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np, qmul_t,
                                         qrot_t, gm, PELV, LSH, LEL, LWR, RSH, REL, RWR,
                                         LBLADE, RBLADE)

UP = np.array([0.0, 1.0, 0.0])
XW = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)


def qconj_t(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def twist_angle_t(qloc, axis):
    proj = (qloc[..., 1:] * axis).sum(-1)
    return 2.0 * torch.atan2(proj, qloc[..., 0])


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def parse_ranges(s):
    return [tuple(int(v) for v in r.split(":")) for r in s.split(",")] if s else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("-o", "--out")
    ap.add_argument("--sym", default="7880:8500,20660:20953,10560:10692")
    ap.add_argument("--relax", default="14869:15556")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--turn-rate", type=float, default=20.0)
    ap.add_argument("--flare", type=float, default=12.0)
    ap.add_argument("--forward", type=float, default=8.0)
    ap.add_argument("--sh-twist-limit", type=float, default=75.0, help="humeral axial twist limit (deg)")
    ap.add_argument("--el-twist-limit", type=float, default=12.0, help="elbow axial twist limit (deg)")
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 100.0
    Gw = fk_world(quats, parent, order)

    sym_idx = np.concatenate([np.arange(a, b) for a, b in parse_ranges(args.sym)]).astype(int)
    relax_idx = np.concatenate([np.arange(a, b) for a, b in parse_ranges(args.relax)]).astype(int)

    def yaw_np(j):
        v = qrot_np(Gw[:, j], np.array([1.0, 0.0, 0.0])); return np.arctan2(v[:, 2], v[:, 0])
    psi = {j: torch.tensor(yaw_np(j)) for j in (LSH, RSH, LEL, REL)}
    psi_pel = yaw_np(PELV)

    # mirror plane + flared/forward rest targets (left flares +lateral, right -lateral)
    lat = qrot_np(Gw[:, PELV], np.array([1.0, 0.0, 0.0])); lat -= UP * (lat @ UP)[:, None]
    lat /= np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9
    fwd = qrot_np(Gw[:, PELV], np.array([0.0, 0.0, 1.0])); fwd -= UP * (fwd @ UP)[:, None]
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9
    tf, tfw = math.tan(math.radians(args.flare)), math.tan(math.radians(args.forward))
    restL = np.array([0., -1., 0.]) + tf * lat + tfw * fwd
    restR = np.array([0., -1., 0.]) - tf * lat + tfw * fwd
    restL /= np.linalg.norm(restL, axis=-1, keepdims=True); restR /= np.linalg.norm(restR, axis=-1, keepdims=True)

    G = {j: torch.tensor(Gw[:, j]) for j in (LSH, RSH, LEL, REL, LWR, RWR, LBLADE, RBLADE, PELV)}
    M = torch.tensor(np.eye(3)[None] - 2.0 * lat[:, :, None] * lat[:, None, :])
    ax = {j: torch.tensor(bax[j]) for j in (LSH, RSH, LEL, REL)}
    SEC = [torch.tensor([0., 0., 1.]), torch.tensor([0., 1., 0.])]
    sym_m = torch.zeros(T, dtype=torch.float64); sym_m[sym_idx] = 1.0
    relax_m = torch.zeros(T, dtype=torch.float64); relax_m[relax_idx] = 1.0
    restL_t, restR_t = torch.tensor(restL), torch.tensor(restR)
    psidot = wrap(torch.tensor(np.roll(psi_pel, -1) - psi_pel))
    turn = (psidot.abs() > math.radians(args.turn_rate) / fps) & (torch.arange(T) < T - 1)
    NB = 12

    def corr(meas, p, psi_s):
        delta = p[0] + p[1]*torch.sin(psi_s) + p[2]*torch.cos(psi_s) + p[3]*torch.sin(2*psi_s) + p[4]*torch.cos(2*psi_s)
        h = -delta / 2.0; z = torch.zeros_like(h)
        qy = torch.stack([torch.cos(h), z, torch.sin(h), z], -1)
        hx, hz = -p[5] / 2.0, -p[6] / 2.0; zc = torch.zeros_like(hx)
        qx = torch.stack([torch.cos(hx), torch.sin(hx), zc, zc])
        qzz = torch.stack([torch.cos(hz), zc, zc, torch.sin(hz)])
        tilt = qmul_t(qx, qzz).expand(meas.shape)
        return qmul_t(tilt, qmul_t(qy, meas))

    def sym_pair(lc, Ls, rc, Rs):
        s = 0.0
        for la, ra in [(ax[Ls], ax[Rs]), (SEC[0], SEC[0]), (SEC[1], SEC[1])]:
            r2 = ((torch.einsum('nij,nj->ni', M, qrot_t(lc, la)) - qrot_t(rc, ra)) ** 2).sum(-1)
            s = s + gm(r2[sym_m > 0], 0.3 ** 2)
        return s / 3.0

    def heading_locked(seg_world, ref_yaw):
        v = qrot_t(seg_world, XW); sy = torch.atan2(v[..., 2], v[..., 0])
        r = wrap(sy - ref_yaw); dr = wrap(torch.roll(r, -1) - r)
        mot = 0.0
        binid = ((sy.detach() + math.pi) // (2 * math.pi / NB)).long() % NB
        for b in range(NB):
            sel = turn & (binid == b)
            if sel.sum() > 30:
                x = psidot[sel]; mot = mot + (x @ dr[sel]) ** 2 / ((x @ x) + 1e-9)
        return mot

    def report_twists(tag, QL):
        Gc = fk_world(QL, parent, order)
        a, b = 4308, 4330
        for nm, sh, fa, bl in [("L", LSH, LEL, LBLADE), ("R", RSH, REL, RBLADE)]:
            shtw = np.degrees(2*np.arctan2((qmul_np(Gc[:, bl]*np.array([1,-1,-1,-1.]), Gc[:, sh])[:, 1:]*bax[sh]).sum(-1),
                                           qmul_np(Gc[:, bl]*np.array([1,-1,-1,-1.]), Gc[:, sh])[:, 0]))
            eltw = np.degrees(2*np.arctan2((qmul_np(Gc[:, sh]*np.array([1,-1,-1,-1.]), Gc[:, fa])[:, 1:]*bax[fa]).sum(-1),
                                           qmul_np(Gc[:, sh]*np.array([1,-1,-1,-1.]), Gc[:, fa])[:, 0]))
            print(f"    {tag} {nm}: shoulder-twist {shtw[a:b].mean():+5.0f}  elbow-twist {eltw[a:b].mean():+5.0f}")

    # ---------------- STAGE 1: shoulders (lockstep L/R) ----------------
    Psh = torch.zeros(2, 7, dtype=torch.float64, requires_grad=True)

    def shoulder_terms():
        upL, upR = corr(G[LSH], Psh[0], psi[LSH]), corr(G[RSH], Psh[1], psi[RSH])
        twL = twist_angle_t(qmul_t(qconj_t(G[LBLADE]), upL), ax[LSH])
        twR = twist_angle_t(qmul_t(qconj_t(G[RBLADE]), upR), ax[RSH])
        lim = math.radians(args.sh_twist_limit)
        twist = (torch.relu(twL.abs() - lim) ** 2).mean() + (torch.relu(twR.abs() - lim) ** 2).mean()
        hang = ((((qrot_t(upL, ax[LSH]) - restL_t) ** 2).sum(-1)[relax_m > 0]).mean()
                + (((qrot_t(upR, ax[RSH]) - restR_t) ** 2).sum(-1)[relax_m > 0]).mean())
        sym = sym_pair(upL, LSH, upR, RSH)
        mot = heading_locked(upL, torch.tensor(psi_pel)) + heading_locked(upR, torch.tensor(psi_pel))
        return sym, hang, twist, mot

    s0 = [float(x) for x in shoulder_terms()]
    b = [v + 1e-9 for v in s0]
    opt = torch.optim.Adam([Psh], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        sym, hang, twist, mot = shoulder_terms()
        loss = sym / b[0] + hang / b[1] + 3.0 * twist / (math.radians(args.sh_twist_limit) ** 2) \
            + mot / b[3] + 0.5 * (Psh[:, 1:5] ** 2).sum() + 0.2 * (Psh ** 2).sum()
        loss.backward(); opt.step()
    s1 = [float(x) for x in shoulder_terms()]
    print("STAGE 1 shoulders  sym {:.3f}->{:.3f}  hang {:.3f}->{:.3f}  twist {:.4f}->{:.4f}  mot {:.4f}->{:.4f}".format(
        s0[0], s1[0], s0[1], s1[1], s0[2], s1[2], s0[3], s1[3]))
    Pshn = Psh.detach()
    upL_c = corr(G[LSH], Pshn[0], psi[LSH]).detach()
    upR_c = corr(G[RSH], Pshn[1], psi[RSH]).detach()

    # ---------------- STAGE 2: elbows given corrected shoulders ----------------
    Pel = torch.zeros(2, 7, dtype=torch.float64, requires_grad=True)
    sh_yawL = torch.atan2(qrot_t(upL_c, XW)[..., 2], qrot_t(upL_c, XW)[..., 0])
    sh_yawR = torch.atan2(qrot_t(upR_c, XW)[..., 2], qrot_t(upR_c, XW)[..., 0])

    def elbow_terms():
        faL, faR = corr(G[LEL], Pel[0], psi[LEL]), corr(G[REL], Pel[1], psi[REL])
        elL = twist_angle_t(qmul_t(qconj_t(upL_c), faL), ax[LEL])
        elR = twist_angle_t(qmul_t(qconj_t(upR_c), faR), ax[REL])
        lim = math.radians(args.el_twist_limit)
        twist = (torch.relu(elL.abs() - lim) ** 2).mean() + (torch.relu(elR.abs() - lim) ** 2).mean()
        sym = sym_pair(faL, LEL, faR, REL)
        mot = heading_locked(faL, sh_yawL) + heading_locked(faR, sh_yawR)
        return sym, twist, mot

    e0 = [float(x) for x in elbow_terms()]
    be = [v + 1e-9 for v in e0]
    opt = torch.optim.Adam([Pel], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        sym, twist, mot = elbow_terms()
        loss = sym / be[0] + 3.0 * twist / (math.radians(args.el_twist_limit) ** 2) + mot / be[2] \
            + 0.5 * (Pel[:, 1:5] ** 2).sum() + 0.2 * (Pel ** 2).sum()
        loss.backward(); opt.step()
    e1 = [float(x) for x in elbow_terms()]
    print("STAGE 2 elbows     sym {:.3f}->{:.3f}  twist {:.4f}->{:.4f}  mot {:.4f}->{:.4f}".format(
        e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]))
    Peln = Pel.detach()
    for ri, nm in [(0, 'Left'), (1, 'Right')]:
        ps, pe = Pshn[ri].numpy(), Peln[ri].numpy()
        print(f"  {nm}: shoulder yaw {math.degrees(ps[0]):+.0f}/{math.degrees(math.hypot(ps[1],ps[2])):.0f}/"
              f"{math.degrees(math.hypot(ps[3],ps[4])):.0f} tilt {math.degrees(math.hypot(ps[5],ps[6])):.0f}"
              f"   elbow yaw {math.degrees(pe[0]):+.0f}/{math.degrees(math.hypot(pe[1],pe[2])):.0f}/"
              f"{math.degrees(math.hypot(pe[3],pe[4])):.0f} tilt {math.degrees(math.hypot(pe[5],pe[6])):.0f}")

    # ---------------- apply ----------------
    def corr_np(meas, p, psi_s):
        delta = p[0] + p[1]*np.sin(psi_s) + p[2]*np.cos(psi_s) + p[3]*np.sin(2*psi_s) + p[4]*np.cos(2*psi_s)
        h = -delta / 2.0
        qy = np.stack([np.cos(h), np.zeros_like(h), np.sin(h), np.zeros_like(h)], -1)
        hx, hz = -p[5]/2.0, -p[6]/2.0
        qt = qmul_np(np.array([math.cos(hx), math.sin(hx), 0., 0.]), np.array([math.cos(hz), 0., 0., math.sin(hz)]))
        return qmul_np(np.broadcast_to(qt, meas.shape), qmul_np(qy, meas))

    psn = {j: yaw_np(j) for j in (LSH, RSH, LEL, REL)}
    upLc = corr_np(Gw[:, LSH], Pshn[0].numpy(), psn[LSH]); upRc = corr_np(Gw[:, RSH], Pshn[1].numpy(), psn[RSH])
    faLc = corr_np(Gw[:, LEL], Peln[0].numpy(), psn[LEL]); faRc = corr_np(Gw[:, REL], Peln[1].numpy(), psn[REL])
    Qc = quats.copy()
    Qc[:, LSH] = qmul_np(Gw[:, LBLADE]*np.array([1,-1,-1,-1.]), upLc)
    Qc[:, RSH] = qmul_np(Gw[:, RBLADE]*np.array([1,-1,-1,-1.]), upRc)
    Qc[:, LEL] = qmul_np(upLc*np.array([1,-1,-1,-1.]), faLc)
    Qc[:, REL] = qmul_np(upRc*np.array([1,-1,-1,-1.]), faRc)
    Qc[:, LWR] = qmul_np(faLc*np.array([1,-1,-1,-1.]), Gw[:, LWR])
    Qc[:, RWR] = qmul_np(faRc*np.array([1,-1,-1,-1.]), Gw[:, RWR])

    print("\nf4308-4330 shoulder/elbow twist (deg):")
    report_twists("orig  ", quats)
    report_twists("chain ", Qc)

    if not args.no_apply:
        out = args.out or args.infile.replace(".npz", "_chainfix.npz")
        save = {k: d[k] for k in d.files}; save['quats'] = Qc
        np.savez(out, **save); print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
