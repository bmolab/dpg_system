"""Whole-body magnetometer heading-deviation estimator (probabilistic / joint MAP).

Direction (vs. correcting the arms by one metric at a time): fit a per-sensor compass-deviation
yaw curve delta_i(psi) = A + B sin psi + C cos psi + D sin 2psi + E cos 2psi for EVERY sensor
at once, under several soft guideposts plus a sparsity prior, so the fit decides which sensors
are magnetized and how much. See diag_magnetometer_deviation.py for the model derivation.

Guideposts (all active simultaneously; none has to be sufficient alone):
  * sym anchors (--sym)   : flagged bilaterally-symmetric held poses -> HARD L/R mirror match.
  * relax anchor (--relax): flagged arms-down rest -> upper arms HARD vertical + symmetric.
  * background symmetry   : faint robust (Geman-McClure) L/R over all frames (a weak prior only).
  * anatomy guardrail     : ONE-SIDED barriers at GENEROUS limits -- knees/elbows are hinges and
                            can't leave their (data-discovered) flexion plane past a margin.
                            This is the only guidepost that acts at every heading, so it is what
                            identifies the heading harmonics on takes whose symmetric poses all
                            sit at one facing.
  * trunk axial           : faint robust consecutive-spine-heading consistency.
  * priors                : L1 sparsity on per-sensor params (-> un-magnetized sensors ~0) and a
                            small L2 on the harmonics (weakly identified -> stay near 0 unless
                            anatomy demands them).
Gauge: pelvis (mirror reference) and Body pinned to delta=0; everything is relative to them.

--apply writes a corrected take (each sensor's world yawed by -delta_i, locals recomputed);
positions are left untouched (orientation-only fix).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from diag_magnetometer_deviation import (
    load_skeleton, fk_world, qrot_np, qmul_np, qmul_t, qrot_t, gm,
    BODY, PELV, LSH, LEL, LWR, RSH, REL, RWR, LBLADE, RBLADE,
    LHIP, LKNEE, LANK, RHIP, RKNEE, RANK, LTOE, RTOE,
    SPINEP, LOWV, MIDV, UPPV, SKULL, IDX_TO_NAME, SYM_PAIRS, TRUNK)

UP = np.array([0.0, 1.0, 0.0])
# hinge joints: (proximal seg, distal seg) -- distal bone must stay in proximal's flexion plane
HINGES = [(LHIP, LKNEE), (RHIP, RKNEE), (LSH, LEL), (RSH, REL)]
# trunk axial-consistency chain -- LOWER/MID spine only; neck+head turn independently in abhinaya
TRUNK_FIT = [PELV, SPINEP, LOWV, MIDV]
# secondary local axes (besides the per-side bone axis) used for FULL-orientation symmetry; their
# X-component is 0 so the L/R mirror convention (right local = left with X negated) leaves them equal
SEC_AXES = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])]


def parse_ranges(s):
    return [tuple(int(v) for v in r.split(":")) for r in s.split(",")] if s else []


def discover_hinge_normal(Gmeas, prox, dist, bone_axis):
    """The hinge axis is the local axis of the proximal segment that the distal bone stays most
    perpendicular to. Discover it from the (measured) data: pick the proximal local axis (of the
    two not along its bone) minimizing mean |distal_dir . axis|."""
    bone = bone_axis[prox]
    cands = []
    for e in np.eye(3):
        if abs(float(e @ bone)) < 0.9:
            cands.append(e)
    dist_dir = qrot_np(Gmeas[:, dist], bone_axis[dist])
    best, best_v = cands[0], 1e9
    for c in cands:
        ax_w = qrot_np(Gmeas[:, prox], c)
        v = np.abs((dist_dir * ax_w).sum(-1)).mean()
        if v < best_v:
            best, best_v = c, v
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--sym", help="symmetric held-pose ranges 'a:b,a:b,...'")
    ap.add_argument("--relax", help="relaxed arms-down range 'a:b'")
    ap.add_argument("--stride", type=int, default=0, help="background frame stride (0=auto ~3000)")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--harmonics", type=int, default=2, choices=[1, 2])
    ap.add_argument("--sym-set", choices=["arms", "limbs"], default="limbs",
                    help="which bilateral pairs constrain symmetry (arms = arms only, when legs/"
                         "head aren't reliably symmetric; hinge anatomy still constrains the legs)")
    ap.add_argument("--l1", type=float, default=2e-3, help="sparsity on per-sensor params")
    ap.add_argument("--l2-harm", type=float, default=5e-3, help="L2 pull on weakly-identified harmonics")
    ap.add_argument("--w-sym", type=float, default=3.0, help="hard anchor symmetry weight")
    ap.add_argument("--w-relax", type=float, default=3.0, help="relaxed arms-vertical weight")
    ap.add_argument("--w-bg", type=float, default=0.3, help="background robust symmetry weight")
    ap.add_argument("--w-anat", type=float, default=0.4, help="anatomy guardrail weight")
    ap.add_argument("--anat-margin", type=float, default=20.0, help="hinge out-of-plane free margin (deg)")
    ap.add_argument("--w-trunk", type=float, default=0.2)
    ap.add_argument("--apply", action="store_true", help="write a corrected take")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    PAIRS = SYM_PAIRS[:3] if args.sym_set == "arms" else SYM_PAIRS   # SYM_PAIRS[:3] = arm chain

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, bone_axis = load_skeleton(def_path)

    d = np.load(args.infile, allow_pickle=True)
    Q = d['quats'].astype(np.float64)
    fps = float(d['mocap_framerate'])
    T = Q.shape[0]
    Gfull = fk_world(Q, parent, order)

    sym_ranges = parse_ranges(args.sym)
    relax_ranges = parse_ranges(args.relax)
    stride = args.stride or max(1, T // 3000)

    sym_idx = np.concatenate([np.arange(a, b) for a, b in sym_ranges]) if sym_ranges else np.array([], int)
    relax_idx = np.concatenate([np.arange(a, b) for a, b in relax_ranges]) if relax_ranges else np.array([], int)
    bg_idx = np.arange(0, T, stride)
    fit_idx = np.unique(np.concatenate([sym_idx, relax_idx, bg_idx]).astype(int))
    is_relax = np.isin(fit_idx, relax_idx)
    # relaxed arms-down frames are bilaterally symmetric too -> fold them into the hard anchors
    # (their verticality is a TILT problem, unfixable by a yaw correction, so we don't anchor on it)
    is_sym = np.isin(fit_idx, sym_idx) | is_relax
    print(f"{Path(args.infile).name}: {T} frames; fit on {len(fit_idx)} "
          f"({is_sym.sum()} symmetric incl. relax, rest background stride {stride})")

    G = Gfull[fit_idx]                                          # (m,37,4) measured world
    xw = qrot_np(G, np.array([1.0, 0.0, 0.0]))
    psi = np.arctan2(xw[..., 2], xw[..., 0])                    # (m,37) measured heading, fixed
    lat = qrot_np(G[:, PELV], np.array([1.0, 0.0, 0.0]))
    lat = lat - UP * (lat @ UP)[:, None]
    lat = lat / (np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9)
    M = np.eye(3)[None] - 2.0 * lat[:, :, None] * lat[:, None, :]

    hinge_normal = {prox: discover_hinge_normal(Gfull, prox, dist, bone_axis)
                    for prox, dist in HINGES}

    # torch
    Gt = torch.tensor(G)
    psit = torch.tensor(psi)
    Mt = torch.tensor(M)
    symmask = torch.tensor(is_sym, dtype=torch.float64)
    ax_t = {j: torch.tensor(bone_axis[j]) for j in range(37)}
    sec_t = [torch.tensor(a) for a in SEC_AXES]
    hinge_t = {p: torch.tensor(v) for p, v in hinge_normal.items()}
    XW = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    H = args.harmonics
    nparam = 1 + 2 * H
    P = torch.zeros(37, nparam, dtype=torch.float64, requires_grad=True)
    fixed = torch.ones(37, 1, dtype=torch.float64)
    fixed[BODY] = 0.0
    fixed[PELV] = 0.0

    def delta(idx):
        p = P[idx] * fixed[idx]
        out = p[0] + p[1] * torch.sin(psit[:, idx]) + p[2] * torch.cos(psit[:, idx])
        if H == 2:
            out = out + p[3] * torch.sin(2 * psit[:, idx]) + p[4] * torch.cos(2 * psit[:, idx])
        return out

    def cworld(idx):
        half = -delta(idx) / 2.0
        z = torch.zeros_like(half)
        qy = torch.stack([torch.cos(half), z, torch.sin(half), z], -1)
        return qmul_t(qy, Gt[:, idx])

    def frame_axes(seg):
        """Corrected world directions of the segment's frame: per-side bone axis + secondary axes.
        Captures azimuth AND twist, so yaw stays observable even when the bone is near-vertical."""
        g = cworld(seg)
        return [qrot_t(g, ax_t[seg])] + [qrot_t(g, a) for a in sec_t]

    def heading_of(seg):
        x = qrot_t(cworld(seg), XW)
        return torch.atan2(x[..., 2], x[..., 0])

    margin = math.sin(math.radians(args.anat_margin))

    def losses():
        sym_hard = bg = 0.0
        denom = symmask.sum() + 1e-6
        for (Ls, _), (Rs, _) in PAIRS:
            aL, aR = frame_axes(Ls), frame_axes(Rs)
            r2 = sum(((torch.einsum('nij,nj->ni', Mt, l) - r) ** 2).sum(-1)
                     for l, r in zip(aL, aR))
            sym_hard = sym_hard + (r2 * symmask).sum() / denom
            bg = bg + gm(r2, 0.3 ** 2)
        sym_hard, bg = sym_hard / len(PAIRS), bg / len(PAIRS)
        # anatomy: one-sided hinge out-of-plane barrier (generous margin)
        anat = 0.0
        for prox, dist in HINGES:
            ax_w = qrot_t(cworld(prox), hinge_t[prox])
            dist_dir = qrot_t(cworld(dist), ax_t[dist])
            s = (dist_dir * ax_w).sum(-1).abs()
            anat = anat + (torch.relu(s - margin) ** 2).mean()
        anat = anat / len(HINGES)
        trunk = 0.0
        for a, b in zip(TRUNK_FIT[:-1], TRUNK_FIT[1:]):
            dd = heading_of(b) - heading_of(a)
            dd = torch.atan2(torch.sin(dd), torch.cos(dd))
            trunk = trunk + gm(dd ** 2, math.radians(15) ** 2)
        trunk = trunk / (len(TRUNK_FIT) - 1)
        return sym_hard, bg, anat, trunk

    def report_resid():
        with torch.no_grad():
            return tuple(float(x) for x in losses())

    r0 = report_resid()
    opt = torch.optim.Adam([P], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        sym_hard, bg, anat, trunk = losses()
        spars = ((P * fixed).abs()).sum()
        harm = ((P[:, 1:] * fixed) ** 2).sum()
        loss = (args.w_sym * sym_hard + args.w_bg * bg
                + args.w_anat * anat + args.w_trunk * trunk
                + args.l1 * spars + args.l2_harm * harm)
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == args.iters - 1:
            print(f"  it{it:4d} sym={float(sym_hard):.4f} bg={float(bg):.4f} "
                  f"anat={float(anat):.4f} trunk={float(trunk):.4f}")
    r1 = report_resid()

    Pn = (P * fixed).detach().numpy()
    rows = []
    for j in range(37):
        if not np.any(Pn[j]):
            continue
        A = Pn[j, 0]; h1 = math.hypot(Pn[j, 1], Pn[j, 2])
        h2 = math.hypot(Pn[j, 3], Pn[j, 4]) if H == 2 else 0.0
        rms = math.sqrt(np.mean(delta(j).detach().numpy() ** 2))
        rows.append((rms, j, A, h1, h2))
    rows.sort(reverse=True)
    print("\nPer-sensor heading-deviation (deg), ranked:")
    print(f"  {'sensor':22s} {'RMS':>6} {'const':>7} {'1stH':>6} {'2ndH':>6}")
    for rms, j, A, h1, h2 in rows:
        print(f"  {IDX_TO_NAME.get(j, '#%d' % j):22s} {math.degrees(rms):6.1f} "
              f"{math.degrees(A):+7.1f} {math.degrees(h1):6.1f} {math.degrees(h2):6.1f}")
    labels = ['sym(hard)', 'bg-sym', 'anatomy', 'trunk']
    print("\nresiduals       before -> after")
    for nm, a, b in zip(labels, r0, r1):
        print(f"  {nm:12s} {a:.4f} -> {b:.4f}")

    if args.apply:
        # correct every sensor's world by Yaw(-delta_i) over the FULL take, recompute locals
        xwf = qrot_np(Gfull, np.array([1.0, 0.0, 0.0]))
        psif = np.arctan2(xwf[..., 2], xwf[..., 0])             # (T,37)
        Gc = Gfull.copy()
        for j in range(37):
            if not np.any(Pn[j]):
                continue
            p = Pn[j]
            dd = p[0] + p[1] * np.sin(psif[:, j]) + p[2] * np.cos(psif[:, j])
            if H == 2:
                dd += p[3] * np.sin(2 * psif[:, j]) + p[4] * np.cos(2 * psif[:, j])
            half = -dd / 2.0
            qy = np.stack([np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half)], -1)
            Gc[:, j] = qmul_np(qy, Gfull[:, j])
        Qc = Q.copy()
        Qc[:, BODY] = Gc[:, BODY]
        for j in order:
            p = parent[j]
            if p >= 0:
                Qc[:, j] = qmul_np(Gc[:, p] * np.array([1, -1, -1, -1.0]), Gc[:, j])
        out = args.out or args.infile.replace(".npz", "_magfit.npz")
        save = {k: d[k] for k in d.files}; save['quats'] = Qc
        np.savez(out, **save)
        print(f"\nwrote corrected take -> {out}")
    else:
        print("\n(diagnostic only; pass --apply to write a corrected take)")


if __name__ == "__main__":
    main()
