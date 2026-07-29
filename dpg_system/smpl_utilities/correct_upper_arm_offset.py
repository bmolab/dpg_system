"""Correct a constant per-upper-arm orientation offset in Shadow-format takes.

Background
----------
Some Shadow IMU takes show the arms hanging biased/twisted while the T-pose
calibration looked fine. Diagnosis localized a *constant* per-arm orientation offset
at the upper-arm sensor (the `Shoulder` joint) — collar/shoulder-blade, spine and
lower-back reference are comparatively clean. On the left arm the offset is close to
a pure rotation about vertical (heading/hard-iron signature).

Each stored quaternion is a sensor-to-sensor *relative* rotation, so the reconstructed
forearm/hand orientations are self-consistent. The fix:
  1. (optional) apply a shared mirror-symmetric twist about the upper-arm bone axis
     (local +X) — the common-mode "deltoid roll" that bilateral symmetry cannot see;
     tuned per take by rendering.
  2. fit a constant per-arm world-frame correction C minimizing left/right mirror
     asymmetry over user-flagged symmetric frames + a relaxed arms-down window where
     the upper arm should hang vertical.
  3. left-multiply the upper-arm global by C; apply the twist about the humerus local
     +X (forearm GLOBAL kept fixed -> deltoid roll); recompute the Shoulder and Elbow
     local quats. Then optional dials: elbow open (per-frame hinge), wrist extend
     (per-frame wrist hinge), hand roll (wtwist), and per-arm abduction / shared
     flex about world axes.

NOTE: correcting the forearm as a separate segment was tried and produced impossible
elbow configurations — don't. The working Subject7 recipe is roughly
`--twist -60 --abduct-l 12 --abduct-r 5 --elbow 28 --wrist 15 --wtwist -15`.
Corrections are per-take unless takes share one calibration.

Assumes the Shadow ("37") joint ordering (body_defs.shadow_joint_index_to_name);
upper-arm bone long axis = local +X (FK-validated). Right arm's local frame mirrors
the left's, so a shared twist/abduct/flex angle is mirror-symmetric.
"""
import argparse
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

# --- Shadow ("37") joint indices ---
PELA, SPIN, LOWV, MIDV, UPPV = 4, 31, 32, 1, 17
LBLADE, LSH, LEL, LWR = 13, 5, 9, 10
RBLADE, RSH, REL, RWR = 27, 19, 23, 24
PARENT = {SPIN: PELA, LOWV: SPIN, MIDV: LOWV, UPPV: MIDV,
          LBLADE: MIDV, LSH: LBLADE, LEL: LSH, LWR: LEL,
          RBLADE: MIDV, RSH: RBLADE, REL: RSH, RWR: REL}
FK_ORDER = [SPIN, LOWV, MIDV, UPPV, LBLADE, LSH, LEL, LWR, RBLADE, RSH, REL, RWR]

UP = np.array([0, 1.0, 0])
DOWN = np.array([0, -1.0, 0])


def qmul(a, b):                                                          # w-first product
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return q * np.array([1, -1, -1, -1.0])


def qrot(q, v):
    qv = np.concatenate([np.zeros(q.shape[:-1] + (1,)), np.broadcast_to(v, q.shape[:-1] + (3,))], -1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def rotvec_to_q(rv):
    r = Rot.from_rotvec(rv).as_quat()                                    # x, y, z, w
    return np.array([r[3], r[0], r[1], r[2]])


def norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def norm_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def forward_kinematics(Q):
    G = Q.copy()
    for j in FK_ORDER:
        G[:, j] = qmul(G[:, PARENT[j]], Q[:, j])
    return G


def head_yaw(g, sgn, hy):
    """Apply the orientation-dependent hard-iron deviation curve as a per-frame yaw about
    world-vertical: delta(psi) = b*sin(psi) + c*cos(psi), psi = the bone azimuth (distal direction
    angle in the horizontal plane, distal axis sgn*+X). hy = (b, c) radians. Removes the
    POSE-VARYING upper-arm L/R asymmetry that a constant C can only average over."""
    if hy[0] == 0.0 and hy[1] == 0.0:
        return g
    bone = qrot(g, np.array([sgn, 0.0, 0.0]))
    psi = np.arctan2(bone[..., 2], bone[..., 0])
    h = (hy[0] * np.sin(psi) + hy[1] * np.cos(psi)) / 2.0      # half-angle for the quaternion
    z = np.zeros_like(h)
    Ry = np.stack([np.cos(h), z, np.sin(h), z], -1)            # yaw about world Y
    return qmul(Ry, g)


def _slice_anchor(G, P, a, b):
    """One mirror anchor (M, nL0, nR0, mL0, mR0) from a WINDOW-AVERAGED symmetric pose."""
    sl = slice(a, b)
    have_shoulders = (P is not None and np.any(P[sl, LSH]) and np.any(P[sl, RSH]))
    have_elbows = (P is not None and np.any(P[sl, LEL]) and np.any(P[sl, REL]))
    if have_shoulders:
        Pm = P[sl].mean(0)
        lat = norm((Pm[LSH] - Pm[RSH]) - UP * np.dot(Pm[LSH] - Pm[RSH], UP))
    else:
        latv = qrot(G[sl, MIDV], np.array([1.0, 0.0, 0.0])).mean(0)
        lat = norm(latv - UP * np.dot(latv, UP))
    M = np.eye(3) - 2 * np.outer(lat, lat)
    if have_shoulders and have_elbows:
        Pm = P[sl].mean(0)
        nL0 = norm(Pm[LEL] - Pm[LSH]); nR0 = norm(Pm[REL] - Pm[RSH])
    else:
        nL0 = norm(qrot(G[sl, LSH], np.array([1.0, 0, 0])).mean(0))
        nR0 = norm(qrot(G[sl, RSH], np.array([-1.0, 0, 0])).mean(0))
    mL0 = norm(qrot(G[sl, LSH], np.array([0, 0, 1.0])).mean(0))
    mR0 = norm(qrot(G[sl, RSH], np.array([0, 0, 1.0])).mean(0))
    return (M, nL0, nR0, mL0, mR0)


def _frame_dirs(G, P, a, b):
    """Per-frame (lat, nL, nR, mL, mR) over a window -- for detecting symmetric instants."""
    sl = slice(a, b)
    have_sh = (P is not None and np.any(P[sl, LSH]) and np.any(P[sl, RSH]))
    have_el = (P is not None and np.any(P[sl, LEL]) and np.any(P[sl, REL]))
    latv = (P[sl, LSH] - P[sl, RSH]) if have_sh else qrot(G[sl, MIDV], np.array([1.0, 0, 0]))
    lat = norm_rows(latv - UP * (latv @ UP)[:, None])
    if have_sh and have_el:
        nL = norm_rows(P[sl, LEL] - P[sl, LSH]); nR = norm_rows(P[sl, REL] - P[sl, RSH])
    else:
        nL = norm_rows(qrot(G[sl, LSH], np.array([1.0, 0, 0])))
        nR = norm_rows(qrot(G[sl, RSH], np.array([-1.0, 0, 0])))
    mL = norm_rows(qrot(G[sl, LSH], np.array([0, 0, 1.0])))
    mR = norm_rows(qrot(G[sl, RSH], np.array([0, 0, 1.0])))
    return lat, nL, nR, mL, mR


def _walk_sym_anchors(G, P, a, b, thresh_deg=8.0, max_anchors=20):
    """Detect symmetric crossings in a relaxed walk-swing window: instants where the mirrored left
    upper-arm direction matches the right (arms passing through the neutral/down crossing of an
    anti-phase swing). Returns a symmetry anchor per crossing + the per-frame mirror residual."""
    lat, nL, nR, mL, mR = _frame_dirs(G, P, a, b)
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
    s = np.degrees(np.arccos(np.clip((np.einsum('nij,nj->ni', M, nL) * nR).sum(1), -1, 1)))
    thr = max(thresh_deg, np.percentile(s, 12))
    idxs = np.flatnonzero(s < thr)
    cand = []
    if len(idxs):
        for run in np.split(idxs, np.flatnonzero(np.diff(idxs) > 1) + 1):
            f = int(run[np.argmin(s[run])])               # most symmetric instant of each crossing
            cand.append((s[f], (M[f], nL[f], nR[f], mL[f], mR[f])))
    else:
        f = int(np.argmin(s)); cand.append((s[f], (M[f], nL[f], nR[f], mL[f], mR[f])))
    cand.sort(key=lambda t: t[0])
    return [anc for _, anc in cand[:max_anchors]], s, len(cand)


def precompute(G, P, sym, relax, walk_relax=True, sym_thresh_deg=8.0):
    """Mirror anchors for the fit. The sym ranges are window-averaged held poses. The relax range
    in this session is a relaxed WALK-swing (no straight-down pose exists; relaxed arms flare and
    sway), so by default we DETECT its symmetric crossings and add each as a symmetry anchor rather
    than averaging the swing or forcing it vertical. walk_relax=False restores the legacy single
    averaged relax anchor. Position/orientation handling per _slice_anchor (Subject7 positions used
    if present, else chest-orientation fallback)."""
    pre = [_slice_anchor(G, P, a, b) for a, b in sym]
    if walk_relax:
        wa, s, ncross = _walk_sym_anchors(G, P, relax[0], relax[1], sym_thresh_deg)
        print(f"correct_upper_arm_offset: relax window treated as walk-swing -- {ncross} symmetric "
              f"crossings detected (median frame residual {np.median(s):.1f} deg), "
              f"using {len(wa)} as symmetry anchors (no verticality term)")
        pre += wa
    else:
        pre.append(_slice_anchor(G, P, relax[0], relax[1]))
    return pre


def fit_corrections(pre, n_sym=None, reg=0.05):
    """Solve per-arm world-frame correction (rotvec) minimizing L/R mirror asymmetry over ALL
    anchors (held sym poses + detected walk-swing crossings). NO verticality term -- relaxed arms
    flare and swing and never hang straight down, so forcing them to [0,-1,0] injects error. A
    small ridge (reg) pins the symmetry-PRESERVING gauge mode (the 'both arms equally rotated'
    freedom that mirror symmetry cannot observe) to identity, so the fit corrects only the
    ASYMMETRIC error (e.g. one upper arm raised relative to the other). (n_sym kept for call
    compatibility; all anchors are now symmetry anchors.)"""
    def resid(x):
        CL = Rot.from_rotvec(x[:3]).as_matrix()
        CR = Rot.from_rotvec(x[3:]).as_matrix()
        r = []
        for (M, nL0, nR0, mL0, mR0) in pre:
            nL, nR, mL, mR = CL @ nL0, CR @ nR0, CL @ mL0, CR @ mR0
            r += list(M @ nL - nR) + list(M @ mL - mR)
        r += list(reg * x)                                # ridge -> unobservable gauge mode to 0
        return r
    sol = least_squares(resid, np.zeros(6), method='lm')
    return sol.x[:3], sol.x[3:]


def fit_heading_yaw(Q, P, cL_rv, cR_rv, sym=None, samples=40, frames=None):
    """Fit a per-arm heading-dependent yaw delta(psi) = b*sin(psi) + c*cos(psi) (the
    once-per-revolution hard-iron deviation curve), on top of the constant upper-arm correction
    cL/cR, by minimizing the per-pose upper-arm L/R mirror asymmetry. psi = upper-arm azimuth
    (distal direction in the horizontal plane). Localised to the humerus only -- doesn't touch the
    forearm. Returns (hy_l, hy_r), each np.array([b, c]) rad. Pass `frames` (explicit symmetric
    instants, e.g. auto-detected crossings spanning many headings) to constrain the curve properly;
    otherwise it samples the `sym` windows (only well-determined if those span enough headings)."""
    if frames is not None:
        idx = np.asarray(sorted(set(int(i) for i in frames)))
    else:
        idx = []
        for a, b in sym:
            idx += list(np.linspace(a, b - 1, samples).astype(int))
        idx = np.array(sorted(set(int(i) for i in idx)))
    Gs = forward_kinematics(Q[idx])
    Ps = P[idx] if P is not None else None
    gL = qmul(np.broadcast_to(rotvec_to_q(cL_rv), Gs[:, LSH].shape), Gs[:, LSH])
    gR = qmul(np.broadcast_to(rotvec_to_q(cR_rv), Gs[:, RSH].shape), Gs[:, RSH])
    nL0 = norm_rows(qrot(gL, np.array([1.0, 0, 0])))            # left distal
    nR0 = norm_rows(qrot(gR, np.array([-1.0, 0, 0])))           # right distal (mirror axis)
    psiL = np.arctan2(nL0[:, 2], nL0[:, 0]); psiR = np.arctan2(nR0[:, 2], nR0[:, 0])
    up = np.array([0, 1.0, 0])
    if Ps is not None and np.any(Ps[:, LSH]) and np.any(Ps[:, RSH]):
        lat = Ps[:, LSH] - Ps[:, RSH]
    else:
        lat = qrot(Gs[:, MIDV], np.array([1.0, 0.0, 0.0]))
    lat = norm_rows(lat - up * (lat @ up)[:, None])
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]

    def yaw(v, a):                                              # rotate dirs about world Y by a
        ca, sa = np.cos(a), np.sin(a)
        return np.stack([v[:, 0] * ca + v[:, 2] * sa, v[:, 1], -v[:, 0] * sa + v[:, 2] * ca], 1)

    def resid(p):
        dL = p[0] * np.sin(psiL) + p[1] * np.cos(psiL)
        dR = p[2] * np.sin(psiR) + p[3] * np.cos(psiR)
        return (np.einsum('nij,nj->ni', M, yaw(nL0, dL)) - yaw(nR0, dR)).ravel()

    x = least_squares(resid, np.zeros(4), method='lm').x
    return np.array([x[0], x[1]]), np.array([x[2], x[3]])


def report(pre, sym, relax, cL_rv, cR_rv):
    """Per-anchor L/R mirror asymmetry before vs after the fit. The first len(sym) anchors are the
    held sym windows; any remaining are detected walk-swing crossings (summarized as a mean)."""
    CL = Rot.from_rotvec(cL_rv).as_matrix(); CR = Rot.from_rotvec(cR_rv).as_matrix()
    print(f"{'anchor':18s} {'asym_before':>11} {'asym_after':>10}")
    walk_b, walk_a = [], []
    for i, (M, nL0, nR0, mL0, mR0) in enumerate(pre):
        sb = math.degrees(math.acos(np.clip(np.dot(M @ nL0, nR0), -1, 1)))
        sa = math.degrees(math.acos(np.clip(np.dot(M @ (CL @ nL0), CR @ nR0), -1, 1)))
        if i < len(sym):
            print(f"  {str(sym[i]):18s} {sb:11.1f} {sa:10.1f}")
        else:
            walk_b.append(sb); walk_a.append(sa)
    if walk_b:
        print(f"  {('walk x%d' % len(walk_b)):18s} {np.mean(walk_b):11.1f} {np.mean(walk_a):10.1f}  (mean)")


def apply_correction(Q, G, P, cL_rv, cR_rv, twist_deg=0.0, abduct_l=0.0, abduct_r=0.0,
                     flex_deg=0.0, elbow_deg=0.0, wrist_deg=0.0, wtwist_deg=0.0,
                     hy_l=(0.0, 0.0), hy_r=(0.0, 0.0)):
    """Left-multiply each upper-arm global by its world correction, then a per-frame world-Y yaw
    delta(psi)=b*sin(psi)+c*cos(psi) (the orientation-dependent hard-iron deviation curve,
    fit by symmetry), then a shared body-frame twist about the upper-arm long axis (+X; the
    deltoid roll, forearm GLOBAL kept fixed -- only the Shoulder + Elbow local quats are
    recomputed). Then per-frame hinge dials for the elbow and wrist, a hand roll, and a per-arm
    abduction / shared flex about world axes that carries the whole arm."""
    CL, CR = rotvec_to_q(cL_rv), rotvec_to_q(cR_rv)
    T = rotvec_to_q(np.array([math.radians(twist_deg), 0.0, 0.0]))     # twist (+X), forearm-fixed
    gLSH = qmul(np.broadcast_to(CL, G[:, LSH].shape), G[:, LSH])
    gRSH = qmul(np.broadcast_to(CR, G[:, RSH].shape), G[:, RSH])
    gLSH = head_yaw(gLSH, 1.0, hy_l)                                    # per-frame heading-yaw correction
    gRSH = head_yaw(gRSH, -1.0, hy_r)
    if twist_deg != 0.0:
        gLSH = qmul(gLSH, np.broadcast_to(T, gLSH.shape))
        gRSH = qmul(gRSH, np.broadcast_to(T, gRSH.shape))
    Qc = Q.copy()
    Qc[:, LSH] = qmul(qconj(G[:, LBLADE]), gLSH)
    Qc[:, RSH] = qmul(qconj(G[:, RBLADE]), gRSH)
    Qc[:, LEL] = qmul(qconj(gLSH), G[:, LEL])     # forearm GLOBAL kept fixed
    Qc[:, REL] = qmul(qconj(gRSH), G[:, REL])
    # per-arm elbow open about the TRUE per-frame hinge = cross(forearm, upperarm) -> pure
    # flexion, no twist/abduction leak; the hand is carried.
    if elbow_deg != 0.0:
        ang = math.radians(elbow_deg)
        for gSH, gfa, jEL in [(gLSH, G[:, LEL], LEL), (gRSH, G[:, REL], REL)]:
            ua = norm_rows(qrot(gSH, np.array([1, 0, 0.])))
            fo = norm_rows(qrot(gfa, np.array([1, 0, 0.])))
            hinge = np.cross(fo, ua)
            n = np.linalg.norm(hinge, axis=-1, keepdims=True)
            straight = (n <= 1e-3)[..., 0]
            hinge = hinge / (n + 1e-12)
            R = np.concatenate([np.cos(ang / 2) * np.ones(hinge.shape[:-1] + (1,)),
                                np.sin(ang / 2) * hinge], -1)
            R[straight] = np.array([1.0, 0, 0, 0])
            Qc[:, jEL] = qmul(qconj(gSH), qmul(R, gfa))
    # wrist extend about the per-frame wrist hinge = cross(hand, forearm); carries the hand.
    if wrist_deg != 0.0:
        ang = math.radians(wrist_deg)
        for gSH, jEL, jWR in [(gLSH, LEL, LWR), (gRSH, REL, RWR)]:
            gEL = qmul(gSH, Qc[:, jEL])
            gWR = qmul(gEL, Qc[:, jWR])
            fo = norm_rows(qrot(gEL, np.array([1, 0, 0.])))
            ha = norm_rows(qrot(gWR, np.array([1, 0, 0.])))
            hinge = np.cross(ha, fo)
            n = np.linalg.norm(hinge, axis=-1, keepdims=True)
            straight = (n <= 1e-3)[..., 0]
            hinge = hinge / (n + 1e-12)
            R = np.concatenate([np.cos(ang / 2) * np.ones(hinge.shape[:-1] + (1,)),
                                np.sin(ang / 2) * hinge], -1)
            R[straight] = np.array([1.0, 0, 0, 0])
            Qc[:, jWR] = qmul(qconj(gEL), qmul(R, gWR))
    if wtwist_deg != 0.0:
        # roll both hands about the hand long axis (local +X)
        W = rotvec_to_q(np.array([math.radians(wtwist_deg), 0.0, 0.0]))
        Qc[:, LWR] = qmul(Qc[:, LWR], np.broadcast_to(W, Qc[:, LWR].shape))
        Qc[:, RWR] = qmul(Qc[:, RWR], np.broadcast_to(W, Qc[:, RWR].shape))
    if abduct_l != 0.0 or abduct_r != 0.0 or flex_deg != 0.0:
        # Per-arm abduction (world fore-aft axis; outward positive) + shared flex (world
        # lateral axis; +forward). The whole arm is carried (shoulder local only).
        up = np.array([0, 1.0, 0])
        if P is not None and np.any(P[:, LSH]) and np.any(P[:, RSH]):
            latv = P[:, LSH] - P[:, RSH]
        else:
            latv = qrot(G[:, MIDV], np.array([1.0, 0.0, 0.0]))
        latv = norm_rows(latv - up * (latv @ up)[:, None])
        fwd = norm_rows(np.cross(np.broadcast_to(up, latv.shape), latv))

        def swing(jSH, jBL, axis, deg):
            h = math.radians(deg) / 2.0
            R = np.concatenate([np.cos(h) * np.ones(axis.shape[:-1] + (1,)), np.sin(h) * axis], -1)
            gSH = qmul(G[:, jBL], Qc[:, jSH])
            Qc[:, jSH] = qmul(qconj(G[:, jBL]), qmul(R, gSH))

        if abduct_l != 0.0:
            swing(LSH, LBLADE, fwd, -abduct_l)
        if abduct_r != 0.0:
            swing(RSH, RBLADE, fwd, +abduct_r)
        if flex_deg != 0.0:
            swing(LSH, LBLADE, latv, flex_deg)
            swing(RSH, RBLADE, latv, flex_deg)
    return Qc


def correct(Q, P, sym, relax, twist=0.0, abduct_l=0.0, abduct_r=0.0, flex=0.0, elbow=0.0,
            wrist=0.0, wtwist=0.0, fit_override=None, verbose=True):
    """Returns (corrected_quats, fit). The fit dict has cl, cr (per-arm upper-arm world
    correction, rotvec) and hy_l, hy_r ([b, c] rad per arm: the orientation-dependent heading-yaw
    deviation curve fit by L/R symmetry over the sym anchors). fit_override=that dict reuses a
    saved fit."""
    G = forward_kinematics(Q)
    if fit_override is None:
        pre = precompute(G, P, sym, relax)
        cl, cr = fit_corrections(pre, len(sym))
        hy_l, hy_r = fit_heading_yaw(Q, P, cl, cr, sym)
        fit = dict(cl=cl, cr=cr, hy_l=hy_l, hy_r=hy_r)
        if verbose:
            report(pre, sym, relax, cl, cr)
            print(f"  heading-yaw amp:  |L|={math.degrees(np.hypot(*hy_l)):.0f}  "
                  f"|R|={math.degrees(np.hypot(*hy_r)):.0f}  deg")
    else:
        fit = dict(fit_override)
    Qc = apply_correction(Q, G, P, fit['cl'], fit['cr'], twist_deg=twist,
                          abduct_l=abduct_l, abduct_r=abduct_r, flex_deg=flex, elbow_deg=elbow,
                          wrist_deg=wrist, wtwist_deg=wtwist,
                          hy_l=fit.get('hy_l', np.zeros(2)), hy_r=fit.get('hy_r', np.zeros(2)))
    return Qc, fit


# Defaults for Subject7 jathiswaram
JATHISWARAM_SYM = [(0, 1721), (3767, 4000), (5512, 5734),
                   (16968, 17082), (21195, 21697), (35762, 36237)]
JATHISWARAM_RELAX = (10430, 11242)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("infile")
    ap.add_argument("-o", "--out")
    ap.add_argument("--sym", help="symmetric ranges 'a:b,a:b,...'")
    ap.add_argument("--relax", help="relaxed arms-down window 'a:b'")
    ap.add_argument("--twist", type=float, default=0.0, help="upper-arm shared twist (deg); tune per take by rendering")
    ap.add_argument("--abduct-l", type=float, default=0.0, help="abduct LEFT arm outward (deg) about world fore-aft axis")
    ap.add_argument("--abduct-r", type=float, default=0.0, help="abduct RIGHT arm outward (deg) about world fore-aft axis")
    ap.add_argument("--flex", type=float, default=0.0, help="swing both arms in the sagittal plane (deg); +forward, -back")
    ap.add_argument("--elbow", type=float, default=0.0, help="open both elbows (deg) about the hinge axis; carries the hands")
    ap.add_argument("--wrist", type=float, default=0.0, help="extend both wrists (deg) about the wrist hinge; reduces over-flexion")
    ap.add_argument("--wtwist", type=float, default=0.0, help="roll both hands about the hand long axis (deg); thumb forward/back")
    ap.add_argument("--save-fit", help="save the fitted per-arm correction (cl,cr) to this .npz")
    ap.add_argument("--load-fit", help="load a saved per-arm correction instead of refitting")
    ap.add_argument("--no-fit", action="store_true", help="skip the symmetry fit (identity C); dial-only tuning")
    ap.add_argument("--sweep", help="emit variants of one dial: explicit 'twist:-40,-50,-60' "
                                    "or range 'twist:-80:20:20' (start:stop:step, stop inclusive). "
                                    "dials: twist,elbow,wrist,wtwist,abl,abr")
    args = ap.parse_args()

    sym = ([tuple(int(v) for v in r.split(":")) for r in args.sym.split(",")]
           if args.sym else JATHISWARAM_SYM)
    relax = (tuple(int(v) for v in args.relax.split(":")) if args.relax else JATHISWARAM_RELAX)

    d = np.load(args.infile, allow_pickle=True)
    fit_override = None
    if args.no_fit:
        fit_override = dict(cl=np.zeros(3), cr=np.zeros(3),
                            hy_l=np.zeros(2), hy_r=np.zeros(2))
    elif args.load_fit:
        z = np.load(args.load_fit)
        ga = lambda k: np.asarray(z[k]) if k in z.files else np.zeros(2)
        fit_override = dict(cl=z['cl'], cr=z['cr'], hy_l=ga('hy_l'), hy_r=ga('hy_r'))

    DIAL_KEYS = {"twist": "twist", "flex": "flex", "elbow": "elbow", "wrist": "wrist",
                 "wtwist": "wtwist", "abl": "abduct_l", "abr": "abduct_r"}
    base = dict(twist=args.twist, abduct_l=args.abduct_l, abduct_r=args.abduct_r,
                flex=args.flex, elbow=args.elbow, wrist=args.wrist, wtwist=args.wtwist)

    def tag_for(p):
        t = f"_tw{p['twist']:+.0f}" if p['twist'] else ""
        for k, lab in [("abduct_l", "abl"), ("abduct_r", "abr"), ("flex", "fl"),
                       ("elbow", "el"), ("wrist", "wr"), ("wtwist", "wt")]:
            if p[k]:
                t += f"_{lab}{p[k]:+.0f}"
        return t

    def run(p, verbose):
        Qc, fit = correct(d['quats'], d['positions'], sym, relax,
                          fit_override=fit_override, verbose=verbose, **p)
        if args.save_fit:
            np.savez(args.save_fit, **fit)
            print(f"saved fit to {args.save_fit}")
        out = args.out or args.infile.replace(".npz", f"_armfix{tag_for(p)}.npz")
        save = {k: d[k] for k in d.files}; save['quats'] = Qc
        np.savez(out, **save); print(f"wrote {out}")

    if args.sweep:
        parts = args.sweep.split(":")
        key = DIAL_KEYS[parts[0].strip()]
        if len(parts) == 4:                                  # range: dial:start:stop:step
            a, b, s = (float(parts[1]), float(parts[2]), float(parts[3]))
            vals = list(np.arange(a, b + (s / 2 if s else 0), s)) if s else [a]
        else:                                                # explicit: dial:v1,v2,v3
            vals = [float(x) for x in parts[1].split(",")]
        for i, v in enumerate(vals):
            p = dict(base); p[key] = v
            run(p, verbose=(i == 0))
    else:
        run(base, verbose=True)


if __name__ == "__main__":
    main()
