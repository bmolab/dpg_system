"""Scan for Shadow-solver twist-flip glitches: sudden bone-twist jumps with a STATIONARY endpoint.

User-observed artifact: shoulders snap back and forth between two bone-twist configurations with no
change in pose shape -- the solver redistributing twist about the bone axis (a twist/gimbal ambiguity)
without moving the joint positions. Signature, distinct from magnetization (smooth) and real motion
(endpoint moves): an implausibly fast twist jump (>>real motion at 100 Hz) while the downstream
endpoint POSITION barely moves, often REVERTING within a few frames (toggle).

Per sensor we track the twist-vs-parent series, flag jumps |Dtwist|>JUMP deg/frame, mark TOGGLES
(opposite-sign jump within 5 frames), and cross-check the endpoint (wrist/hand) position velocity at
jump frames vs the take median. glitch = fast twist jump + stationary endpoint (+ revert).
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
from correct_upper_arm_offset import forward_kinematics, qmul
from diag_magnetometer_deviation import load_skeleton, qrot_np, RSH, REL, RWR, LSH, LEL, LWR, RBLADE, LBLADE

JUMP = 30.0   # deg/frame: faster than physically plausible twist at 100 Hz
def qc(q): return q * np.array([1., -1, -1, -1.])
def tw(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def wrapd(a): return (a + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); args = ap.parse_args()
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    # (joint, parent-for-twist, endpoint joint for position stability)
    chains = [('R-shldr', RSH, RBLADE, RWR), ('L-shldr', LSH, LBLADE, LWR),
              ('R-elbow', REL, RSH, RWR), ('L-elbow', LEL, LSH, LWR)]
    print(f"twist-flip glitch scan (jump > {JUMP:.0f} deg/frame). 'stationary' = endpoint vel < 0.5x take median.")
    print(f"  {'take':14s} " + " ".join(f"{c[0]:>20s}" for c in chains))
    print(f"  {'':14s} " + " ".join(f"{'jmp/tog/statJmp':>20s}" for c in chains))
    tot = {c[0]: [0, 0, 0] for c in chains}
    for f in files:
        d = np.load(f, allow_pickle=True); G = forward_kinematics(d['quats'].astype(np.float64))
        P = d['positions'].astype(np.float64); T = G.shape[0]
        cells = []
        for nm, j, par, ep in chains:
            twj = tw(qmul(qc(G[:, par]), G[:, j]), bax[j])
            dtw = wrapd(np.diff(twj))                       # per-frame twist change
            jump = np.abs(dtw) > JUMP
            # toggles: a jump immediately (within 5f) followed by an opposite-sign jump of similar size
            ji = np.flatnonzero(jump); tog = 0
            for k in ji:
                w = (ji > k) & (ji <= k + 5)
                if np.any(np.sign(dtw[ji[w]]) == -np.sign(dtw[k])): tog += 1
            epv = np.linalg.norm(np.diff(P[:, ep], axis=0), axis=1)   # endpoint position velocity
            med = np.median(epv) + 1e-9
            statjmp = int((jump & (epv < 0.5 * med)).sum())            # fast twist jump + stationary endpoint
            cells.append(f"{int(jump.sum()):4d}/{tog:3d}/{statjmp:4d}")
            tot[nm][0] += int(jump.sum()); tot[nm][1] += tog; tot[nm][2] += statjmp
        print(f"  {os.path.basename(f).replace('Subject7_take_','').replace('_beta.npz','')[:14]:14s} " + " ".join(f"{c:>20s}" for c in cells))
    print(f"  {'TOTAL':14s} " + " ".join(f"{tot[c[0]][0]:6d}/{tot[c[0]][1]:4d}/{tot[c[0]][2]:5d}".rjust(20) for c in chains))
    print("\njmp=fast twist jumps, tog=reverting (snap-back) jumps, statJmp=jump with stationary endpoint")
    print("(the glitch signature). High statJmp/tog on the shoulders = solver twist-flip contamination.")


if __name__ == "__main__":
    main()
