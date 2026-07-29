"""Pool the Stage-1 VPoser implausibility detector across a whole session's takes.

One take only spans the facings its choreography used (padam: ~forward/back), and its
implausibility-vs-heading is confounded by choreography (specific moves at specific facings).
Pooling all *beta.npz originals fixes both: varied takes face varied directions (fills the
heading circle, since the T-pose was the same NEWS orientation every take), and averaging over
many different poses at a given heading washes out choreography -- what survives at a heading
across many poses is the magnetization signature. The heading-DEPENDENT error is a physical
property of the sensor, so it should be consistent take-to-take even if the constant calibration
offset drifts.

Reports: per-take implausibility + heading coverage; session-consensus per-joint suspect ranking
(equal weight per take); and the pooled per-joint implausibility-vs-heading map. No correction.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from diag_pose_plausibility import (shadow_to_body_pose, BODY_JOINTS, BODY_NAMES,
                                     SMPL_TO_SHADOW, EXPR_DIR)
from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, PELV, IDX_TO_NAME
from dpg_system.vae_nodes import VPoser, load_model

NB = 12                                                # heading bins (30 deg)
BIN_LABELS = [f"{-180+30*b:+4d}" for b in range(NB)]


def analyze(path, model, parent, order, stride_target=3000):
    d = np.load(path, allow_pickle=True)
    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    stride = max(1, T // stride_target)
    idx = np.arange(0, T, stride)
    G = fk_world(quats, parent, order)
    px = qrot_np(G[:, PELV], np.array([1.0, 0.0, 0.0]))
    heading = np.degrees(np.arctan2(px[:, 2], px[:, 0]))[idx]
    hbin = ((heading + 180) // 30).astype(int) % NB
    body = shadow_to_body_pose(quats)[idx].reshape(len(idx), 63).astype(np.float32)
    x = torch.tensor(body, requires_grad=True)
    nll = (model.encode(x).mean ** 2).sum(-1)
    nll.sum().backward()
    g = x.grad.view(len(idx), 21, 3).norm(dim=-1).numpy()       # (n,21) attribution
    return nll.detach().numpy(), g, hbin


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="session folder containing *beta.npz originals")
    ap.add_argument("--expr-dir", default=EXPR_DIR)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    if not files:
        print(f"no *beta.npz originals in {args.dir}"); return
    model = load_model(args.expr_dir, model_code=VPoser, remove_words_in_model_weights='vp_model.',
                       disable_grad=True, comp_device='cpu')[0].to('cpu').eval()
    parent, order, _ = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')

    unmapped = np.array([BODY_JOINTS[r] not in SMPL_TO_SHADOW for r in range(21)])
    sum_jb = np.zeros((21, NB))                          # pooled attribution sum per joint,bin
    cnt_b = np.zeros(NB)                                 # pooled frame count per bin
    nll_b = np.zeros(NB)                                 # pooled nll sum per bin
    per_take_jmean = []                                  # equal-weight consensus ranking

    print(f"{'take':42s} {'med nll':>7} {'%impl':>6}  heading bins covered")
    for f in files:
        nll, g, hbin = analyze(f, model, parent, order)
        for b in range(NB):
            sel = hbin == b
            if sel.any():
                sum_jb[:, b] += g[sel].sum(0)
                cnt_b[b] += sel.sum()
                nll_b[b] += nll[sel].sum()
        per_take_jmean.append(g.mean(0))
        cov = "".join("#" if (hbin == b).any() else "." for b in range(NB))
        name = os.path.basename(f).replace("Subject7_take", "").replace("_a_beta.npz", "")
        print(f"  {name:40s} {np.median(nll):7.0f} {100*(nll>96).mean():5.0f}%  {cov}")

    pooled_nll = np.where(cnt_b > 0, nll_b / np.maximum(cnt_b, 1), 0)
    pooled_jb = np.where(cnt_b > 0, sum_jb / np.maximum(cnt_b, 1), 0)
    jmean = np.mean(per_take_jmean, 0)
    jmean[unmapped] = -np.inf
    rank = np.argsort(jmean)[::-1]

    print(f"\nsession heading coverage (frames per 30-deg bin):")
    print("  " + " ".join(f"{l:>6}" for l in BIN_LABELS))
    print("  " + " ".join(f"{int(c):6d}" for c in cnt_b))
    print("  pooled nll: " + " ".join(f"{v:6.0f}" for v in pooled_nll))

    print(f"\nsession-consensus suspect ranking (mean per-joint attribution, equal weight/take):")
    for r in rank[:args.top]:
        sh = IDX_TO_NAME.get(SMPL_TO_SHADOW.get(BODY_JOINTS[r]), '?')
        print(f"  {BODY_NAMES[r]:14s} {jmean[r]:7.2f}   sensor: {sh}")

    print(f"\npooled implausibility vs heading (rows=joint, cols=heading; full circle now):")
    print("  " + " ".join(f"{l:>6}" for l in BIN_LABELS))
    for r in rank[:args.top]:
        row = pooled_jb[r]
        print(f"  {BODY_NAMES[r]:14s} " + " ".join(f"{v:6.1f}" for v in row))
    print("\nA joint whose pooled attribution PEAKS at specific headings (across varied takes) is a")
    print("magnetization signature; one that's flat or tracks frame-count is likely choreography.")


if __name__ == "__main__":
    main()
