"""Stage-1 detector: locate magnetic/calibration distortion by POSE IMPLAUSIBILITY (VPoser).

This does NOT correct anything. It treats physical implausibility as a loose guide (per the
reframing): VPoser -- a VAE pose prior trained on AMASS -- scores any body pose's implausibility
as the latent NLL ||z||^2 (a plausible pose sits near ||z||^2 ~ latentD = 32; large values are
unnatural, capturing axial rotation and joint coupling that bone directions miss). Then:

  1. per-frame ||z||^2                       -> WHEN the pose is implausible
  2. per-joint gradient d||z||^2/d(joint)    -> WHICH joint drives it (localization)
  3. both binned by global heading           -> WHERE in the yaw circle distortion is strongest,
                                                which hints at the magnetization's phase/strength.

Pipeline: Shadow local quats -> SMPL body pose (joints 1..21, axis-angle; via shadow_to_smpl's
mapping) -> VPoser.encode -> z. A high-attribution SMPL joint implicates the Shadow sensor at
that joint and its parent (a sensor's yaw error perturbs both its own and its child's local
rotation), to be disambiguated with the heading map.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dpg_system.vae_nodes import VPoser, load_model
from dpg_system.smpl_utilities.shadow_to_smpl import SHADOW_TO_SMPL, SMPL_JOINT_NAMES
from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, PELV, IDX_TO_NAME

EXPR_DIR = '/Users/drokeby/Dev/human_body_prior_/support_data/training/training_experiments/V02_07'
BODY_JOINTS = list(range(1, 22))                       # SMPL-X body pose = SMPL joints 1..21
BODY_NAMES = [SMPL_JOINT_NAMES[j] for j in BODY_JOINTS]
# SMPL body joint -> the Shadow sensor at that joint
SMPL_TO_SHADOW = {sm: sh for sh, sm in SHADOW_TO_SMPL.items()}


def shadow_to_body_pose(quats):
    """Shadow local quats (T,37,4 wxyz) -> SMPL body pose (T,21,3 axis-angle)."""
    T = quats.shape[0]
    sq = np.zeros((T, 24, 4)); sq[:, :, 0] = 1.0
    for sh, sm in SHADOW_TO_SMPL.items():
        sq[:, sm] = quats[:, sh]
    aa = np.zeros((T, 24, 3))
    for j in range(24):
        qx = np.concatenate([sq[:, j, 1:], sq[:, j, :1]], -1)
        aa[:, j] = Rotation.from_quat(qx).as_rotvec()
    return aa[:, 1:22]                                  # drop pelvis(0) + hands(22,23)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--expr-dir", default=EXPR_DIR, help="VPoser experiment directory")
    ap.add_argument("--stride", type=int, default=0, help="frame stride (0 = auto ~4000)")
    ap.add_argument("--top", type=int, default=8, help="joints to show in the heading map")
    args = ap.parse_args()

    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    stride = args.stride or max(1, T // 4000)
    idx = np.arange(0, T, stride)

    # global heading from pelvis world orientation
    parent, order, _ = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    G = fk_world(quats, parent, order)
    px = qrot_np(G[:, PELV], np.array([1.0, 0.0, 0.0]))
    heading = np.degrees(np.arctan2(px[:, 2], px[:, 0]))[idx]
    hbin = ((heading + 180) // 30).astype(int) % 12

    body = shadow_to_body_pose(quats)[idx].reshape(len(idx), 63).astype(np.float32)
    m = load_model(args.expr_dir, model_code=VPoser, remove_words_in_model_weights='vp_model.',
                   disable_grad=True, comp_device='cpu')[0].to('cpu').eval()

    x = torch.tensor(body, requires_grad=True)
    z = m.encode(x).mean
    nll = (z ** 2).sum(-1)                              # (n,) per-frame implausibility
    nll.sum().backward()
    g = x.grad.view(len(idx), 21, 3).norm(dim=-1).numpy()   # (n,21) per-joint attribution
    nll = nll.detach().numpy()

    print(f"{Path(args.infile).name}: {T} frames, {len(idx)} sampled (stride {stride})")
    print(f"pose implausibility ||z||^2  (plausible ~ 32):  "
          f"median {np.median(nll):.0f}  mean {nll.mean():.0f}  "
          f"90th pct {np.percentile(nll,90):.0f}  max {nll.max():.0f}")
    frac = (nll > 96).mean()                            # > 3x expected = clearly unnatural
    print(f"frames clearly implausible (||z||^2 > 96): {100*frac:.0f}%\n")

    # WHERE: implausibility vs global heading
    print("implausibility by global heading (mean ||z||^2 per 30-deg bin, -180..180):")
    bins = np.array([nll[hbin == b].mean() if (hbin == b).any() else 0 for b in range(12)])
    cnt = np.array([(hbin == b).sum() for b in range(12)])
    labels = [f"{-180+30*b:+4d}" for b in range(12)]
    print("  bin   " + " ".join(f"{l:>5}" for l in labels))
    print("  nll   " + " ".join(f"{v:5.0f}" for v in bins))
    print("  n     " + " ".join(f"{c:5d}" for c in cnt) + "\n")

    # WHICH: per-joint attribution, ranked (mask SMPL joints with no Shadow sensor -> stay
    # identity after conversion, so their implausibility is an artifact, not magnetization)
    jmean = g.mean(0)
    unmapped = np.array([BODY_JOINTS[r] not in SMPL_TO_SHADOW for r in range(21)])
    jmean[unmapped] = -np.inf
    if unmapped.any():
        print("(ignoring unmapped joints: "
              + ", ".join(BODY_NAMES[r] for r in range(21) if unmapped[r]) + ")")
    rank = np.argsort(jmean)[::-1]
    print("per-joint implausibility attribution (mean |grad|), ranked  -> implicated Shadow sensor")
    for r in rank[:args.top]:
        sh = SMPL_TO_SHADOW.get(BODY_JOINTS[r])
        shname = IDX_TO_NAME.get(sh, f'#{sh}')
        print(f"  {BODY_NAMES[r]:14s} {jmean[r]:7.3f}   sensor: {shname}")

    # WHICH x WHERE: top joints' attribution across heading bins
    print(f"\ntop-{args.top} joint attribution by heading bin (rows=joint, cols=heading):")
    print("  " + " ".join(f"{l:>5}" for l in labels))
    for r in rank[:args.top]:
        row = [g[hbin == b, r].mean() if (hbin == b).any() else 0 for b in range(12)]
        print(f"  {BODY_NAMES[r]:14s} " + " ".join(f"{v:5.2f}" for v in row))
    print("\nStage-1 diagnostic only -- no correction. High-attribution joints whose implausibility")
    print("concentrates at particular headings are the magnetization suspects.")


if __name__ == "__main__":
    main()