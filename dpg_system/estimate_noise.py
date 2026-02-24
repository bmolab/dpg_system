#!/usr/bin/env python3
"""
Motion capture noise estimator.

Detects glitchy frames, lists them, scores overall file quality, and
identifies the cleanest continuous sections. Works directly on axis-angle
poses for speed — no SMPL processor needed.

Usage:
    python estimate_noise.py file1.npz [file2.npz ...]
    python estimate_noise.py --dir /path/to/npz_files
    python estimate_noise.py file.npz --json results.json
"""

import numpy as np
import argparse
import os
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple


# ── Thresholds ─────────────────────────────────────────────────────────────
# Calibrated from:
#   Subject_81 (clean, 1 glitch ~frame 491): P99 ang_vel=13, glitch=53
#   Maritsa_Relaxed (very noisy): P99 ang_vel=55, 708 frames >20 rad/s

AV_GLITCH = 35.0           # rad/s — clear glitch (Subject_81 glitch=53)
AV_WARN   = 28.0           # rad/s — suspicious (throwing peaks at ~27)

AA_GLITCH = 1000.0         # rad/s² — Subject_81 glitch=6305
AA_WARN   = 600.0          # rad/s² — throwing peaks at ~890

TA_GLITCH = 80.0           # m/s² — cartwheel landings hit 60+
TA_WARN   = 50.0           # m/s² — hops/landings are 40-60

# Combination weights
W_AV = 1.0
W_AA = 0.4                 # Angular acceleration, deweighted
W_TA = 0.3                 # Translation accel, heavily deweighted
                            # (impacts ≠ glitches without angular evidence)

# Corroboration: translation-only events need angular evidence to flag.
# If ang_vel < this AND ang_acc < AA_WARN, trans_acc alone won't flag.
AV_CORROBORATE = 12.0      # rad/s — very modest angular motion

GLITCH_THRESH  = 0.5       # Per-frame score to flag as glitch
SUSPECT_THRESH = 0.25      # Per-frame score to flag as suspect

# Clean segment: minimum duration to report (seconds)
CLEAN_MIN_DURATION = 1.0


SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# Per-joint threshold multipliers based on physical inertia.
# High-inertia joints (pelvis, spine) keep 1.0× thresholds.
# Low-inertia joints (wrists, hands, feet) can physically move faster,
# so they get higher multipliers (= less sensitive to noise flagging).
JOINT_TOLERANCE = np.array([
    1.0,   # 0  pelvis        — heavy, low-speed
    1.0,   # 1  left_hip      — heavy
    1.0,   # 2  right_hip     — heavy
    1.0,   # 3  spine1        — heavy, central
    1.2,   # 4  left_knee     — moderate
    1.2,   # 5  right_knee    — moderate
    1.0,   # 6  spine2        — heavy, central
    1.5,   # 7  left_ankle    — lighter
    1.5,   # 8  right_ankle   — lighter
    1.0,   # 9  spine3        — heavy, central
    2.0,   # 10 left_foot     — light, fast
    2.0,   # 11 right_foot    — light, fast
    1.0,   # 12 neck          — moderate, central
    1.2,   # 13 left_collar   — moderate
    1.2,   # 14 right_collar  — moderate
    1.2,   # 15 head          — moderate
    1.2,   # 16 left_shoulder — moderate
    1.2,   # 17 right_shoulder— moderate
    1.5,   # 18 left_elbow    — lighter
    1.5,   # 19 right_elbow   — lighter
    2.5,   # 20 left_wrist    — very light, fast
    2.5,   # 21 right_wrist   — very light, fast
    3.0,   # 22 left_hand     — lightest
    3.0,   # 23 right_hand    — lightest
])


@dataclass
class FrameScore:
    frame: int
    score: float
    ang_vel: float          # max across joints (rad/s)
    ang_acc: float          # max across joints (rad/s²)
    trans_acc: float        # root acceleration (m/s²)
    worst_joint: int
    worst_joint_name: str


@dataclass
class CleanSegment:
    """A continuous stretch of clean frames."""
    start: int
    end: int                # inclusive
    n_frames: int
    duration_s: float
    max_score: float        # worst score in this segment
    mean_score: float       # average score in this segment


@dataclass
class FileReport:
    filename: str
    n_frames: int
    fps: float
    duration_s: float
    
    noise_score: float
    classification: str     # clean / moderate / problematic
    
    n_glitch_frames: int
    n_suspect_frames: int
    glitch_fraction: float
    
    max_ang_vel: float
    max_ang_acc: float
    max_trans_acc: float
    
    p95_ang_vel: float
    p99_ang_vel: float
    
    glitch_frames: List[FrameScore] = field(default_factory=list)
    glitch_clusters: List[Tuple[int, int]] = field(default_factory=list)
    clean_segments: List[CleanSegment] = field(default_factory=list)


def compute_frame_scores(poses, trans, fps, n_joints=24):
    T = poses.shape[0]
    aa = poses.reshape(T, -1, 3)[:, :n_joints]
    
    # Per-joint tolerance (broadcast-ready)
    tol = JOINT_TOLERANCE[:n_joints][np.newaxis, :]  # (1, J)
    
    # Angular velocity (rad/s)
    ang_disp = np.linalg.norm(np.diff(aa, axis=0), axis=-1)  # (T-1, J)
    ang_vel = np.vstack([np.zeros((1, n_joints)), ang_disp * fps])
    
    # Normalize by joint tolerance: fast wrist motion → lower effective value
    ang_vel_norm = ang_vel / tol
    
    max_av = np.max(ang_vel_norm, axis=1)       # effective (tolerance-adjusted)
    worst_j = np.argmax(ang_vel_norm, axis=1)
    max_av_raw = np.array([ang_vel[f, worst_j[f]] for f in range(T)])  # raw for reporting
    
    # Angular acceleration (rad/s²)
    ang_acc = np.abs(np.diff(ang_vel, axis=0)) * fps   # (T-1, J)
    ang_acc_norm = ang_acc / tol[:, :ang_acc.shape[1]] if ang_acc.shape[0] > 0 else ang_acc
    max_aa = np.max(ang_acc_norm, axis=1) if ang_acc_norm.shape[0] > 0 else np.zeros(0)
    max_aa = np.concatenate([np.zeros(1), max_aa])
    
    # Raw ang_acc for reporting
    max_aa_raw = np.zeros(T)
    if ang_acc.shape[0] > 0:
        aa_worst = np.argmax(ang_acc_norm, axis=1)
        max_aa_raw[1:] = np.array([ang_acc[f, aa_worst[f]] for f in range(ang_acc.shape[0])])
    
    # Translation acceleration (m/s²)
    tv = np.linalg.norm(np.diff(trans, axis=0), axis=-1) * fps
    ta_raw = np.abs(np.diff(tv)) * fps
    ta = np.concatenate([np.zeros(2), ta_raw])
    
    # Score (using tolerance-adjusted values against thresholds)
    s_av = np.maximum(0, (max_av - AV_WARN) / AV_WARN) * W_AV
    s_aa = np.maximum(0, (max_aa - AA_WARN) / AA_WARN) * W_AA
    s_ta = np.maximum(0, (ta - TA_WARN) / TA_WARN) * W_TA
    
    # Corroboration: suppress translation score when there's no angular evidence.
    # Pure impacts (landings, hops) have high trans_acc but clean rotation.
    no_angular = (max_av < AV_CORROBORATE) & (max_aa < AA_WARN)
    s_ta[no_angular] = 0.0
    
    total = s_av + s_aa + s_ta
    
    scores = []
    for f in range(T):
        wj = int(worst_j[f])
        name = SMPL_JOINT_NAMES[wj] if wj < len(SMPL_JOINT_NAMES) else f'j{wj}'
        scores.append(FrameScore(
            frame=f, score=round(float(total[f]), 4),
            ang_vel=round(float(max_av_raw[f]), 2),
            ang_acc=round(float(max_aa_raw[f]), 1),
            trans_acc=round(float(ta[f]), 2),
            worst_joint=wj, worst_joint_name=name,
        ))
    return scores


def find_clusters(frames, gap=5):
    """Group nearby frame indices into (start, end) clusters."""
    if not frames: return []
    clusters, start, end = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - end <= gap: end = f
        else: clusters.append((start, end)); start = end = f
    clusters.append((start, end))
    return clusters


def find_clean_segments(all_scores, fps, margin=3):
    """
    Find continuous stretches where NO frame is a glitch or suspect.
    
    Args:
        all_scores: array of per-frame scores.
        fps: frameratem for duration calculation.
        margin: extra frames to exclude around each bad frame (avoids
                edge artifacts from glitch spill-over).
    
    Returns:
        List of CleanSegment, sorted longest-first.
    """
    T = len(all_scores)
    # Mark bad frames (glitch + suspect) and expand by margin
    bad = all_scores >= SUSPECT_THRESH
    # Expand bad regions by margin frames on each side
    if margin > 0:
        bad_expanded = bad.copy()
        for m in range(1, margin + 1):
            bad_expanded[m:] |= bad[:-m]           # forward expand
            bad_expanded[:-m] |= bad[m:]           # backward expand
        bad = bad_expanded
    
    # Find contiguous clean runs
    segments = []
    in_clean = False
    start = 0
    
    for f in range(T):
        if not bad[f]:
            if not in_clean:
                start = f
                in_clean = True
        else:
            if in_clean:
                n = f - start
                dur = n / fps
                if dur >= CLEAN_MIN_DURATION:
                    seg_scores = all_scores[start:f]
                    segments.append(CleanSegment(
                        start=start, end=f-1, n_frames=n,
                        duration_s=round(dur, 2),
                        max_score=round(float(np.max(seg_scores)), 4),
                        mean_score=round(float(np.mean(seg_scores)), 4),
                    ))
                in_clean = False
    
    # Handle final segment
    if in_clean:
        n = T - start
        dur = n / fps
        if dur >= CLEAN_MIN_DURATION:
            seg_scores = all_scores[start:]
            segments.append(CleanSegment(
                start=start, end=T-1, n_frames=n,
                duration_s=round(dur, 2),
                max_score=round(float(np.max(seg_scores)), 4),
                mean_score=round(float(np.mean(seg_scores)), 4),
            ))
    
    # Sort by duration (longest first)
    segments.sort(key=lambda s: -s.n_frames)
    return segments


def analyze_file(filepath, verbose=True):
    d = np.load(filepath, allow_pickle=True)
    poses, trans, fps = d['poses'], d['trans'], float(d['mocap_framerate'])
    T = poses.shape[0]
    
    scores = compute_frame_scores(poses, trans, fps)
    
    all_s = np.array([s.score for s in scores])
    all_av = np.array([s.ang_vel for s in scores])
    all_aa = np.array([s.ang_acc for s in scores])
    all_ta = np.array([s.trans_acc for s in scores])
    
    glitch_mask = all_s >= GLITCH_THRESH
    suspect_mask = (all_s >= SUSPECT_THRESH) & ~glitch_mask
    n_g = int(np.sum(glitch_mask))
    n_s = int(np.sum(suspect_mask))
    g_frac = n_g / max(T, 1)
    
    glist = sorted([s for s in scores if s.score >= GLITCH_THRESH], key=lambda x: -x.score)
    clusters = find_clusters(sorted(s.frame for s in glist))
    clean = find_clean_segments(all_s, fps)
    
    peak = float(np.max(all_s)) if T > 0 else 0
    p99 = float(np.percentile(all_s, 99)) if T > 0 else 0
    
    ns = g_frac * 100 + min(peak * 0.01, 5) + p99 * 0.5
    
    if g_frac < 0.005 and peak < 5:
        cls = 'clean'
    elif g_frac < 0.02:
        cls = 'moderate'
    else:
        cls = 'problematic'
    
    r = FileReport(
        filename=os.path.basename(filepath), n_frames=T, fps=fps,
        duration_s=round(T/fps, 1), noise_score=round(ns, 4),
        classification=cls, n_glitch_frames=n_g, n_suspect_frames=n_s,
        glitch_fraction=round(g_frac, 6),
        max_ang_vel=round(float(np.max(all_av)), 1),
        max_ang_acc=round(float(np.max(all_aa)), 0),
        max_trans_acc=round(float(np.max(all_ta)), 1),
        p95_ang_vel=round(float(np.percentile(all_av, 95)), 2),
        p99_ang_vel=round(float(np.percentile(all_av, 99)), 2),
        glitch_frames=glist, glitch_clusters=clusters,
        clean_segments=clean,
    )
    if verbose: print_report(r)
    return r


def print_report(r):
    ic = {'clean': '✅', 'moderate': '⚠️', 'problematic': '❌'}
    
    print(f"\n{'═'*70}")
    print(f"  {r.filename}")
    print(f"  {r.n_frames} frames @ {r.fps:.0f} fps ({r.duration_s}s)")
    print(f"{'═'*70}")
    print(f"  Classification: {ic.get(r.classification)} {r.classification.upper()}")
    print(f"  Noise score:    {r.noise_score:.4f}")
    print(f"  Glitch frames:  {r.n_glitch_frames} ({100*r.glitch_fraction:.2f}%)")
    print(f"  Suspect frames: {r.n_suspect_frames}")
    print()
    print(f"  {'Metric':.<24s}  {'Max':>8s}  {'P95':>8s}  {'P99':>8s}")
    print(f"  {'─'*24}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'Angular velocity':.<24s}  {r.max_ang_vel:8.1f}  {r.p95_ang_vel:8.2f}  {r.p99_ang_vel:8.2f}  rad/s")
    print(f"  {'Angular acceleration':.<24s}  {r.max_ang_acc:8.0f}  {'':>8s}  {'':>8s}  rad/s²")
    print(f"  {'Translation accel.':.<24s}  {r.max_trans_acc:8.1f}  {'':>8s}  {'':>8s}  m/s²")
    
    # ── Glitch regions ─────────────────────────────────────────────────
    if r.glitch_clusters:
        print(f"\n  Glitch regions ({len(r.glitch_clusters)}):")
        for start, end in r.glitch_clusters[:25]:
            n = end - start + 1
            pk = max((s.score for s in r.glitch_frames if start <= s.frame <= end), default=0)
            print(f"    [{start:>6d}–{end:>6d}]  {n:>4d} frames  "
                  f"t={start/r.fps:5.1f}–{end/r.fps:5.1f}s  peak={pk:.1f}")
        if len(r.glitch_clusters) > 25:
            print(f"    ... +{len(r.glitch_clusters)-25} more")
    
    # ── Worst frames ───────────────────────────────────────────────────
    if r.glitch_frames:
        n_show = min(20, len(r.glitch_frames))
        print(f"\n  Worst {n_show} frames:")
        print(f"    {'Frame':>6s}  {'Score':>7s}  {'AngVel':>8s}  {'AngAcc':>8s}  {'TransA':>7s}  Joint")
        for s in r.glitch_frames[:n_show]:
            print(f"    {s.frame:6d}  {s.score:7.2f}  {s.ang_vel:8.1f}  {s.ang_acc:8.0f}  "
                  f"{s.trans_acc:7.1f}  {s.worst_joint_name}")
    
    # ── All glitch frame indices ───────────────────────────────────────
    if r.glitch_frames and len(r.glitch_frames) <= 500:
        indices = sorted(s.frame for s in r.glitch_frames)
        print(f"\n  All {len(indices)} glitch frame indices:")
        for i in range(0, len(indices), 20):
            chunk = indices[i:i+20]
            print(f"    {', '.join(str(x) for x in chunk)}")
    elif r.glitch_frames:
        print(f"\n  Total glitch frames: {len(r.glitch_frames)} (too many to list)")
    
    # ── Clean segments ─────────────────────────────────────────────────
    if r.clean_segments:
        total_clean = sum(s.n_frames for s in r.clean_segments)
        clean_pct = 100 * total_clean / max(r.n_frames, 1)
        print(f"\n  Clean segments ({len(r.clean_segments)} found, "
              f"{total_clean} frames = {clean_pct:.1f}% of file):")
        print(f"    {'Rank':>4s}  {'Frames':>16s}  {'Duration':>8s}  {'MaxScore':>9s}  {'MeanScore':>10s}")
        n_show = min(15, len(r.clean_segments))
        for i, seg in enumerate(r.clean_segments[:n_show]):
            print(f"    {i+1:4d}  [{seg.start:>6d}–{seg.end:>6d}]  "
                  f"{seg.duration_s:6.1f}s   {seg.max_score:9.4f}  {seg.mean_score:10.4f}")
        if len(r.clean_segments) > n_show:
            remaining = r.clean_segments[n_show:]
            rem_frames = sum(s.n_frames for s in remaining)
            print(f"    ... +{len(remaining)} smaller segments ({rem_frames} frames)")
    else:
        print(f"\n  ⚠️  No clean segments found (>= {CLEAN_MIN_DURATION}s without glitches)")
    
    print()


def main():
    p = argparse.ArgumentParser(description='Estimate mocap noise')
    p.add_argument('files', nargs='*')
    p.add_argument('--dir')
    p.add_argument('--json')
    args = p.parse_args()
    
    files = list(args.files)
    if args.dir:
        files += [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir)) if f.endswith('.npz')]
    if not files: p.print_help(); return
    
    reports = []
    for f in files:
        if not os.path.exists(f): print(f"  ⚠️  Not found: {f}"); continue
        try: reports.append(analyze_file(f))
        except Exception as e: print(f"  ❌ Error: {f}: {e}")
    
    if len(reports) > 1:
        print(f"\n{'═'*70}")
        print(f"  SUMMARY ({len(reports)} files)")
        print(f"{'═'*70}")
        print(f"  {'File':<40s}  {'Score':>7s}  {'Class':>12s}  {'Glitches':>8s}  {'Clean%':>6s}")
        for r in sorted(reports, key=lambda x: -x.noise_score):
            e = {'clean':'✅','moderate':'⚠️','problematic':'❌'}.get(r.classification,'?')
            tc = sum(s.n_frames for s in r.clean_segments)
            cp = 100 * tc / max(r.n_frames, 1)
            print(f"  {r.filename:<40s}  {r.noise_score:7.3f}  {e} {r.classification:<10s}  "
                  f"{r.n_glitch_frames:>8d}  {cp:5.1f}%")
    
    if args.json:
        out = []
        for r in reports:
            d = asdict(r)
            d['glitch_frames'] = [asdict(s) for s in r.glitch_frames[:500]]
            d['clean_segments'] = [asdict(s) for s in r.clean_segments]
            out.append(d)
        with open(args.json, 'w') as f: json.dump(out, f, indent=2)
        print(f"\n  Saved to {args.json}")


if __name__ == '__main__':
    main()
