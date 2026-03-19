#!/usr/bin/env python3
"""
Torque-based motion capture noise estimator.

Runs SMPLProcessor torque computation on .npz files (streaming mode,
matching smpl_torque node behaviour) and detects anomalous torque events
that indicate capture noise rather than real movement.

Key detection signals:
  1. Torque surprise  — sudden spikes relative to local rolling context
  2. Torque/velocity   — high torque at low angular velocity
  3. Direction jitter  — rapid sign flips in torque direction
  4. Effort anomaly    — effort (τ/τ_max) spiking relative to local baseline

Usage:
    python estimate_noise_torque.py file1.npz [file2.npz ...]
    python estimate_noise_torque.py --dir /path/to/npz_files
    python estimate_noise_torque.py file.npz --json results.json
    python estimate_noise_torque.py file.npz --mass 70 --gender male
"""

import numpy as np
import argparse
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# Ensure dpg_system is importable when running as a standalone script
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions


# ── SMPL joint names (first 22 used) ─────────────────────────────────────
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

N_JOINTS = 22

# ── Detection thresholds ──────────────────────────────────────────────────

# Rolling window duration (seconds) for local statistics
WINDOW_DURATION = 1.0  # seconds — wider window for more stable local stats

# Minimum dynamic torque gate — fraction of per-joint max_torque.
# Below this fraction, torque is considered baseline noise and excluded from
# surprise/ratio signals.  E.g., 0.02 means ignore torques below 2% of max.
MIN_DYN_TORQUE_FRAC = 0.02

# Torque surprise: |τ - median| / (MAD + ε)
# Calibrated against Subject_81 (clean gait) and Maritsa (noisy static poses).
# The MAD epsilon (0.5 N·m) prevents tiny MAD values from inflating surprise
# when the torque baseline is low but non-zero.
SURPRISE_GLITCH = 20.0     # Definite anomaly
SURPRISE_WARN = 12.0       # Suspicious

# Torque/velocity ratio: |τ_dyn| / max(|ω|, ε)
# High torque at low velocity = suspicious.  Normal gait can produce ratios
# up to ~300 at 120fps during foot contacts.  Real glitches produce 500+.
TV_RATIO_GLITCH = 1000.0   # N·m / (rad/s) — extreme
TV_RATIO_WARN = 500.0      # Suspicious
TV_VELOCITY_FLOOR = 0.05   # rad/s — below this, velocity is "near zero"

# Effort (τ/τ_max) anomaly relative to local context
EFFORT_SPIKE_GLITCH = 4.0  # effort spike vs local median (multiplier)
EFFORT_SPIKE_WARN = 2.5

# Jitter: sign-change rate in rolling window
JITTER_WINDOW = 10         # frames
JITTER_THRESH = 0.5        # fraction of frames with sign changes

# Combined per-frame score weights
W_SURPRISE = 1.0
W_TV_RATIO = 0.5
W_EFFORT = 0.3
W_JITTER = 0.2

# Thresholds for frame classification
GLITCH_THRESH = 0.5
SUSPECT_THRESH = 0.2

# Transience filter: noise glitches are brief (1-5 frames).
# If a joint's torque surprise stays elevated for longer than this, it's
# likely real sustained movement (bouncing, walking) not a glitch.
TRANSIENCE_WINDOW = 10     # frames — runs longer than this get discounted
TRANSIENCE_DISCOUNT = 0.2  # score multiplier for sustained elevations

# Contact-event discount: torque spikes coinciding with contact state changes
# (touchdown, liftoff, hand plant) are expected physics, not noise.
CONTACT_EVENT_WINDOW = 8        # frames before/after transition to discount
CONTACT_EVENT_DISCOUNT = 0.15   # score multiplier during contact events
CONTACT_FORCE_THRESH = 10.0     # N — minimum force to consider as active contact

# Airborne discount: when total contact force is zero (freefall), torque anomalies
# are expected from inertial effects / unmodeled contacts (hand plants, tumbling).
AIRBORNE_WINDOW = 5             # frames before/after airborne phase to discount
AIRBORNE_DISCOUNT = 0.1         # score multiplier during airborne phases

# Kinetic chain: when ANY joint contacts the ground, torque spikes propagate up
# the entire chain to the root.  Built dynamically from the SMPL parent hierarchy.
# Extended to include special contact joints (knuckles, toe tips, finger tips, heels).
SMPL_PARENTS_EXTENDED = [
    -1,  #  0 pelvis
     0,  #  1 left_hip
     0,  #  2 right_hip
     0,  #  3 spine1
     1,  #  4 left_knee
     2,  #  5 right_knee
     3,  #  6 spine2
     4,  #  7 left_ankle
     5,  #  8 right_ankle
     6,  #  9 spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_knuckle  (left_hand)    → parent: left_wrist
    21,  # 23 right_knuckle (right_hand)   → parent: right_wrist
    10,  # 24 left_toe_tip                 → parent: left_foot
    11,  # 25 right_toe_tip                → parent: right_foot
    20,  # 26 left_finger_tip              → parent: left_wrist
    21,  # 27 right_finger_tip             → parent: right_wrist
     7,  # 28 left_heel                    → parent: left_ankle
     8,  # 29 right_heel                   → parent: right_ankle
]
N_CONTACT_JOINTS = len(SMPL_PARENTS_EXTENDED)  # 30

def _build_contact_chain():
    """For each joint, compute the set of all ancestor joints (including self).
    Only includes ancestors within the first N_JOINTS (0-21) for scoring."""
    chain = {}
    for j in range(len(SMPL_PARENTS_EXTENDED)):
        ancestors = set()
        p = j
        # Walk up the hierarchy, collecting ancestors within N_JOINTS
        while p >= 0:
            if p < N_JOINTS:
                ancestors.add(p)
            if p >= len(SMPL_PARENTS_EXTENDED) or SMPL_PARENTS_EXTENDED[p] < 0:
                break
            p = SMPL_PARENTS_EXTENDED[p]
        chain[j] = ancestors
    return chain

CONTACT_CHAIN = _build_contact_chain()

# Clean segment minimum duration
CLEAN_MIN_DURATION = 1.0   # seconds

# Joint importance weights.
# Contact joints (ankles, feet) down-weighted because normal gait contact
# transitions always produce torque spikes.  Spine/hip anomalies are the most
# reliable noise indicators.
JOINT_IMPORTANCE = np.array([
    0.2,   #  0 pelvis       — root, mostly dominated by whole-body motion
    0.3,   #  1 left_hip     — weight-bearing, legit high torque during movement
    0.3,   #  2 right_hip    — weight-bearing, legit high torque during movement
    1.2,   #  3 spine1       — core stability, strong noise indicator
    0.3,   #  4 left_knee    — weight-bearing, legit high torque during bouncing
    0.3,   #  5 right_knee   — weight-bearing, legit high torque during bouncing
    1.2,   #  6 spine2       — core stability, strong noise indicator
    0.3,   #  7 left_ankle   — contact-affected, down-weight
    0.3,   #  8 right_ankle  — contact-affected, down-weight
    1.2,   #  9 spine3       — core stability, strong noise indicator
    0.15,  # 10 left_foot    — contact-affected, heavily down-weight
    0.15,  # 11 right_foot   — contact-affected, heavily down-weight
    0.8,   # 12 neck         — moderate
    0.6,   # 13 left_collar  — moderate
    0.6,   # 14 right_collar — moderate
    0.4,   # 15 head         — moderate
    0.7,   # 16 left_shoulder  — good noise indicator
    0.7,   # 17 right_shoulder — good noise indicator
    0.4,   # 18 left_elbow
    0.4,   # 19 right_elbow
    0.2,   # 20 left_wrist   — very light
    0.2,   # 21 right_wrist  — very light
])


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class FrameScore:
    frame: int
    score: float
    surprise: float          # max torque surprise across joints
    tv_ratio: float          # max torque/velocity ratio
    effort_spike: float      # max effort spike vs local
    jitter_rate: float       # max jitter rate
    worst_joint: int
    worst_joint_name: str


@dataclass
class CleanSegment:
    start: int
    end: int                 # inclusive
    n_frames: int
    duration_s: float
    max_score: float
    mean_score: float


@dataclass
class JointNoiseProfile:
    joint_idx: int
    joint_name: str
    n_glitch_frames: int
    n_suspect_frames: int
    mean_surprise: float
    max_surprise: float
    mean_effort: float
    max_effort: float
    glitch_fraction: float


@dataclass
class StreamBreak:
    frame: int
    time_s: float
    trans_jump: float        # root translation displacement (m)
    pose_jump: float         # max per-joint angular displacement (rad)
    worst_joint: str         # joint with largest angular jump
    break_type: str          # 'translation', 'pose', or 'both'


@dataclass
class CadenceInfo:
    detected: bool
    period: int              # frames (e.g. 3 for Shadow)
    strength: float          # mean autocorrelation at cadence lag
    coverage: float          # fraction of windows showing cadence


@dataclass
class TorqueFileReport:
    filename: str
    n_frames: int
    fps: float
    duration_s: float

    noise_score: float
    classification: str      # clean / moderate / problematic

    n_glitch_frames: int
    n_suspect_frames: int
    glitch_fraction: float

    max_surprise: float
    max_tv_ratio: float
    max_effort: float

    p95_score: float
    p99_score: float

    joint_profiles: List[JointNoiseProfile] = field(default_factory=list)
    glitch_frames: List[FrameScore] = field(default_factory=list)
    glitch_clusters: List[Tuple[int, int]] = field(default_factory=list)
    clean_segments: List[CleanSegment] = field(default_factory=list)
    stream_breaks: List[StreamBreak] = field(default_factory=list)
    cadence: Optional[CadenceInfo] = None
    motion_profile: Optional['MotionProfile'] = None
    ground_contact: Optional['GroundContactInfo'] = None
    classification_detail: str = ''    # multi-dimensional summary line
    recommendations: Optional[dict] = None  # suggested filter params


@dataclass
class MotionProfile:
    mean_vel: float          # mean total body angular velocity (rad/s)
    p50_vel: float
    p90_vel: float
    p99_vel: float
    max_vel: float
    n_still: int             # frames below p25
    n_active: int            # frames above p75
    n_explosive: int         # frames above p95
    complexity: str          # 'static', 'moderate', 'dynamic', 'explosive'


@dataclass
class GroundContactInfo:
    ground_pct: float        # fraction of time on ground
    n_phases: int            # number of distinct ground phases
    longest_phase_s: float   # duration of longest ground phase
    standing_height: float   # estimated standing height (m)
    avg_ground_height: float # average root height during ground phases


# ── Core analysis ─────────────────────────────────────────────────────────

def _rolling_median_mad(data, window):
    """
    Compute rolling median and MAD (Median Absolute Deviation) for a 1D array.
    Uses a causal window (current frame + preceding frames).
    
    Args:
        data: (T,) array
        window: int, window size
    Returns:
        median: (T,) rolling median
        mad: (T,) rolling MAD
    """
    T = len(data)
    median = np.zeros(T)
    mad = np.zeros(T)
    
    for t in range(T):
        start = max(0, t - window + 1)
        chunk = data[start:t + 1]
        m = np.median(chunk)
        median[t] = m
        mad[t] = np.median(np.abs(chunk - m))
    
    return median, mad


def _compute_angular_velocity(poses_aa, fps, n_joints=22):
    """
    Compute per-joint angular velocity magnitude from axis-angle poses.
    
    Args:
        poses_aa: (T, J*3) or (T, J, 3) axis-angle poses
        fps: framerate
        n_joints: number of joints to process
    Returns:
        ang_vel: (T, J) angular velocity magnitudes in rad/s
    """
    T = poses_aa.shape[0]
    aa = poses_aa.reshape(T, -1, 3)[:, :n_joints]
    
    # Frame-to-frame angular displacement
    disp = np.linalg.norm(np.diff(aa, axis=0), axis=-1)  # (T-1, J)
    vel = disp * fps
    # Pad first frame with zeros
    ang_vel = np.vstack([np.zeros((1, n_joints)), vel])
    return ang_vel


# ── Stream break & cadence detection ──────────────────────────────────────

STREAM_BREAK_TRANS_FACTOR = 20.0   # flag if root displacement > factor × median
STREAM_BREAK_POSE_FACTOR = 15.0    # flag if max-joint angle change > factor × median
STREAM_BREAK_RADIUS = 3            # frames around break to discount in scoring
STREAM_BREAK_DISCOUNT = 0.1        # score multiplier for stream break frames

def _detect_stream_breaks(poses, trans, fps):
    """Detect single-frame discontinuities in root position or joint angles.
    
    Returns:
        breaks: list of StreamBreak
        break_mask: (T,) boolean array, True for frames near a break
    """
    T = poses.shape[0]
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    poses_3d = poses.reshape(T, -1, 3)[:, :min(nj, N_JOINTS), :]
    
    # Root translation displacement
    trans_disp = np.linalg.norm(np.diff(trans, axis=0), axis=1)  # (T-1,)
    trans_median = np.median(trans_disp) if len(trans_disp) > 0 else 0
    
    # Per-joint angular displacement, take max across joints
    pose_disp = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2)  # (T-1, J)
    pose_max = np.max(pose_disp, axis=1)  # (T-1,)
    pose_median = np.median(pose_max) if len(pose_max) > 0 else 0
    
    trans_thresh = max(trans_median * STREAM_BREAK_TRANS_FACTOR, 0.05)  # at least 5cm
    pose_thresh = max(pose_median * STREAM_BREAK_POSE_FACTOR, 0.3)     # at least ~17°
    
    breaks = []
    break_frames = set()
    
    for f in range(len(trans_disp)):
        is_trans = trans_disp[f] > trans_thresh
        is_pose = pose_max[f] > pose_thresh
        
        if is_trans or is_pose:
            worst_j_idx = int(np.argmax(pose_disp[f]))
            worst_j_name = SMPL_JOINT_NAMES[worst_j_idx] if worst_j_idx < len(SMPL_JOINT_NAMES) else f'j{worst_j_idx}'
            
            if is_trans and is_pose:
                btype = 'both'
            elif is_trans:
                btype = 'translation'
            else:
                btype = 'pose'
            
            breaks.append(StreamBreak(
                frame=f + 1,  # the discontinuity is AT frame f+1
                time_s=round((f + 1) / fps, 2),
                trans_jump=round(float(trans_disp[f]), 4),
                pose_jump=round(float(pose_max[f]), 4),
                worst_joint=worst_j_name,
                break_type=btype,
            ))
            # Mark frames around the break
            for ff in range(max(0, f + 1 - STREAM_BREAK_RADIUS),
                           min(T, f + 1 + STREAM_BREAK_RADIUS + 1)):
                break_frames.add(ff)
    
    break_mask = np.zeros(T, dtype=bool)
    for ff in break_frames:
        break_mask[ff] = True
    
    return breaks, break_mask


def _detect_cadence(poses, fps, max_period=8):
    """Detect repeating sub-sampling cadence via autocorrelation of velocity.
    
    Returns:
        CadenceInfo or None
    """
    T = poses.shape[0]
    if T < 200:
        return CadenceInfo(detected=False, period=0, strength=0.0, coverage=0.0)
    
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    poses_3d = poses.reshape(T, -1, 3)[:, :min(nj, N_JOINTS), :]
    
    # Velocity magnitude per joint
    vel = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2)  # (T-1, J)
    vel_mean = np.mean(vel, axis=1)  # (T-1,) mean across joints
    
    # Windowed autocorrelation at each lag
    window = 30
    n_windows = max(1, (T - 1) // window)
    
    best_lag = 0
    best_strength = 0
    best_coverage = 0
    
    for lag in range(2, max_period + 1):
        strengths = []
        for w in range(n_windows):
            s = w * window
            e = min(s + window, T - 1)
            if e - s < lag + 3:
                continue
            chunk = vel_mean[s:e]
            chunk_c = chunk - np.mean(chunk)
            var = np.var(chunk_c)
            if var > 1e-10:
                ac = np.mean(chunk_c[lag:] * chunk_c[:-lag]) / var
                strengths.append(ac)
        
        if strengths:
            mean_ac = np.mean(strengths)
            coverage = np.mean([1 if s > 0.3 else 0 for s in strengths])
            if mean_ac > best_strength:
                best_strength = mean_ac
                best_lag = lag
                best_coverage = coverage
    
    detected = best_strength > 0.3 and best_coverage > 0.3
    return CadenceInfo(
        detected=detected,
        period=best_lag,
        strength=round(best_strength, 3),
        coverage=round(best_coverage, 3),
    )


# Walking baseline reference values (from typical clean gait files)
WALKING_BASELINE_MEAN_VEL = 25.0   # rad/s total body angular velocity
WALKING_BASELINE_P90_VEL = 50.0

def _compute_motion_profile(poses, fps):
    """Compute body-level motion statistics from raw poses."""
    T = poses.shape[0]
    if T < 10:
        return MotionProfile(0, 0, 0, 0, 0, 0, 0, 0, 'static')
    
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    poses_3d = poses.reshape(T, -1, 3)[:, :min(nj, N_JOINTS), :]
    dt = 1.0 / fps
    
    # Per-joint angular velocity magnitude, summed across joints
    vel = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2) / dt  # (T-1, J)
    vel_total = np.sum(vel, axis=1)  # (T-1,)
    
    mean_v = float(np.mean(vel_total))
    p50 = float(np.median(vel_total))
    p90 = float(np.percentile(vel_total, 90))
    p99 = float(np.percentile(vel_total, 99))
    max_v = float(np.max(vel_total))
    
    p25_thresh = np.percentile(vel_total, 25)
    p75_thresh = np.percentile(vel_total, 75)
    p95_thresh = np.percentile(vel_total, 95)
    
    n_still = int(np.sum(vel_total < p25_thresh))
    n_active = int(np.sum(vel_total > p75_thresh))
    n_explosive = int(np.sum(vel_total > p95_thresh))
    
    # Classify overall complexity
    if mean_v < 10:
        complexity = 'static'
    elif mean_v < 30:
        complexity = 'moderate'
    elif p90 < 100:
        complexity = 'dynamic'
    else:
        complexity = 'explosive'
    
    return MotionProfile(
        mean_vel=round(mean_v, 1), p50_vel=round(p50, 1),
        p90_vel=round(p90, 1), p99_vel=round(p99, 1),
        max_vel=round(max_v, 1),
        n_still=n_still, n_active=n_active, n_explosive=n_explosive,
        complexity=complexity,
    )


def _analyze_ground_contact(trans, fps):
    """Analyze ground contact phases from root translation height."""
    T = trans.shape[0]
    if T < 10:
        return GroundContactInfo(0, 0, 0, 0, 0)
    
    # SMPL pose files use Z-up
    root_height = trans[:, 2]
    standing_h = float(np.percentile(root_height, 90))
    low_thresh = standing_h * 0.5
    
    is_low = root_height < low_thresh
    
    # Find sustained low phases (>0.5s)
    phases = []
    in_low = False
    start = 0
    for f in range(T):
        if is_low[f] and not in_low:
            in_low = True; start = f
        elif not is_low[f] and in_low:
            in_low = False
            dur = (f - start) / fps
            if dur > 0.5:
                phases.append((start, f - 1, dur))
    if in_low:
        dur = (T - start) / fps
        if dur > 0.5:
            phases.append((start, T - 1, dur))
    
    total_low_frames = sum(e - s + 1 for s, e, _ in phases)
    ground_pct = total_low_frames / T if T > 0 else 0
    longest = max((d for _, _, d in phases), default=0)
    
    # Average height during ground phases
    if phases:
        low_heights = np.concatenate([root_height[s:e+1] for s, e, _ in phases])
        avg_h = float(np.mean(low_heights))
    else:
        avg_h = standing_h
    
    return GroundContactInfo(
        ground_pct=round(ground_pct, 3),
        n_phases=len(phases),
        longest_phase_s=round(longest, 1),
        standing_height=round(standing_h, 3),
        avg_ground_height=round(avg_h, 3),
    )


def _build_classification(report):
    """Build multi-dimensional classification and filter recommendations."""
    parts = []
    recs = {}
    
    # Core noise classification
    ic = {'clean': '✅', 'moderate': '⚠️', 'problematic': '❌'}
    parts.append(f"{ic.get(report.classification, '')} {report.classification.upper()} capture")
    
    # Stream breaks
    if report.stream_breaks:
        parts.append(f"⚡ {len(report.stream_breaks)} stream break(s)")
    
    # Cadence
    if report.cadence and report.cadence.detected:
        parts.append(f"🔄 {report.cadence.period}-frame cadence")
        recs['smooth_input_window'] = report.cadence.period
    
    # Ground work
    if report.ground_contact and report.ground_contact.ground_pct > 0.4:
        pct = int(report.ground_contact.ground_pct * 100)
        parts.append(f"⬇️ {pct}% ground work")
        recs['smooth_contact_forces'] = True
    
    # Motion complexity
    if report.motion_profile:
        mp = report.motion_profile
        if mp.complexity == 'explosive':
            parts.append(f"💥 explosive motion (p90={mp.p90_vel:.0f} rad/s)")
        elif mp.complexity == 'dynamic':
            parts.append(f"🏃 dynamic motion (mean={mp.mean_vel:.0f} rad/s)")
    
    # Velocity-adaptive smoothing recommendation
    if report.classification in ('moderate', 'problematic'):
        if report.motion_profile and report.motion_profile.complexity != 'static':
            recs['enable_velocity_gate'] = True
    
    # acc_smooth as safe default for moderate+ noise
    if report.classification in ('moderate', 'problematic'):
        recs['acc_smooth_window'] = 5
    
    detail = '  '.join(parts)
    return detail, recs


def analyze_file(filepath, total_mass=75.0, gender_override=None, verbose=True):
    """
    Run torque-based noise analysis on a single .npz file.
    
    Args:
        filepath: path to .npz file
        total_mass: body mass in kg (default 75)
        gender_override: override gender from file metadata
        verbose: print report to console
    
    Returns:
        TorqueFileReport
    """
    # ── Load file ─────────────────────────────────────────────────────
    d = np.load(filepath, allow_pickle=True)
    poses = d['poses']
    trans = d['trans']
    
    # Framerate: try multiple keys
    fps = 60.0
    for key in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if key in d:
            fps = float(d[key])
            break
    
    # Betas: optional
    betas = None
    if 'betas' in d:
        betas = np.array(d['betas'], dtype=np.float64)
        if betas.ndim > 1:
            betas = betas.flatten()[:10]
    
    # Gender: optional, with override
    gender = 'neutral'
    if gender_override:
        gender = gender_override
    elif 'gender' in d:
        g = d['gender']
        if hasattr(g, 'item'):
            g = g.item()
        gender = str(g)
        if gender not in ('male', 'female', 'neutral'):
            gender = 'neutral'
    
    T = poses.shape[0]
    dt = 1.0 / fps
    window_frames = max(3, int(WINDOW_DURATION * fps))
    
    if verbose:
        print(f"\n  Loading {os.path.basename(filepath)}: {T} frames @ {fps:.0f} fps, "
              f"gender={gender}, mass={total_mass}kg")
    
    # ── Detect stream breaks (before processing) ─────────────────────
    stream_breaks, break_mask = _detect_stream_breaks(poses, trans, fps)
    if verbose and stream_breaks:
        print(f"  Detected {len(stream_breaks)} stream break(s)")
    
    # ── Detect cadence pattern ────────────────────────────────────────
    cadence = _detect_cadence(poses, fps)
    if verbose and cadence and cadence.detected:
        print(f"  Detected {cadence.period}-frame cadence (strength={cadence.strength:.2f}, coverage={cadence.coverage:.0%})")
    
    # ── Motion profile ────────────────────────────────────────────────
    motion_profile = _compute_motion_profile(poses, fps)
    if verbose:
        print(f"  Motion: {motion_profile.complexity} (mean={motion_profile.mean_vel:.0f}, p90={motion_profile.p90_vel:.0f} rad/s)")
    
    # ── Ground contact analysis ───────────────────────────────────────
    ground_contact = _analyze_ground_contact(trans, fps)
    if verbose and ground_contact.ground_pct > 0.1:
        print(f"  Ground work: {ground_contact.ground_pct:.0%} of file ({ground_contact.n_phases} phases)")
    
    # ── Compute angular velocity from raw poses (before processing) ──
    ang_vel = _compute_angular_velocity(poses, fps)  # (T, J)
    
    # ── Initialize SMPLProcessor ──────────────────────────────────────
    model_path = os.path.dirname(os.path.abspath(__file__))
    processor = SMPLProcessor(
        framerate=fps,
        betas=betas,
        gender=gender,
        total_mass_kg=total_mass,
        model_path=model_path,
    )
    processor.set_axis_permutation('x, z, -y')
    
    # Options matching smpl_torque node defaults
    # IMPORTANT: The CoM One Euro filter params below must match the node's
    # widget defaults, NOT the SMPLProcessingOptions dataclass defaults.
    # The node defaults have com_pos_min_cutoff=999 (position filter OFF),
    # while the dataclass defaults have 8.0 (heavy smoothing that hides spikes).
    options = SMPLProcessingOptions(
        input_type='axis_angle',
        input_up_axis='Y',
        axis_permutation='x, z, -y',
        quat_format='wxyz',
        return_quats=False,
        dt=dt,
        add_gravity=True,
        enable_passive_limits=True,
        enable_apparent_gravity=True,
        floor_enable=True,
        floor_height=0.0,
        floor_tolerance=0.15,
        contact_method='stability_v2',
        world_frame_dynamics=True,
        use_s_curve_spine=True,
        enable_frame_evaluator=True,    # Match node: physics-based contact forces
        # All rate limiting / filtering OFF (match node defaults)
        enable_rate_limiting=False,
        enable_jitter_damping=False,
        enable_kf_smoothing=False,
        enable_velocity_gate=False,
        enable_one_euro_filter=False,
        smooth_input_window=0,
        # CoM One Euro filter params — match smpl_torque node widget defaults
        com_pos_min_cutoff=999.0,   # Position filter OFF (node default)
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,    # Velocity filter — light smoothing
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,     # Acceleration filter
        com_acc_beta=0.8,
    )
    
    # Get max torque array for effort computation
    max_torque_arr = processor.max_torque_array[:N_JOINTS]  # (22, 3)
    max_torque_mag = np.linalg.norm(max_torque_arr, axis=-1)  # (22,)
    max_torque_mag = np.maximum(max_torque_mag, 1.0)  # avoid div-by-zero
    
    # ── Stream frames through processor ───────────────────────────────
    if verbose:
        print(f"  Processing {T} frames in streaming mode...")
    
    dyn_torques = np.zeros((T, N_JOINTS, 3))    # dynamic torque vectors
    net_torques = np.zeros((T, N_JOINTS, 3))     # net torque vectors
    contact_forces = np.zeros((T, N_CONTACT_JOINTS))  # contact pressure per joint (N)
    
    for t in range(T):
        # Reshape single frame: (1, 24, 3) for poses, (1, 3) for trans
        frame_pose = poses[t:t+1]  # (1, 72) or (1, J, 3)
        frame_trans = trans[t:t+1]  # (1, 3)
        
        try:
            res = processor.process_frame(frame_pose, frame_trans, options)
            
            t_dyn = res.get('torques_dyn_vec', None)
            t_net = res.get('torques_vec', None)
            
            if t_dyn is not None:
                n = min(N_JOINTS, t_dyn.shape[1])
                dyn_torques[t, :n] = t_dyn[0, :n]
            if t_net is not None:
                n = min(N_JOINTS, t_net.shape[1])
                net_torques[t, :n] = t_net[0, :n]
            
            # Capture contact forces from frame evaluator
            cp = res.get('contact_pressure', None)
            if cp is not None:
                cp_flat = cp.flatten() if cp.ndim > 1 else cp
                n_cp = min(N_CONTACT_JOINTS, len(cp_flat))
                contact_forces[t, :n_cp] = cp_flat[:n_cp]
                
        except Exception as e:
            # First few frames may produce partial results
            pass
        
        if verbose and (t + 1) % 500 == 0:
            print(f"    frame {t+1}/{T}")
    
    if verbose:
        print(f"  Processing complete. Analysing torques...")
    
    # ── Compute per-joint, per-frame anomaly signals ──────────────────
    
    # Dynamic torque magnitudes: (T, J)
    dyn_mag = np.linalg.norm(dyn_torques, axis=-1)
    net_mag = np.linalg.norm(net_torques, axis=-1)
    
    # Effort: |τ_net| / τ_max  (soft signal)
    effort = net_mag / max_torque_mag[np.newaxis, :]  # (T, J)
    
    # Minimum torque gate: per-joint threshold as fraction of max_torque.
    # This prevents near-baseline torques from producing huge surprise/ratio values.
    min_torque_per_joint = max_torque_mag * MIN_DYN_TORQUE_FRAC  # (J,)
    torque_gate = dyn_mag >= min_torque_per_joint[np.newaxis, :]  # (T, J) boolean
    
    # --- Signal 1: Torque surprise (rolling median/MAD) ---
    surprise = np.zeros((T, N_JOINTS))
    for j in range(N_JOINTS):
        med, mad = _rolling_median_mad(dyn_mag[:, j], window_frames)
        # MAD-based z-score, gated by minimum torque magnitude.
        # The epsilon (0.5) prevents tiny MAD from inflating surprise at low baselines.
        raw_surprise = np.abs(dyn_mag[:, j] - med) / (mad + 0.5)
        surprise[:, j] = raw_surprise * torque_gate[:, j]
    
    # --- Signal 2: Torque/velocity ratio ---
    tv_ratio = np.zeros((T, N_JOINTS))
    for j in range(N_JOINTS):
        vel_j = np.maximum(ang_vel[:, j], TV_VELOCITY_FLOOR)
        raw_tv = dyn_mag[:, j] / vel_j
        tv_ratio[:, j] = raw_tv * torque_gate[:, j]  # gate by min torque
    
    # --- Signal 3: Effort spike relative to local baseline ---
    effort_spike = np.zeros((T, N_JOINTS))
    for j in range(N_JOINTS):
        med, mad = _rolling_median_mad(effort[:, j], window_frames)
        # Ratio: current effort / local median, gated by min torque
        raw_effort_spike = effort[:, j] / (med + 0.01)
        effort_spike[:, j] = raw_effort_spike * torque_gate[:, j]
    
    # --- Signal 4: Torque direction jitter ---
    # Only meaningful when torque magnitude is above baseline —
    # sign flips in near-zero torques are just numerical noise.
    jitter_rate = np.zeros((T, N_JOINTS))
    sign_history = np.zeros((JITTER_WINDOW, N_JOINTS, 3))
    hist_idx = 0
    for t in range(T):
        curr_signs = np.sign(dyn_torques[t, :N_JOINTS])
        sign_history[hist_idx] = curr_signs
        hist_idx = (hist_idx + 1) % JITTER_WINDOW
        
        if t >= JITTER_WINDOW - 1:
            # Count sign changes per joint (across all 3 axes)
            changes = np.sum(
                np.abs(np.diff(sign_history, axis=0)) > 0,
                axis=(0, 2)
            )  # (J,)
            jitter_rate[t] = (changes / (JITTER_WINDOW - 1)) * torque_gate[t]
    
    # ── Combine into per-frame scores ─────────────────────────────────
    
    # Per-joint scores (normalized to 0-1 range, then weighted)
    s_surprise = np.maximum(0, (surprise - SURPRISE_WARN) / SURPRISE_WARN) * W_SURPRISE
    s_tv = np.maximum(0, (tv_ratio - TV_RATIO_WARN) / TV_RATIO_WARN) * W_TV_RATIO
    s_effort = np.maximum(0, (effort_spike - EFFORT_SPIKE_WARN) / EFFORT_SPIKE_WARN) * W_EFFORT
    s_jitter = np.maximum(0, (jitter_rate - JITTER_THRESH) / (1.0 - JITTER_THRESH + 1e-6)) * W_JITTER
    
    # Per-joint combined score: (T, J)
    joint_scores = s_surprise + s_tv + s_effort + s_jitter
    
    # ── Transience filter ─────────────────────────────────────────────
    # Noise glitches are brief bursts; real movement produces sustained torque.
    # Discount scores for frames where elevated torque persists continuously.
    # For each joint, compute the run-length of consecutive non-zero scores.
    for j in range(N_JOINTS):
        elevated = joint_scores[:, j] > 0
        run_len = np.zeros(T)
        count = 0
        for t in range(T):
            if elevated[t]:
                count += 1
            else:
                count = 0
            run_len[t] = count
        # Discount frames that are part of long runs
        sustained = run_len > TRANSIENCE_WINDOW
        joint_scores[sustained, j] *= TRANSIENCE_DISCOUNT
    
    # ── Contact-event discount ────────────────────────────────────────
    # Torque spikes during contact state changes (touchdown, liftoff, hand plant)
    # are expected physics.  Detect transitions in frame-evaluator forces
    # and discount scores for joints in the affected kinetic chain.
    contact_active = contact_forces > CONTACT_FORCE_THRESH  # (T, J) boolean
    contact_event_mask = np.ones((T, N_JOINTS))  # 1.0 = no discount
    
    for contact_j, chain_joints in CONTACT_CHAIN.items():
        if contact_j >= N_CONTACT_JOINTS:
            continue
        # Detect transitions: diff in contact state
        transitions = np.zeros(T, dtype=bool)
        for t in range(1, T):
            if contact_active[t, contact_j] != contact_active[t-1, contact_j]:
                transitions[t] = True
        
        # Expand to window around each transition
        transition_window = np.zeros(T, dtype=bool)
        for t in range(T):
            if transitions[t]:
                lo = max(0, t - CONTACT_EVENT_WINDOW)
                hi = min(T, t + CONTACT_EVENT_WINDOW + 1)
                transition_window[lo:hi] = True
        
        # Also mark frames where contact is actively changing force significantly
        # (force ramp-up/down, not just the boolean transition)
        if np.any(contact_active[:, contact_j]):
            force_diff = np.abs(np.diff(contact_forces[:, contact_j], prepend=0))
            force_changing = force_diff > CONTACT_FORCE_THRESH * 0.5
            transition_window |= force_changing
        
        # Apply discount to all joints in the kinetic chain
        for chain_j in chain_joints:
            if chain_j < N_JOINTS:
                contact_event_mask[transition_window, chain_j] = CONTACT_EVENT_DISCOUNT
    
    joint_scores *= contact_event_mask
    
    # ── Airborne discount ────────────────────────────────────────────
    # When total contact force is zero, the performer is airborne.
    # Torque anomalies during freefall are expected (unmodeled contacts, inertia)
    # not noise.  Discount all joints during airborne + transition frames.
    total_contact = np.sum(contact_forces, axis=1)  # (T,)
    airborne = total_contact < CONTACT_FORCE_THRESH  # (T,) boolean
    
    # Expand airborne mask with transition window
    airborne_expanded = airborne.copy()
    for t in range(T):
        if airborne[t]:
            lo = max(0, t - AIRBORNE_WINDOW)
            hi = min(T, t + AIRBORNE_WINDOW + 1)
            airborne_expanded[lo:hi] = True
    
    # Apply discount to ALL joints during airborne phases
    joint_scores[airborne_expanded] *= AIRBORNE_DISCOUNT
    
    # ── Stream break discount ─────────────────────────────────────────
    # Frames near detected stream breaks are data discontinuities, not noise.
    # Discount their scores heavily so they don't inflate the file-level score.
    if np.any(break_mask):
        joint_scores[break_mask] *= STREAM_BREAK_DISCOUNT
    
    # Apply joint importance weighting
    weighted_scores = joint_scores * JOINT_IMPORTANCE[np.newaxis, :]
    
    # Per-frame score: max across joints (the worst joint drives the frame score)
    frame_scores = np.max(weighted_scores, axis=1)  # (T,)
    worst_joint_per_frame = np.argmax(weighted_scores, axis=1)  # (T,)
    
    # ── Build frame-level results ─────────────────────────────────────
    glitch_mask = frame_scores >= GLITCH_THRESH
    suspect_mask = (frame_scores >= SUSPECT_THRESH) & ~glitch_mask
    n_g = int(np.sum(glitch_mask))
    n_s = int(np.sum(suspect_mask))
    g_frac = n_g / max(T, 1)
    
    # Glitch frame details (sorted by score)
    glitch_list = []
    for t in range(T):
        if frame_scores[t] >= GLITCH_THRESH:
            wj = int(worst_joint_per_frame[t])
            glitch_list.append(FrameScore(
                frame=t,
                score=round(float(frame_scores[t]), 4),
                surprise=round(float(np.max(surprise[t])), 2),
                tv_ratio=round(float(np.max(tv_ratio[t])), 1),
                effort_spike=round(float(np.max(effort_spike[t])), 2),
                jitter_rate=round(float(np.max(jitter_rate[t])), 3),
                worst_joint=wj,
                worst_joint_name=SMPL_JOINT_NAMES[wj] if wj < len(SMPL_JOINT_NAMES) else f'j{wj}',
            ))
    glitch_list.sort(key=lambda x: -x.score)
    
    # Clusters
    clusters = _find_clusters(sorted(s.frame for s in glitch_list))
    
    # Clean segments
    clean = _find_clean_segments(frame_scores, fps)
    
    # ── Per-joint noise profiles ──────────────────────────────────────
    joint_profiles = []
    for j in range(N_JOINTS):
        j_glitch = int(np.sum(joint_scores[:, j] >= GLITCH_THRESH))
        j_suspect = int(np.sum((joint_scores[:, j] >= SUSPECT_THRESH) & (joint_scores[:, j] < GLITCH_THRESH)))
        j_frac = j_glitch / max(T, 1)
        
        prof = JointNoiseProfile(
            joint_idx=j,
            joint_name=SMPL_JOINT_NAMES[j],
            n_glitch_frames=j_glitch,
            n_suspect_frames=j_suspect,
            mean_surprise=round(float(np.mean(surprise[:, j])), 3),
            max_surprise=round(float(np.max(surprise[:, j])), 2),
            mean_effort=round(float(np.mean(effort[:, j])), 4),
            max_effort=round(float(np.max(effort[:, j])), 3),
            glitch_fraction=round(j_frac, 6),
        )
        joint_profiles.append(prof)
    
    # Sort by glitch count (noisiest first)
    joint_profiles.sort(key=lambda p: -(p.n_glitch_frames + p.n_suspect_frames * 0.5))
    
    # ── File-level scoring ────────────────────────────────────────────
    peak = float(np.max(frame_scores)) if T > 0 else 0
    p95 = float(np.percentile(frame_scores, 95)) if T > 0 else 0
    p99 = float(np.percentile(frame_scores, 99)) if T > 0 else 0
    
    # Score: weighted combination of glitch density, peak severity, and P99
    # Scaled so that a file with ~5% glitch frames and moderate peaks ≈ 50
    ns = g_frac * 200 + min(peak * 3, 40) + p99 * 10
    ns = min(100.0, ns)
    
    if g_frac < 0.01 and peak < 2.0:
        cls = 'clean'
    elif g_frac < 0.05 and peak < 5.0:
        cls = 'moderate'
    else:
        cls = 'problematic'
    
    report = TorqueFileReport(
        filename=os.path.basename(filepath),
        n_frames=T, fps=fps,
        duration_s=round(T / fps, 1),
        noise_score=round(ns, 4),
        classification=cls,
        n_glitch_frames=n_g,
        n_suspect_frames=n_s,
        glitch_fraction=round(g_frac, 6),
        max_surprise=round(float(np.max(surprise)), 2) if T > 0 else 0,
        max_tv_ratio=round(float(np.max(tv_ratio)), 1) if T > 0 else 0,
        max_effort=round(float(np.max(effort)), 3) if T > 0 else 0,
        p95_score=round(p95, 4),
        p99_score=round(p99, 4),
        joint_profiles=joint_profiles,
        glitch_frames=glitch_list,
        glitch_clusters=clusters,
        clean_segments=clean,
        stream_breaks=stream_breaks,
        cadence=cadence,
        motion_profile=motion_profile,
        ground_contact=ground_contact,
    )
    
    # Build multi-dimensional classification
    report.classification_detail, report.recommendations = _build_classification(report)
    
    if verbose:
        print_report(report)
    
    return report


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_clusters(frames, gap=5):
    """Group nearby frame indices into (start, end) clusters."""
    if not frames:
        return []
    clusters, start, end = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - end <= gap:
            end = f
        else:
            clusters.append((start, end))
            start = end = f
    clusters.append((start, end))
    return clusters


def _find_clean_segments(scores, fps, margin=3):
    """
    Find continuous stretches where NO frame is glitch or suspect.
    Expands bad regions by margin frames on each side.
    Returns list of CleanSegment, sorted longest-first.
    """
    T = len(scores)
    bad = scores >= SUSPECT_THRESH
    
    if margin > 0:
        bad_expanded = bad.copy()
        for m in range(1, margin + 1):
            bad_expanded[m:] |= bad[:-m]
            bad_expanded[:-m] |= bad[m:]
        bad = bad_expanded
    
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
                    seg_scores = scores[start:f]
                    segments.append(CleanSegment(
                        start=start, end=f - 1, n_frames=n,
                        duration_s=round(dur, 2),
                        max_score=round(float(np.max(seg_scores)), 4),
                        mean_score=round(float(np.mean(seg_scores)), 4),
                    ))
                in_clean = False
    
    if in_clean:
        n = T - start
        dur = n / fps
        if dur >= CLEAN_MIN_DURATION:
            seg_scores = scores[start:]
            segments.append(CleanSegment(
                start=start, end=T - 1, n_frames=n,
                duration_s=round(dur, 2),
                max_score=round(float(np.max(seg_scores)), 4),
                mean_score=round(float(np.mean(seg_scores)), 4),
            ))
    
    segments.sort(key=lambda s: -s.n_frames)
    return segments


# ── Console report ────────────────────────────────────────────────────────

def print_report(r):
    ic = {'clean': '✅', 'moderate': '⚠️', 'problematic': '❌'}
    
    print(f"\n{'═' * 72}")
    print(f"  {r.filename}")
    print(f"  {r.n_frames} frames @ {r.fps:.0f} fps ({r.duration_s}s)")
    print(f"{'═' * 72}")
    print(f"  Classification: {ic.get(r.classification)} {r.classification.upper()}")
    if r.classification_detail:
        print(f"  Detail:         {r.classification_detail}")
    print(f"  Noise score:    {r.noise_score:.1f} / 100")
    print(f"  Glitch frames:  {r.n_glitch_frames} ({100 * r.glitch_fraction:.2f}%)")
    print(f"  Suspect frames: {r.n_suspect_frames}")
    
    # ── Recommendations ────────────────────────────────────────────────
    if r.recommendations:
        rec_str = ', '.join(f'{k}={v}' for k, v in r.recommendations.items())
        print(f"  Recommended:    {rec_str}")
    
    # ── Stream breaks ──────────────────────────────────────────────────
    if r.stream_breaks:
        print(f"\n  ⚡ Stream breaks: {len(r.stream_breaks)} detected")
        print(f"    {'Frame':>7s}  {'Time':>6s}  {'Type':>12s}  {'TransJump':>10s}  {'PoseJump':>10s}  Worst Joint")
        for sb in r.stream_breaks:
            print(f"    {sb.frame:7d}  {sb.time_s:5.1f}s  {sb.break_type:>12s}  "
                  f"{sb.trans_jump:9.4f}m  {sb.pose_jump:9.4f}r  {sb.worst_joint}")
        
        # Show clean segments between breaks
        break_frames = sorted(sb.frame for sb in r.stream_breaks)
        seg_boundaries = [0] + break_frames + [r.n_frames]
        print(f"\n    Continuous segments between breaks:")
        for i in range(len(seg_boundaries) - 1):
            s, e = seg_boundaries[i], seg_boundaries[i + 1]
            if i < len(seg_boundaries) - 2:
                e -= 1  # don't include the break frame
            dur = (e - s) / r.fps
            if dur > 0:
                print(f"      [{s:>7d}–{e:>7d}]  {dur:6.1f}s  ({e-s:>6d} frames)")
    
    # ── Cadence ────────────────────────────────────────────────────────
    if r.cadence and r.cadence.detected:
        print(f"\n  🔄 Cadence: {r.cadence.period}-frame period detected "
              f"(strength={r.cadence.strength:.2f}, {r.cadence.coverage:.0%} of file)")
    elif r.cadence and not r.cadence.detected:
        print(f"\n  ✓  No sub-sampling cadence detected")
    
    # ── Motion profile ─────────────────────────────────────────────────
    if r.motion_profile:
        mp = r.motion_profile
        cplx_ic = {'static': '🧘', 'moderate': '🚶', 'dynamic': '🏃', 'explosive': '💥'}
        print(f"\n  {cplx_ic.get(mp.complexity, '')} Motion: {mp.complexity}")
        print(f"    Total body angular velocity (rad/s):  "
              f"mean={mp.mean_vel:.0f}  p50={mp.p50_vel:.0f}  "
              f"p90={mp.p90_vel:.0f}  p99={mp.p99_vel:.0f}  max={mp.max_vel:.0f}")
        n_total = mp.n_still + mp.n_active + (mp.n_active)  # approx
        print(f"    Frame activity:  "
              f"still={mp.n_still}  active={mp.n_active}  explosive={mp.n_explosive}")
    
    # ── Ground contact ─────────────────────────────────────────────────
    if r.ground_contact and r.ground_contact.ground_pct > 0.05:
        gc = r.ground_contact
        print(f"\n  ⬇️  Ground work: {gc.ground_pct:.0%} of file")
        print(f"    {gc.n_phases} phases, longest={gc.longest_phase_s:.1f}s  "
              f"standing_h={gc.standing_height:.3f}m  avg_ground_h={gc.avg_ground_height:.3f}m")
    print()
    print(f"  {'Metric':.<24s}  {'Max':>10s}  {'P95':>10s}  {'P99':>10s}")
    print(f"  {'─' * 24}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    print(f"  {'Frame score':.<24s}  {float(max(s.score for s in r.glitch_frames) if r.glitch_frames else 0):10.2f}  "
          f"{r.p95_score:10.4f}  {r.p99_score:10.4f}")
    print(f"  {'Torque surprise':.<24s}  {r.max_surprise:10.2f}")
    print(f"  {'Torque/velocity ratio':.<24s}  {r.max_tv_ratio:10.1f}")
    print(f"  {'Max effort (τ/τ_max)':.<24s}  {r.max_effort:10.3f}")
    
    # ── Noisiest joints ───────────────────────────────────────────────
    noisy_joints = [p for p in r.joint_profiles if p.n_glitch_frames > 0 or p.n_suspect_frames > 0]
    if noisy_joints:
        n_show = min(10, len(noisy_joints))
        print(f"\n  Noisiest joints (top {n_show}):")
        print(f"    {'Joint':<18s}  {'Glitch':>7s}  {'Suspect':>8s}  {'MaxSurpr':>9s}  {'MaxEffort':>10s}")
        for p in noisy_joints[:n_show]:
            print(f"    {p.joint_name:<18s}  {p.n_glitch_frames:7d}  {p.n_suspect_frames:8d}  "
                  f"{p.max_surprise:9.2f}  {p.max_effort:10.3f}")
    else:
        print(f"\n  No individual joints had glitch-level scores.")
    
    # ── Glitch regions ────────────────────────────────────────────────
    if r.glitch_clusters:
        print(f"\n  Glitch regions ({len(r.glitch_clusters)}):")
        for start, end in r.glitch_clusters[:25]:
            n = end - start + 1
            pk = max(
                (s.score for s in r.glitch_frames if start <= s.frame <= end),
                default=0
            )
            print(f"    [{start:>6d}–{end:>6d}]  {n:>4d} frames  "
                  f"t={start / r.fps:5.1f}–{end / r.fps:5.1f}s  peak={pk:.2f}")
        if len(r.glitch_clusters) > 25:
            print(f"    ... +{len(r.glitch_clusters) - 25} more")
    
    # ── Worst frames ──────────────────────────────────────────────────
    if r.glitch_frames:
        n_show = min(20, len(r.glitch_frames))
        print(f"\n  Worst {n_show} frames:")
        print(f"    {'Frame':>6s}  {'Score':>7s}  {'Surprise':>9s}  {'T/V ratio':>10s}  "
              f"{'Effort':>7s}  {'Jitter':>7s}  Joint")
        for s in r.glitch_frames[:n_show]:
            print(f"    {s.frame:6d}  {s.score:7.2f}  {s.surprise:9.2f}  {s.tv_ratio:10.1f}  "
                  f"{s.effort_spike:7.2f}  {s.jitter_rate:7.3f}  {s.worst_joint_name}")
    
    # ── Clean segments ────────────────────────────────────────────────
    if r.clean_segments:
        total_clean = sum(s.n_frames for s in r.clean_segments)
        clean_pct = 100 * total_clean / max(r.n_frames, 1)
        print(f"\n  Clean segments ({len(r.clean_segments)} found, "
              f"{total_clean} frames = {clean_pct:.1f}% of file):")
        print(f"    {'Rank':>4s}  {'Frames':>16s}  {'Duration':>8s}  {'MaxScore':>9s}  {'MeanScore':>10s}")
        n_show = min(15, len(r.clean_segments))
        for i, seg in enumerate(r.clean_segments[:n_show]):
            print(f"    {i + 1:4d}  [{seg.start:>6d}–{seg.end:>6d}]  "
                  f"{seg.duration_s:6.1f}s   {seg.max_score:9.4f}  {seg.mean_score:10.4f}")
        if len(r.clean_segments) > n_show:
            remaining = r.clean_segments[n_show:]
            rem_frames = sum(s.n_frames for s in remaining)
            print(f"    ... +{len(remaining)} smaller segments ({rem_frames} frames)")
    else:
        print(f"\n  ⚠️  No clean segments found (>= {CLEAN_MIN_DURATION}s without glitches)")
    
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Torque-based motion capture noise estimator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('files', nargs='*', help='NPZ files to analyse')
    p.add_argument('--dir', help='Directory of NPZ files to analyse')
    p.add_argument('--json', help='Save results to JSON file')
    p.add_argument('--mass', type=float, default=75.0, help='Body mass in kg (default: 75)')
    p.add_argument('--gender', choices=['male', 'female', 'neutral'],
                   help='Override gender (default: read from file)')
    
    args = p.parse_args()
    
    files = list(args.files or [])
    if args.dir:
        files += [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if f.endswith('.npz')
        ]
    
    if not files:
        p.print_help()
        return
    
    reports = []
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"  ⚠️  Not found: {filepath}")
            continue
        try:
            report = analyze_file(
                filepath,
                total_mass=args.mass,
                gender_override=args.gender,
                verbose=True,
            )
            reports.append(report)
        except Exception as e:
            import traceback
            print(f"  ❌ Error: {filepath}: {e}")
            traceback.print_exc()
    
    # ── Multi-file summary ────────────────────────────────────────────
    if len(reports) > 1:
        print(f"\n{'═' * 72}")
        print(f"  SUMMARY ({len(reports)} files)")
        print(f"{'═' * 72}")
        print(f"  {'File':<40s}  {'Score':>7s}  {'Class':>12s}  {'Glitches':>8s}  {'Clean%':>6s}")
        for r in sorted(reports, key=lambda x: -x.noise_score):
            e = {'clean': '✅', 'moderate': '⚠️', 'problematic': '❌'}.get(r.classification, '?')
            tc = sum(s.n_frames for s in r.clean_segments)
            cp = 100 * tc / max(r.n_frames, 1)
            print(f"  {r.filename:<40s}  {r.noise_score:7.1f}  {e} {r.classification:<10s}  "
                  f"{r.n_glitch_frames:>8d}  {cp:5.1f}%")
    
    # ── JSON output ───────────────────────────────────────────────────
    if args.json:
        out = []
        for r in reports:
            d = asdict(r)
            # Limit glitch_frames for JSON
            d['glitch_frames'] = [asdict(s) for s in r.glitch_frames[:500]]
            d['clean_segments'] = [asdict(s) for s in r.clean_segments]
            d['joint_profiles'] = [asdict(jp) for jp in r.joint_profiles]
            out.append(d)
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved to {args.json}")


if __name__ == '__main__':
    main()
