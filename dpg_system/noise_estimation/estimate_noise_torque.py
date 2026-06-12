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


def _to_jsonable(obj):
    """Recursively convert numpy/scalar objects into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


# Ensure dpg_system is importable when running as a standalone script
_this_dir = os.path.dirname(os.path.abspath(__file__))       # noise_estimation/
_package_dir = os.path.dirname(_this_dir)                     # dpg_system/dpg_system/
_project_dir = os.path.dirname(_package_dir)                  # dpg_system/
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

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

# Clean-fraction noise-score penalty.  A long file that fragments down to
# only a small fraction of clean material is automatically very problematic
# regardless of which signal channels fired.  Below the threshold, score
# gains proportionally to (THRESHOLD - clean_fraction).
CLEAN_FRAC_PENALTY_MIN_DURATION_S = 10.0   # only apply to files longer than this
CLEAN_FRAC_PENALTY_THRESHOLD = 0.5         # penalise when clean fraction drops below this
CLEAN_FRAC_PENALTY_SCALE = 60.0            # multiplier on the gap to threshold
CLEAN_FRAC_PROBLEMATIC_THRESHOLD = 0.5     # force "problematic" classification below this

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


MIN_SALVAGEABLE_FRAMES = 500   # minimum frames for a segment to be usable after splitting

@dataclass
class SurgerySegment:
    """A continuous segment between stream breaks that may be usable."""
    start: int               # first frame (inclusive)
    end: int                 # last frame (inclusive)
    n_frames: int
    duration_s: float
    usable: bool             # True if >= MIN_SALVAGEABLE_FRAMES


# ── Corruption zone detection ────────────────────────────────────────────
# Sustained multi-second regions where EXTREMITY joints (shoulder, elbow,
# wrist) produce garbage data (marker dropout, capture failure).
# Only arm-chain joints are checked because leg/pelvis velocity elevation
# during dynamic movement (floor work, dancing) is expected physics.
#
# KEY INSIGHT: Absolute velocity thresholds cannot distinguish fast movement
# from corruption.  We use LOCAL CONTEXT: compare short-term velocity
# against a long-term local baseline.  Corruption = sustained elevation
# relative to the local neighbourhood, NOT relative to the global median.
# This correctly handles:
#   - Take 3: fast arm movement during dance → local baseline also high → no flag
#   - Maritsa: arm teleportation during slow movement → local baseline low → flag
CORRUPTION_MIN_DURATION_FRAMES = 60   # minimum run length (~0.5s at 120fps)
CORRUPTION_SHORT_WINDOW_S = 1.0       # short-term rolling mean window
CORRUPTION_LONG_WINDOW_S = 10.0       # long-term rolling median window (local baseline)
CORRUPTION_LOCAL_RATIO = 4.0          # flag if short/long > this ratio
CORRUPTION_VEL_FLOOR = 3.0            # absolute minimum velocity to flag (rad/s)
# On short files the long-window rolling baseline collapses to roughly the
# length of any sustained motion in the file, so the local-ratio test fires
# on legitimate fast motion (e.g. a punch on a 2-sec file).  Below this
# duration the rolling baseline cannot give a meaningful "is this anomalous?"
# signal, so corruption detection is skipped entirely.  Whole-file p95 is
# NOT used as a reference because in pervasively-corrupted files p95 is
# inflated by the corruption itself and would hide it.
CORRUPTION_MIN_FILE_S = 5.0           # below this duration, skip corruption detection
CORRUPTION_MERGE_GAP_S = 1.0          # merge zones separated by less than this
# Only these joints are checked for corruption (arm chain).
# Leg/pelvis/spine activity during dynamic movement is NOT corruption.
CORRUPTION_JOINTS = {16, 17, 18, 19, 20, 21}  # L/R shoulder, elbow, wrist

# Single-frame spike detection: a dropped/corrupted frame surrounded on both
# sides by much lower velocity.  The neighbour ratio is the principled
# discriminant — it normalises for local movement speed, so a dropped frame
# during fast motion and one during slow motion produce similar ratios.
# The velocity floor is only a guard against near-stillness numerical noise
# inflating the ratio via a tiny denominator; it is not a signal in itself.
SPIKE_NEIGHBOR_RATIO = 3.0   # vel[t] / max(vel[t-1], vel[t+1]) threshold
SPIKE_VEL_FLOOR = 2.0        # rad/s — guard against near-stillness noise only

# Spike-density gate for clean segments: sustained low-level arm-chain
# noise (many sub-significant spikes in a short window) indicates flickery
# arm data even when no individual spike passes the severity gate.  This
# gate is applied only to clean-segment determination, not to noise_score
# or classification.
ARM_CHAIN_JOINT_INDICES = {13, 14, 16, 17, 18, 19, 20, 21}  # collars + shoulders + elbows + wrists
SPIKE_DENSITY_WINDOW_S = 1.0    # rolling-window length for the density count
# Threshold calibrated empirically: vigorous running arm action (C6 post-416)
# peaks at 8 unique arm-chain spike frames per 1-s window.  Sustained
# marker flicker (Maritsa) typically runs 12–27.  Threshold 10 cleanly
# separates "legitimate fast arm motion" from "arm marker flicker".
SPIKE_DENSITY_MIN_COUNT = 10    # unique-frame count within window to trigger

# Spike-frame severity: per-joint contribution weighted by how independent
# that joint's motion is from the rest of the body.  Core joints (pelvis,
# spine) only move when the whole body moves, so anomalies there are strong
# evidence of a real glitch.  Distal joints (wrists, collars) move fast
# during normal arm motion, so anomalies there are weak evidence.
SPIKE_SEVERITY_JOINT_WEIGHT = {
    'pelvis': 1.0, 'spine1': 1.0, 'spine2': 1.0, 'spine3': 1.0,
    'neck': 0.7, 'head': 0.7,
    'left_hip': 0.5, 'right_hip': 0.5,
    'left_knee': 0.5, 'right_knee': 0.5,
    'left_ankle': 0.5, 'right_ankle': 0.5,
    'left_shoulder': 0.5, 'right_shoulder': 0.5,
    'left_elbow': 0.5, 'right_elbow': 0.5,
    'left_foot': 0.5, 'right_foot': 0.5,
    'left_collar': 0.2, 'right_collar': 0.2,
    'left_wrist': 0.2, 'right_wrist': 0.2,
}
# A per-joint contribution must clear this absolute threshold to count
# toward the "n core/mid qualifying" gate.
SPIKE_SEVERITY_QUALIFY_CONTRIB = 2.0
# A spike frame counts as "significant" (fold into clean-segment bad mask)
# when EITHER (a) wsev >= MAJOR threshold with ≥ N qualifying core/mid
# joints, or (b) wsev >= STRONG threshold with ≥ N qualifying core/mid.
# Both criteria require multi-joint involvement to avoid flagging
# arm-only fast-motion bursts.
SPIKE_SEVERITY_STRONG_THRESHOLD = 10.0
SPIKE_SEVERITY_MIN_QUALIFYING = 2

# Pure-arm-cluster gate: a frame with N+ arm-chain joint entries AND zero
# non-arm-chain entries is the signature of one-side marker failure.
# Real coordinated motion (running, side step, gesture) involves the
# spine/pelvis/legs; pure-arm activity is suspicious.  Catches synchronized
# arm events that the severity gate misses on files where the per-joint
# p95 reference is inflated by pervasive corruption (e.g. Maritsa frame 994).
PURE_ARM_CLUSTER_MIN_JOINTS = 3

# Isolated arm-impulse gate (clean-segment fragmentation only):
# Arm-chain spikes flanked by near-stillness on both sides are the
# signature of out-of-context marker jumps — even at modest velocity.
# A real arm motion peak has at least one elevated neighbour (build-up
# or follow-through).
#
# To distinguish marker glitches (e.g. HumanEva ThrowCatch_3) from
# sampling-cadence artifacts (e.g. Shadow 100→60 pulldown), a single
# isolated impulse is given the benefit of the doubt: a frame counts as
# disqualifying only when (a) ≥2 arm-chain joints fire isolated impulses
# at the SAME frame, OR (b) there's another isolated-impulse frame within
# the temporal-clustering window.
#
# In validation: ThrowCatch clean-region impulses cluster (median gap
# 13-84 frames @ 120 fps); Shadow clean-region impulses are sparse
# (median gap 115 frames @ 60 fps), so most don't pass the cluster test.
#
# Does NOT affect noise_score or classification — only fragmentation
# for clean_segments and clean_section_score.
ISOLATED_ARM_IMPULSE_VEL_FLOOR = 2.5         # rad/s — magnitude floor on v[t]
ISOLATED_ARM_IMPULSE_NEIGHBOR_CEILING = 2.0  # rad/s — max(v[t-1], v[t+1]) ceiling
ISOLATED_ARM_IMPULSE_CLUSTER_WINDOW_S = 1.5  # seconds — temporal-clustering window
ISOLATED_ARM_IMPULSE_MIN_JOINTS_SAME_FRAME = 2  # multi-joint co-occurrence threshold

# Spike-frame consolidation: many spikes packed into a short span are one
# contaminated section, not that many independent dropped frames — reporting
# them individually buries the structure.  Spike frames separated by
# <= SPIKE_CLUSTER_GAP_S are merged into one cluster; a cluster with
# >= SPIKE_CLUSTER_CONTAMINATED_MIN distinct spike frames is reported as a
# contaminated section (one issue), the rest as isolated spikes.
# Reporting/JSON only — does not affect noise_score or classification.
SPIKE_CLUSTER_GAP_S = 0.25            # merge gap (≈30 frames @ 120 fps)
SPIKE_CLUSTER_CONTAMINATED_MIN = 3    # distinct spike frames → contaminated

@dataclass
class SpikeFrame:
    """A single dropped/corrupted frame identified by the neighbour-ratio test."""
    frame: int
    joint_idx: int
    joint_name: str
    velocity: float        # rad/s at the spike frame
    neighbor_ratio: float  # vel[t] / max(vel[t-1], vel[t+1])


@dataclass
class SpikeCluster:
    """A group of spike frames in close temporal proximity.

    Dense clusters (contaminated=True) represent a contaminated section
    of the capture; sparse clusters are genuinely isolated dropped frames.
    """
    start: int               # first spike frame in cluster
    end: int                 # last spike frame in cluster
    n_frames: int            # span (end - start + 1)
    duration_s: float
    n_spike_frames: int      # distinct frames with spike entries
    n_spike_records: int     # per-joint spike entries
    spike_density: float     # n_spike_frames / n_frames
    joints: List[str]        # affected joints, most spike records first
    max_velocity: float      # peak spike velocity in cluster (rad/s)
    contaminated: bool       # n_spike_frames >= SPIKE_CLUSTER_CONTAMINATED_MIN


@dataclass
class CorruptionZone:
    """A sustained period of corrupted data on specific joints."""
    start: int               # first frame (inclusive)
    end: int                 # last frame (inclusive)
    n_frames: int
    duration_s: float
    joints: List[str]        # affected joint names
    joint_indices: List[int] # affected joint indices
    mean_vel: float          # mean velocity during corruption (rad/s)
    max_vel: float           # peak velocity during corruption


@dataclass
class ExcisionInfo:
    """Summary of corruption zones requiring excision (section removal)."""
    n_zones: int
    total_corrupted_frames: int
    corrupted_fraction: float
    zones: List[CorruptionZone] = field(default_factory=list)


@dataclass
class SurgeryInfo:
    """Stream break and corruption analysis — separate from noise quality scoring.
    
    Two types of structural issues:
    1. Stream breaks — whole-body discontinuities → split the file
    2. Corruption zones — sustained joint-level garbage → excise sections
    """
    n_breaks: int
    needs_surgery: bool           # True if breaks or corruption present
    segments: List[SurgerySegment] = field(default_factory=list)
    n_usable_segments: int = 0    # segments >= MIN_SALVAGEABLE_FRAMES
    usable_frames: int = 0        # total frames in usable segments
    usable_fraction: float = 0.0  # fraction of file that is usable
    recommendation: str = ''      # 'none', 'split', 'excise', 'split+excise', 'discard'
    excision: Optional[ExcisionInfo] = None


@dataclass
class TorqueFileReport:
    filename: str
    n_frames: int
    fps: float
    duration_s: float

    noise_score: float
    classification: str      # clean / moderate / problematic

    # Rates noise level WITHIN the identified clean segments only.
    # Complements noise_score (which rates the whole file): if you
    # extract just the clean sections, how usable is that data?
    # Low = the recoverable data is genuinely clean; high = even
    # the clean sections have sub-significant flicker.
    clean_section_score: float

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
    spike_frames: List[SpikeFrame] = field(default_factory=list)
    spike_clusters: List[SpikeCluster] = field(default_factory=list)
    surgery: Optional[SurgeryInfo] = None
    cadence: Optional[CadenceInfo] = None
    motion_profile: Optional['MotionProfile'] = None
    ground_contact: Optional['GroundContactInfo'] = None
    classification_detail: str = ''    # multi-dimensional summary line
    recommendations: Optional[dict] = None  # suggested filter params

    # Kinematic pose-level lenses (operate on raw arm-joint motion, separate
    # from the torque scorer): teleport (non-physical jump), flicker (synced
    # marker jitter), zigzag (buzz on a moving joint), ROM (candy-wrapper).
    teleport_max_arm_vel: float = 0.0       # peak arm-joint angular velocity (rad/s)
    teleport_inconsistent_rate: float = 0.0 # sub-60 inconsistent-frame rate (diagnostic only)
    n_teleport_frames: int = 0              # excised non-physical (>40 rad/s) frames
    flicker_rate: float = 0.0               # synchronized-flicker frames / second
    flicker_peak: int = 0                   # max flicker frames in any 1-second window
    n_flicker_frames: int = 0
    zigzag_severity: float = 0.0            # peak windowed zigzag intensity (0-1)
    zigzag_contribution: float = 0.0        # points it added to noise_score
    n_rom_frames: int = 0                   # impossible-rotation (candy-wrapper) frames


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
STREAM_BREAK_RADIUS = 20           # frames around break to exclude from scoring
STREAM_BREAK_DISCOUNT = 0.1        # score multiplier for stream break frames

# Minimum joints that must teleport simultaneously to qualify as a stream break.
# Single/few-joint teleportation (e.g., wrist marker dropout) is NOISE, not a
# data discontinuity.  Real stream breaks (concatenated takes, capture resets)
# affect the whole skeleton including the torso.
MIN_JOINTS_FOR_BREAK = 8
# Per-joint threshold for "this joint teleported": factor × per-joint median
JOINT_BREAK_FACTOR = 10.0
# Local-adaptive threshold for multi-joint breaks.  During rapid motion the
# per-joint pose displacement is naturally elevated and a static
# global-median threshold over-fires (e.g. 60/100 Hz cadence drops during
# fast movement on Shadow data produce 18 false breaks).  The threshold
# at each frame is the MAX of the global threshold and a local rolling-
# median × factor, so high-motion regions get correspondingly higher bars
# without affecting low-motion regions.
JOINT_BREAK_LOCAL_WINDOW_S = 1.0       # rolling-median window length
JOINT_BREAK_LOCAL_FACTOR = 5.0         # multiplier on local rolling median
# Core joint indices that MUST be involved for a pose-based stream break.
# A limb-chain glitch (wrist→elbow→shoulder) can pull 5+ joints without
# affecting the core.  Real capture resets always move the torso.
CORE_JOINTS = {0, 1, 2, 3, 6, 9}  # pelvis, L/R hip, spine1, spine2, spine3

def _detect_stream_breaks(poses, trans, fps):
    """Detect whole-body discontinuities (concatenated takes, capture resets).
    
    A stream break requires EITHER:
      - Root translation jump > threshold, OR
      - At least MIN_JOINTS_FOR_BREAK joints teleporting AND at least one
        core joint (pelvis/spine/hip) involved
    
    Single-joint or single-limb-chain teleportation is NOT a stream break.
    
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
    
    # Per-joint angular displacement
    pose_disp = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2)  # (T-1, J)
    
    # Per-joint median displacement for threshold computation
    joint_medians = np.median(pose_disp, axis=0)  # (J,)
    joint_medians = np.maximum(joint_medians, 0.01)  # floor to avoid div-by-zero
    joint_thresholds_global = joint_medians * JOINT_BREAK_FACTOR  # (J,)

    # Local-rolling-median per-joint thresholds.  Compares each frame's
    # joint displacement against the median over a surrounding window.
    # This prevents fast natural motion from being mistaken for break-like
    # multi-joint teleportation: during a fast movement, ALL pose_disp
    # values are elevated, raising the local threshold accordingly.
    w_local = max(3, int(JOINT_BREAK_LOCAL_WINDOW_S * fps))
    half_w = w_local // 2
    n_frames_for_local = pose_disp.shape[0]
    local_joint_medians = np.empty_like(pose_disp)
    for t in range(n_frames_for_local):
        lo = max(0, t - half_w)
        hi = min(n_frames_for_local, t + half_w + 1)
        local_joint_medians[t] = np.median(pose_disp[lo:hi], axis=0)
    local_joint_thresholds = local_joint_medians * JOINT_BREAK_LOCAL_FACTOR
    # Effective threshold at each frame = max of global and local
    joint_thresholds = np.maximum(joint_thresholds_global[None, :], local_joint_thresholds)

    # Whole-skeleton threshold (for the max-across-joints metric)
    pose_max = np.max(pose_disp, axis=1)  # (T-1,)
    pose_median = np.median(pose_max) if len(pose_max) > 0 else 0
    
    trans_thresh = max(trans_median * STREAM_BREAK_TRANS_FACTOR, 0.05)  # at least 5cm
    pose_thresh = max(pose_median * STREAM_BREAK_POSE_FACTOR, 0.3)     # at least ~17°
    
    # Translation-shift threshold: lower than full stream break, but requires
    # low pose change.  Catches sensor reference switches where the body
    # shifts position without changing pose.
    TRANS_SHIFT_FACTOR = 10.0     # trans > 10× median
    TRANS_SHIFT_POSE_MAX = 2.0    # pose < 2× pose median
    trans_shift_thresh = max(trans_median * TRANS_SHIFT_FACTOR, 0.03)  # at least 3cm
    pose_shift_ceil = pose_median * TRANS_SHIFT_POSE_MAX
    
    breaks = []
    break_frames = set()
    
    for f in range(len(trans_disp)):
        is_trans = trans_disp[f] > trans_thresh
        
        # Translation-shift: high trans displacement, low pose change.
        # The body teleports in space but maintains the same pose.
        is_trans_shift = (trans_disp[f] > trans_shift_thresh and
                          pose_max[f] < pose_shift_ceil and
                          not is_trans)  # don't double-count with full break
        
        # Count how many joints teleported in this frame (using per-frame
        # adaptive threshold: max of global static and local-rolling)
        joints_exceeded = pose_disp[f] > joint_thresholds[f]  # (J,) boolean
        exceeded_indices = set(np.where(joints_exceeded)[0].tolist())
        n_joints_jumped = len(exceeded_indices)
        
        # Whole-body = enough joints AND at least one core joint involved
        has_core = bool(exceeded_indices & CORE_JOINTS)
        is_whole_body = n_joints_jumped >= MIN_JOINTS_FOR_BREAK and has_core
        
        # A stream break is a WHOLE-BODY discontinuity:
        # either root translation jump OR many joints teleporting together
        # OR a translation-shift (position jump, same pose)
        if is_trans or is_whole_body or is_trans_shift:
            worst_j_idx = int(np.argmax(pose_disp[f]))
            worst_j_name = SMPL_JOINT_NAMES[worst_j_idx] if worst_j_idx < len(SMPL_JOINT_NAMES) else f'j{worst_j_idx}'
            
            if is_trans and is_whole_body:
                btype = 'both'
            elif is_trans:
                btype = 'translation'
            elif is_trans_shift:
                btype = 'trans_shift'
            else:
                btype = 'multi_joint'
            
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
    
    # Surgery (stream breaks + corruption)
    if report.surgery and report.surgery.needs_surgery:
        s = report.surgery
        if s.recommendation == 'split':
            parts.append(f"✂️ {s.n_breaks} break(s) → split ({s.n_usable_segments} usable, {s.usable_fraction:.0%})")
        elif s.recommendation == 'excise':
            ex = s.excision
            parts.append(f"🔪 {ex.n_zones} corruption zone(s) → excise ({ex.corrupted_fraction:.0%} corrupted)")
        elif s.recommendation == 'split+excise':
            ex = s.excision
            parts.append(f"✂️🔪 {s.n_breaks} break(s) + {ex.n_zones} corruption zone(s)")
        elif s.recommendation == 'discard':
            parts.append(f"⚡ {s.n_breaks} break(s) → discard")

    # Kinematic pose-level lenses (teleport / flicker / zigzag / candy-wrapper)
    if report.teleport_max_arm_vel > TELEPORT_VEL_DEFINITE:
        parts.append(f"🛰️ teleport: {report.teleport_max_arm_vel:.0f} rad/s arm jump (non-physical)")
    if report.flicker_rate >= FLICKER_RATE_LOCK or report.flicker_peak >= FLICKER_PEAK_LOCK:
        parts.append(f"📳 flicker: {report.flicker_rate:.1f}/s synchronized arm jitter")
    if report.zigzag_contribution >= 8:
        parts.append(f"〰️ zigzag buzz (intensity {report.zigzag_severity:.2f})")
    if report.n_rom_frames >= ROM_MIN_FRAMES:
        parts.append(f"🥨 candy-wrapper: {report.n_rom_frames} impossible wrist-rotation frame(s)")

    # Single-frame spikes — consolidated: dense clusters become
    # contaminated sections, the remainder are isolated spikes
    if report.spike_frames:
        contaminated = [c for c in report.spike_clusters if c.contaminated]
        n_isolated = sum(c.n_spike_frames for c in report.spike_clusters
                         if not c.contaminated)
        bits = []
        if contaminated:
            total_s = sum(c.duration_s for c in contaminated)
            bits.append(f"{len(contaminated)} contaminated section(s) "
                        f"({total_s:.1f}s)")
        if n_isolated:
            bits.append(f"{n_isolated} isolated spike(s)")
        joint_set = sorted({s.joint_name for s in report.spike_frames})
        joints = ', '.join(joint_set) if len(joint_set) <= 6 else f"{len(joint_set)} joints"
        parts.append(f"⚡ {' + '.join(bits)} ({joints})")

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


def analyze_file(filepath, total_mass=75.0, gender_override=None,
                 smooth_input_window=0, verbose=True):
    """
    Run torque-based noise analysis on a single .npz file.
    
    Args:
        filepath: path to .npz file
        total_mass: body mass in kg (default 75)
        gender_override: override gender from file metadata
        smooth_input_window: Savitzky-Golay input smoothing window (0=off, 3/5/7).
            Use 3 for files with 3-frame cadence sub-sampling artifacts.
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
        if isinstance(g, bytes):
            g = g.decode('utf-8')
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
    
    # ── Detect single-frame spikes (dropped/corrupted frames) ────────────
    spike_frames = _detect_spike_frames(poses, fps)
    teleport_mask, teleport_score, teleport_rate, max_arm_vel = _detect_teleports(poses, fps)
    flicker_mask, flicker_rate, flicker_peak = _detect_flicker(poses, fps)
    zigzag_Z, zigzag_peak, zigzag_frame = _detect_zigzag(poses, fps)
    rom_mask, n_rom = _detect_rom(poses, fps)
    spike_clusters = _cluster_spike_frames(spike_frames, fps)
    if verbose and spike_frames:
        unique_frames = len({s.frame for s in spike_frames})
        n_contaminated = sum(1 for c in spike_clusters if c.contaminated)
        print(f"  Detected {unique_frames} spike frame(s) in "
              f"{len(spike_clusters)} cluster(s), {n_contaminated} contaminated "
              f"({', '.join(sorted({s.joint_name for s in spike_frames}))})")

    # Significant spike frames — used both for clean-segment fragmentation
    # and as pose-level corroboration for "hard" glitches below.
    significant_spike_frame_set = _significant_spike_frames(spike_frames, poses, fps)

    # ── Detect corruption zones (raw pose level, before torque filtering)
    # Done early so corruption frames can be excluded from torque scoring.
    # The torque pipeline's SG filtering absorbs arm corruption, making it
    # invisible to the torque scorer.  We detect it here on raw poses.
    corruption_zones, corruption_mask = _detect_corruption_zones(poses, fps)
    
    # Merge break + corruption into a unified structural exclusion mask.
    # Frames in either category are STRUCTURAL issues, not motion quality
    # issues, and must be excluded from torque-based noise scoring.
    structural_mask = break_mask | corruption_mask
    structural_mask = structural_mask | teleport_mask  # TELEPORT: excise non-physical arm frames
    structural_mask = structural_mask | flicker_mask  # FLICKER: excise synchronized jitter frames
    structural_mask = structural_mask | (zigzag_Z > ZIGZAG_FLOOR)  # ZIGZAG: exclude sustained buzz
    structural_mask = structural_mask | rom_mask  # ROM: excise impossible wrist poses
    
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
    # model_path must contain smplh/ directory with SMPLH_{GENDER}.pkl
    model_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dpg_system/
    processor = SMPLProcessor(
        framerate=fps,
        betas=betas,
        gender=gender,
        total_mass_kg=total_mass,
        model_path=model_path,
    )
    processor.set_axis_permutation('x, z, -y')
    
    # Options matching smpl_torque node defaults (as of 2026-05-06).
    # These must mirror the node's live widget values, NOT the
    # SMPLProcessingOptions dataclass defaults.
    #
    # For noise detection we keep rate limiting / jitter / KF OFF
    # so we see raw anomalies, but all physics settings (contact method,
    # logodds params, CoM filters, acc smoothing) match the live node
    # so that torques are computed identically.
    options = SMPLProcessingOptions(
        input_type='axis_angle',
        input_up_axis='Y',
        axis_permutation='x, z, -y',
        quat_format='wxyz',
        return_quats=False,
        dt=dt,

        # Physics
        add_gravity=True,
        enable_passive_limits=True,
        enable_apparent_gravity=True,
        use_s_curve_spine=True,
        world_frame_dynamics=True,

        # Floor / contact
        floor_enable=True,
        floor_height=0.0,
        floor_tolerance=0.15,
        contact_method='logodds_valved',
        enable_body_contacts=True,

        # LogOdds evidence streams (match node defaults)
        logodds_enable_height=True,
        logodds_enable_kinematic=True,
        logodds_enable_structural=True,
        logodds_enable_divergence=True,
        logodds_weight_height=1.0,
        logodds_weight_kinematic=0.5,
        logodds_weight_structural=1.0,
        logodds_weight_divergence=1.0,
        logodds_decay_rate=0.90,
        logodds_struct_force_ema_alpha=1.0,
        logodds_struct_relief_logodds=0.3,

        # All rate limiting / filtering OFF — we want RAW anomalies
        enable_rate_limiting=False,
        enable_jitter_damping=False,
        enable_kf_smoothing=False,
        enable_velocity_gate=False,
        enable_one_euro_filter=False,
        smooth_input_window=smooth_input_window,
        smooth_contact_forces=False,

        # Acceleration smoothing (Savitzky-Golay derivative window)
        acc_smooth_window=7,
        torque_smooth_window=0,

        # CoM One Euro filter params — match node widget defaults
        com_pos_min_cutoff=999.0,   # Position filter OFF
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,    # Velocity: light smoothing
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,     # Acceleration
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
    
    # ── Velocity-gated effort discount ─────────────────────────────────
    # Noise glitches produce high effort at LOW angular velocity (the joint
    # isn't actually moving).  Real dynamic maneuvers (cartwheels, jumps)
    # produce high effort at HIGH angular velocity (the joint IS moving).
    # Discount the effort signal when velocity is above a threshold.
    EFFORT_VEL_GATE = 3.0     # rad/s — above this, effort is expected
    EFFORT_VEL_SCALE = 10.0   # rad/s — at this velocity, full discount
    vel_discount = np.clip(
        1.0 - (ang_vel[:, :N_JOINTS] - EFFORT_VEL_GATE) / EFFORT_VEL_SCALE,
        0.1, 1.0
    )  # (T, J): 1.0 when slow (suspicious), 0.1 when fast (expected)
    s_effort *= vel_discount
    
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
    
    # ── Dynamic motion discount ───────────────────────────────────────
    # When the whole body is in fast motion (cartwheels, flips, jumps),
    # high torque across joints is expected physics.  Noise glitches
    # happen at ANY activity level, but legitimate dynamics only at high
    # activity.  Discount scores proportionally to how far above median
    # the per-frame total body velocity is.
    body_vel_total = np.sum(ang_vel[:, :N_JOINTS], axis=1)  # (T,) rad/s
    vel_p50 = np.percentile(body_vel_total, 50) if T > 0 else 0
    vel_p90 = np.percentile(body_vel_total, 90) if T > 0 else 0
    # Gate: starts discounting at p75 of the file's velocity distribution
    vel_gate = np.percentile(body_vel_total, 75) if T > 0 else 0
    vel_range = max(vel_p90 - vel_gate, 5.0)  # avoid div-by-zero
    # Discount: 1.0 at vel_gate, 0.15 at vel_p90+
    DYNAMIC_MOTION_DISCOUNT_MIN = 0.15
    dynamic_discount = np.clip(
        1.0 - (body_vel_total - vel_gate) / vel_range * (1.0 - DYNAMIC_MOTION_DISCOUNT_MIN),
        DYNAMIC_MOTION_DISCOUNT_MIN, 1.0
    )  # (T,)
    joint_scores *= dynamic_discount[:, np.newaxis]  # broadcast to (T, J)
    
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
    
    # ── Load-bearing discount ─────────────────────────────────────────
    # When a contact joint is actively bearing weight (not just transitioning),
    # high torque in the kinetic chain is expected physics (e.g., shoulder
    # torque during a cartwheel hand plant).  Apply a proportional discount
    # based on the fraction of body weight carried by that chain.
    LOAD_BEARING_THRESH = 20.0     # N — minimum force for load-bearing
    LOAD_BEARING_DISCOUNT = 0.1    # score multiplier when bearing full weight
    
    for contact_j, chain_joints in CONTACT_CHAIN.items():
        if contact_j >= N_CONTACT_JOINTS:
            continue
        
        force_j = contact_forces[:, contact_j]  # (T,)
        bearing = force_j > LOAD_BEARING_THRESH  # (T,) boolean
        
        if np.any(bearing):
            # Discount proportional to force fraction (normalized to total mass)
            total_weight = 75.0 * 9.81  # ~735 N
            force_frac = np.clip(force_j / total_weight, 0, 1)  # (T,)
            # Blend: full weight → LOAD_BEARING_DISCOUNT, zero weight → 1.0
            load_discount = 1.0 - force_frac * (1.0 - LOAD_BEARING_DISCOUNT)
            
            for chain_j in chain_joints:
                if chain_j < N_JOINTS:
                    # Take the minimum of existing discount and load-bearing discount
                    contact_event_mask[bearing, chain_j] = np.minimum(
                        contact_event_mask[bearing, chain_j],
                        load_discount[bearing]
                    )
    
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
    
    # ── Structural exclusion (breaks + corruption) ─────────────────────
    # Stream breaks and corruption zones are STRUCTURAL issues, not
    # motion quality issues.  Exclude them entirely from the noise
    # score computation.  They are reported separately via SurgeryInfo.
    if np.any(structural_mask):
        joint_scores[structural_mask] = 0
    
    # Apply joint importance weighting
    weighted_scores = joint_scores * JOINT_IMPORTANCE[np.newaxis, :]
    
    # Per-frame score: max across joints (the worst joint drives the frame score)
    frame_scores = np.max(weighted_scores, axis=1)  # (T,)
    worst_joint_per_frame = np.argmax(weighted_scores, axis=1)  # (T,)
    
    # ── Build frame-level results ─────────────────────────────────────
    # Exclude structural frames (breaks + corruption) from glitch/suspect counts
    non_structural = ~structural_mask
    soft_glitch_mask = (frame_scores >= GLITCH_THRESH) & non_structural
    suspect_mask = (frame_scores >= SUSPECT_THRESH) & ~soft_glitch_mask & non_structural
    n_s = int(np.sum(suspect_mask))
    T_valid = int(np.sum(non_structural))  # frames used for scoring

    # "Hard" glitch corroboration — only frames where the elevated torque
    # score has independent pose-level evidence (significant spike at the
    # frame, in a corruption zone, or in a stream break) are counted as
    # glitches.  Score elevation alone without corroboration is a "soft"
    # motion impulse (the torque scorer's FP signature on dynamic
    # core motion, e.g. C24's spine torque during a side step).
    corroboration_mask = structural_mask.copy()
    for f in significant_spike_frame_set:
        if 0 <= f < T:
            corroboration_mask[f] = True
    # Note: structural_mask frames are excluded from the score elevation
    # check (non_structural), so re-include them via corroboration explicitly.
    glitch_mask = (frame_scores >= GLITCH_THRESH) & corroboration_mask
    n_g = int(np.sum(glitch_mask))
    # Count over the full file (T), since corroborated structural frames
    # are now valid glitches even though they were excluded from non_structural
    g_frac = n_g / max(T, 1)

    # Glitch frame details (sorted by score) — list HARD glitches only
    glitch_list = []
    for t in range(T):
        if glitch_mask[t]:
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
    
    # Clean segments — use structural mask so break AND corruption frames
    # are excluded from the "clean" analysis; also fold in spike_frames
    # that pass the severity gate (multi-joint, above-envelope) so a
    # synchronized whole-body glitch doesn't sit inside a "clean" stretch.
    # Minor wrist twitches (single-joint, sub-envelope) are intentionally
    # not folded — they would over-fragment otherwise-clean files.
    #
    # Additionally, OR in a "density mask" that catches sustained
    # low-level arm-chain noise (the Maritsa "flickers in shoulders /
    # pecs / arms" pattern), which by design doesn't pass the
    # individual-spike severity gate.
    # significant_spike_frame_set was computed earlier (right after spike
    # detection) and used for hard-glitch corroboration.  Reuse here.
    density_mask = _arm_spike_density_mask(spike_frames, T, fps)
    clean_bad_mask = structural_mask | density_mask
    # Sparse out-of-context arm impulses — single-joint, modest velocity,
    # both neighbours nearly stationary.  Fragments clean segments only.
    isolated_arm_impulses = _isolated_arm_impulse_frames(spike_frames, fps)
    fragment_spike_frames = significant_spike_frame_set | isolated_arm_impulses
    clean = _find_clean_segments(
        frame_scores, fps,
        corruption_mask=clean_bad_mask,
        spike_frame_indices=fragment_spike_frames,
    )
    
    # ── Per-joint noise profiles ──────────────────────────────────────
    joint_profiles = []
    for j in range(N_JOINTS):
        # Exclude structural frames from per-joint stats
        j_scores_valid = joint_scores[non_structural, j]
        j_glitch = int(np.sum(j_scores_valid >= GLITCH_THRESH))
        j_suspect = int(np.sum((j_scores_valid >= SUSPECT_THRESH) & (j_scores_valid < GLITCH_THRESH)))
        j_frac = j_glitch / max(T_valid, 1)
        
        surprise_valid = surprise[non_structural, j]
        effort_valid = effort[non_structural, j]
        prof = JointNoiseProfile(
            joint_idx=j,
            joint_name=SMPL_JOINT_NAMES[j],
            n_glitch_frames=j_glitch,
            n_suspect_frames=j_suspect,
            mean_surprise=round(float(np.mean(surprise_valid)), 3),
            max_surprise=round(float(np.max(surprise_valid)), 2),
            mean_effort=round(float(np.mean(effort_valid)), 4),
            max_effort=round(float(np.max(effort_valid)), 3),
            glitch_fraction=round(j_frac, 6),
        )
        joint_profiles.append(prof)
    
    # Sort by glitch count (noisiest first)
    joint_profiles.sort(key=lambda p: -(p.n_glitch_frames + p.n_suspect_frames * 0.5))
    
    # ── File-level scoring (non-break frames only) ────────────────────
    scores_valid = frame_scores[non_structural] if T_valid > 0 else frame_scores
    p95 = float(np.percentile(scores_valid, 95)) if T_valid > 0 else 0

    # Peak / p99 — restrict to HARD glitch frames so motion-induced FP
    # impulses (high frame score from dynamic core motion, e.g. C24 spine
    # during a side step) don't drive these stats.  Falls back to overall
    # stats if no hard glitches exist — gives a sensible peak/p99 of zero.
    if n_g > 0:
        peak = float(np.max(frame_scores[glitch_mask]))
        p99 = float(np.percentile(frame_scores[glitch_mask], 99))
    else:
        peak = 0.0
        p99 = 0.0

    # Raw signal severity — restrict to HARD glitch frames so motion
    # impulse FPs don't drive the max-surprise stat either.
    if n_g > 0:
        max_raw_surprise = float(np.max(surprise[glitch_mask]))
        max_raw_effort = float(np.max(effort_spike[glitch_mask]))
    else:
        max_raw_surprise = 0.0
        max_raw_effort = 0.0
    
    # Score: multi-component to handle different noise profiles.
    #
    # Component 1: Glitch FRACTION — how much of the valid file is noisy
    score_frac = g_frac * 200
    
    # Component 2: Peak weighted frame score — worst-case (after discounts)
    score_peak = min(peak * 3, 30)
    
    # Component 3: P99 of frame scores — captures persistent noise
    score_p99 = p99 * 10
    
    # Component 4: Raw surprise severity — captures extreme teleportation
    score_raw = min(max_raw_surprise * 0.05, 25)
    
    # Component 5: Absolute glitch count — more glitches = more problematic
    score_count = min(n_g * 0.1, 10) if n_g > 3 else 0
    
    ns = score_frac + score_peak + score_p99 + score_raw + score_count
    ns = min(100.0, ns)
    
    if n_g <= 3 and peak < 1.5:
        cls = 'clean'
    elif g_frac < 0.03 and peak < 5.0 and max_raw_surprise < 300:
        cls = 'moderate'
    else:
        cls = 'problematic'

    # If the file is classified clean but has real pose-level anomalies
    # (significant spike events, corruption zones, or stream breaks),
    # bump to moderate.  These are real events even if the torque scorer
    # didn't elevate at those frames.
    n_pose_anomalies = (
        len(significant_spike_frame_set)
        + len(corruption_zones)
        + len(stream_breaks)
    )
    if cls == 'clean' and n_pose_anomalies >= 1:
        cls = 'moderate'
    
    # ── Surgery analysis (uses corruption zones computed earlier) ─────────
    surgery = _analyze_surgery(stream_breaks, T, fps, corruption_zones=corruption_zones)
    
    # Component 6: Corruption fraction — raw pose level evidence that the
    # torque pipeline's filtering has absorbed.  This is the pre-filtering
    # signal that would otherwise be invisible to the torque scorer.
    corruption_frac = surgery.excision.corrupted_fraction if surgery.excision else 0.0
    score_corruption = corruption_frac * 50  # 50% corruption → +25 points
    ns += score_corruption
    ns = min(100.0, ns)

    # Zigzag (buzz) — continuous contribution, soft ramp above a dead-band floor.
    zigzag_contribution = min(max(0.0, (zigzag_peak - ZIGZAG_FLOOR) * ZIGZAG_SCALE), ZIGZAG_CAP)
    ns += zigzag_contribution
    ns = min(100.0, ns)

    # Component 7: Single-frame spike fraction.  Each spike is a real
    # dropped frame.  Scaled by spike density (fraction of file affected)
    # rather than absolute count, so long files with many spikes are
    # penalised proportionally instead of saturating at a 15-point cap.
    n_spike_frames = len({s.frame for s in spike_frames})
    spike_fraction = n_spike_frames / max(T, 1)
    score_spikes = spike_fraction * 300
    ns += score_spikes
    ns = min(100.0, ns)

    # Component 8: Pose-level anomaly density.  Significant spikes,
    # corruption zones, and stream breaks are the channels we trust to
    # flag REAL physical anomalies.  Scaled by file length (anomalies
    # per frame) and uncapped so a long file with proportional anomaly
    # density gets the same per-unit weight as a short file would.
    n_significant_spikes = len(significant_spike_frame_set)
    n_pose_anomalies = (
        n_significant_spikes
        + len(corruption_zones)
        + len(stream_breaks)
    )
    score_anomalies = (n_pose_anomalies / max(T, 1)) * 300
    ns += score_anomalies
    ns = min(100.0, ns)

    # Component 9: Clean-fraction penalty on long files.  A long file that
    # only yields a small clean fraction is fundamentally compromised —
    # the score reflects how little usable material survives even if
    # individual channels are modest.
    clean_total = sum(s.n_frames for s in clean)
    clean_frac = clean_total / max(T, 1)
    file_seconds = T / float(fps) if fps > 0 else 0.0
    if (file_seconds >= CLEAN_FRAC_PENALTY_MIN_DURATION_S
            and clean_frac < CLEAN_FRAC_PENALTY_THRESHOLD):
        score_fragmentation = (CLEAN_FRAC_PENALTY_THRESHOLD - clean_frac) * CLEAN_FRAC_PENALTY_SCALE
        ns += score_fragmentation
        ns = min(100.0, ns)

    # Corruption fraction also bumps classification.
    # >10% arm corruption means the file is structurally compromised.
    if corruption_frac > 0.10:
        cls = 'problematic'

    # Teleportation -> problematic (plausibility-ceiling lens): either a single
    # near-certain (non-physical) jump, or pervasive moderate teleportation.
    # Sub-60 inconsistency dropped from the lock: it added 0 DanceDB recall
    # (>60 + flicker + zigzag + ROM cover it) and was the sole source of FPs on
    # fast REPETITIVE actions (shake/clap/punch), which are kinematically
    # indistinguishable from sub-60 teleports.  Only the physical-ceiling
    # criterion locks; teleport_rate stays exposed as a diagnostic.
    if max_arm_vel > TELEPORT_VEL_DEFINITE:
        cls = 'problematic'
    if flicker_rate >= FLICKER_RATE_LOCK or flicker_peak >= FLICKER_PEAK_LOCK:
        cls = 'problematic'
    if n_rom >= ROM_MIN_FRAMES:
        cls = 'problematic'

    # Low clean fraction on a long file is also automatically problematic.
    if (file_seconds >= CLEAN_FRAC_PENALTY_MIN_DURATION_S
            and clean_frac < CLEAN_FRAC_PROBLEMATIC_THRESHOLD):
        cls = 'problematic'

    # ── Clean-section score ───────────────────────────────────────────
    # Rates noise level WITHIN the clean segments.  Tells the user: if
    # I extract just the clean sections, how usable is the data?
    #
    # By construction, clean segments contain no pose-level anomalies
    # (significant spikes, corruption zones, stream breaks are all
    # excluded), so under the hard-glitch concept there are zero hard
    # glitches within clean.  What's left to measure is sub-significant
    # spike density — frames flagged by the spike detector but didn't
    # qualify as significant (single-joint wrist twitches, isolated arm
    # blips, etc.).  These are what the user calls "flickers."
    #
    # Per-frame torque-score components are NOT used — they have the
    # same dynamic-motion FP signature within clean sections that we
    # already excluded from noise_score (C24-style spine impulses).
    clean_frame_idx = np.zeros(T, dtype=bool)
    for seg in clean:
        clean_frame_idx[seg.start:seg.end + 1] = True
    n_clean_frames = int(clean_frame_idx.sum())
    if n_clean_frames > 0:
        clean_n_spike_unique = len({s.frame for s in spike_frames
                                    if 0 <= s.frame < T and clean_frame_idx[s.frame]})
        cs = (clean_n_spike_unique / n_clean_frames) * 300
        clean_section_score = round(min(100.0, cs), 4)
    else:
        clean_section_score = 0.0

    report = TorqueFileReport(
        filename=os.path.basename(filepath),
        n_frames=T, fps=fps,
        duration_s=round(T / fps, 1),
        noise_score=round(ns, 4),
        classification=cls,
        clean_section_score=clean_section_score,
        n_glitch_frames=n_g,
        n_suspect_frames=n_s,
        glitch_fraction=round(g_frac, 6),
        max_surprise=round(max_raw_surprise, 2),
        max_tv_ratio=round(float(np.max(tv_ratio)), 1) if T > 0 else 0,
        max_effort=round(float(np.max(effort)), 3) if T > 0 else 0,
        p95_score=round(p95, 4),
        p99_score=round(p99, 4),
        joint_profiles=joint_profiles,
        glitch_frames=glitch_list,
        glitch_clusters=clusters,
        clean_segments=clean,
        stream_breaks=stream_breaks,
        spike_frames=spike_frames,
        spike_clusters=spike_clusters,
        surgery=surgery,
        cadence=cadence,
        motion_profile=motion_profile,
        ground_contact=ground_contact,
    )
    
    report.teleport_max_arm_vel = round(float(max_arm_vel), 1)
    report.teleport_inconsistent_rate = round(float(teleport_rate), 2)
    report.n_teleport_frames = int(teleport_mask.sum())
    report.flicker_rate = round(float(flicker_rate), 2)
    report.flicker_peak = int(flicker_peak)
    report.n_flicker_frames = int(flicker_mask.sum())
    report.zigzag_severity = round(float(zigzag_peak), 3)
    report.zigzag_contribution = round(float(zigzag_contribution), 1)
    report.n_rom_frames = int(n_rom)
    # Build multi-dimensional classification
    report.classification_detail, report.recommendations = _build_classification(report)
    
    if verbose:
        print_report(report)
    
    return report


# ── Helpers ───────────────────────────────────────────────────────────────

# ── Plausibility-ceiling teleport lens ────────────────────────────────
# A human arm joint cannot rotate faster than ~20-30 rad/s.  Per-frame arm
# angular velocity beyond DEFINITE (60) is non-physical -> locks problematic.
# The sub-60 inconsistency rate (narrow, unfilled hump from a slow baseline)
# is computed as a diagnostic only: it is kinematically indistinguishable
# from fast repetitive motion (shake/clap/punch) so it does NOT lock.
TELEPORT_ARM_JOINTS  = {13, 14, 16, 17, 18, 19, 20, 21}
TELEPORT_VEL_CAND    = 30.0   # rad/s — only evaluate frames above this (below = plausible)
TELEPORT_VEL_HARD    = 40.0   # rad/s — frames above this excised
TELEPORT_VEL_DEFINITE= 60.0   # rad/s — physically impossible -> lock regardless of shape
TELEPORT_WINDOW      = 8      # frames each side for the consistency window
TELEPORT_HMW_MAX     = 4      # half-max hump width <= this AND ...
TELEPORT_FILL_MAX    = 0.28   # ... hump fill < this AND ...
TELEPORT_BASE_MAX    = 8.0    # ... local baseline velocity < this (out-of-character) => teleport
TELEPORT_RATE_LOCK   = 0.3    # inconsistent-frame rate (/s) to force "problematic"


def _detect_teleports(poses, fps):
    """Physical-plausibility teleport lens on the arm chain.

    A teleport is a high-velocity DISCONTINUITY; a real fast move (throw, punch)
    is high velocity that is CONTINUOUS — the peak is reached through a gradual
    accel ramp and left through a decel ramp, so it sits atop a multi-frame
    velocity hump.  We measure that hump with two WINDOW-INTEGRATED quantities
    (robust, not a fragile +/-1 ratio): HMW = half-max width (#frames near peak)
    and fill = hump area / (excess * window).  A candidate frame is teleport-like
    only when BOTH are small (narrow, unfilled hump).

    Returns (teleport_mask, n_inconsistent, inconsistent_rate, max_arm_vel).
    """
    T = poses.shape[0]
    mask = np.zeros(T, dtype=bool)
    if T < 3:
        return mask, 0.0, 0.0, 0.0
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    P = poses.reshape(T, -1, 3)[:, :nj, :]
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) * fps   # (T-1, J)
    arm = [j for j in TELEPORT_ARM_JOINTS if j < v.shape[1]]
    if not arm:
        return mask, 0.0, 0.0, 0.0
    v_arm = v[:, arm].max(axis=1)                          # (T-1,)
    L = len(v_arm)
    W = TELEPORT_WINDOW
    n_inconsistent = 0
    for t in np.where(v_arm > TELEPORT_VEL_CAND)[0]:
        j = arm[int(v[t, arm].argmax())]
        vj = v[:, j]
        a = max(0, t - W); b = min(L, t + W + 1)
        win = vj[a:b]; pk = vj[t]; pkpos = t - a
        # robust local baseline: window median excluding the peak +/-2
        m = np.ones(len(win), dtype=bool)
        for k in range(pkpos - 2, pkpos + 3):
            if 0 <= k < len(m):
                m[k] = False
        base = float(np.median(win[m])) if m.any() else 0.0
        exc = pk - base
        if exc < 1e-6:
            continue
        hmw = int(((win - base) >= 0.5 * exc).sum())
        fill = float(np.clip(win - base, 0.0, None).sum() / (exc * len(win)))
        # teleport-like = narrow, unfilled hump AND out of character: the spike
        # erupts from a slow local baseline.  A real fast action (throw, punch)
        # has a fast baseline around its peak, so it is spared.
        if hmw <= TELEPORT_HMW_MAX and fill < TELEPORT_FILL_MAX and base < TELEPORT_BASE_MAX:
            n_inconsistent += 1  # diagnostic only; sub-60 frames are NOT excised (may be real motion)
    # also excise clearly-impossible frames (so a smooth-looking but >40 rad/s
    # candy-wrapper unwind is still removed)
    for t in np.where(v_arm > TELEPORT_VEL_HARD)[0]:
        mask[t] = True
        if t + 1 < T:
            mask[t + 1] = True
    dur = T / float(fps) if fps > 0 else 0.0
    inconsistent_rate = n_inconsistent / dur if dur > 0 else 0.0
    return mask, float(n_inconsistent), float(inconsistent_rate), float(v_arm.max())


# ── Flicker lens ──────────────────────────────────────────────────────
# Low-amplitude marker jitter: a bad mocap-solve frame perturbs SEVERAL
# arm joints on the SAME single frame (a relative-velocity spike from a
# near-still baseline).  Real motion recruits joints in sequence, so it
# never synchronises spikes across joints.  Absolute velocity stays low
# (these are NOT teleports), so the teleport lens is blind to them.
FLICKER_FLOOR = 2.0    # rad/s — guard against numerical noise
FLICKER_RATIO = 3.0    # vel[t] / max(neighbours) — relative spike
FLICKER_VCAP  = 25.0   # rad/s — flicker is LOW absolute velocity (teleport lens has the rest)
FLICKER_SYNC  = 3      # >= this many arm joints spiking on one frame = a bad-solve frame
FLICKER_RATE_LOCK = 2.0  # synchronized-flicker frames / second to force "problematic"
FLICKER_PEAK_LOCK = 8    # ... or this many within any 1-second window (localized burst)


def _detect_flicker(poses, fps):
    """Synchronized low-amplitude arm flicker (marker jitter).

    Returns (flicker_mask, flicker_rate, flicker_peak):
      flicker_mask  (T,) bool — frames where >= FLICKER_SYNC arm joints spike
      flicker_rate  float    — flicker frames per second
      flicker_peak  int      — max flicker frames within any 1-second window
    """
    T = poses.shape[0]
    mask = np.zeros(T, dtype=bool)
    if T < 5:
        return mask, 0.0, 0
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    P = poses.reshape(T, -1, 3)[:, :nj, :]
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) * fps   # (T-1, J)
    arm = [j for j in TELEPORT_ARM_JOINTS if j < v.shape[1]]
    L = v.shape[0]
    spk = np.zeros((L, len(arm)), dtype=bool)
    for jj, j in enumerate(arm):
        vj = v[:, j]
        nm = np.maximum(vj[:-2], vj[2:])
        cond = (vj[1:-1] >= FLICKER_FLOOR) & (vj[1:-1] < FLICKER_VCAP) & (nm > 1e-6) & \
               (vj[1:-1] / np.maximum(nm, 1e-9) >= FLICKER_RATIO)
        spk[1:-1, jj] = cond
    sync = spk.sum(axis=1) >= FLICKER_SYNC          # (L,) bad-solve frames
    mask[:L] = sync
    dur = T / float(fps) if fps > 0 else 0.0
    flicker_rate = int(sync.sum()) / dur if dur > 0 else 0.0
    idx = np.where(sync)[0]; w = int(fps); peak = 0
    for f in idx:
        peak = max(peak, int(((idx >= f) & (idx < f + w)).sum()))
    return mask, float(flicker_rate), int(peak)


# ── Zigzag lens ───────────────────────────────────────────────────────
# High-frequency direction reversal on a MOVING joint (the wrist buzzing /
# zigzagging while it travels) — flicker superimposed on real motion, which
# the sync-flicker lens (needs a still baseline) and the teleport lens (needs
# a smooth high-velocity jump) both miss.  Built as a CONTINUOUS, window-
# integrated intensity, not a threshold: per frame, a reversal weight
# (1-cos)/2 (0=smooth, 1=full reversal) times a soft motion weight, smoothed
# over a window so one legit turning-point washes out but a sustained buzz
# accumulates.  Contributes to noise_score on a soft ramp above a dead-band.
ZIGZAG_W     = 8
ZIGZAG_V0    = 8.0     # rad/s — soft motion-weight knee (reversals on a still joint = noise, ignored)
ZIGZAG_VS    = 3.0
ZIGZAG_FLOOR = 0.12    # dead-band: clean motion sits below this (clean p90~0.08)
ZIGZAG_SCALE = 50.0    # noise_score points per unit intensity above the floor
ZIGZAG_CAP   = 30.0


def _detect_zigzag(poses, fps):
    """Continuous arm zigzag intensity Z[t] (windowed soft reversal x motion).
    Returns (Z, peak_Z, peak_frame)."""
    T = poses.shape[0]
    Zall = np.zeros(T)
    if T < 5:
        return Zall, 0.0, 0
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    P = poses.reshape(T, -1, 3)[:, :nj, :]
    arm = [j for j in TELEPORT_ARM_JOINTS if j < P.shape[1]]
    kern = np.ones(2 * ZIGZAG_W + 1) / (2 * ZIGZAG_W + 1)
    for j in arm:
        dv = np.diff(P[:, j, :], axis=0)
        n = np.linalg.norm(dv, axis=1)
        spd = n * fps
        if len(dv) < 3:
            continue
        num = (dv[:-1] * dv[1:]).sum(axis=1)
        den = n[:-1] * n[1:]
        c = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        rw = (1.0 - c) / 2.0                                  # reversal weight, continuous
        mspd = np.minimum(spd[:-1], spd[1:])                  # both sides must move
        mw = 1.0 / (1.0 + np.exp(-(mspd - ZIGZAG_V0) / ZIGZAG_VS))
        z = rw * mw
        Z = np.convolve(z, kern, mode='same')
        Zall[1:1 + len(Z)] = np.maximum(Zall[1:1 + len(Z)], Z)
    return Zall, float(Zall.max()), int(Zall.argmax())


# ── ROM / static-pose lens ────────────────────────────────────────────
# A wrist cannot be >150 deg from neutral (anatomical max ~90-100; trusted-
# clean p99=98, never >150).  A SUSTAINED impossible wrist rotation is a
# candy-wrapper twist artifact — invisible to every velocity lens when it
# does not correct.  Excise the impossible span; lock if sustained.
ROM_WRIST_JOINTS = (20, 21)
ROM_MAX_DEG      = 150.0
ROM_MIN_FRAMES   = 3       # sustained -> lock


def _detect_rom(poses, fps):
    """Sustained anatomically-impossible wrist rotation (candy-wrapper).
    Returns (rom_mask, n_impossible_frames)."""
    T = poses.shape[0]
    mask = np.zeros(T, dtype=bool)
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    P = poses.reshape(T, -1, 3)[:, :nj, :]
    for j in ROM_WRIST_JOINTS:
        if j >= P.shape[1]:
            continue
        mag = np.degrees(np.linalg.norm(P[:, j, :], axis=1))
        mask |= (mag > ROM_MAX_DEG)
    return mask, int(mask.sum())


def _detect_spike_frames(poses, fps):
    """Detect single dropped/corrupted frames via the neighbour-ratio test.

    A real sudden movement has at least one high-velocity neighbour (build-up
    or follow-through).  A dropped frame is flanked by low velocity on BOTH
    sides, so vel[t] / max(vel[t-1], vel[t+1]) is large.

    Returns a list of SpikeFrame (one entry per joint per spike frame).
    """
    T = poses.shape[0]
    if T < 3:
        return []

    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    poses_3d = poses.reshape(T, -1, 3)[:, :nj, :]
    vel = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2) * fps  # (T-1, J)

    spikes = []
    for j in range(nj):
        v = vel[:, j]
        # Skip first and last velocity frames to always have two neighbours
        for t in range(1, len(v) - 1):
            if v[t] < SPIKE_VEL_FLOOR:
                continue
            neighbor_max = max(v[t - 1], v[t + 1])
            if neighbor_max < 1e-6:
                continue
            ratio = v[t] / neighbor_max
            if ratio >= SPIKE_NEIGHBOR_RATIO:
                name = SMPL_JOINT_NAMES[j] if j < len(SMPL_JOINT_NAMES) else f'j{j}'
                spikes.append(SpikeFrame(
                    frame=t,
                    joint_idx=j,
                    joint_name=name,
                    velocity=round(float(v[t]), 1),
                    neighbor_ratio=round(float(ratio), 2),
                ))

    # Sort by frame then joint for stable output
    spikes.sort(key=lambda s: (s.frame, s.joint_idx))
    return spikes


def _cluster_spike_frames(spike_frames, fps):
    """Consolidate spike frames into temporal clusters.

    Spike frames separated by <= SPIKE_CLUSTER_GAP_S merge into one
    cluster; clusters with >= SPIKE_CLUSTER_CONTAMINATED_MIN distinct
    spike frames are flagged as contaminated sections.

    Returns a list of SpikeCluster ordered by start frame.
    """
    if not spike_frames:
        return []

    gap = max(1, int(round(SPIKE_CLUSTER_GAP_S * fps)))
    by_frame = {}
    for s in spike_frames:
        by_frame.setdefault(s.frame, []).append(s)

    frames = sorted(by_frame)
    groups = [[frames[0]]]
    for f in frames[1:]:
        if f - groups[-1][-1] <= gap:
            groups[-1].append(f)
        else:
            groups.append([f])

    clusters = []
    for g in groups:
        entries = [s for f in g for s in by_frame[f]]
        joint_counts = {}
        for s in entries:
            joint_counts[s.joint_name] = joint_counts.get(s.joint_name, 0) + 1
        span = g[-1] - g[0] + 1
        clusters.append(SpikeCluster(
            start=g[0], end=g[-1],
            n_frames=span,
            duration_s=round(span / fps, 2),
            n_spike_frames=len(g),
            n_spike_records=len(entries),
            spike_density=round(len(g) / span, 3),
            joints=sorted(joint_counts, key=lambda n: -joint_counts[n]),
            max_velocity=round(max(s.velocity for s in entries), 1),
            contaminated=len(g) >= SPIKE_CLUSTER_CONTAMINATED_MIN,
        ))
    return clusters


def _isolated_arm_impulse_frames(spike_frames, fps):
    """Identify isolated arm-chain spikes that disqualify clean segments.

    A candidate is any arm-chain spike whose two neighbours are both
    near-stationary (max(v[t-1], v[t+1]) <= ceiling) and whose magnitude
    is above the velocity floor.  But a candidate is only DISQUALIFYING
    when it co-occurs with another candidate — either at the same frame
    (≥2 joints) or within the temporal-clustering window.  Lone
    candidates in long clean spans are presumed sampling artifacts.

    Uses values already stored on SpikeFrame: max(v[t-1], v[t+1]) is
    recovered as velocity / neighbor_ratio.  No pose recomputation needed.

    Returns a set of frame indices.  Caller is expected to union this with
    significant_spike_frame_set when fragmenting clean segments.
    """
    # Per-frame joint count of candidate impulses
    per_frame_count = {}
    for s in spike_frames:
        if s.joint_idx not in ARM_CHAIN_JOINT_INDICES:
            continue
        if s.velocity < ISOLATED_ARM_IMPULSE_VEL_FLOOR:
            continue
        if s.neighbor_ratio < 1e-6:
            continue
        neighbor_max = s.velocity / s.neighbor_ratio
        if neighbor_max <= ISOLATED_ARM_IMPULSE_NEIGHBOR_CEILING:
            per_frame_count[s.frame] = per_frame_count.get(s.frame, 0) + 1

    if not per_frame_count:
        return set()

    candidate_frames = sorted(per_frame_count.keys())
    window = max(1, int(ISOLATED_ARM_IMPULSE_CLUSTER_WINDOW_S * fps))

    disqualifying = set()
    n = len(candidate_frames)
    for i, f in enumerate(candidate_frames):
        # Multi-joint at same frame is disqualifying on its own
        if per_frame_count[f] >= ISOLATED_ARM_IMPULSE_MIN_JOINTS_SAME_FRAME:
            disqualifying.add(f)
            continue
        # Otherwise, require another candidate frame within the window
        if i > 0 and f - candidate_frames[i - 1] <= window:
            disqualifying.add(f)
        elif i < n - 1 and candidate_frames[i + 1] - f <= window:
            disqualifying.add(f)
    return disqualifying


def _arm_spike_density_mask(spike_frames, T, fps):
    """Build a bad mask from arm-chain spike-frame density.

    Sustained low-level arm-chain noise (multiple sub-significant spikes
    spread across a short window) indicates flickery arm-marker data even
    when no individual spike clears the severity gate.  Counts unique
    frames with arm-chain spike entries in a rolling window; flags any
    window whose count crosses SPIKE_DENSITY_MIN_COUNT.

    This catches the "flickers of torque on shoulders, pecs, and arms"
    pattern found in known-corrupted long files (e.g. Maritsa).  It is
    *not* triggered by isolated multi-joint events (e.g. a single
    side-step burst): those have high joint count at one instant but
    only 1–2 unique frames in the density window.
    """
    if not spike_frames:
        return np.zeros(T, dtype=bool)

    has_arm_spike = np.zeros(T, dtype=bool)
    for s in spike_frames:
        if s.joint_idx in ARM_CHAIN_JOINT_INDICES and 0 <= s.frame < T:
            has_arm_spike[s.frame] = True

    # Clamp window to T — np.convolve in 'same' mode returns max(M, N)
    # elements, so a kernel longer than the input would produce an
    # output bigger than T and break the downstream OR with structural_mask.
    window = max(1, min(int(SPIKE_DENSITY_WINDOW_S * fps), T))
    kernel = np.ones(window, dtype=int)
    density = np.convolve(has_arm_spike.astype(int), kernel, mode='same')
    return density >= SPIKE_DENSITY_MIN_COUNT


def _significant_spike_frames(spike_frames, poses, fps):
    """Identify spike frames severe enough to fragment clean segments.

    Returns a set of frame indices.  A frame is "significant" when its
    per-joint contributions sum to >= SPIKE_SEVERITY_STRONG_THRESHOLD AND
    >= SPIKE_SEVERITY_MIN_QUALIFYING core/mid joints (weight >= 0.5)
    each contribute >= SPIKE_SEVERITY_QUALIFY_CONTRIB.

    Per-joint contribution = joint_weight * max(0, vel/joint_p95 - 1) * ratio

    Multi-joint involvement requirement excludes arm-only fast-motion
    bursts (which produce high single-joint wsev but are real motion).
    """
    if not spike_frames:
        return set()

    T = poses.shape[0]
    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    poses_3d = poses.reshape(T, -1, 3)[:, :nj, :]
    vel = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2) * fps
    joint_p95 = np.maximum(np.percentile(vel, 95, axis=0), 0.5)

    by_frame = {}
    for s in spike_frames:
        by_frame.setdefault(s.frame, []).append(s)

    significant = set()
    for f, entries in by_frame.items():
        wsev = 0.0
        n_qualifying = 0
        n_arm_chain = 0
        n_non_arm_chain = 0
        for e in entries:
            w = SPIKE_SEVERITY_JOINT_WEIGHT.get(e.joint_name, 0.5)
            p95 = float(joint_p95[e.joint_idx]) if e.joint_idx < len(joint_p95) else 0.5
            excess = max(0.0, e.velocity / p95 - 1.0)
            contrib = w * excess * e.neighbor_ratio
            wsev += contrib
            if w >= 0.5 and contrib >= SPIKE_SEVERITY_QUALIFY_CONTRIB:
                n_qualifying += 1
            if e.joint_idx in ARM_CHAIN_JOINT_INDICES:
                n_arm_chain += 1
            else:
                n_non_arm_chain += 1
        # Standard severity gate
        if wsev >= SPIKE_SEVERITY_STRONG_THRESHOLD and n_qualifying >= SPIKE_SEVERITY_MIN_QUALIFYING:
            significant.add(f)
            continue
        # Pure-arm-cluster gate: signature of one-side marker failure
        if n_arm_chain >= PURE_ARM_CLUSTER_MIN_JOINTS and n_non_arm_chain == 0:
            significant.add(f)
    return significant


def _detect_corruption_zones(poses, fps):
    """Detect sustained joint-level corruption (marker dropout, capture failure).
    
    Uses LOCAL CONTEXT: compares each joint's short-term velocity against its
    own long-term local baseline (rolling median).  This correctly distinguishes
    between fast movement (both short and long terms are high → low ratio) and
    corruption during slow movement (short term spikes while baseline is low
    → high ratio).
    
    Returns:
        zones: list of CorruptionZone
        corruption_mask: (T,) boolean array, True for frames in any corruption zone
    """
    T = poses.shape[0]
    if T < CORRUPTION_MIN_DURATION_FRAMES * 2:
        return [], np.zeros(T, dtype=bool)
    # Skip corruption detection on files too short for the rolling baseline
    # to settle — the long window would collapse to roughly the activity
    # length itself, causing legitimate motion bursts (punches, quick gestures)
    # to be mis-flagged as sustained corruption.
    if T / float(fps) < CORRUPTION_MIN_FILE_S:
        return [], np.zeros(T, dtype=bool)

    nj = poses.shape[1] // 3 if poses.ndim == 2 else poses.shape[1]
    nj = min(nj, N_JOINTS)
    poses_3d = poses.reshape(T, -1, 3)[:, :nj, :]
    
    # Per-joint angular velocity magnitude (rad/s)
    vel = np.linalg.norm(np.diff(poses_3d, axis=0), axis=2) * fps  # (T-1, J)
    
    # Window sizes.  Clamp w_long to half the file so that on short clips the
    # baseline reflects a genuine local neighbourhood rather than the whole file.
    w_short = max(3, int(CORRUPTION_SHORT_WINDOW_S * fps))
    w_long = max(w_short * 2, min(int(CORRUPTION_LONG_WINDOW_S * fps), (T - 1) // 2))
    merge_gap = int(CORRUPTION_MERGE_GAP_S * fps)
    
    short_kernel = np.ones(w_short) / w_short
    
    # Per-joint: find sustained locally-anomalous velocity runs.
    # Only check arm-chain joints.
    per_joint_runs = {}  # j -> list of (start, end)
    
    for j in sorted(CORRUPTION_JOINTS):
        v = vel[:, j]

        # Short-term rolling mean: captures current activity
        short_mean = np.convolve(v, short_kernel, mode='same')

        # Long-term rolling median: captures local baseline.
        # Uses a strided approach for efficiency on long files.
        # For each frame, the local baseline is the median velocity over
        # the surrounding w_long frames.
        half_long = w_long // 2
        long_median = np.empty_like(v)
        for t in range(len(v)):
            lo = max(0, t - half_long)
            hi = min(len(v), t + half_long + 1)
            long_median[t] = np.median(v[lo:hi])

        # Floor the local baseline to avoid division by zero and to avoid
        # flagging noise during near-stillness
        long_median_floor = np.maximum(long_median, 0.5)

        # Local surprise ratio: how anomalous is the current velocity
        # relative to the local baseline?
        ratio = short_mean / long_median_floor

        # Flag frames where BOTH conditions hold:
        #   1. Local surprise ratio is high (short >> long)
        #   2. Absolute velocity exceeds floor (to avoid stillness noise)
        high = (ratio > CORRUPTION_LOCAL_RATIO) & (short_mean > CORRUPTION_VEL_FLOOR)
        
        # Find runs of CORRUPTION_MIN_DURATION_FRAMES+ frames
        runs = []
        start = None
        for t in range(len(high)):
            if high[t]:
                if start is None:
                    start = t
            else:
                if start is not None and (t - start) >= CORRUPTION_MIN_DURATION_FRAMES:
                    runs.append((start, t - 1))
                start = None
        if start is not None and (len(high) - start) >= CORRUPTION_MIN_DURATION_FRAMES:
            runs.append((start, len(high) - 1))
        
        # Merge nearby runs
        merged = []
        for s, e in runs:
            if merged and (s - merged[-1][1]) <= merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        
        if merged:
            per_joint_runs[j] = merged
    
    if not per_joint_runs:
        return [], np.zeros(T, dtype=bool)
    
    # Build unified corruption zones by merging overlapping per-joint runs
    # across the arm chain (wrist+elbow+shoulder often corrupt together)
    all_intervals = []
    for j, runs in per_joint_runs.items():
        for s, e in runs:
            all_intervals.append((s, e, j))
    
    all_intervals.sort()
    
    # Merge overlapping/adjacent intervals, tracking which joints are involved
    zones = []
    if all_intervals:
        cur_s, cur_e, _ = all_intervals[0]
        cur_joints = set()
        for s, e, j in all_intervals:
            if s <= cur_e + merge_gap:
                cur_e = max(cur_e, e)
                cur_joints.add(j)
            else:
                if cur_joints:
                    zones.append((cur_s, cur_e, cur_joints))
                cur_s, cur_e = s, e
                cur_joints = {j}
        if cur_joints:
            zones.append((cur_s, cur_e, cur_joints))
    
    # Build CorruptionZone objects
    result = []
    corruption_mask = np.zeros(T, dtype=bool)
    
    for s, e, joint_set in zones:
        n = e - s + 1
        dur = n / fps
        joint_indices = sorted(joint_set)
        joint_names = [SMPL_JOINT_NAMES[j] if j < len(SMPL_JOINT_NAMES) else f'j{j}'
                       for j in joint_indices]
        
        # Velocity stats across affected joints during this zone
        zone_vels = vel[s:min(e + 1, len(vel)), :][:, joint_indices]
        mean_v = float(np.mean(zone_vels))
        max_v = float(np.max(zone_vels))
        
        result.append(CorruptionZone(
            start=s, end=e, n_frames=n,
            duration_s=round(dur, 1),
            joints=joint_names,
            joint_indices=joint_indices,
            mean_vel=round(mean_v, 1),
            max_vel=round(max_v, 0),
        ))
        corruption_mask[s:min(e + 1, T)] = True
    
    return result, corruption_mask


def _analyze_surgery(stream_breaks, T, fps, corruption_zones=None):
    """Analyze stream breaks and corruption zones to determine salvageability.
    
    Returns SurgeryInfo with segment boundaries and a recommendation:
      - 'none':          no issues, file is intact
      - 'split':         breaks present, usable segments exist
      - 'excise':        corruption zones present, sections need removal
      - 'split+excise':  both breaks and corruption
      - 'discard':       no usable segments remain
    """
    has_breaks = bool(stream_breaks)
    has_corruption = bool(corruption_zones)
    
    if not has_breaks and not has_corruption:
        return SurgeryInfo(
            n_breaks=0, needs_surgery=False,
            recommendation='none',
        )
    
    # ── Break segments ────────────────────────────────────────────────
    segments = []
    usable_frames = 0
    n_usable = 0
    
    if has_breaks:
        break_frames = sorted(b.frame for b in stream_breaks)
        boundaries = [0] + break_frames + [T]
        
        for i in range(len(boundaries) - 1):
            s = boundaries[i]
            e = boundaries[i + 1]
            if i < len(boundaries) - 2:
                e -= 1
            n = e - s
            if n <= 0:
                continue
            dur = n / fps
            usable = n >= MIN_SALVAGEABLE_FRAMES
            segments.append(SurgerySegment(
                start=s, end=e, n_frames=n,
                duration_s=round(dur, 1), usable=usable,
            ))
            if usable:
                usable_frames += n
                n_usable += 1
    
    usable_frac = usable_frames / max(T, 1) if has_breaks else 1.0
    
    # ── Excision info ─────────────────────────────────────────────────
    excision = None
    if has_corruption:
        total_corrupted = sum(z.n_frames for z in corruption_zones)
        excision = ExcisionInfo(
            n_zones=len(corruption_zones),
            total_corrupted_frames=total_corrupted,
            corrupted_fraction=round(total_corrupted / max(T, 1), 4),
            zones=corruption_zones,
        )
    
    # ── Recommendation ────────────────────────────────────────────────
    if has_breaks and has_corruption:
        rec = 'split+excise' if n_usable > 0 else 'discard'
    elif has_breaks:
        rec = 'split' if n_usable > 0 else 'discard'
    elif has_corruption:
        rec = 'excise'
    else:
        rec = 'none'
    
    return SurgeryInfo(
        n_breaks=len(stream_breaks) if stream_breaks else 0,
        needs_surgery=True,
        segments=segments,
        n_usable_segments=n_usable,
        usable_frames=usable_frames,
        usable_fraction=round(usable_frac, 4),
        recommendation=rec,
        excision=excision,
    )


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


def _find_clean_segments(scores, fps, margin=3, corruption_mask=None,
                         spike_frame_indices=None):
    """
    Find continuous stretches with NO pose-level anomalies — i.e. no
    significant spike, no corruption-zone frame, no stream-break frame.
    Expands bad regions by margin frames on each side.

    Per-frame torque scores are NOT consulted: in dynamic-motion files
    (side steps, punches, jumps) the torque scorer routinely flags
    legitimate motion impulses on core joints as "glitches" — but those
    are real movement, not bad data. The pose-level detectors (spike,
    corruption, stream-break) catch true pose anomalies.  Per-frame
    score remains in noise_score and classification; just not here.

    corruption_mask: optional (T,) boolean array from _detect_corruption_zones.
        Frames in corruption zones are marked bad regardless of their torque
        score, because the torque pipeline's filtering absorbs the evidence
        of arm corruption before scoring.

    spike_frame_indices: optional iterable of frame indices to mark bad.
        These are typically *significant* spike frames (multi-joint
        synchronized, above-envelope) that the torque score absorbed via
        acceleration smoothing.  Caller is expected to pre-filter by
        severity — passing every spike would over-fragment otherwise-clean
        files due to minor wrist twitches.

    Returns list of CleanSegment, sorted longest-first.
    """
    T = len(scores)
    bad = np.zeros(T, dtype=bool)

    # Adaptive minimum segment duration: on short files the standard
    # CLEAN_MIN_DURATION (1.0 s) would reject every gap, even when the
    # file is overall clean.  Scale down for files where 1.0 s would be
    # an unreasonable fraction of total length.
    file_seconds = T / float(fps) if fps > 0 else 0.0
    effective_min_duration = min(CLEAN_MIN_DURATION, file_seconds / 4.0)

    # Merge corruption mask — these frames have pre-filtering evidence
    # of capture failure that the torque scorer cannot see
    if corruption_mask is not None and len(corruption_mask) >= T:
        bad = bad | corruption_mask[:T]

    # Fold significant spike frames into the bad mask.  The torque score
    # can absorb single-frame discontinuities via acceleration smoothing,
    # so frames that the spike detector flagged as severe need to be
    # added explicitly to prevent contaminating clean segments.
    if spike_frame_indices:
        for f in spike_frame_indices:
            if 0 <= f < T:
                bad[f] = True

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
                if dur >= effective_min_duration:
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
    print(f"  Noise score:    {r.noise_score:.1f} / 100   (file overall)")
    print(f"  Clean-section:  {r.clean_section_score:.1f} / 100   (within clean segments only)")
    print(f"  Glitch frames:  {r.n_glitch_frames} ({100 * r.glitch_fraction:.2f}%)")
    print(f"  Suspect frames: {r.n_suspect_frames}")
    
    # ── Recommendations ────────────────────────────────────────────────
    if r.recommendations:
        rec_str = ', '.join(f'{k}={v}' for k, v in r.recommendations.items())
        print(f"  Recommended:    {rec_str}")
    
    # ── Surgery ─────────────────────────────────────────────────────────
    if r.surgery and r.surgery.needs_surgery:
        si = r.surgery
        rec_icon = {'split': '✂️', 'excise': '🔪', 'split+excise': '✂️🔪', 'discard': '⚡'}
        print(f"\n  {rec_icon.get(si.recommendation, '⚡')} Surgery: "
              f"recommendation: {si.recommendation.upper()}")
        
        # Stream breaks
        if si.n_breaks > 0:
            print(f"\n    Stream breaks: {si.n_breaks}")
            if si.recommendation in ('split', 'split+excise'):
                print(f"    {si.n_usable_segments} usable segment(s), "
                      f"{si.usable_frames} frames ({si.usable_fraction:.0%} of file)")
            
            print(f"    {'Frame':>7s}  {'Time':>6s}  {'Type':>12s}  {'TransJump':>10s}  {'PoseJump':>10s}  Worst Joint")
            for sb in r.stream_breaks:
                print(f"    {sb.frame:7d}  {sb.time_s:5.1f}s  {sb.break_type:>12s}  "
                      f"{sb.trans_jump:9.4f}m  {sb.pose_jump:9.4f}r  {sb.worst_joint}")
            
            # Show segments between breaks with usability flag
            if si.segments:
                print(f"\n    Segments between breaks:")
                for seg in si.segments:
                    flag = '✅' if seg.usable else '❌'
                    print(f"      {flag}  [{seg.start:>7d}–{seg.end:>7d}]  "
                          f"{seg.duration_s:6.1f}s  ({seg.n_frames:>6d} frames)")
        
        # Corruption zones
        if si.excision:
            ex = si.excision
            print(f"\n    Corruption zones: {ex.n_zones} detected "
                  f"({ex.total_corrupted_frames} frames, {ex.corrupted_fraction:.0%} of file)")
            for z in ex.zones:
                joints_str = ', '.join(z.joints[:4])
                if len(z.joints) > 4:
                    joints_str += f' +{len(z.joints)-4}'
                print(f"      [{z.start:>7d}–{z.end:>7d}]  {z.duration_s:5.1f}s  "
                      f"mean={z.mean_vel:.0f} max={z.max_vel:.0f} rad/s  {joints_str}")
    elif r.stream_breaks:
        # Fallback: show raw breaks if no surgery info
        print(f"\n  ⚡ Stream breaks: {len(r.stream_breaks)} detected")

    # ── Single-frame spikes (consolidated) ─────────────────────────────
    if r.spike_frames:
        clusters = r.spike_clusters or _cluster_spike_frames(r.spike_frames, r.fps)
        contaminated = [c for c in clusters if c.contaminated]
        isolated = [c for c in clusters if not c.contaminated]
        n_sf = len({s.frame for s in r.spike_frames})
        print(f"\n  ⚡ Spike frames: {n_sf} dropped/corrupted frame(s) → "
              f"{len(contaminated)} contaminated section(s) + "
              f"{sum(c.n_spike_frames for c in isolated)} isolated")
        if contaminated:
            print(f"    Contaminated sections:")
            print(f"    {'Frames':>17s}  {'Dur':>6s}  {'Spikes':>6s}  "
                  f"{'Density':>7s}  {'MaxVel':>7s}  Joints")
            for c in contaminated:
                joints_str = ', '.join(c.joints[:4])
                if len(c.joints) > 4:
                    joints_str += f' +{len(c.joints) - 4}'
                print(f"    [{c.start:>6d}–{c.end:>6d}]  {c.duration_s:5.1f}s  "
                      f"{c.n_spike_frames:6d}  {c.spike_density:7.2f}  "
                      f"{c.max_velocity:7.1f}  {joints_str}")
        if isolated:
            iso_frames = {f for c in isolated
                          for f in range(c.start, c.end + 1)}
            print(f"    Isolated spikes:")
            print(f"    {'Frame':>7s}  {'Joint':<16s}  {'Vel (rad/s)':>11s}  {'NeighbourRatio':>14s}")
            for s in r.spike_frames:
                if s.frame in iso_frames:
                    print(f"    {s.frame:7d}  {s.joint_name:<16s}  {s.velocity:11.1f}  {s.neighbor_ratio:14.1f}×")
    
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
    p.add_argument('--recursive', action='store_true',
                   help='Walk directory tree recursively (for AMASS)')
    p.add_argument('--json', help='Save results to JSON file. When --dir + --recursive '
                   'are used, treat this as an output directory and write one JSON per '
                   'top-level subdirectory.')
    p.add_argument('--checkpoint', help='Checkpoint file for resume (JSON). '
                   'Results are written incrementally after each file. '
                   'On restart, already-processed files are skipped.')
    p.add_argument('--mass', type=float, default=75.0, help='Body mass in kg (default: 75)')
    p.add_argument('--gender', choices=['male', 'female', 'neutral'],
                   help='Override gender (default: read from file)')
    p.add_argument('--smooth-window', type=int, default=0,
                   help='Input smoothing window (0=off, 3/5/7). '
                   'Use 3 for files with 3-frame cadence artifacts.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-file console reports (summary only)')
    
    args = p.parse_args()

    reports = []   # all reports processed this session (for end-of-run summary)
    n_skipped = 0
    n_errors = 0

    if args.dir and args.recursive:
        # ── Recursive mode: process subdir-by-subdir ──────────────────
        abs_dir = os.path.abspath(args.dir)

        # Group files by top-level subdirectory
        subdir_files = {}
        for root, _, f_names in os.walk(args.dir):
            for f in sorted(f_names):
                if f.endswith('.npz'):
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, abs_dir)
                    top = rel.split(os.sep)[0] if os.sep in rel else '__root__'
                    subdir_files.setdefault(top, []).append(fp)

        if not subdir_files:
            p.print_help()
            return

        n_total = sum(len(v) for v in subdir_files.values())

        # Load checkpoint — new format: {"completed_dirs": [...], "files": {...}}
        # Falls back to migrating the old list format.
        completed_dirs = set()
        ckpt_files = {}   # abs_path -> report_dict for the current in-progress dir
        if args.checkpoint and os.path.exists(args.checkpoint):
            try:
                with open(args.checkpoint, 'r') as fh:
                    ckpt = json.load(fh)
                if isinstance(ckpt, dict):
                    completed_dirs = set(ckpt.get('completed_dirs', []))
                    ckpt_files = ckpt.get('files', {})
                else:
                    for entry in ckpt:
                        ckpt_files[entry.get('filepath', '')] = entry
                print(f"  Resuming: {len(completed_dirs)} dirs complete, "
                      f"{len(ckpt_files)} files cached for current dir")
            except Exception as e:
                print(f"  ⚠️  Could not load checkpoint: {e}")

        if args.json:
            os.makedirs(args.json, exist_ok=True)

        file_idx = 0
        for subdir_name in sorted(subdir_files.keys()):
            subdir_fp_list = subdir_files[subdir_name]

            if subdir_name in completed_dirs:
                n_skipped += len(subdir_fp_list)
                file_idx += len(subdir_fp_list)
                print(f"\n  ✓ {subdir_name}  (already complete, skipping)")
                continue

            print(f"\n{'─' * 60}")
            print(f"  Directory: {subdir_name}  ({len(subdir_fp_list)} files)")
            print(f"{'─' * 60}")

            subdir_reports = []
            for filepath in subdir_fp_list:
                abs_path = os.path.abspath(filepath)
                file_idx += 1

                if abs_path in ckpt_files:
                    subdir_reports.append(ckpt_files[abs_path])
                    n_skipped += 1
                    print(f"  [{file_idx}/{n_total}] {os.path.basename(filepath)}  (resumed)")
                    continue

                if not os.path.exists(filepath):
                    print(f"  ⚠️  Not found: {filepath}")
                    continue

                print(f"\n  [{file_idx}/{n_total}] {os.path.basename(filepath)}")
                try:
                    report = analyze_file(
                        filepath,
                        total_mass=args.mass,
                        gender_override=args.gender,
                        smooth_input_window=args.smooth_window,
                        verbose=not args.quiet,
                    )
                    report_dict = _to_jsonable(asdict(report))
                    report_dict['filepath'] = abs_path
                    report_dict['glitch_frames'] = report_dict['glitch_frames'][:500]
                    subdir_reports.append(report_dict)

                    if args.checkpoint:
                        ckpt_files[abs_path] = report_dict
                        _write_checkpoint(args.checkpoint, {
                            'completed_dirs': sorted(completed_dirs),
                            'files': ckpt_files,
                        })
                except Exception as e:
                    import traceback
                    print(f"  ❌ Error: {filepath}: {e}")
                    traceback.print_exc()
                    n_errors += 1

            # Subdir complete — write its JSON, then slim the checkpoint
            if args.json and subdir_reports:
                sorted_group = sorted(subdir_reports, key=lambda x: -x.get('noise_score', 0))
                out_path = os.path.join(args.json, subdir_name + '.json')
                with open(out_path, 'w') as fh:
                    json.dump(sorted_group, fh, indent=2)
                print(f"\n  Saved {len(sorted_group)} reports → {out_path}")

            completed_dirs.add(subdir_name)
            for fp in subdir_fp_list:
                ckpt_files.pop(os.path.abspath(fp), None)
            if args.checkpoint:
                _write_checkpoint(args.checkpoint, {
                    'completed_dirs': sorted(completed_dirs),
                    'files': ckpt_files,
                })

            reports.extend(subdir_reports)

    else:
        # ── Flat mode: positional files and/or a single directory ─────
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

        # Load checkpoint (list format; also accepts new dict format)
        completed = {}  # abs_path -> report dict
        if args.checkpoint and os.path.exists(args.checkpoint):
            try:
                with open(args.checkpoint, 'r') as fh:
                    existing = json.load(fh)
                entries = existing if isinstance(existing, list) else list(existing.get('files', {}).values())
                for entry in entries:
                    fp = entry.get('filepath', entry.get('filename', ''))
                    completed[fp] = entry
                print(f"  Resuming: {len(completed)} files already processed, "
                      f"{len(files) - len(completed)} remaining")
            except Exception as e:
                print(f"  ⚠️  Could not load checkpoint: {e}")

        reports = list(completed.values())
        n_total = len(files)

        for i, filepath in enumerate(files):
            abs_path = os.path.abspath(filepath)
            if abs_path in completed:
                n_skipped += 1
                continue

            if not os.path.exists(filepath):
                print(f"  ⚠️  Not found: {filepath}")
                continue

            print(f"\n  [{i+1}/{n_total}] {os.path.basename(filepath)}")
            try:
                report = analyze_file(
                    filepath,
                    total_mass=args.mass,
                    gender_override=args.gender,
                    smooth_input_window=args.smooth_window,
                    verbose=not args.quiet,
                )
                report_dict = _to_jsonable(asdict(report))
                report_dict['filepath'] = abs_path
                report_dict['glitch_frames'] = report_dict['glitch_frames'][:500]
                reports.append(report_dict)

                if args.checkpoint:
                    completed[abs_path] = report_dict
                    _write_checkpoint(args.checkpoint, list(completed.values()))

            except Exception as e:
                import traceback
                print(f"  ❌ Error: {filepath}: {e}")
                traceback.print_exc()
                n_errors += 1

        if args.json:
            sorted_reports = sorted(reports, key=lambda x: -x.get('noise_score', 0))
            with open(args.json, 'w') as fh:
                json.dump(sorted_reports, fh, indent=2)
            print(f"\n  Saved {len(sorted_reports)} reports to {args.json}")

    # ── Multi-file summary ────────────────────────────────────────────
    if len(reports) > 1:
        print(f"\n{'═' * 72}")
        print(f"  SUMMARY ({len(reports)} files, {n_skipped} resumed, {n_errors} errors)")
        print(f"{'═' * 72}")
        print(f"  {'File':<40s}  {'Score':>7s}  {'Class':>12s}  {'Glitches':>8s}  {'Clean%':>6s}  {'Surgery':>10s}")

        sorted_reports = sorted(reports, key=lambda x: -x.get('noise_score', 0))
        for r in sorted_reports:
            cls = r.get('classification', '?')
            e = {'clean': '✅', 'moderate': '⚠️', 'problematic': '❌'}.get(cls, '?')
            clean_segs = r.get('clean_segments', [])
            tc = sum(s.get('n_frames', 0) for s in clean_segs)
            nf = max(r.get('n_frames', 1), 1)
            cp = 100 * tc / nf
            fname = r.get('filename', '?')
            ns = r.get('noise_score', 0)
            ng = r.get('n_glitch_frames', 0)
            surg = r.get('surgery', None) or {}
            surg_rec = surg.get('recommendation', 'none') if surg else 'none'
            surg_str = {'none': '—', 'split': '✂️ split', 'excise': '🔪 excise',
                        'split+excise': '✂️🔪', 'discard': '⚡discard'}.get(surg_rec, '—')
            print(f"  {fname:<40s}  {ns:7.1f}  {e} {cls:<10s}  {ng:>8d}  {cp:5.1f}%  {surg_str:>10s}")


def _write_checkpoint(path, data):
    """Atomic checkpoint write via temp file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
