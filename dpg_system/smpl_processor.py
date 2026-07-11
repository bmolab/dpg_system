import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
import sys
import time
from dataclasses import dataclass
import logging
from dpg_system.one_euro_filter import OneEuroFilter
from dpg_system.contact_consensus import ContactConsensus, ConsensusOptions, PARENTS as CONSENSUS_PARENTS

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("Warning: smplx or torch not found. Dynamic anthropometry will use approximate heuristics.")


# Module-level cache for loaded smplx body model objects.  Keyed by
# (absolute_model_path, gender_tag).  The smplx body model is purely
# read-only constant data (mesh templates, skinning weights, joint
# regressor) — instantiating one is expensive (disk read + pickle
# deserialization + torch tensor construction takes ~5-10 s).  Caching
# the loaded model lets in-process batch runs reuse it across files
# without paying that cost per file.
#
# SAFETY: The smplx model is stateless under forward calls (just a
# parametric mesh function); it does NOT accumulate state between
# files.  All EMA / sequential-filter state in SMPLProcessor lives on
# the SMPLProcessor instance itself (self.prob_prev_*, self.prev_zmp_*,
# self._ang_sg_cache, ...) and resets when a new SMPLProcessor is
# constructed.
_smplx_model_cache = {}


def _get_cached_smplx_model(model_path, g_tag):
    """Return a smplx body model for (model_path, gender_tag).
    Loads once per process and caches; subsequent calls return the
    same instance."""
    abs_path = os.path.abspath(model_path)
    key = (abs_path, g_tag)
    if key not in _smplx_model_cache:
        _smplx_model_cache[key] = smplx.create(
            model_path=model_path,
            model_type='smplh',
            gender=g_tag,
            num_betas=10,
            ext='pkl',
        )
    return _smplx_model_cache[key]


class NumpySmartClampKF:
    """
    A Linear Kalman Filter with 'Smart Innovation Clamping' for smoothing
    multi-dimensional signals while rejecting sudden spikes.
    
    This filter limits the magnitude of innovation (correction) that can be applied
    per frame, preventing glitches from teleporting the filter state while still
    tracking smooth motion accurately.
    
    Used for dynamic torque smoothing in the rate limiter.
    """
    def __init__(self, dt, num_streams, dim):
        """
        Args:
            dt: Time step (seconds)
            num_streams: Number of independent streams (e.g., 22 joints)
            dim: Dimensionality per stream (e.g., 3 for xyz torque vectors)
        """
        self.dt = dt
        self.num_streams = num_streams
        self.dim = dim
        self.state_dim = 2 * dim  # Position + Velocity
        
        # State: [Position, Velocity] per stream
        # Shape: (num_streams, state_dim)
        self.x = np.zeros((num_streams, self.state_dim))
        
        # Error Covariance P: (num_streams, state_dim, state_dim)
        self.P = np.tile(np.eye(self.state_dim) * 0.1, (num_streams, 1, 1))
        
        # State Transition Matrix F: [[I, dt*I], [0, I]]
        self.F = np.eye(self.state_dim)
        self.F[:dim, dim:] = np.eye(dim) * dt
        
        # Measurement Matrix H: [I, 0] - we only measure position
        self.H = np.zeros((dim, self.state_dim))
        self.H[:dim, :dim] = np.eye(dim)
        
        # Process and Measurement Noise (defaults, updated by update_params)
        self.Q = np.eye(self.state_dim) * 0.01
        self.R = np.eye(dim) * 0.1
        
        # Innovation clamp radius
        self.clamp_radius = 15.0
        
        # Flags
        self._initialized = False
    
    def update_params(self, responsiveness, smoothness, clamp_radius, dt):
        """
        Update filter parameters.
        
        Args:
            responsiveness: Process noise (higher = faster tracking, 1-100)
            smoothness: Measurement noise (higher = smoother output, 0.1-10)
            clamp_radius: Max innovation magnitude per frame
            dt: Time step
        """
        self.dt = dt
        self.clamp_radius = clamp_radius
        
        # Update F with new dt
        self.F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        
        # Build Q (Process Noise Covariance)
        # Position variance scales with dt^2, velocity variance scales with 1
        pos_var = (responsiveness * dt) ** 2
        vel_var = responsiveness ** 2
        q_diag = np.concatenate([np.full(self.dim, pos_var), np.full(self.dim, vel_var)])
        self.Q = np.diag(q_diag)
        
        # Build R (Measurement Noise Covariance)
        self.R = np.eye(self.dim) * smoothness
    
    def predict(self):
        """
        Prediction step: propagate state and covariance forward.
        """
        if not self._initialized:
            return
        
        # x = F @ x
        self.x = self.x @ self.F.T
        
        # P = F @ P @ F^T + Q (applied per stream)
        for i in range(self.num_streams):
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q
    
    def update(self, z):
        """
        Update step with smart innovation clamping.
        
        Args:
            z: Measurement array of shape (num_streams, dim)
            
        Returns:
            Filtered position estimate of shape (num_streams, dim)
        """
        # Initialize on first measurement
        if not self._initialized:
            self.x[:, :self.dim] = z
            self._initialized = True
            return z.copy()
        
        # Innovation: y = z - H @ x = z - x[:, :dim]
        x_pos = self.x[:, :self.dim]
        y = z - x_pos
        
        # Smart Clamping: limit innovation magnitude per stream
        if self.clamp_radius > 0:
            innovation_mag = np.linalg.norm(y, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, self.clamp_radius / innovation_mag)
            y = y * scale
        
        # Kalman Gain: K = P @ H^T @ (H @ P @ H^T + R)^-1
        # Since H = [I, 0], this simplifies significantly
        for i in range(self.num_streams):
            P_pos = self.P[i, :self.dim, :self.dim]  # Top-left block
            S = P_pos + self.R
            S_inv = np.linalg.inv(S + np.eye(self.dim) * 1e-9)
            
            # K = P @ H^T @ S_inv = P[:, :dim] @ S_inv
            K = self.P[i, :, :self.dim] @ S_inv
            
            # Update state: x = x + K @ y
            self.x[i] += K @ y[i]
            
            # Update covariance: P = (I - K @ H) @ P
            KH = np.zeros((self.state_dim, self.state_dim))
            KH[:, :self.dim] = K
            self.P[i] = (np.eye(self.state_dim) - KH) @ self.P[i]
        
        return self.x[:, :self.dim].copy()
    
    def reset(self):
        """Reset filter state."""
        self.x.fill(0)
        self.P = np.tile(np.eye(self.state_dim) * 0.1, (self.num_streams, 1, 1))
        self._initialized = False


@dataclass
class NoiseStats:
    """
    Tracks noise events detected by the rate limiter for file quality evaluation.
    
    Tracks both event counts AND severity (magnitude of violations) for accurate
    quality scores even when comparing files with different noise characteristics.
    """
    # Cumulative counts
    total_frames: int = 0
    spike_detections: int = 0        # PASS 1: Teleport/impossible movement detections
    rate_limit_clamps: int = 0       # PASS 3: Rate limit exceeded events
    jitter_damping_events: int = 0   # PASS 4: Jitter damping applied
    innovation_clamps: int = 0       # PASS 5: KF innovation exceeded threshold
    
    # Cumulative SEVERITY (magnitude of violations in N·m)
    spike_severity: float = 0.0          # Sum of excess torque above threshold
    rate_limit_severity: float = 0.0     # Sum of clamped magnitudes
    innovation_severity: float = 0.0     # Sum of clamped innovation magnitudes
    
    # PEAK severity (worst individual events)
    max_spike_severity: float = 0.0      # Worst single spike magnitude
    max_frame_severity: float = 0.0      # Worst single frame total severity
    spike_severity_list: list = None     # List of all spike severities for percentile analysis
    
    # Per-joint cumulative (for identifying problematic joints)
    joint_spike_counts: np.ndarray = None
    joint_spike_severity: np.ndarray = None
    joint_rate_limit_counts: np.ndarray = None
    joint_jitter_counts: np.ndarray = None
    
    def __post_init__(self):
        # Initialize per-joint arrays (24 joints)
        if self.joint_spike_counts is None:
            self.joint_spike_counts = np.zeros(24, dtype=np.int32)
        if self.joint_spike_severity is None:
            self.joint_spike_severity = np.zeros(24, dtype=np.float64)
        if self.joint_rate_limit_counts is None:
            self.joint_rate_limit_counts = np.zeros(24, dtype=np.int32)
        if self.joint_jitter_counts is None:
            self.joint_jitter_counts = np.zeros(24, dtype=np.int32)
        if self.spike_severity_list is None:
            self.spike_severity_list = []
    
    def reset(self):
        """Reset all statistics."""
        self.total_frames = 0
        self.spike_detections = 0
        self.rate_limit_clamps = 0
        self.jitter_damping_events = 0
        self.innovation_clamps = 0
        self.spike_severity = 0.0
        self.rate_limit_severity = 0.0
        self.innovation_severity = 0.0
        self.max_spike_severity = 0.0
        self.max_frame_severity = 0.0
        self.spike_severity_list = []
        self.joint_spike_counts.fill(0)
        self.joint_spike_severity.fill(0.0)
        self.joint_rate_limit_counts.fill(0)
        self.joint_jitter_counts.fill(0)
    
    def get_noise_score(self):
        """
        Compute a normalized noise score (0-100).
        Higher = noisier file.
        
        Calibrated so that:
        - Clean files: 0-20
        - Acceptable with noise (like Cartwheel): 30-50
        - Problematic (extreme teleports like Maritsa): 70-100
        
        Key insight: Extreme teleports (>500 N·m spikes) are the most problematic
        and should dominate the score over continuous mild noise.
        """
        if self.total_frames == 0:
            return 0.0
        
        # Average severity component (capped at 20 points)
        # This captures continuous noise but shouldn't dominate
        avg_severity_per_frame = (
            self.spike_severity * 1.0 +           
            self.rate_limit_severity * 0.2 +      
            self.innovation_severity * 0.1        
        ) / self.total_frames
        avg_score = min(20.0, avg_severity_per_frame * 0.2)
        
        # Peak severity is the KEY differentiator for extreme teleports
        # Use threshold-based scoring:
        # - 0-100 N·m: mild (0-15 points)
        # - 100-300 N·m: moderate (15-35 points)  <- Cartwheel ~307
        # - 300-600 N·m: concerning (35-55 points)
        # - 600+ N·m: severe (55-80 points)  <- Maritsa ~1126
        peak = self.max_spike_severity
        if peak <= 100:
            peak_score = peak * 0.15  # 0-15
        elif peak <= 300:
            peak_score = 15 + (peak - 100) * 0.10  # 15-35
        elif peak <= 600:
            peak_score = 35 + (peak - 300) * 0.067  # 35-55
        else:
            peak_score = 55 + min(25, (peak - 600) * 0.05)  # 55-80 (cap at 80)
        
        # P99 spike severity (captures consistency of severe spikes)
        p99_score = 0.0
        if len(self.spike_severity_list) >= 20:
            p99 = np.percentile(self.spike_severity_list, 99)
            p99_score = min(15.0, p99 / 20.0)
        
        # Combined score
        combined = avg_score + peak_score + p99_score
        
        score = min(100.0, combined)
        return score
    
    def get_report(self):
        """Return a summary dict of noise statistics."""
        # Compute percentiles if enough data
        p95_spike = 0.0
        p99_spike = 0.0
        if len(self.spike_severity_list) >= 10:
            p95_spike = np.percentile(self.spike_severity_list, 95)
            p99_spike = np.percentile(self.spike_severity_list, 99)
        
        return {
            'total_frames': self.total_frames,
            'noise_score': self.get_noise_score(),
            # Counts
            'spike_detections': self.spike_detections,
            'rate_limit_clamps': self.rate_limit_clamps,
            'jitter_damping_events': self.jitter_damping_events,
            'innovation_clamps': self.innovation_clamps,
            # Severity (total N·m)
            'spike_severity': self.spike_severity,
            'rate_limit_severity': self.rate_limit_severity,
            'innovation_severity': self.innovation_severity,
            # PEAK metrics (identifies extreme teleports)
            'max_spike_severity': self.max_spike_severity,
            'max_frame_severity': self.max_frame_severity,
            'spike_severity_p95': p95_spike,
            'spike_severity_p99': p99_spike,
            # Per-frame metrics
            'spikes_per_frame': self.spike_detections / max(1, self.total_frames),
            'severity_per_frame': (self.spike_severity + self.rate_limit_severity) / max(1, self.total_frames),
            'noisiest_joints': self._get_noisiest_joints()
        }
    
    def _get_noisiest_joints(self, top_n=5):
        """Return indices of the noisiest joints (by severity)."""
        # Use severity, not counts, for ranking
        total_per_joint = self.joint_spike_severity + self.joint_rate_limit_counts * 0.5
        sorted_indices = np.argsort(total_per_joint)[::-1]
        return sorted_indices[:top_n].tolist()

@dataclass
class SMPLProcessingOptions:
    # --- Input / Coordinates ---
    input_type: str = 'axis_angle'
    input_up_axis: str = 'Y'
    quat_format: str = 'xyzw' # 'xyzw' (scipy) or 'wxyz'
    axis_permutation: str = None # "x,y,z"
    return_quats: bool = False
    
    # --- Physics / Dynamics ---
    dt: float = 1.0/60.0
    add_gravity: bool = True
    enable_passive_limits: bool = True # Enabled for Structural Support
    enable_apparent_gravity: bool = True
    torque_output_frame: str = 'local'  # 'local' (parent frame) or 'world'
    
    # --- Filtering / Signal Processing ---
    enable_one_euro_filter: bool = False
    acc_smooth_window: int = 0   # 0=off, 3/5/7 = Savitzky-Golay derivative window
    torque_smooth_window: int = 0  # 0=off, 3/5/7 = SG output smoothing window
    adaptive_effort_smooth: bool = True  # effort-adaptive EMA: heavy at low effort, light at high
    adaptive_effort_lo: float = 0.1      # effort fraction (||efforts_net||): below this → alpha_min
    adaptive_effort_hi: float = 0.5      # effort fraction (||efforts_net||): above this → alpha_max
    adaptive_effort_alpha_min: float = 0.05  # EMA alpha at low effort (~20-frame window)
    adaptive_effort_alpha_max: float = 0.5   # EMA alpha at high effort (~11 Hz cutoff at 100fps)
    # Signal-change branch: raises alpha when median(|Δeffort|) over a K-frame
    # window is high, letting fast low-effort oscillations through without
    # admitting single-frame glitch spikes (a spike contributes 2 large deltas
    # out of K=5 → median picks the small 3rd value, gate stays closed).
    adaptive_effort_change_window: int = 5   # K: sliding window for median |Δeff|
    adaptive_effort_change_lo: float = 0.02  # |Δeff|/frame: below this → no change boost
    adaptive_effort_change_hi: float = 0.10  # |Δeff|/frame: above this → full change boost
    # Coherence gate (opt-in, default OFF): lift the per-joint adaptive-effort
    # alpha_max cap toward 1.0 for coordinated, ballistic (popping-like) limb
    # motion, so genuine sharp accents survive the EMA without opening the gate
    # to spatially-local noise (soft-tissue ringing, per-IMU magnetometer jitter)
    # or temporally-isolated spikes (sensor glitches, cadence steps). Two soft
    # valves multiplied: coherence (min normalised angular speed across the
    # joint's kinematic neighbours) × envelope (trailing windowed-mean/peak
    # angular speed). Only limb joints are gated; others stay canonical.
    # See PIPELINE_INVESTIGATION_NOTES.md (2026-07-06).
    coherence_gate_enable: bool = False   # master on/off
    coherence_gate_strength: float = 1.0  # scale on the lift [0,1]
    coherence_gate_coh_lo: float = 0.7    # min-neighbour normalised speed → gate opens
    coherence_gate_coh_hi: float = 1.8    # → full coherence vote
    coherence_gate_env_lo: float = 0.30   # trailing mean/peak speed → gate opens
    coherence_gate_env_hi: float = 0.65   # → full envelope vote
    coherence_gate_env_window: int = 5    # (deprecated; superseded by _env_window_ms)
    coherence_gate_env_window_ms: float = 50.0  # envelope window duration (fps-scaled)
    coherence_gate_abs_lo: float = 400.0  # deg/s: own-speed floor below which gate→0
    coherence_gate_abs_hi: float = 1000.0 # deg/s: full absolute-speed vote
    # Hill sharpening on the 3-valve product (opt-in): gate = p^n/(p^n+p50^n).
    # The product of three partial valves compresses (a real accent yields
    # ~0.3-0.5, quiet-material flicker ~0.01-0.12), so at strength 1 the lift
    # is subtle; strength>1 "fixes" this by linear-clamp saturation, which
    # also passes the largest flickers (isolated OEF transients on quiet
    # 250 fps material). The Hill valve instead steepens the mapping around
    # p50: decisive on real accents, still soft, suppresses the flicker band.
    # Use with strength=1. n=0 disables (legacy linear strength path).
    coherence_gate_hill_n: float = 0.0    # Hill coefficient; 0 = off, try 3
    coherence_gate_hill_p50: float = 0.15 # product value mapping to gate 0.5
    coherence_gate_smooth_ms: float = 25.0  # gate EMA time constant (fps-aware); 0=off
    # Asymmetric gate EMA (opt-in): separate attack (gate rising) and release
    # (gate falling) time constants. A 1-2 frame pop at 100 fps is over before
    # a symmetric 25 ms EMA opens the gate, so the front-end cutoff is still
    # low during the frames that carry the accent's energy. Fast attack lets
    # the cutoff open DURING the impulse; slow release keeps the ramp-down
    # smooth (the OEF-transient concern that motivated the EMA). Negative →
    # fall back to the symmetric coherence_gate_smooth_ms (validated default).
    coherence_gate_attack_ms: float = -1.0   # gate-rising time constant; <0 = symmetric
    coherence_gate_release_ms: float = -1.0  # gate-falling time constant; <0 = symmetric
    # Adaptive front-end filter (opt-in, default OFF → legacy fixed-window
    # pipeline is byte-identical). Replaces the fixed `smooth_input_window`
    # moving-average with a per-joint One Euro Filter on the pose whose cutoff
    # is driven by the coherence gate (NOT by raw speed — speed alone can't tell
    # a 1-frame optical glitch from a real accent; both are fast). gate 0 → low
    # cutoff (heavy smoothing, glitch-safe); gate 1 → high cutoff (accent
    # preserved, no double-pipeline / state-continuity issue since it is one
    # recursive filter with a moving cutoff). Requires the gate; auto-enables
    # its computation. See PIPELINE_INVESTIGATION_NOTES.md (2026-07-06).
    adaptive_frontend: bool = False       # master on/off for the gated OEF front-end
    # Calibrated (2026-07-06) so the gate-CLOSED baseline matches the legacy
    # (5-frame MA, ~10 Hz passband) quiet/clean-section torque statistics
    # (RMS + jitter) while still suppressing 1-frame glitches; cutoff_hi set so
    # full-gate popping recovery stays >= legacy peak. Raising cutoff_lo past
    # ~6 Hz starts leaking 1-frame glitches (single-pole OEF rejects impulses
    # less than legacy's boxcar MA at matched passband).
    adaptive_frontend_cutoff_lo: float = 4.0    # Hz cutoff at gate=0 (~legacy clean baseline)
    adaptive_frontend_cutoff_hi: float = 18.0   # Hz cutoff at gate=1 (accent passes)
    adaptive_frontend_beta: float = 0.0         # OEF speed term; keep 0 so cutoff is gate-driven only
    adaptive_frontend_trans_window: int = 5     # trans MA window when the OEF replaces pose smoothing
                                                # (trans is never OEF-gated; without this, setting
                                                # smooth_input_window=0 would leave trans unsmoothed)
    smooth_contact_forces: bool = False  # proximity-adaptive contact force smoothing
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.0
    
    # --- Floor / Environment ---
    floor_enable: bool = True
    floor_height: float = 0.0
    floor_tolerance: float = 0.15

    contact_method: str = 'logodds_valved'  # 'logodds' or 'logodds_valved'

    enable_body_contacts: bool = False   # Extend contact detection to knees, elbows, head, pelvis
    
    # --- Log-odds contact options ---
    logodds_enable_height: bool = True
    logodds_enable_kinematic: bool = True     # Unified kinematic stream (approach angle + td + settled)
    logodds_enable_structural: bool = True    # Frame evaluator structural necessity
    logodds_enable_divergence: bool = True    # Foot-CoM relative velocity divergence
    # Legacy enables (for backward compatibility / A/B testing)
    logodds_enable_vertical_kinematic: bool = False
    logodds_enable_hspeed: bool = False
    logodds_enable_equilibrium: bool = False
    logodds_enable_velocity: bool = False
    logodds_enable_trajectory: bool = False
    logodds_enable_touchdown: bool = False
    # Per-stream weights
    logodds_weight_height: float = 1.0
    logodds_weight_kinematic: float = 1.0
    logodds_weight_structural: float = 1.0
    logodds_weight_divergence: float = 1.0
    # Legacy weights
    logodds_weight_vertical_kinematic: float = 1.0
    logodds_weight_hspeed: float = 1.0
    logodds_weight_equilibrium: float = 1.0
    logodds_weight_velocity: float = 1.0
    logodds_weight_trajectory: float = 1.0
    logodds_weight_touchdown: float = 1.0
    # Accumulator
    logodds_decay_rate: float = 0.90
    # Structural-stream EMA on per-foot forces (1.0 = no smoothing)
    logodds_struct_force_ema_alpha: float = 1.0
    # Effort-relief prior: hand candidates near a high-strain spine joint
    # (lever arm on lean side) get a small structural positive even when
    # the FE's ZMP solution would assign them no load. Default OFF.
    logodds_fe_relief_enable: bool = False
    logodds_fe_relief_strain_threshold: float = 25.0
    logodds_struct_relief_logodds: float = 0.3
    
    # --- Torque Rate Limiting ---
    enable_rate_limiting: bool = False
    rate_limit_strength: float = 1.0  # Multiplier for per-joint rate limits
    enable_jitter_damping: bool = False  # Pass 4: sign-flip oscillation damping
    enable_velocity_gate: bool = False   # Suppress dynamic torque at low angular velocity
    enable_kf_smoothing: bool = False  # SmartClampKF filter for dynamic torque (disabled: corrupts torques_vec after re-init)
    kf_responsiveness: float = 10.0   # How quickly KF tracks changes (higher = faster)
    kf_smoothness: float = 1.0        # Process noise (higher = trusts new data more)
    kf_clamp_radius: float = 15.0     # Max innovation per frame (Nm)
    
    # --- World-Frame Dynamics ---
    world_frame_dynamics: bool = True  # CoM-based dynamic torque
    com_pos_min_cutoff: float = 8.0    # Base One Euro min_cutoff for CoM position filter (scaled by 1/√mass)
    com_pos_beta: float = 0.05         # Base One Euro beta for CoM position filter
    com_vel_min_cutoff: float = 3.0    # Base One Euro min_cutoff for CoM velocity filter (scaled by 1/√mass)
    com_vel_beta: float = 0.05         # Base One Euro beta for CoM velocity filter
    com_acc_min_cutoff: float = 2.0    # Base One Euro min_cutoff for CoM acceleration filter (999 = disabled)
    com_acc_beta: float = 0.8          # Base One Euro beta — high for adaptive responsiveness during impacts
    smooth_input_window: int = 0       # Causal moving average window for pose+trans input (0 = off, 3 = recommended for 33Hz cadence removal)
    # Independent trans smoothing window, in ms (fps-scaled). The trans sensor
    # is a body-mounted projecting mass that oscillates ("flops") during
    # extreme — especially vertical — movement, and trans tolerates filtering
    # far better than joint rotations do. So trans gets its own fixed window,
    # decoupled from the pose MA and NEVER driven by the coherence gate (the
    # flop occurs exactly when the gate is open). 0 = legacy: trans follows
    # smooth_input_window / adaptive_frontend_trans_window. When set (>0) it
    # supersedes both for trans in every mode (streaming and batch).
    smooth_trans_window_ms: float = 0.0
    zmp_sg_window: int = 0             # SG derivative window for ZMP acceleration (0 = off/use One Euro chain, 11+ = SG window).
                                       # When enabled, gets acceleration directly from COM position via Savitzky-Golay
                                       # 2nd derivative, bypassing the noisy pos→vel→acc finite difference chain.


    
    # --- Spine Geometry ---
    use_s_curve_spine: bool = True      # Use biomechanical S-curve spine instead of SMPL cantilevered spine

class SMPLProcessor:
    @staticmethod
    def _aa_to_quat_batch(aa):
        """Convert axis-angle (N, 3) to quaternions (N, 4) [w, x, y, z]."""
        angles = np.linalg.norm(aa, axis=1, keepdims=True)  # (N, 1)
        safe = (angles > 1e-8).flatten()
        quats = np.zeros((aa.shape[0], 4))
        quats[:, 0] = 1.0  # Identity for zero rotations
        if np.any(safe):
            half = angles[safe] * 0.5
            axes = aa[safe] / angles[safe]
            quats[safe, 0] = np.cos(half).flatten()
            quats[safe, 1:] = axes * np.sin(half)
        return quats
    
    @staticmethod
    def _quat_to_aa_batch(quats):
        """Convert quaternions (N, 4) [w, x, y, z] to axis-angle (N, 3)."""
        # Ensure w >= 0 for consistent conversion
        flip = quats[:, 0] < 0
        q = quats.copy()
        q[flip] *= -1
        
        w = np.clip(q[:, 0], -1.0, 1.0)
        half_angle = np.arccos(w)  # (N,)
        angle = 2.0 * half_angle
        sin_ha = np.sin(half_angle)
        
        aa = np.zeros((q.shape[0], 3))
        safe = sin_ha > 1e-8
        if np.any(safe):
            aa[safe] = q[safe, 1:] / sin_ha[safe, np.newaxis] * angle[safe, np.newaxis]
        return aa
    
    @staticmethod
    def _slerp_batch(q0, q1, t):
        """Vectorized quaternion slerp: interpolate (N, 4) arrays at parameter t in [0, 1]."""
        dots = np.sum(q0 * q1, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        
        omega = np.arccos(dots)  # (N,)
        sin_omega = np.sin(omega)
        
        result = np.zeros_like(q0)
        
        # Near-identity: linear interpolation (avoid division by zero)
        linear = sin_omega < 1e-6
        if np.any(linear):
            result[linear] = q0[linear] * (1 - t) + q1[linear] * t
            # Normalize
            norms = np.linalg.norm(result[linear], axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            result[linear] /= norms
        
        # Standard slerp
        slerp = ~linear
        if np.any(slerp):
            s0 = np.sin((1 - t) * omega[slerp]) / sin_omega[slerp]
            s1 = np.sin(t * omega[slerp]) / sin_omega[slerp]
            result[slerp] = q0[slerp] * s0[:, np.newaxis] + q1[slerp] * s1[:, np.newaxis]
        
        return result

    @staticmethod
    def _spine_s_curve_positions(pelvis_pos, neck_pos, bone_offsets=None):
        """
        Compute biomechanical S-curve spine positions between pelvis and neck.
        
        Models lumbar lordosis (forward curve at spine1) and thoracic kyphosis
        (backward curve at spine3). Returns dict of {joint_idx: position}.
        
        Args:
            pelvis_pos: (3,) pelvis position
            neck_pos: (3,) neck position
            bone_offsets: optional dict {3: offset_3, 6: offset_6, 9: offset_9, 12: offset_12}
                          Original SMPL bone offsets for proportional t-value computation.
                          If None, uses even spacing (0.25 intervals).
        """
        spine_dir = neck_pos - pelvis_pos
        spine_len = np.linalg.norm(spine_dir)
        if spine_len < 1e-4:
            # Degenerate case — return straight positions
            return {0: pelvis_pos, 3: pelvis_pos, 6: pelvis_pos,
                    9: pelvis_pos, 12: neck_pos}
        
        spine_unit = spine_dir / spine_len
        
        # Forward direction: perpendicular to spine in the sagittal plane.
        # In SMPL T-pose (y-up), forward/anterior is +z.
        # Compute as: world_forward projected perpendicular to spine_unit.
        world_fwd = np.array([0.0, 0.0, 1.0])  # SMPL T-pose: face/anterior is +z
        fwd = world_fwd - np.dot(world_fwd, spine_unit) * spine_unit
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-4:
            # Spine is horizontal — use y as forward
            world_fwd = np.array([0.0, 1.0, 0.0])
            fwd = world_fwd - np.dot(world_fwd, spine_unit) * spine_unit
            fwd_len = np.linalg.norm(fwd)
        fwd = fwd / fwd_len if fwd_len > 1e-4 else np.zeros(3)
        
        # Compute t-values: proportional to original bone lengths if available
        spine_joints = [0, 3, 6, 9, 12]
        if bone_offsets is not None:
            bone_lengths = [np.linalg.norm(bone_offsets[sj]) for sj in [3, 6, 9, 12]]
            total_path = sum(bone_lengths)
            if total_path > 1e-4:
                cumul = 0.0
                t_values = [0.0]
                for bl in bone_lengths:
                    cumul += bl
                    t_values.append(cumul / total_path)
            else:
                t_values = [0.0, 0.25, 0.50, 0.75, 1.0]
        else:
            t_values = [0.0, 0.25, 0.50, 0.75, 1.0]
        
        # Anterior displacement (positive = forward, lordosis; negative = kyphosis)
        # sin(2π*t) gives S-curve: 0 at endpoints, lordosis/kyphosis in between
        lordosis_amp = 0.020   # 2.0 cm peak lumbar lordosis
        kyphosis_amp = 0.015   # 1.5 cm peak thoracic kyphosis
        
        positions = {}
        for sj, t in zip(spine_joints, t_values):
            straight = pelvis_pos + t * spine_dir
            s_disp = np.sin(2 * np.pi * t)
            if s_disp > 0:
                disp = s_disp * lordosis_amp   # forward (lordosis)
            else:
                disp = s_disp * kyphosis_amp   # backward (kyphosis)
            positions[sj] = straight + fwd * disp
        
        return positions
    
    def __init__(self, framerate, betas=None, gender='neutral', model_path=None, total_mass_kg=75.0):
        """
        Initialize the SMPLProcessor.
        
        Args:
            framerate (float): Motion capture framerate.
            betas (np.ndarray): Shape coefficients (usually size 10 or more).
            gender (str): 'male', 'female', or 'neutral'.
            model_path (str): Optional path to SMPL model file.
            total_mass_kg (float): Approximate total body mass for physics calculations.
        """
        self.framerate = framerate
        self.betas = betas if betas is not None else np.zeros(10)
        self.gender = gender
        self.model_path = model_path
        self.total_mass_kg = total_mass_kg
        self.current_zmp = np.zeros(3) # Initialize to safe default
        
        # Standard SMPL joint names (first 24)
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # Passive Limits (Soft Limit in Radians, Stiffness)
        # Angles are "distance from neutral". Neutral = 0.
        # This is a simplification (isotropic limits).
        self.joint_limits = {
            'pelvis': {'limit': np.radians(180), 'k': 0.0}, # Root is free
            'neck': {'limit': np.radians(30), 'k': 20.0}, 
            'spine': {'limit': np.radians(30), 'k': 50.0}, 
            
            # Collar: Asymmetric Hinge limits for structural support
            # Axis 2 (Z-axis) carries the Gravity Load for Elevation (Up/Down).
            # Left (+Z Up): Min Limit at 0.05. Natural (-0.13) < 0.05 -> Supported.
            # Right (-Z Up): Max Limit at -0.05. Natural (-0.003) > -0.05 -> Supported.
            # Tightened limits (+/- 5 deg) to ensure support even near neutral.
            'left_collar': {'type': 'hinge', 'axis': 2, 'min': 0.05, 'max': 0.8, 'k': 2000.0}, 
            'right_collar': {'type': 'hinge', 'axis': 2, 'min': -0.8, 'max': -0.05, 'k': 2000.0},
            
            # Shoulders: Flexible
            'shoulder': {'limit': np.radians(90), 'k': 10.0}, 
            
            # Elbows: Hinge with Extension Stop
            # Left Flexion is Negative (approx), Right is Positive
            # Locked Axes: 0 (Twist/Pronation - assume passive?), 2 (Valgus/Varus - definitely passive)
            # Max=0.0 (Hard Stop), K=500.0 (Linear Stiffness).
            'left_elbow': {'type': 'hinge', 'axis': 1, 'min': -2.8, 'max': 0.0, 'k': 500.0, 'locked_axes': [0, 2]},
            'right_elbow': {'type': 'hinge', 'axis': 1, 'min': -0.0, 'max': 2.8, 'k': 500.0, 'locked_axes': [0, 2]},
            
            # Knees: Similar Hinge (assuming Axis 0 usually?)
            'knee': {'limit': np.radians(160), 'k': 50.0},
            
            'hip': {'limit': np.radians(90), 'k': 30.0},
            'default': {'limit': np.radians(180), 'k': 1.0}
        }
        
        # We are interested in the first 22 joints (indices 0-21)
        self.target_joint_count = 22
        
        self.use_s_curve_spine = True  # Default; toggled by options at runtime
        self.limb_data = self._compute_limb_properties()
        self.skeleton_offsets = self._compute_skeleton_offsets()
        self.max_torques = self._compute_max_torque_profile()
        self.max_torque_array = self._compute_max_torque_array()
        self._precompute_inertia_tables()

        # Initialize consensus contact detection
        self._init_contact_consensus()

        self.perm_basis = None
        self.perm_basis_rot = None
        self.reset_physics_state()

        # Snapshot the attribute set of a freshly-initialised processor. Every
        # per-frame streaming state holder (kinematics rings, CoM/gravity/contact
        # filters, coherence-gate normalisation, OEF, ...) is created lazily on
        # the first frame, so anything NOT in this set was created by streaming
        # and must be removed on reset — otherwise a new sequence differences
        # against the previous sequence's state (large frame-0 torque spike) or,
        # for the gate's cumulative-mean normalisation, is scored against the
        # previous file's statistics. Robust to state names changing.
        self._init_attr_snapshot = frozenset(self.__dict__)

    def _compute_limb_properties_from_model(self):
        """
        Attempts to load SMPL model via smplx and calculate exact limb lengths from betas.
        Returns:
            lengths (dict): Dictionary of limb lengths if successful, else None.
        """
        if not SMPLX_AVAILABLE:
            return None
            
        # User has smplh models in CWD/smplh/SMPLH_{GENDER}.pkl
        # smplx.create(model_path='.', model_type='smplh') expects exactly this structure.
        model_path = '.' 
        if hasattr(self, 'model_path') and self.model_path:
             model_path = self.model_path
             
        # Map gender. Default to MALE if neutral requested but likely only M/F exist for SMPLH
        gender_map = {'male': 'MALE', 'female': 'FEMALE'}
        g_tag = gender_map.get(self.gender, 'MALE') # Fallback to MALE for neutral
        
        try:
            # Use the module-level cache so the smplx model is loaded
            # exactly once per (model_path, gender) per process — saves
            # ~5-10 s per file in batch runs.
            model = _get_cached_smplx_model(model_path, g_tag)

            betas_tensor = torch.zeros(1, 10)
            if self.betas is not None:
                b = torch.tensor(self.betas, dtype=torch.float32)
                if b.numel() > 10: b = b[:10]
                betas_tensor[0, :b.numel()] = b
                
            output = model(betas=betas_tensor)
            
            # SMPLH has 52 joints (includes hands). First 24 are body joints.
            # We only care about the first 24 for our skeleton.
            joints = output.joints[0].detach().cpu().numpy() # (52, 3)
            # joints = joints[:24, :] # DO NOT SLICE YET! We need >24 check later.
            
            # Calculate Lengths
            # Indices:
            # 0: Pelvis
            # 1: L_Hip, 2: R_Hip
            # 3: Spine1
            # 4: L_Knee, 5: R_Knee
            # 6: Spine2
            # 7: L_Ankle, 8: R_Ankle
            # 9: Spine3
            # 10: L_Foot, 11: R_Foot
            # 12: Neck
            # 13: L_Collar, 14: R_Collar
            # 15: Head
            # 16: L_Shoulder, 17: R_Shoulder
            # 18: L_Elbow, 19: R_Elbow
            # 20: L_Wrist, 21: R_Wrist
            # 22: L_Hand, 23: R_Hand
            
            def dist(i, j):
                return float(np.linalg.norm(joints[i] - joints[j]))
                
            lengths = {}
            lengths['pelvis_width'] = dist(1, 2)
            lengths['upper_leg'] = (dist(1, 4) + dist(2, 5)) / 2.0
            lengths['lower_leg'] = (dist(4, 7) + dist(5, 8)) / 2.0
            
            # Compute Exact Offsets (24, 3) relative to parents
            # Parent indices (standard SMPL):
            # 0: -1 (Root)
            # 1: 0, 2: 0
            # 3: 0
            # 4: 1, 5: 2
            # 6: 3
            # 7: 4, 8: 5
            # 9: 6
            # 10: 7, 11: 8
            # 12: 9
            # 13: 9, 14: 9
            # 15: 12
            # 16: 13, 17: 14
            # 18: 16, 19: 17
            # 20: 18, 21: 19
            # 22: 20, 23: 21
            # Virtual End Effectors:
            # 24: L_Toe (Child of 10)
            # 25: R_Toe (Child of 11)
            # 26: L_Fingertip (Child of 22/L_Hand)
            # 27: R_Fingertip (Child of 37/R_Hand)
            # 27: R_Fingertip (Child of 23/R_Hand -- Replaced 37)
            # 28: L_Heel (Child of 7/L_Ankle)
            # 29: R_Heel (Child of 8/R_Ankle)
            parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 10, 11, 22, 23, 7, 8]
            self.parents = parents # Persist for RNE Physics
            
            # Note: We need 30 offsets now
            model_offsets = np.zeros((30, 3))
            
            # Use joints directly (World/Rest Pose).
            # Assume Root is at origin or doesn't matter for relative offsets.
            # But Root offset (0) is usually 0,0,0 or position relative to world origin? 
            # In our system, root translation is separate. 
            # Offsets[0] is usually 0. Or hip center rel to pelvis? 
            # Usually Offsets[0] is 0,0,0.
            
            for i in range(1, 24):
                p = parents[i]
                
                # Handle Hand Override for SMPL-H
                if joints.shape[0] > 24:
                    child_idx = i
                    if i == 22: child_idx = 22 # Left Hand Base
                    if i == 23: child_idx = 37 # Right Hand Base (remapped)
                    
                    # Store vector: Child - Parent
                    # Parent is standard index.
                    offset_vec = joints[child_idx] - joints[parents[i]]
                    model_offsets[i] = offset_vec
                else:
                    offset_vec = joints[i] - joints[parents[i]]
                    model_offsets[i] = offset_vec

            # --- Virtual Extensions (Indices 24-29) ---
            # Toes (24, 25): Child of 10, 11
            # User feedback: "Toe tip positions seem to be directly below foot position in rest pose."
            # This implies the previous (0, -1, 0) offset was visually wrong (vertical).
            # The Mesh Foot likely points FORWARD (Local Z) relative to the Ankle joint frame.
            # Updated to Z-forward offset.
            foot_vec = np.array([0.0, 0.0, 1.0]) 
            foot_len = 0.15 # Approx toe length extension from ankle
            model_offsets[24] = foot_vec * foot_len # L_Toe
            model_offsets[25] = foot_vec * foot_len # R_Toe
            
            # Fingertips (26, 27): Child of Hand (22, 37)
            # Extrapolate direction from Wrist->Hand.
            # L_Hand (22) Parent is 20 (Wrist).
            # R_Hand (37) Parent is 21 (Wrist).
            # Vector 20->22 is Wrist->Knuckle.
            # Fingertip should continue this vector.
            # If 20->22 is present in model_offsets[22].
            l_knuckle_vec = model_offsets[22]
            if np.linalg.norm(l_knuckle_vec) > 1e-4:
                l_dir = l_knuckle_vec / np.linalg.norm(l_knuckle_vec)
            else:
                l_dir = np.array([1.0, 0.0, 0.0])
                
            model_offsets[26] = l_dir * 0.08 # 8cm finger
            
            r_knuckle_vec = model_offsets[23]
            if np.linalg.norm(r_knuckle_vec) > 1e-4:
                r_dir = r_knuckle_vec / np.linalg.norm(r_knuckle_vec)
            else:
                r_dir = np.array([-1.0, 0.0, 0.0])
                
            model_offsets[27] = r_dir * 0.08 # 8cm finger
            
            # Heels (28, 29): Child of 7, 8 (Ankle)
            # Offset Down (~3cm) and Back (~7cm).
            # Ankle frame usually has Z forward? 
            # We assume Foot Vector [0,0,1] is Forward.
            # So Back is [0,0,-1].
            heel_vec = np.array([0.0, -0.03, -0.07])
            model_offsets[28] = heel_vec # L_Heel
            model_offsets[29] = heel_vec # R_Heel
                
            # Fix for SMPL-H (52 joints) vs SMPL (24 joints)
            # SMPL: 20->22 (L_Hand), 21->23 (R_Hand)
            # SMPL-H: 22 and 23 are Left Fingers. Right Fingers start around 37.
            if joints.shape[0] > 24:
                # SMPL-H Logic
                l_hand_len = dist(20, 22)
                r_hand_len = dist(21, 37)
                lengths['foot'] = (dist(7, 10) + dist(8, 11)) / 2.0 
            else:
                # Standard SMPL Logic
                l_hand_len = dist(20, 22)
                r_hand_len = dist(21, 23)
                lengths['foot'] = (dist(7, 10) + dist(8, 11)) / 2.0
                
            lengths['hand'] = (l_hand_len + r_hand_len) / 2.0
            
            lengths['spine_segment'] = (dist(0, 3) + dist(3, 6) + dist(6, 9)) / 3.0
            # Individual spine segments (for accurate visualization)
            lengths['spine_lower'] = dist(0, 3)     # pelvis -> spine1
            lengths['spine_mid'] = dist(3, 6)        # spine1 -> spine2
            lengths['spine_upper'] = dist(6, 9)      # spine2 -> spine3
            lengths['spine_to_neck'] = dist(9, 12)   # spine3 -> neck
            lengths['neck'] = dist(12, 15) # Neck -> Head
            lengths['head'] = 0.20 
            
            lengths['shoulder_width'] = dist(16, 17) 
            lengths['collar'] = (dist(13, 16) + dist(14, 17)) / 2.0
            
            lengths['upper_arm'] = (dist(16, 18) + dist(17, 19)) / 2.0
            lengths['lower_arm'] = (dist(18, 20) + dist(19, 21)) / 2.0
            
            # Extend Lengths keys?
            # Or reliance on offsets is enough?
            # User wants "output as index 24 and 25 of both limb length and joint positions"
            # So 'limb_lengths' array must be 28.
            # The node calculates it from offsets.
            
            # Message only once? Or standard print
            # print(f"Loaded SMPL-H model ({g_tag}) for accurate anthropometry.")
            
            # --- Compute true segment CoM offsets from mesh ---
            # Each vertex is assigned to the joint with the highest LBS weight.
            # The centroid of those vertices gives the segment's center of mass.
            # Store as offset from joint position (in T-pose coordinates).
            try:
                vertices = output.vertices[0].detach().cpu().numpy()  # (6890, 3)
                lbs_weights = model.lbs_weights.detach().cpu().numpy()  # (6890, N_joints)
                n_body_joints = min(24, lbs_weights.shape[1])
                
                # Assign each vertex to its dominant joint
                seg_assignment = np.argmax(lbs_weights[:, :n_body_joints], axis=1)
                
                segment_com_offsets = np.zeros((n_body_joints, 3))
                segment_com_world = np.zeros((n_body_joints, 3))
                for j in range(n_body_joints):
                    verts_j = vertices[seg_assignment == j]
                    if len(verts_j) > 0:
                        com_world = np.mean(verts_j, axis=0)
                        segment_com_world[j] = com_world
                        segment_com_offsets[j] = com_world - joints[j]
                    else:
                        segment_com_offsets[j] = np.zeros(3)
                        segment_com_world[j] = joints[j]
                
                # print(f"  Computed {n_body_joints} segment CoM offsets from mesh ({len(vertices)} vertices)")
                
                # --- Per-joint directional surface extents ---
                # For each joint, compute how far NEARBY mesh vertices extend
                # in each direction (±X, ±Y, ±Z) from the joint center.
                # Uses LBS weight threshold (not argmax) to avoid including
                # distant vertices that inflate extents.
                # Uses 95th percentile instead of absolute min/max for
                # robustness against vertex outliers.
                # Shape: (24, 3, 2) = [joint, axis, (min_extent, max_extent)]
                LBS_WEIGHT_THRESH = 0.3
                PERCENTILE = 95
                
                joint_surface_extents = np.zeros((n_body_joints, 3, 2))
                # Also compute near-surface distances per joint per axis.
                # Used for body contacts where we want the skin/tissue 
                # thickness, not the full limb extent.
                # SMPL joints are regressed from vertices, so some vertices
                # sit AT the joint center. We use the 10th percentile of
                # absolute offsets per axis direction to capture the
                # near-surface shell while ignoring co-located vertices.
                # Shape: (24, 3, 2) = [joint, axis, (neg_10pct, pos_10pct)]
                NEAR_SURFACE_PERCENTILE = 10
                joint_surface_min_dists = np.full((n_body_joints, 3, 2), 0.03)
                
                for j in range(n_body_joints):
                    # Select vertices strongly associated with this joint
                    mask = lbs_weights[:, j] > LBS_WEIGHT_THRESH
                    verts_j = vertices[mask]
                    
                    if len(verts_j) < 3:
                        verts_j = vertices[seg_assignment == j]
                    
                    if len(verts_j) > 0:
                        offsets = verts_j - joints[j]
                        # Max extents (95th percentile) — for foot/hand contacts
                        joint_surface_extents[j, :, 0] = np.percentile(offsets, 100 - PERCENTILE, axis=0)
                        joint_surface_extents[j, :, 1] = np.percentile(offsets, PERCENTILE, axis=0)
                        # Near-surface distances (10th percentile of |offset|)
                        # per axis direction — for body contacts
                        for axis in range(3):
                            neg_mask = offsets[:, axis] < -0.001  # exclude co-located vertices
                            pos_mask = offsets[:, axis] > 0.001
                            if np.any(neg_mask):
                                abs_neg = np.abs(offsets[neg_mask, axis])
                                joint_surface_min_dists[j, axis, 0] = -np.percentile(abs_neg, NEAR_SURFACE_PERCENTILE)
                            if np.any(pos_mask):
                                joint_surface_min_dists[j, axis, 1] = np.percentile(offsets[pos_mask, axis], NEAR_SURFACE_PERCENTILE)
                    else:
                        # Default: ±3cm padding
                        joint_surface_extents[j, :, 0] = -0.03
                        joint_surface_extents[j, :, 1] = 0.03
                
                self._joint_surface_extents = joint_surface_extents
                self._joint_surface_min_dists = joint_surface_min_dists
                
                # ── Correct spine CoM offsets for S-curve spine ────────────
                # SMPL spine positions serve mesh deformation, not biomechanics.
                # Recompute spine CoM offsets relative to biomechanical S-curve
                # positions (lumbar lordosis + thoracic kyphosis).
                if self.use_s_curve_spine:
                    spine_chain = [0, 3, 6, 9, 12]
                    if len(joints) > max(spine_chain):
                        # Build bone_offsets dict from original SMPL joints
                        bone_offsets = {}
                        for sj in [3, 6, 9, 12]:
                            parent = {3: 0, 6: 3, 9: 6, 12: 9}[sj]
                            bone_offsets[sj] = joints[sj] - joints[parent]
                        s_curve = SMPLProcessor._spine_s_curve_positions(
                            joints[0], joints[12], bone_offsets)
                        for sj in spine_chain:
                            # Mesh CoM world position = actual_pos + actual_offset
                            com_world = joints[sj] + segment_com_offsets[sj]
                            # Recompute offset relative to S-curve position
                            segment_com_offsets[sj] = com_world - s_curve[sj]
                
                # ── Clamp limb CoM offsets to biomechanical norms ──────────
                # LBS vertex assignment inflates limb segment CoMs because mesh
                # blending assigns distant vertices to joints. Clamp limb offsets
                # to de Leva (1996) proximal CoM fractions along the bone.
                # Spine/trunk joints are LEFT UNTOUCHED — their perpendicular
                # offsets capture the cantilever geometry we need.
                
                # SMPL parent indices
                smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
                
                # Spine/trunk joints: preserve offsets as-is
                trunk_joints = {0, 3, 6, 9, 12, 15}
                
                # De Leva proximal CoM fractions (from parent joint toward child)
                # Conservative max — real is ~0.43-0.58, use 0.50 as cap
                max_com_fraction = 0.50
                
                # Build children map
                children_map = {}
                for ci in range(n_body_joints):
                    pi = smpl_parents[ci] if ci < len(smpl_parents) else -1
                    if pi >= 0:
                        children_map.setdefault(pi, []).append(ci)
                
                for j in range(n_body_joints):
                    if j in trunk_joints:
                        continue  # preserve spine/trunk offsets
                    
                    kids = children_map.get(j, [])
                    if not kids:
                        # Leaf joint (hands, feet, head) — clamp magnitude
                        mag = np.linalg.norm(segment_com_offsets[j])
                        if mag > 0.05:
                            segment_com_offsets[j] *= 0.05 / mag
                        continue
                    
                    # Use first child to define bone direction
                    child_j = kids[0]
                    if child_j >= n_body_joints:
                        continue
                    bone_vec = joints[child_j] - joints[j]
                    bone_len = np.linalg.norm(bone_vec)
                    if bone_len < 1e-4:
                        continue
                    
                    bone_dir = bone_vec / bone_len
                    max_offset = max_com_fraction * bone_len
                    
                    # Project CoM offset onto bone direction
                    offset = segment_com_offsets[j]
                    proj = np.dot(offset, bone_dir)
                    
                    # Clamp projection along bone
                    proj_clamped = np.clip(proj, 0, max_offset)
                    
                    # Keep small perpendicular component for realism,
                    # but clamp it to avoid LBS artifacts
                    perp = offset - proj * bone_dir
                    perp_mag = np.linalg.norm(perp)
                    max_perp = 0.02  # 2cm max perpendicular offset for limbs
                    if perp_mag > max_perp:
                        perp = perp * (max_perp / perp_mag)
                    
                    segment_com_offsets[j] = proj_clamped * bone_dir + perp
                
                # Debug: print spine segment CoMs
                spine_names = {0: 'Pelvis', 3: 'Spine1', 6: 'Spine2', 9: 'Spine3', 12: 'Neck'}
                for idx, name in spine_names.items():
                    if idx < n_body_joints:
                        off = segment_com_offsets[idx]
                        # print(f"    {name}: CoM offset = [{off[0]:.4f}, {off[1]:.4f}, {off[2]:.4f}]")
                
            except Exception as e:
                print(f"  Warning: Could not compute segment CoMs from mesh: {e}")
                segment_com_offsets = None
            
            return {'lengths': lengths, 'offsets': model_offsets, 'segment_com_offsets': segment_com_offsets}
            
        except Exception as e:
            # Only print warning once per session ideally, but for now simple print
            print(f"Failed to load SMPL-H model: {e}. Falling back to heuristics.")
            print(f"DEBUG: Ensure 'smplh' folder is in {os.path.abspath(model_path)} and contains SMPLH_{g_tag}.pkl")
            return None

    def _compute_limb_properties(self):
        """
        Approximates limb lengths and masses.
        """
        model_lengths = self._compute_limb_properties_from_model()

        
        # Anthropometric data (de Leva 1996) - approximate segment mass fractions
        # These sums usually add up to ~1.0. 
        # Note: SMPL joints don't map 1:1 to de Leva segments perfectly but we approximate.
        mass_fractions = {
            'pelvis': 0.142, # Pelvis + Lower trunk
            'spine': 0.15,   # Mid + Upper trunk
            'head': 0.069,   # Head + Neck
            'thigh': 0.14,   # Both thighs (0.1 each usually? No, singular is ~0.14? Check de Leva.)
                             # Winter: Thigh is 0.1 per leg. 
                             # Let's use simplified percent per SINGLE limb.
            'upper_leg': 0.10, 
            'lower_leg': 0.0465,
            'foot': 0.0145,
            'upper_arm': 0.028,
            'lower_arm': 0.016,
            'hand': 0.006
        }
        
        model_data = self._compute_limb_properties_from_model()
        
        offsets = None
        segment_com_offsets = None
        if model_data is not None:
             if 'lengths' in model_data:
                 lengths = model_data['lengths']
                 offsets = model_data.get('offsets')
                 segment_com_offsets = model_data.get('segment_com_offsets')
             else:
                 # Legacy fallback if I missed something (unlikely)
                 lengths = model_data
        else:
            # Approximate lengths (in meters) for a standard ~1.7m human.
            # We can try to adjust this based on betas[0] (height correlation) lightly if we want.
            # Beta[0] is roughly height/scaling. +1 sigma is taller.
            # A very rough approximation: scale = 1.0 + (beta[0] * 0.06)
            scale_factor = 1.0
            if self.betas is not None and len(self.betas) > 0:
                scale_factor += self.betas[0] * 0.06
                
            if self.gender == 'female':
                scale_factor *= 0.92
            
            defaults = {
                'pelvis_width': 0.25,
                'upper_leg': 0.45,
                'lower_leg': 0.42,
                'foot': 0.20, # length
                'spine_segment': 0.12, # average per spine joint (backward compat)
                'spine_lower': 0.13,   # pelvis -> spine1
                'spine_mid': 0.14,     # spine1 -> spine2
                'spine_upper': 0.06,   # spine2 -> spine3
                'spine_to_neck': 0.21, # spine3 -> neck
                'neck': 0.10,
                'head': 0.20,
                'shoulder_width': 0.40,
                'collar': 0.15,
                'upper_arm': 0.30,
                'lower_arm': 0.28,
                'hand': 0.18
            }
            lengths = {k: v * scale_factor for k, v in defaults.items()}
            
        # Masses
        # Base Mass (Neutral)
        m_base = self.total_mass_kg
        
        # Adjust for Shape (Betas)
        # Beta[0]: Stature (Height/Size). +1 sigma ~ +8% mass.
        # Beta[1]: Girth (BMI/Fat). +1 sigma ~ +20% mass (Volume scalling).
        scale_mass = 1.0
        if self.betas is not None and len(self.betas) > 1:
            scale_mass += self.betas[0] * 0.08
            scale_mass += self.betas[1] * 0.20
            
        m = m_base * max(0.4, scale_mass) # Safety clip (min 40% of base)
        
        # print(f"Computed Total Mass: {m:.2f} kg (Base: {m_base}, Betas: {self.betas[:2] if self.betas is not None else 'None'})")

        masses = {
            'pelvis': m * mass_fractions['pelvis'],
            'spine': m * mass_fractions['spine'] / 3.0, # split across 3 spine joints
            'head': m * mass_fractions['head'],
            'upper_leg': m * mass_fractions['upper_leg'],
            'lower_leg': m * mass_fractions['lower_leg'],
            'foot': m * mass_fractions['foot'],
            'upper_arm': m * mass_fractions['upper_arm'],
            'lower_arm': m * mass_fractions['lower_arm'],
            'hand': m * mass_fractions['hand']
        }

        # Weights for segment mass distribution (approximate)
        self.mass_fractions = mass_fractions # Store for later use
        
        result = {'lengths': lengths, 'masses': masses}
        if offsets is not None:
             result['offsets'] = offsets
        if segment_com_offsets is not None:
             result['segment_com_offsets'] = segment_com_offsets
        return result

    def _compute_skeleton_offsets(self):
        """
        Compute static local offsets for the skeleton based on limb lengths.
        Returns:
            offsets (np.array): (30, 3) Local position vectors for each joint in parent frame.
            Includes 24 real joints + 6 virtual (24-25:toes, 26-27:fingers, 28-29:heels)
        """
        # If we have exact model offsets, use them!
        if isinstance(self.limb_data, dict) and 'offsets' in self.limb_data:
            model_offsets = self.limb_data['offsets']
            # Ensure we have 30 joints
            if model_offsets.shape[0] < 30:
                offsets = np.zeros((30, 3))
                offsets[:model_offsets.shape[0]] = model_offsets
            else:
                offsets = model_offsets.copy()  # Copy to avoid mutating cached data
            
            # ── Replace SMPL spine with biomechanical S-curve ──────────
            # SMPL spine is cantilevered (for mesh deformation), not suitable
            # for torque computation. Replace with anatomical S-curve:
            # lumbar lordosis + thoracic kyphosis.
            if self.use_s_curve_spine:
                spine_chain = [3, 6, 9, 12]  # spine1, spine2, spine3, neck
                parents_map = {3: 0, 6: 3, 9: 6, 12: 9}
                
                # Reconstruct actual spine positions to get pelvis→neck endpoints
                spine_positions = {0: np.zeros(3)}  # pelvis at origin
                for sj in spine_chain:
                    pj = parents_map[sj]
                    spine_positions[sj] = spine_positions[pj] + offsets[sj]
                
                # Compute S-curve target positions (proportional to original bone lengths)
                bone_offsets = {sj: offsets[sj].copy() for sj in spine_chain}
                s_curve = SMPLProcessor._spine_s_curve_positions(
                    spine_positions[0], spine_positions[12], bone_offsets)
                
                # Convert S-curve positions back to parent-relative offsets
                for sj in spine_chain:
                    pj = parents_map[sj]
                    offsets[sj] = s_curve[sj] - s_curve[pj]
            
            # Add virtual joint offsets if missing (zeros)
            if model_offsets.shape[0] <= 24 or np.allclose(offsets[24], 0):
                foot_len = self.limb_data['lengths'].get('foot', 0.15) if isinstance(self.limb_data, dict) else 0.15
                toe_dir = np.array([0.0, -0.2, 1.0])
                toe_dir = toe_dir / np.linalg.norm(toe_dir) * foot_len
                offsets[24] = toe_dir  # L_toe from L_foot
                offsets[25] = toe_dir  # R_toe from R_foot
                offsets[26] = np.array([0.08, 0.0, 0.0])   # L_fingertip
                offsets[27] = np.array([-0.08, 0.0, 0.0])  # R_fingertip
            
            # Always override heel offsets — model values target calcaneus center,
            # not the ground contact point. We need the bottom of the heel.
            # Use mesh surface extents for body-specific calibration when available.
            extents = getattr(self, '_joint_surface_extents', None)
            if extents is not None:
                # Use the ankle's mesh extent: -Y = below ankle, -Z = behind ankle
                # Average L and R ankle extents for robustness
                l_ankle_down = abs(extents[7, 1, 0])   # L_Ankle -Y extent
                r_ankle_down = abs(extents[8, 1, 0])   # R_Ankle -Y extent
                l_ankle_back = abs(extents[7, 2, 0])   # L_Ankle -Z extent
                r_ankle_back = abs(extents[8, 2, 0])   # R_Ankle -Z extent
                
                # Heel offset: go to the bottom of the mesh, slightly behind
                # Use 95% of downward extent (the mesh bottom IS the floor contact)
                # and 70% of backward extent (heel is behind ankle but not at edge)
                l_heel = np.array([0.0, -l_ankle_down * 0.95, -l_ankle_back * 0.70])
                r_heel = np.array([0.0, -r_ankle_down * 0.95, -r_ankle_back * 0.70])
                offsets[28] = l_heel
                offsets[29] = r_heel
            else:
                # Fallback: fixed heuristic offset
                heel_dir = np.array([0.0, -1.0, -0.25])
                heel_dir = heel_dir / np.linalg.norm(heel_dir) * 0.063
                offsets[28] = heel_dir
                offsets[29] = heel_dir
            return offsets
        
        # Fallback to heuristic reconstruction
        offsets = np.zeros((30, 3))
        
        for i in range(30):
            if i < 24:
                node_name = self.joint_names[i]
            else:
                extra_names = ['l_toe', 'r_toe', 'l_fingertip', 'r_fingertip',
                               'l_heel', 'r_heel']
                node_name = extra_names[i-24]
            
            # Default logic
            length = 0.1
            if 'knee' in node_name: length = self.limb_data['lengths']['upper_leg']
            elif 'ankle' in node_name: length = self.limb_data['lengths']['lower_leg']
            elif 'foot' in node_name: length = 0.05
            elif 'elbow' in node_name: length = self.limb_data['lengths']['upper_arm']
            elif 'wrist' in node_name: length = self.limb_data['lengths']['lower_arm']
            elif 'hand' in node_name: length = self.limb_data['lengths']['hand']
            elif 'spine' in node_name: length = self.limb_data['lengths']['spine_segment']
            elif 'neck' in node_name or 'head' in node_name: length = self.limb_data['lengths']['neck']
            
            # Virtual Extensions
            elif 'toe' in node_name: length = 0.15
            elif 'fingertip' in node_name: length = 0.08
            elif 'heel' in node_name: length = 0.05

            # Define Base Local Position (unrotated)
            lx, ly, lz = 0.0, -1.0, 0.0 # Down
            
            if 'hip' in node_name:
                w = self.limb_data['lengths']['pelvis_width'] * 0.5
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx = dir_x * w
                ly = -0.05
                lz = 0.0
                length = 1.0 
                
            elif 'spine' in node_name or 'neck' in node_name or 'head' in node_name:
                 lx, ly, lz = 0.0, 1.0, 0.0
                 
            elif 'shoulder' in node_name or 'collar' in node_name: 
                if 'collar' in node_name:
                    # Spine->Collar Offset (Base of Neck to Clavicle Start)
                    length = self.limb_data['lengths']['collar'] * 0.33
                else:
                    # Collar->Shoulder Offset (Clavicle Length)
                    length = self.limb_data['lengths']['collar']
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx = dir_x
                ly = 0.0
                lz = 0.0
                
            elif 'elbow' in node_name or 'wrist' in node_name:
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'hand' in node_name:
                length = self.limb_data['lengths']['hand']
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'foot' in node_name:
                length = self.limb_data['lengths']['foot']
                # Corrected: Point Forward (+Z) and slightly Down (-Y)
                lx, ly, lz = 0.0, -0.2, 1.0
            
            elif 'heel' in node_name:
                # Heel: primarily Downward (-Y) with slight Backward (-Z) from ankle
                # Ankle joint center is ~3.5cm above heel ground contact point
                lx, ly, lz = 0.0, -0.9, -0.4
            
            base_vec = np.array([lx, ly, lz])
            if 'hip' in node_name:
                local_pos = base_vec
            else:
                norm = np.linalg.norm(base_vec)
                if norm > 1e-6:
                    base_vec = base_vec / norm
                local_pos = base_vec * length
            
            offsets[i] = local_pos
            
        return offsets

    def _update_physics_state(self, world_pos, pose_data_aa, options):
        """
        Updates physics state (Velocity, CoM, ZMP) for the current frame.
        Removes legacy contact probability logic.
        """
        F = world_pos.shape[0]
        J = world_pos.shape[1]
        dt = options.dt
        y_dim = getattr(self, 'internal_y_dim', 1)
        
        # --- 1. Velocity Smoothing State ---
        if not hasattr(self, 'prob_smoothed_vel') or self.prob_smoothed_vel is None or self.prob_smoothed_vel.shape[0] != J:
             self.prob_smoothed_vel = np.zeros((J, 3), dtype=np.float32)
             self.prob_prev_world_pos = None

        if not options.floor_enable:
             return
             
        floor_height = options.floor_height

        # --- Use Effective Position (Tips Override) ---
        effective_pos = world_pos.copy()
        if hasattr(self, 'temp_tips') and self.temp_tips:
             for j, t_pos in self.temp_tips.items():
                  if j < J:
                       effective_pos[:, j, :] = t_pos

        # --- Velocity Calculation ---
        if not hasattr(self, 'prob_prev_world_pos') or self.prob_prev_world_pos is None:
             lin_vel = np.zeros_like(effective_pos)
             self.prob_prev_world_pos = effective_pos[0].copy() if effective_pos.ndim == 3 else effective_pos.copy()
        elif F > 1:
             # Batch mode approximation
             lin_vel = np.zeros_like(effective_pos)
             lin_vel[1:] = (effective_pos[1:] - effective_pos[:-1]) / dt
             self.prob_prev_world_pos = effective_pos[-1].copy()
        else:
             lin_vel = (effective_pos - self.prob_prev_world_pos) / dt
             self.prob_prev_world_pos = effective_pos.copy()
             
        # Temporal Smoothing
        if F == 1:
             current_vel = lin_vel[0] # (J, 3)
             alpha_v = 0.3  # Reduced from 0.8 for more stable velocity tracking
             self.prob_smoothed_vel = self.prob_smoothed_vel * (1.0 - alpha_v) + current_vel * alpha_v
        
        # --- 2. CoM & ZMP State ---
        
        # CoM Access
        if self.current_com is not None:
            com = self.current_com # (F, 3) 
            if com.ndim == 2: c = com[0] # Assume F=1/streaming for state mostly
            else: c = com
        else:
            c = self._compute_full_body_com(world_pos[0:1])[0]

        # CoM Dynamics (Acceleration)
        #
        # Two paths:
        #   A) SG derivative (zmp_sg_window > 0): Fits a local quadratic to a
        #      ring buffer of COM positions and reads off the 2nd derivative.
        #      Single-step pos→acc avoids chaining two finite differences,
        #      which amplifies noise by ×fps².
        #   B) Legacy (zmp_sg_window == 0): Chained finite differences
        #      pos→vel→acc with One Euro filter on acceleration.
        
        sg_win = getattr(options, 'zmp_sg_window', 0)
        
        if not hasattr(self, 'prob_prev_com') or self.prob_prev_com is None:
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = np.zeros_like(c)
            self.prob_prev_com_acc = np.zeros_like(c)
            self.com_acc_filter = OneEuroFilter(min_cutoff=0.3, beta=0.1, d_cutoff=1.0)
            # SG ring buffer for COM positions (allocated on first use)
            self._zmp_sg_ring = None
            self._zmp_sg_ptr = 0
            self._zmp_sg_cnt = 0
            self._zmp_sg_coeffs = None
            
            com_vel = np.zeros_like(c)
            com_acc = np.zeros_like(c)
        elif sg_win >= 5:
            # ── Path A: SG derivative ──────────────────────────────────
            # Ensure odd window
            if sg_win % 2 == 0:
                sg_win += 1
            
            # Allocate / resize ring buffer
            if self._zmp_sg_ring is None or self._zmp_sg_ring.shape[0] != sg_win:
                self._zmp_sg_ring = np.tile(c, (sg_win, 1))  # (W, 3)
                self._zmp_sg_ptr = 0
                self._zmp_sg_cnt = 0
                self._zmp_sg_coeffs = None
            
            # Push current COM position into ring buffer
            self._zmp_sg_ring[self._zmp_sg_ptr] = c
            self._zmp_sg_ptr = (self._zmp_sg_ptr + 1) % sg_win
            self._zmp_sg_cnt = min(self._zmp_sg_cnt + 1, sg_win)
            
            N = self._zmp_sg_cnt
            
            if N >= 5:
                # Compute SG 2nd-derivative coefficients (cached)
                # For a quadratic fit, the 2nd derivative coefficients
                # are the same regardless of data — they only depend on
                # window size and dt.
                if self._zmp_sg_coeffs is None or self._zmp_sg_coeffs[0] != N or self._zmp_sg_coeffs[1] != dt:
                    # Build least-squares 2nd derivative kernel.
                    # For polyorder=2, the 2nd derivative at the center
                    # of a window of size N with spacing dt is:
                    #   d²f/dt² ≈ Σ_k  c_k · f_k
                    # where c_k are the SG coefficients for deriv=2.
                    from scipy.signal import savgol_coeffs
                    # savgol_coeffs returns filter coefficients for the
                    # given derivative.  We evaluate at the LAST point
                    # (causal) by using pos=N-1.
                    sg_c = savgol_coeffs(N, polyorder=min(2, N - 1),
                                         deriv=2, delta=dt, pos=N - 1)
                    self._zmp_sg_coeffs = (N, dt, sg_c)
                
                _, _, sg_c = self._zmp_sg_coeffs
                
                # Reorder ring buffer oldest → newest
                if N == sg_win:
                    ordered = np.roll(self._zmp_sg_ring, -self._zmp_sg_ptr, axis=0)  # (W, 3)
                else:
                    # Partial fill: take the N most recent entries
                    idxs = [(self._zmp_sg_ptr - N + k) % sg_win for k in range(N)]
                    ordered = self._zmp_sg_ring[idxs]  # (N, 3)
                
                # Apply coefficients: acc = Σ c_k · pos_k  (for each xyz)
                com_acc = sg_c @ ordered  # (3,) — 2nd derivative at current time
                
                # Velocity via finite difference (still needed for state tracking)
                com_vel = (c - self.prob_prev_com) / dt
            else:
                # Not enough samples yet — fall back to zero
                com_vel = (c - self.prob_prev_com) / dt
                com_acc = np.zeros_like(c)
            
            # Store raw acceleration for plausibility
            self._raw_com_acc = com_acc.copy()
            
            # Update state
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = com_vel.copy()
            self.prob_prev_com_acc = com_acc.copy()
        else:
            # ── Path B: Legacy chained finite differences ──────────────
            com_vel = (c - self.prob_prev_com) / dt
                 
            # Acceleration
            raw_acc = (com_vel - self.prob_prev_com_vel) / dt
            
            # Store raw acceleration for plausibility (unfiltered)
            self._raw_com_acc = raw_acc.copy()
            
            if not hasattr(self, 'com_acc_filter'):
                self.com_acc_filter = OneEuroFilter(min_cutoff=0.3, beta=0.1, d_cutoff=1.0)

            if dt > 0:
                self.com_acc_filter._freq = 1.0 / dt
            
            com_acc = self.com_acc_filter(raw_acc)
            
            # Update State
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = com_vel.copy()
            self.prob_prev_com_acc = com_acc.copy()
            
        # ZMP Calculation
        plane_dims = [d for d in [0, 1, 2] if d != y_dim]
        
        # h = CoM Height relative to floor
        com_h = c[y_dim] - floor_height
        g = 9.81
        
        # Offset = - (h/g) * a_horz
        acc_horz = com_acc[plane_dims] 
        zmp_offset = - (com_h / g) * acc_horz
        
        # Static Projection
        p_zmp_static = c[plane_dims]
        p_zmp = p_zmp_static + zmp_offset
        
        # Store ZMP
        if not hasattr(self, 'current_zmp') or self.current_zmp is None:
             self.current_zmp = np.zeros(3) # Streaming assumption
        
        if self.current_zmp.ndim == 1:
            self.current_zmp[plane_dims] = p_zmp
            self.current_zmp[y_dim] = floor_height
        else:
            # Should handle batch if needed, but prob_prev state implies F=1
            self.current_zmp[-1, plane_dims] = p_zmp
            self.current_zmp[-1, y_dim] = floor_height
        
        # ZMP Smoothing
        ZMP_ALPHA = 0.1 
        
        if not hasattr(self, 'prev_zmp_smooth'):
            self.prev_zmp_smooth = self.current_zmp.copy()
        
        if self.prev_zmp_smooth.shape != self.current_zmp.shape:
             self.prev_zmp_smooth = self.current_zmp.copy()
        
        if self.current_zmp.ndim == 1:
            self.current_zmp = self.prev_zmp_smooth * (1.0 - ZMP_ALPHA) + self.current_zmp * ZMP_ALPHA
            self.prev_zmp_smooth = self.current_zmp.copy()
        else:
            # Simple batch smooth?
            pass

    def _compute_probabilistic_contacts_stability_v2(self, F, J, world_pos, options):
        """Consensus-driven contact determination with crisp state machine.
        
        Architecture:
          Layer 1: Consensus probabilities (existing) → soft per-joint confidence
          Layer 2: Foot-group state machine with hysteresis → crisp ON/OFF
          Layer 3: Immediate weight distribution → pressure from CoM proximity
        
        No EMA on pressure, no redistribution, no overlapping signals.
        """
        # --- Layer 1: Get consensus probabilities ---
        consensus_probs = self._compute_probabilistic_contacts_consensus(
            F, J, world_pos, options
        )
        
        # Axis setup
        if options.input_up_axis == 'Y':
            y_dim = 1
            plane_dims = [0, 2]
        else:
            y_dim = 2
            plane_dims = [0, 1]
        
        floor_height = options.floor_height
        g_mag = 9.81
        total_mass = self.total_mass_kg
        
        # --- Compute support fraction from CoM acceleration ---
        prev_com = getattr(self, '_prev_com_for_stability', None)
        com_acc = getattr(self, 'prob_prev_com_acc', None)
        
        if prev_com is None or com_acc is None:
            return consensus_probs
        
        com = prev_com[0] if prev_com.ndim > 1 else prev_com
        
        g_vec = np.zeros(3)
        g_vec[y_dim] = -g_mag
        F_required = total_mass * (com_acc - g_vec)
        F_support_up = F_required[y_dim]
        support_fraction = np.clip(F_support_up / (total_mass * g_mag), 0.0, 1.0)
        
        # --- Dedicated RAW CoM acceleration (no One-Euro filter) ---
        # Computed from finite differences of the CoM position available here.
        # This bypasses ALL filtering to capture true free-fall acceleration.
        dt_s = 1.0 / max(self.framerate, 1.0)
        if not hasattr(self, '_v2_raw_prev_com') or self._v2_raw_prev_com is None:
            self._v2_raw_prev_com = com.copy()
            self._v2_raw_prev_vel = np.zeros(3)
            self._v2_raw_com_acc = np.zeros(3)
        else:
            raw_vel = (com - self._v2_raw_prev_com) / max(dt_s, 1e-6)
            self._v2_raw_com_acc = (raw_vel - self._v2_raw_prev_vel) / max(dt_s, 1e-6)
            self._v2_raw_prev_com = com.copy()
            self._v2_raw_prev_vel = raw_vel.copy()
        
        # Smooth support fraction via one-euro filter
        # (adaptive: smooth when stable, responsive during rapid hops)
        if not hasattr(self, '_v2_support_oef') or self._v2_support_oef is None:
            self._v2_support_oef = OneEuroFilter(
                min_cutoff=0.8, beta=0.2, d_cutoff=1.0,
                framerate=self.framerate
            )
            self._v2_support_frac = support_fraction
        self._v2_support_frac = float(self._v2_support_oef(support_fraction))
        smoothed_support = self._v2_support_frac
        
        if smoothed_support < 0.05:
            # Airborne — zero all pressure
            self._stability_computed_pressure = np.zeros(J)
            if hasattr(self, '_stability_pressure'):
                self._stability_pressure = np.zeros(J)
            return consensus_probs
        
        # --- Layer 2: Foot-group state machine with hysteresis ---
        
        # Define contact groups
        # Each group: (name, joint_indices)
        FOOT_GROUPS = {
            'LF': [10],   # L_foot  (heels 28/29 added in weight distribution)
            'RF': [11],   # R_foot  (ankle rotation positions heel, but ankle is not a contact point)
        }
        HAND_GROUPS = {
            'LH': [20, 22],  # L_wrist, L_hand
            'RH': [21, 23],  # R_wrist, R_hand
        }
        # Other joints are individual
        GROUPED_JOINTS = set()
        for joints in FOOT_GROUPS.values():
            GROUPED_JOINTS.update(joints)
        for joints in HAND_GROUPS.values():
            GROUPED_JOINTS.update(joints)
        # Ankles (7, 8) are NOT contact candidates — heel virtual joints (28, 29)
        # are the actual rear contact points, positioned by ankle rotation.
        # Exclude ankles from becoming individual groups.
        GROUPED_JOINTS.update({7, 8})
        # Toes (24, 25) are ignored entirely
        
        ALL_GROUPS = {}
        ALL_GROUPS.update(FOOT_GROUPS)
        ALL_GROUPS.update(HAND_GROUPS)
        # Add individual joints (knees, elbows, etc.) as single-member groups
        for j in range(min(J, 24)):
            if j not in GROUPED_JOINTS:
                ALL_GROUPS[f'J{j}'] = [j]
        
        # State init
        if not hasattr(self, '_v2_group_state') or self._v2_group_state is None:
            self._v2_group_state = {}      # group_name → bool (on/off)
            self._v2_group_frames = {}     # group_name → frames in transition
            self._v2_group_planted_h = {}  # group_name → lowest height while ON
            self._v2_group_prev_min_h = {} # group_name → previous frame's min_h
            self._v2_group_settled_frames = {} # group_name → frames at settled height
        
        # Hysteresis thresholds
        THRESH_ON  = 0.45  # consensus must exceed this to turn ON (primary)
        THRESH_OFF = 0.30  # consensus must drop below this to turn OFF
        FRAMES_ON  = 1     # consecutive frames above threshold to confirm ON
        FRAMES_OFF = 3     # consecutive frames below threshold to confirm OFF
        
        # Backup threshold: any non-zero probability + near floor = backup candidate
        THRESH_BACKUP = 0.01  # minimum consensus to be a backup candidate
        BACKUP_MAX_HEIGHT = 0.12  # max height (above floor) to be a backup
        
        # Height gate: even with high consensus, if ALL joints in group
        # are above 35cm, don't engage
        HEIGHT_GATE = 0.35
        
        # Liftoff plausibility parameters
        LIFT_HEIGHT_BAND = 0.05   # m — delta_h range for lift_height signal (0.0→0.05 maps to 0→1)
        LIFT_VEL_SCALE = 0.3      # m/s — upward velocity for full lift_velocity signal
        SETTLED_VEL_THRESH = 0.02 # m/s — velocity below this = settled
        SETTLED_FRAMES_REQ = 5    # frames at settled velocity to declare settled
        PLAUS_HEIGHT_GATE_MIN = 0.02  # m — HEIGHT_GATE floor at max plausibility
        
        # Foot group names for plausibility (only foot groups participate)
        FOOT_PLAUS_GROUPS = {'LF', 'RF'}
        
        # Heel joint mapping for push-off detection
        HEEL_MAP = {'LF': 28, 'RF': 29}
        FOOT_MAP = {'LF': 10, 'RF': 11}
        
        dt = 1.0 / max(self.framerate, 1.0)
        
        confirmed_groups = {}  # group_name → True if ON
        backup_groups = {}     # group_name → True if eligible as backup
        group_min_heights = {} # group_name → minimum joint height above floor
        group_positions = {}   # group_name → representative hz position
        
        # Calculate 'active_floor' to be immune to global root drift.
        # It is the minimum raw height among all currently 'ON' joints.
        active_floor = floor_height
        on_heights = []
        for gname in ALL_GROUPS:
            if self._v2_group_state.get(gname, False):
                joints = ALL_GROUPS[gname]
                for j in joints:
                    if j < J:
                        on_heights.append(world_pos[0, j, y_dim])
        if len(on_heights) > 0:
            active_floor = min(active_floor, min(on_heights))
        
        # First pass: compute min heights for all groups (needed for alt_support check)
        group_data = {}  # gname → (group_prob, min_h, best_pos, valid_joints, h_above)
        for gname, joints in ALL_GROUPS.items():
            valid_joints = [j for j in joints if j < J]
            if not valid_joints:
                continue
            
            group_prob = max(consensus_probs[0, j] for j in valid_joints)
            
            min_h = float('inf')
            active_h_above = float('inf')
            best_pos = None
            for j in valid_joints:
                if hasattr(self, 'temp_tips') and j in self.temp_tips:
                    pos_j = self.temp_tips[j][0] if self.temp_tips[j].ndim > 1 else self.temp_tips[j]
                    h = pos_j[y_dim]
                else:
                    pos_j = world_pos[0, j]
                    h = pos_j[y_dim]
                h_above_floor = h - floor_height
                h_above_active = h - active_floor
                if h_above_floor < min_h:
                    min_h = h_above_floor
                    active_h_above = h_above_active
                    best_pos = pos_j[plane_dims].copy()
            
            group_min_heights[gname] = min_h
            if best_pos is not None:
                group_positions[gname] = best_pos
            group_data[gname] = (group_prob, min_h, best_pos, valid_joints, active_h_above)
        
        # --- CoM velocity for direction signal ---
        # Used in plausibility to predict whether weight is arriving at
        # or departing from each foot (inverted pendulum model).
        com_hz = com[plane_dims]
        com_vel_raw = getattr(self, 'prob_prev_com_vel', None)
        com_vel_hz = np.zeros(2)
        if com_vel_raw is not None:
            if com_vel_raw.ndim == 1:
                com_vel_hz = com_vel_raw[plane_dims]
            elif com_vel_raw.ndim == 2:
                com_vel_hz = com_vel_raw[0, plane_dims]
        
        # Second pass: state machine with plausibility
        for gname, (group_prob, min_h, best_pos, valid_joints, h_above) in group_data.items():
            prev_state = self._v2_group_state.get(gname, False)
            prev_frames = self._v2_group_frames.get(gname, 0)
            
            # --- Liftoff plausibility (foot groups only, when ON) ---
            plausibility = 0.0
            if prev_state and gname in FOOT_PLAUS_GROUPS:
                planted_h = self._v2_group_planted_h.get(gname, h_above)
                prev_h_above = self._v2_group_prev_min_h.get(gname, h_above)
                settled_frames = self._v2_group_settled_frames.get(gname, 0)
                
                # Update planted height (track minimum relative height while ON)
                planted_h = min(planted_h, h_above)
                
                # Vertical velocity of lowest joint (relative to active support)
                v_up = (h_above - prev_h_above) / max(dt, 1e-6)
                
                # Settled detection: foot rose but velocity → 0
                delta_h = h_above - planted_h
                if delta_h > 0.01 and abs(v_up) < SETTLED_VEL_THRESH:
                    settled_frames += 1
                    if settled_frames >= SETTLED_FRAMES_REQ:
                        # Accept new baseline — foot has settled (tiptoe/pointe)
                        planted_h = h_above
                        settled_frames = 0
                else:
                    settled_frames = 0
                
                # Lifting evidence
                if delta_h > 0.005 and v_up > 0.01:
                    # Foot is above planted AND still rising
                    lift_height = min(1.0, max(0.0, delta_h / LIFT_HEIGHT_BAND))
                    lift_velocity = min(1.0, max(0.0, v_up / LIFT_VEL_SCALE))
                    lifting = max(lift_height, lift_velocity)
                else:
                    lifting = 0.0
                
                # Heel-lifted context (push-off indicator)
                heel_j = HEEL_MAP.get(gname)
                foot_j = FOOT_MAP.get(gname)
                heel_lifted = 0.0
                if heel_j is not None and foot_j is not None:
                    if hasattr(self, 'temp_tips') and heel_j in self.temp_tips:
                        h_heel = self.temp_tips[heel_j][0, y_dim] if self.temp_tips[heel_j].ndim > 1 else self.temp_tips[heel_j][y_dim]
                    else:
                        h_heel = world_pos[0, min(heel_j, world_pos.shape[1]-1), y_dim]
                    if hasattr(self, 'temp_tips') and foot_j in self.temp_tips:
                        h_foot = self.temp_tips[foot_j][0, y_dim] if self.temp_tips[foot_j].ndim > 1 else self.temp_tips[foot_j][y_dim]
                    else:
                        h_foot = world_pos[0, foot_j, y_dim]
                    heel_diff = (h_heel - floor_height) - (h_foot - floor_height)
                    if heel_diff > 0.02:  # heel is > 2cm above foot
                        heel_lifted = min(1.0, heel_diff / 0.08)  # full at 8cm
                
                # Boost lifting signal when heel is lifted (push-off context)
                lifting = lifting * (0.5 + 0.5 * heel_lifted)
                
                # No-support-needed: during free fall / bouncing, the body
                # doesn't need full ground support — liftoff is more plausible.
                # Uses RAW (unfiltered) CoM acceleration to capture true free-fall
                # that the filtered signal misses during short airborne phases.
                # Safety: this value is multiplied by `lifting`, so noise in the raw
                # signal only affects decisions when the foot IS kinematically rising.
                raw_com_acc = getattr(self, '_v2_raw_com_acc', None)
                if raw_com_acc is not None:
                    g_vec_y = -9.81
                    raw_F_up = self.total_mass_kg * (raw_com_acc[y_dim] - g_vec_y)
                    raw_sf = np.clip(raw_F_up / (self.total_mass_kg * 9.81), 0.0, 1.5)
                    no_support_needed = max(0.0, min(1.0, (0.95 - raw_sf) / 0.45))
                else:
                    no_support_needed = 0.0
                
                # Alternative support: is the OTHER foot group ON and grounded?
                # Also considers the other foot's descent velocity (incoming foot).
                alt_support = 0.0
                other_group = 'RF' if gname == 'LF' else 'LF'
                other_state = self._v2_group_state.get(other_group, False)
                other_h = group_min_heights.get(other_group, float('inf'))
                other_prev_h = self._v2_group_prev_min_h.get(other_group, other_h)
                other_vel = (other_h - other_prev_h) / max(dt, 1e-6)
                
                if other_state and other_h < 0.05:
                    # Other foot is ON and well grounded
                    alt_support = 1.0
                elif other_state and other_h < 0.12:
                    # Other foot is ON but somewhat elevated
                    alt_support = 0.5
                elif not other_state and other_h < 0.05:
                    # Other foot is OFF but very near floor — likely about to land
                    alt_support = 0.7
                elif not other_state and other_h < 0.20 and other_vel < -0.15:
                    # Other foot is OFF, below 20cm, and descending toward floor
                    # Estimate time to arrival: h / |vel|
                    time_to_floor = other_h / max(abs(other_vel), 0.01)
                    if time_to_floor < 0.10:  # Arrives in <100ms
                        alt_support = 0.9
                    elif time_to_floor < 0.25:  # Arrives in <250ms
                        alt_support = 0.7
                    elif time_to_floor < 0.50:  # Arrives in <500ms
                        alt_support = 0.4
                elif not other_state and other_h < 0.12:
                    # Other foot is OFF but approaching floor (no strong velocity)
                    alt_support = 0.3
                
                # FE low-force signal: if the frame evaluator says this foot
                # bears very little weight, liftoff is more plausible even
                # without strong kinematic lifting evidence.
                fe_low_force = 0.0
                fe_group_force = getattr(self, '_fe_group_force', {})
                fe_f = fe_group_force.get(gname, -1)
                if fe_f >= 0:  # FE has computed forces
                    if fe_f < 2.0:  # Less than 2 kg — essentially unsupported
                        fe_low_force = 0.8
                    elif fe_f < 5.0:  # Less than 5 kg — lightly loaded
                        fe_low_force = 0.4
                    elif fe_f < 10.0:  # Less than 10 kg — moderately loaded
                        fe_low_force = 0.2
                
                # Height-consensus evidence: foot is elevated AND raw consensus
                # has declined, even if not actively rising (v_up ≈ 0).
                # This catches the case where a foot is at 12cm+ but hovering
                # rather than rising, so the kinematic `lifting` signal is weak.
                # Must use RAW consensus, not group_prob (which is clamped at 0.95
                # for ON groups by the state machine output).
                height_evidence = 0.0
                if delta_h > 0.03:  # Foot is >3cm above planted baseline
                    h_signal = min(1.0, delta_h / 0.08)  # Full at 8cm
                    # Get raw consensus for this group's joints
                    raw_cons = getattr(self, '_raw_consensus_probs', None)
                    if raw_cons is not None:
                        raw_1d = raw_cons[0] if raw_cons.ndim > 1 else raw_cons
                        raw_group_prob = max(raw_1d[j] for j in valid_joints if j < len(raw_1d))
                    else:
                        raw_group_prob = group_prob
                    # Low raw consensus = strong evidence foot is off floor
                    cons_signal = max(0.0, 1.0 - raw_group_prob / 0.3)  # Full at prob=0
                    height_evidence = h_signal * cons_signal
                
                plausibility = max(
                    lifting * max(alt_support, no_support_needed, fe_low_force),
                    height_evidence * max(alt_support, fe_low_force, 0.3)
                )
                
                # --- CoM direction signal (inverted pendulum) ---
                # Predicts whether weight is arriving at or departing from this
                # foot by projecting the CoM horizontal velocity onto the
                # CoM→foot direction vector.
                #   dot > 0 → CoM approaching foot → weight arriving → suppress lift
                #   dot < 0 → CoM departing foot  → weight leaving  → encourage lift
                #
                # This distinguishes tiptoe support (CoM stationary/approaching
                # the RF) from genuine liftoff (CoM shifting away from LF).
                com_dir_signal = 0.0  # default: neutral
                if best_pos is not None:
                    foot_hz = best_pos  # already plane_dims
                    com_to_foot = foot_hz - com_hz
                    dist = np.linalg.norm(com_to_foot)
                    if dist > 0.01:  # avoid divide-by-zero for foot under CoM
                        com_to_foot_dir = com_to_foot / dist
                        # Project CoM velocity onto CoM→foot direction
                        approach_speed = np.dot(com_vel_hz, com_to_foot_dir)
                        # Normalize: ±0.3 m/s → ±1.0 signal
                        com_dir_signal = np.clip(approach_speed / 0.3, -1.0, 1.0)
                
                # Apply CoM direction modulation:
                # - Approaching (signal > 0): plausibility *= (1 - 0.6*signal)
                #   At full approach (signal=1.0): plausibility *= 0.4 (strong suppression)
                # - Departing (signal < 0): plausibility *= (1 + 0.4*|signal|)
                #   At full departure (signal=-1.0): plausibility *= 1.4 (moderate boost)
                if com_dir_signal > 0:
                    # CoM approaching this foot — suppress liftoff
                    plausibility *= (1.0 - 0.6 * com_dir_signal)
                elif com_dir_signal < 0:
                    # CoM departing this foot — boost liftoff
                    plausibility *= (1.0 + 0.4 * abs(com_dir_signal))
                
                # FE high-force suppression: if the FE says this foot bears
                # significant weight AND it was promoted via necessity override,
                # it is NOT lifting off — cap plausibility.
                # This prevents tiptoe/relevé (ball-of-foot) from triggering
                # false liftoff via height_evidence (foot joint is 12cm+ above
                # active floor even though it's supporting the body).
                #
                # Only apply when under necessity hold — normal ON feet need
                # their plausibility to be free to build so genuine lifts are
                # not suppressed by stale FE force distribution.
                necessity_held = getattr(self, '_v2_necessity_hold', {}).get(gname, 0) > 0
                if necessity_held and fe_f > 15.0:  # More than 15 kg — clearly load-bearing
                    plausibility = min(plausibility, 0.1)
                elif necessity_held and fe_f > 8.0:  # More than 8 kg — likely load-bearing
                    plausibility = min(plausibility, 0.25)
                
                # Store plausibility for downstream use (e.g., FE override)
                if not hasattr(self, '_v2_group_plausibility'):
                    self._v2_group_plausibility = {}
                self._v2_group_plausibility[gname] = plausibility
                
                # Store updated tracking
                self._v2_group_planted_h[gname] = planted_h
                self._v2_group_settled_frames[gname] = settled_frames
            
            # Store current min_h for next frame's velocity computation
            self._v2_group_prev_min_h[gname] = min_h
            
            # --- State machine with plausibility-modulated thresholds ---
            # Modulate HEIGHT_GATE, FRAMES_OFF, and THRESH_OFF based on plausibility
            eff_height_gate = HEIGHT_GATE - (HEIGHT_GATE - PLAUS_HEIGHT_GATE_MIN) * plausibility
            eff_frames_off = max(1, int(round(FRAMES_OFF * (1.0 - 0.67 * plausibility))))
            # Raise THRESH_OFF: at max plausibility, even moderate consensus isn't
            # enough to hold the contact ON (the body doesn't need it)
            eff_thresh_off = THRESH_OFF + (0.65 - THRESH_OFF) * plausibility
            
            if prev_state:  # Currently ON
                # Check if this group is under necessity hold protection
                necessity_hold = getattr(self, '_v2_necessity_hold', {}).get(gname, 0)
                if necessity_hold > 0:
                    # Protected: decrement hold, do not eject
                    if not hasattr(self, '_v2_necessity_hold'):
                        self._v2_necessity_hold = {}
                    self._v2_necessity_hold[gname] = necessity_hold - 1
                    self._v2_group_frames[gname] = 0
                elif group_prob < eff_thresh_off or min_h > eff_height_gate:
                    prev_frames += 1
                    if prev_frames >= eff_frames_off:
                        self._v2_group_state[gname] = False
                        self._v2_group_frames[gname] = 0
                    else:
                        self._v2_group_frames[gname] = prev_frames
                else:
                    self._v2_group_frames[gname] = 0

            else:  # Currently OFF
                # Decay stale plausibility: when a group is OFF, the liftoff
                # plausibility from the moment it was ejected should not persist
                # forever. Decay it toward 0 so the Necessity Override can
                # eventually re-acquire the contact if physics demands it.
                if gname in FOOT_PLAUS_GROUPS:
                    if not hasattr(self, '_v2_group_plausibility'):
                        self._v2_group_plausibility = {}
                    old_plaus = self._v2_group_plausibility.get(gname, 0.0)
                    # Exponential decay: halves roughly every 3 frames (~50ms at 60fps)
                    # Fast decay ensures necessity override can re-acquire quickly
                    # once genuine liftoff evidence dissipates.
                    PLAUS_DECAY = 0.80
                    self._v2_group_plausibility[gname] = old_plaus * PLAUS_DECAY
                
                if group_prob > THRESH_ON and min_h < HEIGHT_GATE:
                    prev_frames += 1
                    if prev_frames >= FRAMES_ON:
                        self._v2_group_state[gname] = True
                        self._v2_group_frames[gname] = 0
                        # Initialize planted tracking on ON transition
                        self._v2_group_planted_h[gname] = h_above
                        self._v2_group_settled_frames[gname] = 0
                    else:
                        self._v2_group_frames[gname] = prev_frames
                else:
                    self._v2_group_frames[gname] = 0
            
            confirmed_groups[gname] = self._v2_group_state.get(gname, False)
            
            # Backup eligibility: near floor (raw height) + foot group only.
            # No consensus dependency — handles cases where velocity penalties
            # wrongly zero out a foot that's genuinely on the floor.
            # Only foot groups (LF, RF, LA, RA) can be backups.
            FOOT_GROUP_NAMES = {'LF', 'RF', 'LA', 'RA'}
            if not confirmed_groups[gname] and gname in FOOT_GROUP_NAMES:
                backup_groups[gname] = min_h < BACKUP_MAX_HEIGHT
            else:
                backup_groups[gname] = False
        
        # --- Backup promotion: check if confirmed support is adequate ---
        # Only promote backups when:
        # 1. Body needs significant support (support_frac > 0.3)
        # 2. CoM is moving fast (high dynamic demand, not just walking)
        # 3. CoM is far from confirmed support centroid
        COM_SPEED_THRESH = 0.5  # m/s — below this, single-foot support is fine
        
        com_hz = com[plane_dims]
        com_vel_tmp = getattr(self, 'prob_prev_com_vel', None)
        com_speed_hz = 0.0
        if com_vel_tmp is not None and com_vel_tmp.ndim >= 1:
            cv_hz = com_vel_tmp[plane_dims] if com_vel_tmp.ndim == 1 else com_vel_tmp[0, plane_dims]
            com_speed_hz = np.linalg.norm(cv_hz)
            eff_com = com_hz + cv_hz * 0.05
        else:
            eff_com = com_hz
        
        if smoothed_support > 0.3 and com_speed_hz > COM_SPEED_THRESH:
            # Centroid of confirmed contacts
            confirmed_positions = []
            for gname, is_on in confirmed_groups.items():
                if is_on and gname in group_positions:
                    confirmed_positions.append(group_positions[gname])
            
            if confirmed_positions:
                support_centroid = np.mean(confirmed_positions, axis=0)
                com_to_support = np.linalg.norm(eff_com - support_centroid)
                
                SUPPORT_GAP = 0.20
                
                if com_to_support > SUPPORT_GAP:
                    # Find best backup group: closest to the effective CoM
                    best_backup = None
                    best_dist = float('inf')
                    for gname, is_backup in backup_groups.items():
                        if is_backup and gname in group_positions:
                            d = np.linalg.norm(eff_com - group_positions[gname])
                            if d < best_dist:
                                best_dist = d
                                best_backup = gname
                    
                    if best_backup is not None:
                        confirmed_groups[best_backup] = True
                        self._v2_group_state[best_backup] = True
                        self._v2_group_frames[best_backup] = 0
        
        # --- No-foot-contact fallback ---
        # A person cannot hover. If NO foot groups are confirmed but the
        # body clearly needs support, promote the lowest near-floor foot.
        # This handles extended dropouts during rapid footwork where
        # consensus temporal smoothing can't recover fast enough.
        FOOT_GROUPS_SET = {'LF', 'RF'}
        any_foot_on = any(confirmed_groups.get(g, False) for g in FOOT_GROUPS_SET)
        
        if not any_foot_on and smoothed_support > 0.3:
            # Find the lowest foot backup
            best_foot = None
            best_h = float('inf')
            for gname in FOOT_GROUPS_SET:
                if backup_groups.get(gname, False) and gname in group_min_heights:
                    if group_min_heights[gname] < best_h:
                        best_h = group_min_heights[gname]
                        best_foot = gname
            
            if best_foot is not None:
                confirmed_groups[best_foot] = True
                self._v2_group_state[best_foot] = True
                self._v2_group_frames[best_foot] = 0
        
        # --- Layer 3: Weight distribution among confirmed contacts ---
        
        # Effective CoM with velocity prediction
        COM_LOOKAHEAD = 0.03
        com_hz = com[plane_dims]
        com_vel = getattr(self, 'prob_prev_com_vel', None)
        if com_vel is not None and com_vel.ndim >= 1:
            com_vel_hz = com_vel[plane_dims] if com_vel.ndim == 1 else com_vel[0, plane_dims]
            effective_com_hz = com_hz + com_vel_hz * COM_LOOKAHEAD
        else:
            effective_com_hz = com_hz
        
        # Collect confirmed contact positions
        candidates = []
        for gname, is_on in confirmed_groups.items():
            if not is_on:
                continue
            for j in ALL_GROUPS[gname]:
                if j >= J:
                    continue
                if hasattr(self, 'temp_tips') and j in self.temp_tips:
                    pos = self.temp_tips[j][0]
                else:
                    pos = world_pos[0, j]
                h = pos[y_dim] - floor_height
                if h < HEIGHT_GATE:  # Only include joints actually near floor
                    candidates.append({
                        'j': j,
                        'pos_hz': pos[plane_dims],
                        'h': h,
                        'group': gname,
                    })
        
        # Also add heel virtual joints for confirmed foot groups
        for gname in ['LF', 'RF']:
            if confirmed_groups.get(gname, False):
                heel_j = 28 if gname == 'LF' else 29
                if heel_j < J:
                    if hasattr(self, 'temp_tips') and heel_j in self.temp_tips:
                        pos = self.temp_tips[heel_j][0]
                    else:
                        pos = world_pos[0, heel_j]
                    h = pos[y_dim] - floor_height
                    candidates.append({
                        'j': heel_j,
                        'pos_hz': pos[plane_dims],
                        'h': h,
                        'group': gname,
                    })
        
        target_pressure = np.zeros(J)
        
        if candidates:
            # --- Two-stage weight distribution ---
            # Stage 1: Group-level split (how much total pressure per foot group)
            # Uses centroid of each group's candidates vs CoM projection
            SIGMA_GROUP = 0.20  # 20cm — wider to reduce L/R pressure sensitivity to COM sway
            # At centroids 15cm each from CoM: 50/50
            # At 0cm vs 30cm: 94%/6%
            # At 0cm vs 50cm: 100%/0%
            
            # Collect group centroids from candidates
            group_candidates = {}  # gname -> list of candidate indices
            group_centroids = {}   # gname -> mean horizontal position
            for idx, c in enumerate(candidates):
                g = c['group']
                if g not in group_candidates:
                    group_candidates[g] = []
                group_candidates[g].append(idx)
            
            for g, idxs in group_candidates.items():
                positions = np.array([candidates[i]['pos_hz'] for i in idxs])
                group_centroids[g] = np.mean(positions, axis=0)
            
            # Gaussian weight per group based on centroid distance to CoM
            group_weights = {}
            for g, centroid in group_centroids.items():
                dist = np.linalg.norm(centroid - effective_com_hz)
                group_weights[g] = np.exp(-0.5 * (dist / SIGMA_GROUP)**2)
            
            total_group_weight = sum(group_weights.values())
            if total_group_weight > 1e-12:
                group_fractions = {g: w / total_group_weight for g, w in group_weights.items()}
            else:
                group_fractions = {g: 1.0 / len(group_weights) for g in group_weights}
            
            # Stage 2: Within each group, distribute equally among candidate joints
            desired_total = total_mass * smoothed_support
            mass_fractions = np.zeros(len(candidates))
            for g, idxs in group_candidates.items():
                per_joint = group_fractions[g] / max(len(idxs), 1)
                for i in idxs:
                    mass_fractions[i] = per_joint
            
            for idx, c in enumerate(candidates):
                target_pressure[c['j']] = mass_fractions[idx] * desired_total
            
            # Heel/Toe split: when heel lifts above foot, shift pressure to foot
            HEEL_LIFT_BAND = 0.05
            foot_heel_pairs = [
                (10, 28, 7),  # (L_foot, L_heel, L_ankle for fallback height)
                (11, 29, 8),  # (R_foot, R_heel, R_ankle for fallback height)
            ]
            for foot_j, heel_j, ankle_j_fallback in foot_heel_pairs:
                combined = target_pressure[foot_j] + target_pressure[heel_j]
                if combined <= 0:
                    continue
                
                if hasattr(self, 'temp_tips') and heel_j in self.temp_tips:
                    h_heel = self.temp_tips[heel_j][0, y_dim] if self.temp_tips[heel_j].ndim > 1 else self.temp_tips[heel_j][y_dim]
                else:
                    h_heel = world_pos[0, ankle_j_fallback, y_dim]
                
                if hasattr(self, 'temp_tips') and foot_j in self.temp_tips:
                    h_foot = self.temp_tips[foot_j][0, y_dim] if self.temp_tips[foot_j].ndim > 1 else self.temp_tips[foot_j][y_dim]
                else:
                    h_foot = world_pos[0, foot_j, y_dim]
                
                diff = (h_heel - floor_height) - (h_foot - floor_height)
                if diff <= 0:
                    continue
                
                # t_blend: 0 at diff=0, 1.0 when diff >= HEEL_LIFT_BAND
                t_blend = min(1.0, diff / HEEL_LIFT_BAND)
                
                # Shift pressure from heel toward foot as heel lifts
                # At t_blend=1.0, all pressure goes to foot
                foot_frac = target_pressure[foot_j] / max(combined, 1e-8)
                heel_frac = target_pressure[heel_j] / max(combined, 1e-8)
                target_pressure[foot_j] = combined * (t_blend + (1.0 - t_blend) * foot_frac)
                target_pressure[heel_j] = combined * (1.0 - t_blend) * heel_frac
            
            # Normalize to desired total
            raw_total = np.sum(target_pressure)
            if raw_total > 1e-6:
                target_pressure *= (desired_total / raw_total)
        
        # --- Direct pressure scaling by raw support fraction ---
        # Physics: if the body is in free fall, GRF must be zero regardless of
        # what the state machine thinks. Use a lightly-filtered raw support
        # fraction (3-frame moving average) to scale pressure directly.
        #
        # Corroboration: the raw CoM acceleration can be noisy (especially from
        # lateral motion coupling in sidestepping). True free fall requires:
        #   1. raw_acc_y ≈ -9.81 (acceleration consistent with gravity only)
        #   2. Contact joints actually elevated above the floor
        # If the acceleration is wildly different from -g, or feet are on the
        # ground, the "free fall" signal is noise — don't scale pressure.
        raw_com_acc = getattr(self, '_v2_raw_com_acc', None)
        if raw_com_acc is not None:
            raw_acc_y = raw_com_acc[y_dim]
            
            # Free-fall trajectory consistency: how close is the acceleration to -g?
            # True free fall: raw_acc_y ≈ -9.81 → deviation ≈ 0
            # Noise from lateral motion: raw_acc_y = -30 → deviation = 20
            # Consistency is 1.0 within ±3 m/s², drops to 0 beyond ±8 m/s²
            deviation = abs(raw_acc_y - (-g_mag))
            freefall_consistency = np.clip(1.0 - (deviation - 3.0) / 5.0, 0.0, 1.0)
            
            # Compute raw support fraction, constrained by consistency
            raw_sf_instant = np.clip(
                (raw_acc_y + g_mag) / g_mag,
                0.0, 1.5
            )
            # When consistency is low, pull raw_sf toward 1.0 (no effect)
            effective_sf = raw_sf_instant + (1.0 - freefall_consistency) * (1.0 - raw_sf_instant)
            
            # 3-frame moving average to guard against momentary glitches
            if not hasattr(self, '_raw_sf_history') or self._raw_sf_history is None:
                self._raw_sf_history = [effective_sf] * 3
            self._raw_sf_history.append(effective_sf)
            if len(self._raw_sf_history) > 3:
                self._raw_sf_history.pop(0)
            filtered_raw_sf = np.mean(self._raw_sf_history)
            
            # Foot-height corroboration: how high are the contact joints?
            # If feet are on the floor, don't trust the free-fall signal.
            min_contact_h = float('inf')
            for gname in ['LF', 'RF']:
                for j in ALL_GROUPS[gname]:
                    if j < J:
                        jh = world_pos[0, j, y_dim] - floor_height
                        min_contact_h = min(min_contact_h, jh)
            if min_contact_h == float('inf'):
                min_contact_h = 0.0
            # Corroboration: 0 when feet at floor (<1cm), 1.0 when all feet >4cm
            height_corroboration = np.clip((min_contact_h - 0.01) / 0.03, 0.0, 1.0)
            
            # Blend: at full corroboration, use filtered_raw_sf for scaling.
            # At zero corroboration (feet on ground), pressure_scale = 1.0 (no scaling).
            raw_pressure_scale = min(1.0, max(0.0, filtered_raw_sf / 0.5))
            pressure_scale = 1.0 - height_corroboration * (1.0 - raw_pressure_scale)
            target_pressure *= pressure_scale
        
        # Store as computed pressure (immediate — no EMA)
        self._stability_computed_pressure = target_pressure.copy()
        
        # Also set _stability_pressure for compatibility with other code
        if not hasattr(self, '_stability_pressure') or self._stability_pressure is None or len(self._stability_pressure) != J:
            self._stability_pressure = np.zeros(J)
        self._stability_pressure = target_pressure.copy()
        
        # Store raw consensus probs (before boost) for frame evaluator
        self._raw_consensus_probs = consensus_probs.copy()
        
        # Boost consensus probs for confirmed contacts
        for gname, is_on in confirmed_groups.items():
            if is_on:
                for j in ALL_GROUPS[gname]:
                    if j < J and target_pressure[j] > 1.0:
                        boost = min(0.95, target_pressure[j] / (total_mass * 0.3))
                        consensus_probs[0, j] = max(consensus_probs[0, j], boost)
        
        return consensus_probs


    def _evaluate_dynamic_frame(self, F, world_pos, tips, options, contact_probs=None):
        """Run the DynamicFrameEvaluator on current contacts (read-only).
        
        Uses consensus sensory probabilities (soft, forgiving) to identify
        candidates, then lets the physics evaluator determine which are
        actually needed. Independent from V2's state machine decisions.
        
        Does NOT modify any contact decisions.
        """
        from dpg_system.dynamic_frame_evaluator import DynamicFrameEvaluator
        
        # Lazy initialization
        if not hasattr(self, '_frame_evaluator') or self._frame_evaluator is None:
            seg_masses = self._seg_mass if hasattr(self, '_seg_mass') else np.ones(24)
            self._frame_evaluator = DynamicFrameEvaluator(
                total_mass=self.total_mass_kg,
                segment_masses=seg_masses,
                num_joints=world_pos.shape[1]
            )
        
        # Determine up axis and floor
        if options.input_up_axis == 'Y':
            y_dim = 1
            plane_dims = [0, 2]
        else:
            y_dim = 2
            plane_dims = [0, 1]
        floor_height = getattr(self, '_inferred_floor_height', None)
        if floor_height is None:
            floor_height = getattr(options, 'floor_height', 0.0) or 0.0
        
        # --- Effective floor estimation ---
        # If the body needs support (not in freefall), the floor cannot
        # be lower than the lowest joint. Any gap between lowest joints
        # and floor_height represents drift/calibration error.
        # Compute from previous frame's data (available now); store for
        # consensus to use on the next frame.
        EFFECTIVE_FLOOR_CEILING = 0.5  # Only consider joints below 0.5m
        eff_floor = getattr(self, '_effective_floor_height', None)
        if eff_floor is not None:
            floor_height = eff_floor
        
        J = world_pos.shape[1]
        
        # Get joint positions for frame 0 (single-frame mode)
        pos = world_pos[0]  # (J, 3)
        
        # Use tip positions where available
        positions = pos.copy()
        if tips:
            for j_idx, tip_pos in tips.items():
                if j_idx < J:
                    p = tip_pos[0] if tip_pos.ndim > 1 else tip_pos
                    positions[j_idx] = p
        
        # Get CoM and acceleration
        com = getattr(self, '_prev_com_for_stability', None)
        if com is None:
            com = getattr(self, 'current_com', None)
        if com is not None and com.ndim > 1:
            com = com[0]
        if com is None:
            # Fallback: compute from joint positions
            com = np.mean(positions[:min(J, 24)], axis=0)
        
        # --- EMA-subtraction CoM acceleration (frame evaluator only) ---
        # Uses the difference between fast and slow EMA of CoM position
        # as a velocity proxy, then differentiates once for acceleration.
        # This avoids double-differentiation noise amplification.
        # Other systems (stability_v2, etc.) keep using prob_prev_com_acc.
        ALPHA_FAST = 0.5
        ALPHA_SLOW = 0.05
        dt_s = getattr(self, '_dynamics_dt', 1.0 / max(self.framerate, 1.0))
        
        fe_state = getattr(self, '_fe_ema_state', None)
        if fe_state is None:
            self._fe_ema_state = {
                'com_fast': com.copy(),
                'com_slow': com.copy(),
                'prev_vel_proxy': np.zeros(3),
            }
            com_acc = np.zeros(3)
        else:
            # Update EMAs
            fe_state['com_fast'] = ALPHA_FAST * com + (1 - ALPHA_FAST) * fe_state['com_fast']
            fe_state['com_slow'] = ALPHA_SLOW * com + (1 - ALPHA_SLOW) * fe_state['com_slow']
            
            # Velocity proxy = fast - slow (proportional to velocity)
            vel_proxy = fe_state['com_fast'] - fe_state['com_slow']
            
            # Acceleration = d(vel_proxy)/dt (one differentiation only)
            acc_proxy = (vel_proxy - fe_state['prev_vel_proxy']) / dt_s
            fe_state['prev_vel_proxy'] = vel_proxy.copy()
            
            # Scale to physical units
            # For EMA subtraction: scale = dt * ((1-α_s)/α_s - (1-α_f)/α_f)
            scale = dt_s * ((1 - ALPHA_SLOW) / ALPHA_SLOW - (1 - ALPHA_FAST) / ALPHA_FAST)
            com_acc = acc_proxy / scale if abs(scale) > 1e-10 else np.zeros(3)
            
            # Amplitude inflation: the EMA subtraction bandpass attenuates
            # acceleration at walking frequency (~1Hz) by ~50%. This makes
            # ZMP sit too close to CoM projection, distributing force too
            # evenly. Inflate to restore ~90% of true ZMP displacement.
            # This doesn't hurt noise rejection because noise is at much
            # higher frequencies where the filter still attenuates strongly.
            ACC_GAIN = 1.8
            com_acc = com_acc * ACC_GAIN
        
        # --- Effective floor estimation ---
        # If body needs support (not in freefall), the floor cannot be lower
        # than the lowest joint. Compute from current positions + acceleration.
        g_mag = 9.81
        g_vec = np.zeros(3)
        g_vec[y_dim] = -g_mag
        f_required_up = (75.0 * (com_acc - g_vec))[y_dim]  # approx check
        
        if f_required_up > 0.1 * 75.0 * g_mag:
            # Body needs support — find lowest joints
            base_floor = getattr(options, 'floor_height', 0.0) or 0.0
            low_heights = []
            for j in range(min(J, 24)):
                h = positions[j, y_dim]
                if h - base_floor < EFFECTIVE_FLOOR_CEILING:
                    low_heights.append(h)
            if low_heights:
                min_joint_h = min(low_heights)
                new_eff = max(base_floor, min_joint_h)
                prev_eff = getattr(self, '_effective_floor_height', None)
                prev_min_h = getattr(self, '_eff_floor_prev_min_h', None)
                
                if prev_eff is not None:
                    # Compute velocity of lowest joint
                    min_h_velocity = 0.0
                    if prev_min_h is not None:
                        min_h_velocity = (min_joint_h - prev_min_h) / dt_s
                    
                    delta = new_eff - prev_eff
                    if delta > 0:
                        # Upward: only if lowest joint is stationary (drift).
                        # A rising joint means the body is leaving the floor,
                        # not the floor moving. Floors don't rise.
                        if min_h_velocity < 0.1:  # m/s — essentially stationary
                            new_eff = prev_eff + delta
                        else:
                            new_eff = prev_eff  # Don't update — body is lifting off
                    # Downward: always allow — floor can't be above where joints are
                
                self._eff_floor_prev_min_h = min_joint_h
                self._effective_floor_height = new_eff
                floor_height = new_eff
        
        # --- Compute per-joint surface distances ---
        # Transform T-pose surface extents into current pose to get
        # direction-aware joint-to-skin distances for floor contact
        surface_dists = None
        extents = getattr(self, '_joint_surface_extents', None)
        global_rots = getattr(self, '_prev_global_rots', None)
        if extents is not None and global_rots is not None:
            floor_normal = np.zeros(3)
            floor_normal[y_dim] = 1.0
            grots = global_rots[0] if global_rots.ndim == 4 else global_rots
            surface_dists = DynamicFrameEvaluator.compute_effective_surface_distances(
                extents, grots, floor_normal, num_joints=min(J, 24)
            )
            # Extend to virtual joints with defaults
            if len(surface_dists) < J:
                sd_full = np.full(J, 0.03)
                sd_full[:len(surface_dists)] = surface_dists
                surface_dists = sd_full
        
        # --- Gather candidate contacts ---
        # Two independent sources, unioned together:
        # 1. Consensus probability > low threshold (sensory evidence)
        # 2. Joint surface close to floor (geometric plausibility,
        #    using directional surface offsets to handle padding)
        CANDIDATE_THRESHOLD = 0.05
        
        raw_probs = getattr(self, '_raw_consensus_probs', None)
        if raw_probs is None and contact_probs is not None:
            raw_probs = contact_probs
        
        candidate_joints = set()
        # Exclude toes (24, 25) — IMU cannot measure toe bend, positions unreliable
        # Exclude ankles (7, 8) — heel virtual joints (28, 29) serve as ankle's
        # contact proxy. Having ankle + foot + heel as three co-located contacts
        # causes jittery force flickering between them.
        EXCLUDED_JOINTS = {7, 8, 24, 25}
        
        # Source 1: Consensus probability
        if raw_probs is not None:
            probs_1d = raw_probs[0] if raw_probs.ndim > 1 else raw_probs
            
            com = getattr(self, '_prev_com_for_stability', None)
            if com is None:
                com = getattr(self, 'current_com', None)
            if com is not None and com.ndim > 1:
                com = com[0]
            if com is None:
                com = positions[0]
            com_hz = com[plane_dims]
            
            for j in range(min(len(probs_1d), J)):
                prob = probs_1d[j]
                if prob > CANDIDATE_THRESHOLD and j not in EXCLUDED_JOINTS:
                    candidate_joints.add(j)
        
        # Compute joint heights for evaluator use early
        joint_heights = np.zeros(J)
        for j in range(J):
            joint_heights[j] = positions[j, y_dim] - floor_height
        
        # Source 2: Geometric proximity — foot/heel joints near the active
        # floor are candidates even when consensus probability is suppressed
        # (e.g. ball-of-foot stance where heel is elevated but ball touches).
        # This prevents the deadlock where low consensus → not a candidate →
        # FE assigns 0 force → Necessity Override can't fire → stays OFF.
        # 
        # CRITICAL: Use the height of the lowest currently-ON foot as the
        # reference, not the FE floor. A foot 7cm above the other planted foot
        # is lifted, not a proximity candidate. A foot 12cm above the other 
        # planted foot in a relevé IS a candidate (ball-of-foot stance).
        GEO_PROXIMITY_HEIGHT = 0.15  # 15cm above active support base
        GEO_FOOT_JOINTS = {10, 11, 28, 29}  # foot + heel joints only
        v2_state = getattr(self, '_v2_group_state', {})
        # Find the lowest ON foot joint as reference
        on_foot_heights = []
        FOOT_GROUP_JOINTS = {'LF': [10, 28], 'RF': [11, 29]}
        for gname, joints in FOOT_GROUP_JOINTS.items():
            if v2_state.get(gname, False):
                for j in joints:
                    if j < J:
                        on_foot_heights.append(positions[j, y_dim])
        geo_ref_floor = min(on_foot_heights) if on_foot_heights else floor_height
        
        for j in GEO_FOOT_JOINTS:
            if j < J and j not in EXCLUDED_JOINTS and j not in candidate_joints:
                h_above_ref = positions[j, y_dim] - geo_ref_floor
                if h_above_ref < GEO_PROXIMITY_HEIGHT:
                    candidate_joints.add(j)
        
        # --- Chain dominance pruning (soft) ---
        # Within each kinematic chain, prune ancestors that are WELL ABOVE
        # the chain's lowest contact. But keep ancestors near the floor —
        # crawling has both knee and foot in contact simultaneously.
        CHAINS = [
            [16, 18, 20, 22],  # Left arm: shoulder → elbow → wrist → hand
            [17, 19, 21, 23],  # Right arm: shoulder → elbow → wrist → hand
            [1, 4, 7, 10, 28], # Left leg: hip → knee → ankle → foot → heel
            [2, 5, 8, 11, 29], # Right leg: hip → knee → ankle → foot → heel
            [0, 3, 6, 9],      # Spine: pelvis → spine1 → spine2 → spine3
        ]
        CHAIN_PRUNE_MARGIN = 0.15  # Only prune if >15cm above lowest contact
        for chain in CHAINS:
            chain_candidates = [j for j in chain if j in candidate_joints and j < J]
            if len(chain_candidates) > 1:
                lowest_height = min(joint_heights[j] for j in chain_candidates)
                # Prune only joints well above the lowest contact point
                for j in chain_candidates:
                    if joint_heights[j] - lowest_height > CHAIN_PRUNE_MARGIN:
                        candidate_joints.discard(j)
        
        # --- Clamp CoM acceleration to physical limits ---
        # Prevents impossible force spikes from noisy data.
        # ±3g = ±29.4 m/s² is generous; covers jumping/landing.
        MAX_ACC = 3.0 * 9.81  # 3g
        acc_mag = np.linalg.norm(com_acc)
        if acc_mag > MAX_ACC:
            com_acc = com_acc * (MAX_ACC / acc_mag)
        
        # --- Compute per-joint horizontal velocities ---
        # Used by the evaluator to penalize moving joints (likely airborne)
        plane_dims = [0, 2] if y_dim == 1 else [0, 1]
        joint_velocities = np.zeros(J)
        prev_pos = getattr(self, '_fe_prev_positions', None)
        if prev_pos is not None and prev_pos.shape == positions.shape:
            delta = positions - prev_pos
            for j in range(J):
                joint_velocities[j] = np.linalg.norm(delta[j][plane_dims]) / dt_s
        self._fe_prev_positions = positions.copy()
        
        # --- Gather consensus probabilities ---
        raw_probs = getattr(self, '_raw_consensus_probs', None)
        consensus_1d = None
        if raw_probs is not None:
            consensus_1d = raw_probs[0] if raw_probs.ndim > 1 else raw_probs
        
        # --- Descent suppression ---
        # Suppress contact probability for joints with sustained downward
        # velocity, preventing premature detection during slow foot descent.
        # Uses asymmetric EMA: slow tracking of descent (tau_down=80ms),
        # fast recovery when velocity decreases (tau_up=15ms).
        # Adaptive scale ensures recovery time is speed-independent.
        # EMA is reset to 0 while a joint is in active contact.
        DS_TAU_DOWN = 0.12   # slow EMA for descent tracking
        DS_TAU_UP = 0.04     # fast EMA for recovery
        DS_BASE_SCALE = 0.03 # m/s — minimum scale (transition width)
        DS_FRAC = 0.5        # adaptive scale fraction
        
        ds_alpha_down = 1.0 - np.exp(-dt_s / DS_TAU_DOWN)
        ds_alpha_up = 1.0 - np.exp(-dt_s / DS_TAU_UP)
        
        # Deceleration detection: a separate symmetric EMA pair.
        # The asymmetric suppression EMA's fast recovery (tau_up=40ms) masks
        # deceleration. Two symmetric EMAs with different taus produce a clean
        # decel signal: positive during sustained deceleration (contact),
        # near-zero during gradual velocity changes (slow descent).
        DS_TAU_DECEL_SLOW = 0.080  # 80ms — moderate lag
        DS_TAU_DECEL_FAST = 0.020  # 20ms — tracks quickly
        DS_DECEL_SCALE = 0.10  # m/s difference to fully disable suppression
        ds_alpha_decel_slow = 1.0 - np.exp(-dt_s / DS_TAU_DECEL_SLOW)
        ds_alpha_decel_fast = 1.0 - np.exp(-dt_s / DS_TAU_DECEL_FAST)
        
        # Initialize per-joint state if needed
        if not hasattr(self, '_ds_vy_ema'):
            self._ds_vy_ema = np.zeros(J)
        if not hasattr(self, '_ds_decel_slow'):
            self._ds_decel_slow = np.zeros(J)
        if not hasattr(self, '_ds_decel_fast'):
            self._ds_decel_fast = np.zeros(J)
        if not hasattr(self, '_ds_prev_abs_heights'):
            self._ds_prev_abs_heights = None
        # Resize if joint count changed
        if len(self._ds_vy_ema) < J:
            self._ds_vy_ema = np.zeros(J)
        if len(self._ds_decel_slow) < J:
            self._ds_decel_slow = np.zeros(J)
        if len(self._ds_decel_fast) < J:
            self._ds_decel_fast = np.zeros(J)
        
        # Absolute heights (not floor-relative) for velocity computation
        # — immune to effective floor height drift between frames
        abs_heights = np.array([positions[j, y_dim] for j in range(J)])
        
        if consensus_1d is not None and self._ds_prev_abs_heights is not None:
            consensus_1d = consensus_1d.copy()  # don't modify original
            ds_suppressed = set()  # track which joints get suppressed
            prev_active = getattr(self, '_fe_prev_contacts', None) or set()
            for j in list(candidate_joints):
                if j >= J:
                    continue
                # Compute raw vertical velocity from absolute positions
                # (not floor-relative, to avoid phantom velocity from floor drift)
                raw_vy = (abs_heights[j] - self._ds_prev_abs_heights[j]) / dt_s
                
                # Impact detection: if velocity jumps sharply positive FROM
                # a descending state (e.g., landing: -2.0 → 0), reset EMA.
                # Only trigger when EMA was negative (joint was descending),
                # NOT for spikes from ~0 (push-off noise).
                DS_IMPACT_DELTA = 0.5  # m/s — minimum velocity jump for impact
                if (raw_vy - self._ds_vy_ema[j] > DS_IMPACT_DELTA and
                        self._ds_vy_ema[j] < -0.1):
                    self._ds_vy_ema[j] = raw_vy
                    self._ds_decel_slow[j] = raw_vy
                    self._ds_decel_fast[j] = raw_vy
                else:
                    # Asymmetric EMA update (for suppression)
                    if raw_vy > self._ds_vy_ema[j]:
                        alpha = ds_alpha_up    # fast recovery
                    else:
                        alpha = ds_alpha_down  # slow descent tracking
                    self._ds_vy_ema[j] += alpha * (raw_vy - self._ds_vy_ema[j])
                    # Symmetric decel pair (for deceleration detection)
                    self._ds_decel_slow[j] += ds_alpha_decel_slow * (raw_vy - self._ds_decel_slow[j])
                    self._ds_decel_fast[j] += ds_alpha_decel_fast * (raw_vy - self._ds_decel_fast[j])
                
                # Ascent ejection: if the joint is moving upward rapidly
                # and sustainedly, it cannot be in contact — eject.
                # Require BOTH fast and slow decel EMAs to agree on ascent
                # to avoid false ejection from single-frame velocity glitches
                # (e.g., noisy mocap during push-off).
                DS_ASCENT_THRESHOLD = 0.5  # m/s — minimum upward velocity to eject
                DS_ASCENT_SLOW_THRESHOLD = 0.25  # m/s — slow EMA must also confirm
                if (self._ds_decel_fast[j] > DS_ASCENT_THRESHOLD and
                        self._ds_decel_slow[j] > DS_ASCENT_SLOW_THRESHOLD):
                    consensus_1d[j] = 0.0
                    candidate_joints.discard(j)
                    ds_suppressed.add(j)
                    continue
                
                # Skip suppression for joints active in previous frame
                # (prevents post-landing flicker from velocity oscillation)
                if j in prev_active:
                    continue
                
                # Skip suppression for joints near or below the floor —
                # but only if the descent didn't start from well above
                # (EMA strongly negative means established descent from height)
                DS_FLOOR_MARGIN = 0.03  # 3cm
                if joint_heights[j] < DS_FLOOR_MARGIN and self._ds_vy_ema[j] > -2.0 * DS_BASE_SCALE:
                    continue
                
                # Adaptive scale and suppression
                vy_ema = self._ds_vy_ema[j]
                adaptive_scale = max(DS_BASE_SCALE, DS_FRAC * abs(vy_ema))
                suppression = np.clip(1.0 + vy_ema / adaptive_scale, 0.0, 1.0)
                
                # Deceleration detection: if the fast decel EMA is significantly
                # less negative than the slow one, a sustained deceleration is
                # happening (contact imminent). Scale down suppression.
                decel_signal = self._ds_decel_fast[j] - self._ds_decel_slow[j]
                if decel_signal > 0 and vy_ema < 0:
                    decel_factor = np.clip(decel_signal / DS_DECEL_SCALE, 0.0, 1.0)
                    suppression = 1.0 - (1.0 - suppression) * (1.0 - decel_factor)
                
                consensus_1d[j] *= suppression
                
                # Remove from candidates if fully suppressed
                if suppression < 0.01:
                    candidate_joints.discard(j)
                    ds_suppressed.add(j)
        else:
            ds_suppressed = set()
        
        # Store absolute heights for next frame's velocity computation
        self._ds_prev_abs_heights = abs_heights.copy()
        
        # --- Previous frame's active contacts and seed ---
        prev_active = getattr(self, '_fe_prev_contacts', None)
        prev_seed = getattr(self, '_fe_prev_seed', None)
        
        # --- Run the evaluator (pass 1: normal candidates) ---
        result = self._frame_evaluator.evaluate_and_refine(
            candidate_joints=candidate_joints,
            joint_positions=positions,
            com=com,
            com_acc=com_acc,
            floor_height=floor_height,
            up_axis=y_dim,
            max_iterations=3,
            all_joint_heights=joint_heights,
            surface_distances=surface_dists,
            consensus_probs=consensus_1d,
            joint_velocities=joint_velocities,
            prev_active_contacts=prev_active,
            prev_seed=prev_seed,
            excluded_joints=EXCLUDED_JOINTS,
        )
        
        # --- Contact recovery: residual-jump detection ---
        # If a previously-active contact dropped from the candidate set and
        # the residual jumped, try adding it back. This catches cases where
        # consensus briefly dips below the candidate threshold but the contact
        # is still physically needed.
        RECOVERY_PROB_THRESHOLD = 0.01  # minimum consensus to consider recovery
        RECOVERY_RESIDUAL_RATIO = 0.70  # recovered contact must reduce residual to 70%
        
        current_residual = np.linalg.norm(result.residual)
        prev_residual = getattr(self, '_fe_prev_residual', current_residual)
        current_active_reps = set(result.per_contact_force.keys())
        
        # Find previously-active contacts that dropped out
        if prev_active is not None:
            dropped = prev_active - current_active_reps - ds_suppressed
            
            if dropped and current_residual > prev_residual * 1.3:
                # Residual jumped — try recovering dropped contacts
                recovery_candidates = set()
                probs_1d = raw_probs[0] if raw_probs is not None and raw_probs.ndim > 1 else raw_probs
                for j in dropped:
                    if j not in EXCLUDED_JOINTS and j < J:
                        p = probs_1d[j] if probs_1d is not None and j < len(probs_1d) else 0
                        if p > RECOVERY_PROB_THRESHOLD:
                            recovery_candidates.add(j)
                
                if recovery_candidates:
                    # Re-run with the dropped contacts added back
                    expanded = candidate_joints | recovery_candidates
                    test_result = self._frame_evaluator.evaluate_and_refine(
                        candidate_joints=expanded,
                        joint_positions=positions,
                        com=com,
                        com_acc=com_acc,
                        floor_height=floor_height,
                        up_axis=y_dim,
                        max_iterations=3,
                        all_joint_heights=joint_heights,
                        surface_distances=surface_dists,
                        consensus_probs=consensus_1d,
                        joint_velocities=joint_velocities,
                        prev_active_contacts=prev_active,
                        prev_seed=prev_seed,
                        excluded_joints=EXCLUDED_JOINTS,
                    )
                    test_residual = np.linalg.norm(test_result.residual)
                    # Accept recovery if it substantially reduces the residual
                    if test_residual < current_residual * RECOVERY_RESIDUAL_RATIO:
                        result = test_result
        
        self._fe_prev_residual = np.linalg.norm(result.residual)
        
        # --- Static lever sanity cap ---
        # During weight transfers, the ZMP-based force distribution can assign
        # 50-70% of body weight to a foot the CoM has moved away from. The
        # static lever rule (CoM projection) provides a ground-truth upper
        # bound: a foot far from the CoM projection cannot bear more than its
        # lever fraction suggests. Cap FE forces accordingly.
        LEVER_CAP_MULTIPLIER = 1.3  # Allow up to 1.3x the static prediction
        LEVER_CAP_MIN_FRAC = 0.15   # Don't cap below this fraction
        
        # Foot group definitions: group_name -> list of joint indices
        FE_FOOT_GROUPS = {'LF': [10, 28], 'RF': [11, 29]}
        
        total_fe_force = sum(max(0, f) for f in result.per_contact_force.values())
        if total_fe_force > 1.0 and com is not None:
            # Compute static lever fractions
            foot_positions_hz = {}
            foot_forces = {}
            for gname, joints in FE_FOOT_GROUPS.items():
                active_in_group = [j for j in joints if j in result.per_contact_force and result.per_contact_force[j] > 0]
                if active_in_group:
                    # Use force-weighted position of active joints in this group
                    pos_hz = np.zeros(2)
                    f_total = 0.0
                    for j in active_in_group:
                        pos_hz += positions[j, plane_dims] * result.per_contact_force[j]
                        f_total += result.per_contact_force[j]
                    if f_total > 0:
                        pos_hz /= f_total
                    foot_positions_hz[gname] = pos_hz
                    foot_forces[gname] = f_total
            
            if len(foot_positions_hz) == 2 and 'LF' in foot_positions_hz and 'RF' in foot_positions_hz:
                lf_hz = foot_positions_hz['LF']
                rf_hz = foot_positions_hz['RF']
                d_total = np.linalg.norm(lf_hz - rf_hz)
                if d_total > 0.05:  # Feet sufficiently separated
                    d_com_to_rf = np.linalg.norm(com_hz - rf_hz)
                    d_com_to_lf = np.linalg.norm(com_hz - lf_hz)
                    # Static lever: fraction borne by each foot
                    static_lf = np.clip(d_com_to_rf / d_total, 0, 1)
                    static_rf = 1.0 - static_lf
                    
                    for gname, static_frac in [('LF', static_lf), ('RF', static_rf)]:
                        cap_frac = max(LEVER_CAP_MIN_FRAC, static_frac * LEVER_CAP_MULTIPLIER)
                        max_force = cap_frac * total_fe_force
                        current_force = foot_forces.get(gname, 0)
                        
                        if current_force > max_force and current_force > 0:
                            scale = max_force / current_force
                            for j in FE_FOOT_GROUPS[gname]:
                                if j in result.per_contact_force:
                                    result.per_contact_force[j] *= scale
                                    if j < len(result.force_array):
                                        result.force_array[j] *= scale
                            # Update necessity
                            for j in FE_FOOT_GROUPS[gname]:
                                if j in result.per_contact_force:
                                    f_val = result.per_contact_force[j]
                                    if f_val > 2.0:
                                        result.necessity[j] = 'necessary'
                                    elif f_val > 0.5:
                                        result.necessity[j] = 'marginal'
                                    else:
                                        result.necessity[j] = 'unnecessary'
        
        # Store per-group FE force for state machine use
        fe_group_force = {}
        for gname, joints in FE_FOOT_GROUPS.items():
            fe_group_force[gname] = sum(
                max(0, result.per_contact_force.get(j, 0)) for j in joints
            )
        self._fe_group_force = fe_group_force
        
        # Track active contacts and seed for next frame
        self._fe_prev_contacts = set(
            j for j, f in result.per_contact_force.items() if f > 0
        )
        self._fe_prev_seed = getattr(result, 'seed', None)
        
        # Reset descent EMA for active contacts that weren't suppressed
        # (genuine grounded contacts — prevents post-landing flicker)
        for j in self._fe_prev_contacts:
            if j < len(self._ds_vy_ema) and j not in ds_suppressed:
                self._ds_vy_ema[j] = 0.0
        
        # --- Proximity-adaptive contact force smoothing ---
        # When contacts are close together, the ZMP lever rule is
        # ill-conditioned: small ZMP shifts → large force redistribution.
        # Smooth the force distribution proportional to contact proximity.
        if getattr(options, 'smooth_contact_forces', False):
            active_force_joints = [j for j, fv in result.per_contact_force.items() if fv > 0.01]
            if len(active_force_joints) >= 2:
                # Minimum pairwise distance between active contacts
                positions = world_pos[0] if world_pos.ndim > 2 else world_pos
                min_dist = float('inf')
                for i_idx in range(len(active_force_joints)):
                    for j_idx in range(i_idx + 1, len(active_force_joints)):
                        ji, jj = active_force_joints[i_idx], active_force_joints[j_idx]
                        if ji < positions.shape[0] and jj < positions.shape[0]:
                            d = np.linalg.norm(positions[ji] - positions[jj])
                            min_dist = min(min_dist, d)
                
                # Adaptive alpha: close contacts → strong smoothing (low alpha)
                D_REF = 0.3  # ~30cm — heel-to-foot is ~15-20cm
                ALPHA_MIN = 0.15  # strongest smoothing (when contacts very close)
                alpha = max(ALPHA_MIN, min(1.0, min_dist / D_REF))
                
                # EMA on force_array, then rescale to preserve total force
                prev_forces = getattr(self, '_fe_smooth_forces', None)
                raw_forces = result.force_array.copy()
                total_raw = np.sum(raw_forces)
                
                if prev_forces is not None and prev_forces.shape == raw_forces.shape and total_raw > 0.01:
                    smoothed = prev_forces + alpha * (raw_forces - prev_forces)
                    # Rescale to preserve total force
                    total_smooth = np.sum(smoothed)
                    if total_smooth > 0.01:
                        smoothed *= total_raw / total_smooth
                    result.force_array = smoothed
                    # Update per_contact_force dict to match
                    for j in result.per_contact_force:
                        if j < len(smoothed):
                            result.per_contact_force[j] = max(0.0, float(smoothed[j]))
                
                self._fe_smooth_forces = result.force_array.copy()
        
        self._frame_eval_result = result


    def _init_contact_consensus(self):
        """Initialize the consensus contact detection module."""
        # Build per-joint segment masses from limb_data
        limb_masses = self.limb_data['masses']
        segment_masses = np.zeros(24)
        for idx in range(24):
            name = self.joint_names[idx]
            m = 0.0
            if 'pelvis' in name: m = limb_masses['pelvis']
            elif 'hip' in name: m = limb_masses['upper_leg']
            elif 'knee' in name: m = limb_masses['lower_leg']
            elif 'ankle' in name: m = limb_masses['foot']
            elif 'spine' in name: m = limb_masses['spine']
            elif 'neck' in name: m = limb_masses.get('head', 1.0) * 0.2
            elif 'head' in name: m = limb_masses['head'] * 0.8
            elif 'collar' in name: m = limb_masses['upper_arm'] * 0.2
            elif 'shoulder' in name: m = limb_masses['upper_arm'] * 0.8
            elif 'elbow' in name: m = limb_masses['lower_arm']
            elif 'wrist' in name: m = limb_masses['hand']
            segment_masses[idx] = m
        
        self._consensus = ContactConsensus(
            parents=list(self._get_hierarchy()[:24]),
            segment_masses=segment_masses,
            max_torques=self.max_torque_array if hasattr(self, 'max_torque_array') else None,
            num_joints=30  # 24 real + 6 virtual
        )
    
    def _compute_probabilistic_contacts_consensus(self, F, J, world_pos, options):
        """
        Consensus-based contact detection.
        
        Delegates to ContactConsensus module which combines sensory,
        structural, dynamic, and torque plausibility evidence.
        """
        contact_probs = np.zeros((F, J))
        
        if not options.floor_enable:
            return contact_probs
            
        # Priority: effective floor (physics-derived) > inferred floor > static
        floor_for_consensus = options.floor_height
        if hasattr(self, '_inferred_floor_height') and self._inferred_floor_height is not None:
            floor_for_consensus = self._inferred_floor_height
        if hasattr(self, '_effective_floor_height') and self._effective_floor_height is not None:
            floor_for_consensus = self._effective_floor_height
        
        consensus_opts = ConsensusOptions(
            floor_height=floor_for_consensus,
            up_axis=getattr(self, 'internal_y_dim', 1),
        )
        
        for f in range(F):
            pos = world_pos[f]  # (J, 3)
            
            # Get CoM - may not be available yet on first frames
            if self.current_com is not None:
                com = self.current_com[f] if self.current_com.ndim > 1 else self.current_com
            else:
                # Compute a simple CoM from the pelvis (joint 0)
                com = pos[0].copy()
            
            # Feed previous frame torques for plausibility checking
            if hasattr(self, '_prev_torque_vecs') and self._prev_torque_vecs is not None:
                self._consensus.set_prev_torques(self._prev_torque_vecs)
            
            # Get previous frame contact history if available
            prev_history = None
            if f > 0:
                prev_history = contact_probs[f-1].copy()
            elif hasattr(self, '_raw_consensus_probs') and self._raw_consensus_probs is not None:
                prev_history = self._raw_consensus_probs[0].copy() if self._raw_consensus_probs.ndim > 1 else self._raw_consensus_probs.copy()
                
            probs = self._consensus.compute_contacts(pos, com, options.dt, consensus_opts, contact_history=prev_history)
            
            # Ensure output size matches
            contact_probs[f, :min(J, len(probs))] = probs[:min(J, len(probs))]

        return contact_probs

    # ------------------------------------------------------------------
    # Patch-based contact detection (Phase 1)
    # ------------------------------------------------------------------

    # Predefined multi-anchor patches.


    def _compute_probabilistic_contacts_logodds(self, F, J, world_pos, options, enable_valving=False):
        """Log-odds continuous contact estimation.
        
        Uses additive log-odds accumulation with temporal decay.
        No state machine, no vetoes — continuous intensity output.
        When enable_valving=True, cross-stream context is used to
        adjust stream contributions (data-driven rules).
        """
        from dpg_system.contact_logodds import (
            LogOddsContactEstimator, LogOddsContactOptions, ALL_GROUPS
        )
        
        # Lazy-init estimator
        if not hasattr(self, '_logodds_estimator') or self._logodds_estimator is None:
            # Get frame evaluator for structural stream
            frame_eval = getattr(self, '_frame_evaluator', None)
            seg_masses = getattr(self, '_seg_mass', None)
            self._logodds_estimator = LogOddsContactEstimator(
                framerate=self.framerate,
                total_mass_kg=self.total_mass_kg,
                segment_masses=seg_masses,
                frame_evaluator=frame_eval,
            )
        
        # Late-bind frame evaluator if it wasn't available at construction
        if (self._logodds_estimator.structural_stream.evaluator is None
                and hasattr(self, '_frame_evaluator')
                and self._frame_evaluator is not None):
            self._logodds_estimator.structural_stream.set_evaluator(
                self._frame_evaluator)
        
        # Create a DynamicFrameEvaluator on-demand if the structural stream
        # still has no evaluator. Without this, the structural stream outputs
        # 0.0 for all groups, leaving kinematic noise unchecked.
        if self._logodds_estimator.structural_stream.evaluator is None:
            from dpg_system.dynamic_frame_evaluator import DynamicFrameEvaluator
            seg_masses = getattr(self, '_seg_mass', None)
            if seg_masses is None:
                seg_masses = np.ones(24)
            n_j = world_pos.shape[1] if world_pos.ndim == 3 else world_pos.shape[0]
            fe = DynamicFrameEvaluator(
                total_mass=self.total_mass_kg,
                segment_masses=seg_masses,
                num_joints=n_j
            )
            self._frame_evaluator = fe
            self._logodds_estimator.structural_stream.set_evaluator(fe)
        
        # Build options
        lo_opts = LogOddsContactOptions(
            up_axis=1 if options.input_up_axis == 'Y' else 2,
            # 3-stream architecture enables
            enable_height=options.logodds_enable_height,
            enable_kinematic=options.logodds_enable_kinematic,
            enable_structural=options.logodds_enable_structural,
            enable_divergence=options.logodds_enable_divergence,
            # Legacy enables
            enable_vertical_kinematic=options.logodds_enable_vertical_kinematic,
            enable_hspeed=options.logodds_enable_hspeed,
            enable_equilibrium=options.logodds_enable_equilibrium,
            enable_velocity=options.logodds_enable_velocity,
            enable_trajectory=options.logodds_enable_trajectory,
            enable_touchdown=options.logodds_enable_touchdown,
            # 3-stream weights
            weight_height=options.logodds_weight_height,
            weight_kinematic=options.logodds_weight_kinematic,
            weight_structural=options.logodds_weight_structural,
            weight_divergence=options.logodds_weight_divergence,
            # Legacy weights
            weight_vertical_kinematic=options.logodds_weight_vertical_kinematic,
            weight_hspeed=options.logodds_weight_hspeed,
            weight_equilibrium=options.logodds_weight_equilibrium,
            weight_velocity=options.logodds_weight_velocity,
            weight_trajectory=options.logodds_weight_trajectory,
            weight_touchdown=options.logodds_weight_touchdown,
            # Accumulator
            decay_rate=options.logodds_decay_rate,
            struct_force_ema_alpha=options.logodds_struct_force_ema_alpha,
            fe_relief_enable=options.logodds_fe_relief_enable,
            fe_relief_strain_threshold=options.logodds_fe_relief_strain_threshold,
            struct_relief_logodds=options.logodds_struct_relief_logodds,
            enable_valving=enable_valving,
            # Body contacts
            enable_body_contacts=options.enable_body_contacts,
        )
        
        # Get CoM state
        com = getattr(self, '_prev_com_for_stability', None)
        com_vel = getattr(self, 'prob_prev_com_vel', None)
        com_acc = getattr(self, 'prob_prev_com_acc', None)
        
        if com is None:
            return np.zeros((F, J))
        
        com = com[0] if com.ndim > 1 else com
        if com_vel is not None:
            com_vel = com_vel[0] if com_vel.ndim > 1 else com_vel
        else:
            com_vel = np.zeros(3)
        if com_acc is not None:
            com_acc = com_acc[0] if com_acc.ndim > 1 else com_acc
        else:
            com_acc = np.zeros(3)
        
        dt = 1.0 / max(self.framerate, 1.0)
        pos = world_pos[0] if world_pos.ndim == 3 else world_pos
        
        # Use adaptive floor
        floor_h = options.floor_height
        if hasattr(self, '_inferred_floor_height') and self._inferred_floor_height is not None:
            floor_h = self._inferred_floor_height
        
        # Compute surface distances for lever angle correction
        surface_dists = None
        extents = getattr(self, '_joint_surface_extents', None)
        min_dists_data = getattr(self, '_joint_surface_min_dists', None)
        global_rots = getattr(self, '_prev_global_rots', None)
        if extents is not None and global_rots is not None:
            from dpg_system.dynamic_frame_evaluator import DynamicFrameEvaluator
            from dpg_system.contact_logodds import PRIMARY_GROUPS, BODY_GROUPS
            floor_normal = np.zeros(3)
            floor_normal[1] = 1.0  # Y-up after axis permutation
            grots = global_rots[0] if global_rots.ndim == 4 else global_rots
            n24 = min(J, 24)
            # Max extents (for foot/hand contacts — full limb reach to floor)
            sd_max = DynamicFrameEvaluator.compute_effective_surface_distances(
                extents, grots, floor_normal, num_joints=n24)
            # Min distances (for body contacts — closest skin surface)
            if min_dists_data is not None:
                sd_min = DynamicFrameEvaluator.compute_min_surface_distances(
                    min_dists_data, grots, floor_normal, num_joints=n24)
            else:
                sd_min = np.full(n24, 0.03)
            
            # Build hybrid surface distances per joint:
            # - Primary joints (feet/hands): max extent (full reach to sole/palm)
            # - Pelvis (j=0): max extent (butt surface is far side, ~13cm)
            # - Head (j=15): min extent (forehead/face is near side;
            #   max extent would over-correct when head tilts forward,
            #   projecting the 13cm scalp toward the floor)
            # - Knees, elbows: min extent (kneecap/flesh is near side, ~3-4cm)
            # - Hips (j=1,2): max extent (joint center is deep; contact
            #   surface is buttock on far side, like pelvis in standing)
            # - Pelvis (j=0): min extent. The sd_max (~19cm buttock projection)
            #   is only valid in neutral standing. In a squat/crouch the thighs
            #   compress against the buttock, drastically reducing the effective
            #   offset. Using sd_min (~2.5cm) prevents false positives during
            #   squats while still detecting actual seated contact (pelvis
            #   drops to 5-10cm raw → 2.5-7.5cm corrected → contact zone).
            USE_MAX_JOINTS = set()
            for joints in PRIMARY_GROUPS.values():
                USE_MAX_JOINTS.update(joints)
            USE_MAX_JOINTS.update({1, 2})  # Hips only — deep joints
            
            surface_dists = sd_min.copy()  # default to min (skin thickness)
            for j in USE_MAX_JOINTS:
                if j < len(surface_dists):
                    surface_dists[j] = sd_max[j]
            
            if len(surface_dists) < J:
                sd_full = np.zeros(J)  # Virtual joints (24-29) are already
                # at mesh contact surfaces — no surface correction needed
                sd_full[:len(surface_dists)] = surface_dists
                surface_dists = sd_full

        # Previous frame's world-frame gravity torques, for the structural
        # stream's effort-relief prior. Strain is slow-varying so the
        # one-frame lag is fine; first frame has none and the prior no-ops.
        prev_grav = getattr(self, '_last_gravity_torques_world', None)
        if (prev_grav is not None and prev_grav.ndim == 3
                and prev_grav.shape[0] >= 1):
            prev_grav_2d = prev_grav[0]
        else:
            prev_grav_2d = None

        result = self._logodds_estimator.process_frame(
            pos, com, com_vel, com_acc,
            floor_h, dt, lo_opts,
            raw_com_acc=getattr(self, '_raw_com_acc', None),
            surface_dists=surface_dists,
            gravity_torque_vecs=prev_grav_2d,
        )
        
        # Store pressure for the stab_press path
        self._stability_computed_pressure = result.pressure_array
        
        # Store contact state for downstream
        self._eq_group_state = result.contact_state
        
        # Store log-odds result for diagnostics
        self._logodds_result = result
        
        # Forward the structural stream's FE result for frame_eval outputs
        # (ZMP, forces, necessity, support polygon)
        struct_fe = getattr(self._logodds_estimator.structural_stream,
                            'last_eval_result', None)
        self._frame_eval_result = struct_fe
        
        # Build contact probability output using intensity
        contact_probs = np.zeros((F, J))
        group_joints = dict(ALL_GROUPS)
        for gname, intensity in result.intensity.items():
            if intensity > 0.05 and gname in group_joints:
                for j in group_joints[gname]:
                    if j < J:
                        contact_probs[0, j] = intensity
        
        return contact_probs

    def _update_adaptive_floor(self, world_pos, tips, working_probs, options):
        """Update the adaptive floor height estimate.
        
        Extracted from the legacy pressure pipeline so ALL contact methods
        benefit from floor adaptation. Uses confirmed contacts, stability
        pressure, and minimum foot height as fallbacks.
        
        Args:
            world_pos: (F, J, 3) joint positions in world space
            tips: dict of joint_index → tip position arrays
            working_probs: (F, J) contact probabilities
            options: SMPLProcessingOptions
        
        Returns:
            float: current inferred floor height
        """
        yd = getattr(self, 'internal_y_dim', 1)
        n_joints = world_pos.shape[1]
        
        # Compute current joint heights (using tips where available)
        curr_heights = np.zeros(n_joints)
        for j in range(n_joints):
            if j in tips:
                curr_heights[j] = tips[j][0, yd] if tips[j].ndim > 1 else tips[j][yd]
            elif j < world_pos.shape[1]:
                curr_heights[j] = world_pos[0, j, yd]
        
        # --- Adaptive floor height estimation ---
        # Maintain a running estimate of the actual floor height from
        # the lowest confirmed-contact joints.
        _floor_age = getattr(self, '_floor_adapt_age', 0)
        if _floor_age < 60:
            # First ~1 second: converge quickly to planted-foot height
            FLOOR_ALPHA = 0.15
            FLOOR_MAX_CHANGE = 0.01   # 10mm per frame
        else:
            FLOOR_ALPHA = 0.02        # Very slow EMA
            FLOOR_MAX_CHANGE = 0.002  # Max 2mm per frame
        self._floor_adapt_age = _floor_age + 1
        
        if not hasattr(self, '_inferred_floor_height') or self._inferred_floor_height is None:
            # Initialize from minimum foot height (not from options)
            foot_joints = [j for j in [10, 11] if j < len(curr_heights)]
            if foot_joints:
                self._inferred_floor_height = min(
                    curr_heights[j] for j in foot_joints)
            else:
                self._inferred_floor_height = options.floor_height
        
        # Update floor estimate from high-confidence contact joints
        all_contact_joints = [7, 8, 10, 11]
        for vi in [24, 25, 28, 29]:
            if vi < n_joints:
                all_contact_joints.append(vi)
        
        confirmed_heights = []
        for j in all_contact_joints:
            if j < working_probs.shape[1] and working_probs[0, j] > 0.5:
                confirmed_heights.append(curr_heights[j])
        
        # LogOdds contacts: include ALL active groups (hands, knees,
        # pelvis, etc.) so the floor adapts when body is on the ground
        # with legs in the air.
        lo_result = getattr(self, '_logodds_result', None)
        if lo_result is not None and hasattr(lo_result, 'intensity'):
            from dpg_system.contact_logodds import get_active_groups
            for gname, joints in get_active_groups(options).items():
                if lo_result.intensity.get(gname, 0.0) > 0.5:
                    valid = [j for j in joints if j < len(curr_heights)]
                    if valid:
                        # Use lowest joint in the group
                        min_h = min(curr_heights[j] for j in valid)
                        confirmed_heights.append(min_h)

        # Fallback for stability/equilibrium/unified methods: use computed pressure
        if not confirmed_heights:
            stab_press = getattr(self, '_stability_computed_pressure', None)
            if stab_press is not None:
                for j in all_contact_joints:
                    if j < len(stab_press) and stab_press[j] > 1.0:
                        confirmed_heights.append(curr_heights[j])
        
        # Secondary fallback: track minimum foot height (very slow)
        # This ensures floor adapts even during single-foot stance
        if not confirmed_heights:
            foot_joints = [j for j in [10, 11] if j < len(curr_heights)]
            if foot_joints:
                min_foot_h = min(curr_heights[j] for j in foot_joints)
                # Only adapt toward foot if it's reasonably close to current floor
                if min_foot_h < self._inferred_floor_height + options.floor_tolerance:
                    confirmed_heights.append(min_foot_h)
        
        if confirmed_heights:
            lowest_confirmed = min(confirmed_heights)
            raw_update = (
                self._inferred_floor_height * (1 - FLOOR_ALPHA) +
                lowest_confirmed * FLOOR_ALPHA
            )
            # Clamp change per frame for stability
            delta = raw_update - self._inferred_floor_height
            delta = np.clip(delta, -FLOOR_MAX_CHANGE, FLOOR_MAX_CHANGE)
            self._inferred_floor_height += delta
        
        return self._inferred_floor_height



    def _refine_contacts_from_torque_discontinuity(
            self, contact_probs, torques_vec, world_pos, parents, options, tips):
        """
        Torque-discontinuity contact refinement.
        
        Detects sudden torque jumps between frames and tests whether
        contact changes can explain them:
        - If a new contact caused a torque burst → demote it
        - If a lost contact caused a torque burst → promote it
        - If contacts make no difference → burst is real motion, leave it
        
        Args:
            contact_probs: (F, J) current contact probabilities
            torques_vec: (F, J, 3) active torque vectors from current iteration
            world_pos: (F, J, 3) world positions
            parents: parent index array
            options: processing options
            tips: tip positions dict
            
        Returns:
            refined_probs: (F, J) adjusted contact probabilities
        """
        F, J = contact_probs.shape
        refined = contact_probs.copy()
        
        # Need previous frame torques and contacts for comparison
        if not hasattr(self, '_prev_active_torques') or self._prev_active_torques is None:
            return refined
        if not hasattr(self, '_prev_contact_probs') or self._prev_contact_probs is None:
            return refined
        
        prev_torques = self._prev_active_torques  # (J, 3)
        prev_probs = self._prev_contact_probs      # (J,)
        
        yd = getattr(self, 'internal_y_dim', 1)
        
        for f in range(F):
            curr_torques = torques_vec[f]  # (J, 3)
            curr_probs = contact_probs[f]   # (J,)
            
            # Detect torque discontinuity per joint
            torque_delta = curr_torques[:min(J, prev_torques.shape[0])] - prev_torques[:min(J, curr_torques.shape[0])]
            delta_mag = np.linalg.norm(torque_delta, axis=-1)  # (J,)
            
            # Threshold: scale with dt so it's frame-rate independent
            # A discontinuity of 50 N·m/frame at 120fps = 6000 N·m/s rate
            DISC_THRESHOLD = 30.0  # N·m per frame (tune-able)
            
            # Find joints with significant torque jumps
            problem_joints = np.where(delta_mag > DISC_THRESHOLD)[0]
            
            if len(problem_joints) == 0:
                continue
            
            # For each problem joint, check if contact changes on
            # descendant joints could explain the torque burst
            for pj in problem_joints:
                if pj >= self.target_joint_count:
                    continue
                
                burst_vec = torque_delta[pj]  # (3,) the torque jump
                burst_mag = delta_mag[pj]
                
                # Find descendant joints (joints whose contact forces
                # create torque on this joint via kinematic chain)
                descendants = self._get_descendant_joints(pj, parents, J)
                
                # Check which descendants had significant contact changes
                best_fix = None
                best_reduction = 0.0
                
                for dj in descendants:
                    if dj >= J:
                        continue
                    
                    dp = curr_probs[dj] - prev_probs[min(dj, len(prev_probs)-1)]
                    
                    if abs(dp) < 0.02:  # No meaningful change
                        continue
                    
                    # Estimate the torque contribution of this contact
                    # Contact force ≈ mass_fraction * g * up_direction
                    # Torque on pj ≈ cross(r, F) where r = pos[dj] - pos[pj]
                    if dj in tips:
                        pos_dj = tips[dj][f]
                    else:
                        pos_dj = world_pos[f, dj]
                    pos_pj = world_pos[f, pj]
                    
                    r = pos_dj - pos_pj  # lever arm
                    
                    # Estimate contact force direction (upward = supporting weight)
                    f_dir = np.zeros(3)
                    f_dir[yd] = 1.0  # upward
                    
                    # Approximate torque contribution magnitude
                    # proportional to contact_prob * body_mass_fraction * g
                    mass_frac = curr_probs[dj] * 0.3  # rough approximation
                    f_mag = mass_frac * self.total_mass_kg * 9.81
                    f_vec = f_dir * f_mag
                    
                    tau_est = np.cross(r, f_vec)  # estimated torque from this contact
                    
                    # Does removing/reducing this contact explain the burst?
                    # If the contact appeared (dp > 0) and its torque aligns
                    # with the burst direction → demoting it would reduce burst
                    alignment = np.dot(tau_est, burst_vec) / (burst_mag + 1e-6)
                    
                    if dp > 0.05 and alignment > 0:
                        # New contact appeared and its torque aligns with burst
                        # → likely a false contact causing the burst
                        reduction = min(alignment, burst_mag)
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_fix = ('demote', dj, dp)
                    
                    elif dp < -0.05 and alignment < 0:
                        # Contact disappeared and its absence aligns with burst
                        # → likely a missed contact
                        reduction = min(abs(alignment), burst_mag)
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_fix = ('promote', dj, abs(dp))
                
                # Apply the best fix if it explains a meaningful portion
                if best_fix is not None and best_reduction > DISC_THRESHOLD * 0.3:
                    action, fix_joint, prob_change = best_fix
                    
                    if action == 'demote':
                        # Reduce the contact probability
                        scale = min(1.0, best_reduction / (burst_mag + 1e-6))
                        refined[f, fix_joint] *= (1.0 - scale * 0.5)
                    
                    elif action == 'promote':
                        # Boost the contact probability toward its previous value
                        prev_p = prev_probs[min(fix_joint, len(prev_probs)-1)]
                        scale = min(1.0, best_reduction / (burst_mag + 1e-6))
                        refined[f, fix_joint] = max(
                            refined[f, fix_joint],
                            prev_p * scale * 0.5
                        )
        
        return refined
    
    def _get_descendant_joints(self, joint_idx, parents, J):
        """Get all descendant joints (children, grandchildren, etc.)."""
        descendants = []
        for j in range(J):
            # Walk up from j to see if joint_idx is an ancestor
            curr = j
            while curr != -1 and curr != joint_idx:
                if curr < len(parents):
                    curr = parents[curr]
                else:
                    curr = -1
            if curr == joint_idx and j != joint_idx:
                descendants.append(j)
        return descendants




    def _compute_max_torque_array(self):
        """
        Convert dictionary profile to per-joint max torque array.
        Returns:
            arr (np.array): (24, 3) Max torque vector for each joint.
        """
        arr = np.zeros((24, 3))
        
        # Biometric Estimates (Approximate)
        # Coordinate Systems (SMPL T-Pose):
        # Legs (Hip/Knee/Ankle): Bone along Y. X=Flex/Ext, Y=Twist, Z=Abd/Add.
        # Arms (Shldr/Elbow/Wrist): Bone along X. X=Twist, Y=Flex/Ext, Z=Abd/Add.
        # Spine: Bone along Y. X=Flex/Ext, Y=Twist, Z=LatBend.
        
        # Default fallback (Isotropic)
        default_t = 100.0
        
        for i in range(24):
            name = self.joint_names[i]
            
            # Start with isotropic or specific vector
            val = default_t
            
            if 'pelvis' in name: val = [500.0, 500.0, 500.0]
            elif 'hip' in name: val = [300.0, 300.0, 300.0]
            elif 'knee' in name: val = [250.0, 250.0, 250.0]
            elif 'ankle' in name: val = [100.0, 100.0, 100.0]
            elif 'foot' in name: val = [40.0, 40.0, 40.0]
            
            elif 'spine' in name: val = [250.0, 250.0, 250.0]
            elif 'neck' in name: val = [50.0, 50.0, 50.0]
            elif 'head' in name: val = [50.0, 50.0, 50.0]
            
            elif 'collar' in name: val = [60.0, 40.0, 60.0]      # Scapular pro/retraction
            elif 'shoulder' in name: val = [30.0, 100.0, 60.0]   # X=twist~30, Y=flex/ext~100 (athletic peak), Z=abd~40-60
            elif 'elbow' in name: val = [10.0, 40.0, 8.0]        # Flexion dominant
            elif 'wrist' in name: val = [8.0, 15.0, 10.0]        # Small muscles
            elif 'hand' in name: val = [3.0, 5.0, 3.0]           # Terminal segment
            
            # Feature: User Overrides via set_max_torque (stored in dict)
            # If user set a value in self.max_torques, use it.
            # But keys in dict are generic 'knee', 'hip'. 
            # We must look up by specific joint name or generic key.
            # Efficiency: Reverse lookup?
            # Or iterate dict... slow.
            # Since self.max_torques was populated with defaults before,
            # we should clear the defaults from that dict and only store overrides?
            # Or just check if the dict has a value for this specific key?
            
            # Actually, the previous implementation checked generic keys in order.
            # Let's preserve that logic for Overrides, but use these defaults if missing.
            
            # Check for user overrides in self.max_torques
            # self.max_torques currently contains... whatever we init'd it with.
            # We should probably initialize self.max_torques to these new defaults in __init__ instead?
            # Or just use the defaults here if dict lookup fails.
            
            # Let's map these defaults to the dictionary in __init__ properly, 
            # so user editing the dict works as expected.
            # But this function *reads* the dict.
            
            # Let's just use the priority logic to pull from dict, assuming dict is populated.
            # Update: I will update _compute_max_torque_profile (the dict init) instead!
            # It's cleaner.
            
            arr[i] = val 
            
        return arr

    def _compute_max_torque_profile(self):
        """
        Returns a dictionary of biometric max torque vectors (N-m).
        """
        # Coordinate Systems (SMPL T-Pose):
        # Legs (Hip/Knee/Ankle): Bone Y. X=Flex/Ext, Y=Twist, Z=Abd/Add.
        # Arms (Shldr/Elbow/Wrist): Bone X. X=Twist, Y=Flex/Ext, Z=Abd/Add.
        # Spine: Bone Y. X=Flex/Ext, Y=Twist, Z=LatBend.

        max_t = {
            'pelvis': [500.0, 500.0, 500.0],
            'spine': [250.0, 250.0, 250.0],
            'hip': [300.0, 300.0, 300.0],
            'knee': [250.0, 250.0, 250.0],
            'ankle': [100.0, 100.0, 100.0],
            'foot': [40.0, 40.0, 40.0],
            'neck': [50.0, 50.0, 50.0],
            'head': [50.0, 50.0, 50.0],
            'collar': [60.0, 40.0, 60.0],       # Scapular pro/retraction
            'shoulder': [30.0, 100.0, 60.0],    # X=twist~30, Y=flex/ext~100 (athletic peak), Z=abd~40-60
            'elbow': [10.0, 40.0, 8.0],          # Flexion dominant
            'wrist': [8.0, 15.0, 10.0],          # Small muscles
            'hand': [3.0, 5.0, 3.0]              # Terminal segment
        }
        
        scale = 1.0
        if self.gender == 'female':
            scale = 0.7
            
        return {k: np.array(v) * scale for k, v in max_t.items()}

    # def _compute_max_torque_array(self):
    #     """
    #     Convert dictionary profile to per-joint max torque array.
    #     Returns:
    #         arr (np.array): (24, 3) Max torque vector for each joint.
    #     """
    #     arr = np.zeros((24, 3))
    #     for i in range(24):
    #         name = self.joint_names[i]
    #         max_t = 100.0 # Default scalar
    #
    #         # Priority Logic (matches original process_frame)
    #         if 'pelvis' in name: max_t = self.max_torques.get('pelvis', 500.0)
    #         elif 'hip' in name: max_t = self.max_torques.get('hip', 300.0)
    #         elif 'knee' in name: max_t = self.max_torques.get('knee', 250.0)
    #         elif 'ankle' in name: max_t = self.max_torques.get('ankle', 40.0)
    #         elif 'foot' in name: max_t = self.max_torques.get('foot', 30.0)
    #         elif 'spine' in name: max_t = self.max_torques.get('spine', 400.0)
    #         elif 'neck' in name: max_t = self.max_torques.get('neck', 50.0)
    #         elif 'head' in name: max_t = self.max_torques.get('head', 15.0)
    #         elif 'collar' in name: max_t = self.max_torques.get('collar', 500.0)
    #         elif 'shoulder' in name: max_t = self.max_torques.get('shoulder', 120.0)
    #         elif 'elbow' in name: max_t = self.max_torques.get('elbow', 80.0)
    #         elif 'wrist' in name: max_t = self.max_torques.get('wrist', 20.0)
    #         elif 'hand' in name: max_t = self.max_torques.get('hand', 10.0)
    #
    #         # Handle Vector or Scalar
    #         if np.ndim(max_t) == 0:
    #              arr[i, :] = max_t
    #         else:
    #              # Check length
    #              v = np.array(max_t)
    #              if v.shape == (3,):
    #                   arr[i, :] = v
    #              else:
    #                   arr[i, :] = v[0] # Fallback?
    #
    #     return arr

    def set_max_torque(self, joint_name_filter, value):
        """
        Manually updates max torque for joints matching the filter.
        Args:
            joint_name_filter (str): Substring to match (e.g., 'neck').
            value (float or list[3]): New max torque value/vector.
        """
        count = 0
        # Check value format
        # If passed as string, list, or array from DPG
        processed_val = value
        
        # Try to clean input
        if hasattr(value, '__len__') and not isinstance(value, str):
             if len(value) == 3:
                  processed_val = np.array(value, dtype=float)
             elif len(value) == 1:
                  processed_val = float(value[0])
        else:
             processed_val = float(value)
             
        # 1. Update Profile Dict
        for k in self.max_torques:
            if joint_name_filter in k:
                self.max_torques[k] = processed_val
                count += 1
        
        # 2. Update Cached Array
        if hasattr(self, 'max_torque_array'):
            self.max_torque_array = self._compute_max_torque_array()
            
        return count

    def set_full_max_torque_profile(self, profile_array):
        """
        Set valid max torques for all joints at once.
        Args:
            profile_array: (24, 3) or (24,) array.
        """
        arr = np.array(profile_array)
        
        if arr.shape == (24, 3):
             self.max_torque_array = arr
        elif arr.shape == (24,) or arr.shape == (24, 1):
             # Broadcast to 3D
            if arr.shape == (24, 1): arr = arr[:, 0]
            self.max_torque_array = np.zeros((24, 3))
            self.max_torque_array[:, 0] = arr
            self.max_torque_array[:, 1] = arr
            self.max_torque_array[:, 2] = arr
        else:
             print(f"SMPLProcessor: Invalid max torque profile shape {arr.shape}")

    def _compute_torque_rate_limits(self):
        """
        Compute per-joint torque rate limits based on biomechanical capabilities.
        
        Returns:
            rate_limits (np.array): (24,) Max torque change per frame at 60 FPS (N·m/frame).
            
        Note:
            These values represent the maximum physically plausible rate of torque change.
            Larger joints (hips, spine) can sustain higher absolute torques but change more slowly.
            Smaller joints (wrists, fingers) have lower absolute limits but faster dynamics.
        """
        # Defaults indexed by joint group
        # Values are N·m per frame at 60 FPS reference rate
        defaults = {
            'pelvis': 20.0,
            'hip': 30.0,
            'spine': 15.0,
            'knee': 25.0,
            'ankle': 15.0,
            'foot': 10.0,
            'neck': 10.0,
            'head': 8.0,
            'collar': 12.0,
            'shoulder': 20.0,
            'elbow': 12.0,
            'wrist': 8.0,
            'hand': 5.0
        }
        
        rate_limits = np.zeros(self.target_joint_count)
        
        for i in range(self.target_joint_count):
            name = self.joint_names[i]
            
            # Find matching key by substring
            matched = False
            for key, val in defaults.items():
                if key in name:
                    rate_limits[i] = val
                    matched = True
                    break
            
            if not matched:
                rate_limits[i] = 15.0  # Fallback default
        
        return rate_limits

    def reset_physics_state(self):
        """Reset internal state for frame-by-frame physics calculation."""
        self.prev_pose_aa = None
        self.prev_pose_raw = None # For input spike detection
        self.prev_vel_aa_raw = None # For accel spike detection
        self.prev_acc_raw = None # For sign-flip detection
        self.last_frame_spiked = False # To prevent stuck rejection on fast moves
        self.prev_vel_aa = None
        self.prev_acc_aa = None
        
        # Dual Path (Effort) State
        self.prev_pose_q_effort = None
        self.prev_vel_aa_effort = None
        
        self.prev_world_pos = None # For Linear Velocity calculation
        self.prev_tips = None # For Tip Velocity calculation
        # Vectorized OneEuroFilter for 72 channels (24 joints * 3)
        self.pose_filter = None 
        self.prev_efforts = None # For spike rejection 
        
        # CoM Dynamics
        self.prev_com_pos = None
        self.prev_com_vel = None
        
        # Filter for CoM acceleration (to derive ZMP)
        # Reduced min_cutoff to 4.0 (from 10.0) to reduce jitter while maintaining responsiveness
        self.com_accel_filter = OneEuroFilter(framerate=self.framerate, min_cutoff=4.0, beta=0.05) 
        
        # Output Smoothing State for Contact Weights (Left, Right)
        # Stores the previous w_foot value (0.0 to 1.0)
        self.prev_w_foot = np.array([0.5, 0.5]) # Initialized to mid-foot 
        
        # Pitch History for Velocity Calculation
        self.prev_foot_pitch_arr = {10: 0.0, 11: 0.0} 
        
        # Probabilistic Contact State
        self.prob_prev_world_pos = None
        self.prob_contact_probs = None
        self.prob_smoothed_vel = None
        
        # Torque Rate Limiter State
        self.prev_dynamic_torque = None  # (24, 3) previous frame's dynamic torque vectors
        self.torque_rate_limits = self._compute_torque_rate_limits()  # (24,) per-joint limits 
        
        # Jitter Detection State
        # Track sign of torque components over a short window to detect oscillation
        JITTER_WINDOW = 8  # frames to track
        self.torque_sign_history = None  # (window, joints, 3) circular buffer of signs
        self.jitter_history_idx = 0  # current write position in circular buffer
        
        # SmartClampKF Filter for Dynamic Torque Smoothing
        self.dynamic_torque_kf = None  # Initialized on first use
        
        # Noise Statistics Tracking
        self.noise_stats = NoiseStats()
        
        # Consensus Contact Detection State
        self._prev_torque_vecs = None
        self._prev_active_torques = None
        self._prev_contact_probs = None

        self._prev_joint_heights = None
        self._inferred_floor_height = None
        if hasattr(self, '_consensus'):
            self._consensus.reset()

        # --- Coherence-gate / adaptive-front-end state ---
        # These carry cross-frame memory and MUST be cleared per file/sequence.
        # The cumulative-mean normalisation (_cg_speed_sum/_cnt) is the critical
        # one: unlike the short-window rings it NEVER self-corrects, so a second
        # file loaded into the same processor would be normalised against the
        # previous file's speed statistics. Each read site re-inits on None /
        # shape-mismatch, so clearing the holders is sufficient.
        self._cg_prev_aa = None
        self._cg_speed_sum = None
        self._cg_speed_cnt = 0
        self._cg_env_ring = None
        self._cg_env_ptr = 0
        self._cg_env_cnt = 0
        self._cg_gate_smooth = None
        self._cg_gate = None
        self._frontend_oef = None
        self._frontend_oef_n = 0
        self._input_smooth_ring = None

        # --- Adaptive-effort EMA state (same cross-file concern) ---
        self._adapt_smooth_tv = None
        self._adapt_smooth_en = None
        self._adapt_smooth_deff_ring = None
        self._adapt_smooth_deff_ptr = 0
        self._adapt_smooth_deff_cnt = 0
        self._per_joint_alpha_max = None

        # Robust catch-all: remove every attribute created since __init__ (all
        # per-frame streaming state — kinematics rings, CoM/gravity/contact
        # filters, coherence-gate normalisation, OEF, precomputed lazy tables).
        # Each re-inits lazily on the next frame, so the processor becomes
        # indistinguishable from a fresh one. Subsumes the explicit clears above
        # (kept for documentation of the load-bearing gate-normalisation reset).
        snap = getattr(self, '_init_attr_snapshot', None)
        if snap is not None:
            for _k in [k for k in self.__dict__ if k not in snap]:
                delattr(self, _k)

    def reset_noise_stats(self):
        """Reset noise statistics for a new file evaluation."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            self.noise_stats.reset()
        else:
            self.noise_stats = NoiseStats()
    
    def get_noise_report(self):
        """Get cumulative noise statistics report."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            return self.noise_stats.get_report()
        return None
    
    def get_noise_score(self):
        """Get current noise score (0-100)."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            return self.noise_stats.get_noise_score()
        return 0.0


    def _get_hierarchy(self):
        """Defines parent-child relationships for SMPL 24 joints + Virtual."""
        if hasattr(self, 'parents'):
             return self.parents
             
        # Fallback (Should not happen if __init__ run)
        parents = [-1,  0,  0,  0,  1,  2, 
                    3,  4,  5,  6,  7,  8, 
                    9,  9,  9, 12, 13, 14, 
                   16, 17, 18, 19, 20, 21,
                   10, 11, 22, 23, 7, 8] # Included 28, 29 (Heels). 27->23.
        return parents

    def _precompute_inertia_tables(self):
        """
        Pre-compute static inertia lookup tables at init time.
        Eliminates redundant tree traversals during per-frame processing.
        """
        parents = self._get_hierarchy()
        limb_masses = self.limb_data['masses']
        limb_lengths = self.limb_data['lengths']
        
        # Build children lookup
        children_of = {i: [] for i in range(30)}
        for i in range(30):
            p = parents[i]
            if p >= 0:
                children_of[p].append(i)
        self._hierarchy_children = children_of
        
        # Build subtree membership for each joint (including self)
        self._subtree_members = {}
        for j in range(24):
            members = []
            stack = [j]
            while stack:
                curr = stack.pop()
                if curr < 24:
                    members.append(curr)
                stack.extend(children_of.get(curr, []))
            self._subtree_members[j] = members
        
        # Per-segment mass and length arrays (24,)
        seg_mass = np.zeros(24)
        seg_length = np.zeros(24)
        seg_is_leaf_skip = np.zeros(24, dtype=bool)  # True for 'hand' joints (mass handled by wrist)
        
        for idx in range(24):
            name = self.joint_names[idx]
            m = 0.0
            l = 0.0
            
            if 'pelvis' in name: m, l = limb_masses['pelvis'], 0.1
            elif 'hip' in name: m, l = limb_masses['upper_leg'], limb_lengths['upper_leg']
            elif 'knee' in name: m, l = limb_masses['lower_leg'], limb_lengths['lower_leg']
            elif 'ankle' in name: m, l = limb_masses['foot'], limb_lengths['foot']
            elif 'spine' in name: m, l = limb_masses['spine'], limb_lengths['spine_segment']
            elif 'neck' in name: m, l = limb_masses.get('head', 1.0)*0.2, limb_lengths['neck']
            elif 'head' in name: m, l = limb_masses['head']*0.8, limb_lengths['head']
            elif 'collar' in name: m, l = limb_masses['upper_arm']*0.2, 0.1
            elif 'shoulder' in name: m, l = limb_masses['upper_arm']*0.8, limb_lengths['upper_arm']
            elif 'elbow' in name: m, l = limb_masses['lower_arm'], limb_lengths['lower_arm']
            elif 'wrist' in name: m, l = limb_masses['hand'], limb_lengths['hand']
            elif 'hand' in name:
                seg_is_leaf_skip[idx] = True
                continue
            elif 'foot' in name: m, l = limb_masses['foot'] * 0.4, 0.08
            
            seg_mass[idx] = m
            seg_length[idx] = l
        
        self._seg_mass = seg_mass
        self._seg_length = seg_length
        self._seg_local_inertia = (1.0/12.0) * seg_mass * (seg_length**2)
        self._seg_is_leaf_skip = seg_is_leaf_skip
        self._seg_has_children = np.array([len(children_of.get(i, [])) > 0 for i in range(24)])
        
        # For leaf nodes, store parent index for fallback direction
        self._seg_leaf_parent = np.array([parents[i] if not self._seg_has_children[i] else -1 for i in range(24)])
    
    def _compute_all_subtree_inertias(self, world_positions):
        """
        Compute effective moment of inertia for all target joints in one call.
        Uses precomputed tables from _precompute_inertia_tables.
        
        Vectorized: eliminates nested Python loops by pre-flattening the
        (pivot, segment) pairs into flat arrays and processing all at once.
        
        Args:
            world_positions: (F, J, 3) world positions
            
        Returns:
            inertias: (F, target_joint_count) effective inertia per joint
        """
        # Lazily build the flattened scatter tables on first call
        if not hasattr(self, '_inertia_scatter_built'):
            self._build_inertia_scatter_tables()
        
        F = world_positions.shape[0]
        
        # --- Step 1: Compute all segment COMs (batch) ---
        # All positions including virtual joints
        J_pos = world_positions.shape[1]
        
        # Start with com = joint position (default for segments without special handling)
        # Start with com = joint position (default for segments without special handling)
        # Non-leaf COMs: midpoint between joint and mean of children (vectorized)
        end_pos_all = self._compute_nonleaf_end_positions(world_positions)  # (F, 24, 3)
        seg_com = (world_positions[:, :24, :] + end_pos_all) * 0.5  # (F, 24, 3)
        
        # Leaf COMs: joint + normalize(joint - parent) * (length / 2)
        if len(self._leaf_indices) > 0:
            leaf_pos = world_positions[:, self._leaf_indices, :]      # (F, n_leaf, 3)
            parent_pos = world_positions[:, self._leaf_parents, :]    # (F, n_leaf, 3)
            dir_vec = leaf_pos - parent_pos                    # (F, n_leaf, 3)
            norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)  # (F, n_leaf, 1)
            norm_safe = np.maximum(norm, 1e-6)
            dir_n = dir_vec / norm_safe                        # (F, n_leaf, 3)
            seg_com[:, self._leaf_indices, :] = leaf_pos + dir_n * self._leaf_half_lengths[np.newaxis, :, np.newaxis]
        
        # --- Step 2: Compute r² for all (pivot, segment) pairs ---
        pivot_pos = world_positions[:, self._scatter_pivots, :]  # (F, K, 3)
        seg_com_k = seg_com[:, self._scatter_segs, :]            # (F, K, 3)
        
        r_vec = seg_com_k - pivot_pos                             # (F, K, 3)
        r_sq = np.sum(r_vec * r_vec, axis=-1)                    # (F, K)
        
        # Parallel axis theorem: I = I_local + m * r²
        pair_inertia = self._scatter_i_local + self._scatter_mass * r_sq  # (F, K)
        
        # --- Step 3: Scatter-add to per-pivot totals ---
        inertias = np.zeros((F, self.target_joint_count))
        np.add.at(inertias, (slice(None), self._scatter_out_idx), pair_inertia)
        
        return inertias
    
    def _build_inertia_scatter_tables(self):
        """Pre-build flat arrays for vectorized inertia computation.
        
        Flattens the nested (pivot_joint, subtree_segment) structure into
        parallel arrays so the entire computation can be done without Python loops.
        """
        # --- Non-leaf segment tables (for COM computation) ---
        nl_indices = []
        nl_child_starts = []
        nl_child_ends = []
        nl_child_flat = []
        
        for idx in range(24):
            if self._seg_is_leaf_skip[idx] or self._seg_mass[idx] <= 0:
                continue
            if self._seg_has_children[idx]:
                children = self._hierarchy_children[idx]
                nl_indices.append(idx)
                nl_child_starts.append(len(nl_child_flat))
                nl_child_flat.extend(children)
                nl_child_ends.append(len(nl_child_flat))
        
        self._nl_indices = nl_indices
        self._nl_child_starts = nl_child_starts
        self._nl_child_ends = nl_child_ends
        self._nl_child_flat = np.array(nl_child_flat, dtype=np.intp) if nl_child_flat else np.array([], dtype=np.intp)
        
        # --- Leaf segment tables (for COM computation) ---
        leaf_indices = []
        leaf_parents = []
        leaf_half_lengths = []
        
        for idx in range(24):
            if self._seg_is_leaf_skip[idx] or self._seg_mass[idx] <= 0:
                continue
            if not self._seg_has_children[idx]:
                p_idx = self._seg_leaf_parent[idx]
                if p_idx != -1:
                    leaf_indices.append(idx)
                    leaf_parents.append(p_idx)
                    leaf_half_lengths.append(self._seg_length[idx] * 0.5)
        
        self._leaf_indices = np.array(leaf_indices, dtype=np.intp)
        self._leaf_parents = np.array(leaf_parents, dtype=np.intp)
        self._leaf_half_lengths = np.array(leaf_half_lengths, dtype=np.float64)
        
        # --- Scatter tables: flatten all (pivot, segment) pairs ---
        scatter_pivots = []
        scatter_segs = []
        scatter_out_idx = []
        scatter_mass = []
        scatter_i_local = []
        
        for j in range(self.target_joint_count):
            for idx in self._subtree_members[j]:
                if self._seg_is_leaf_skip[idx]:
                    continue
                m = self._seg_mass[idx]
                if m <= 0:
                    continue
                scatter_pivots.append(j)
                scatter_segs.append(idx)
                scatter_out_idx.append(j)
                scatter_mass.append(m)
                scatter_i_local.append(self._seg_local_inertia[idx])
        
        self._scatter_pivots = np.array(scatter_pivots, dtype=np.intp)
        self._scatter_segs = np.array(scatter_segs, dtype=np.intp)
        self._scatter_out_idx = np.array(scatter_out_idx, dtype=np.intp)
        self._scatter_mass = np.array(scatter_mass, dtype=np.float64)
        self._scatter_i_local = np.array(scatter_i_local, dtype=np.float64)
        
        # --- Child-mean weight matrix for vectorized end-position computation ---
        # For each of 24 joints, the "end position" is the mean of its children.
        # Build a sparse (24, J_max) weight matrix W such that:
        #   end_pos = W @ all_positions  (matrix multiply over joint dim)
        # For joints with 1 child: W[i, child] = 1.0
        # For joints with N children: W[i, child_k] = 1/N
        # For leaf joints: W[i, :] = 0 (unused)
        J_max = 30  # max joint index including virtual
        W = np.zeros((24, J_max), dtype=np.float64)
        for idx in range(24):
            children = self._hierarchy_children.get(idx, [])
            if children:
                w = 1.0 / len(children)
                for c in children:
                    if c < J_max:
                        W[idx, c] = w
        self._child_mean_weights = W  # (24, J_max)
        # Mask of which joints actually have children (for selective application)
        self._has_children_mask = np.array([len(self._hierarchy_children.get(i, [])) > 0
                                            for i in range(24)])
        
        self._inertia_scatter_built = True
    
    def _compute_nonleaf_end_positions(self, world_positions):
        """Compute end-of-segment positions for all 24 joints in one batched op.
        
        For non-leaf joints: end_pos = mean(children positions)
        For leaf joints: returns the joint's own position (unused but safe).
        
        Args:
            world_positions: (F, J, 3) with J >= 24
        Returns:
            end_pos: (F, 24, 3) end positions for each segment
        """
        F = world_positions.shape[0]
        J = world_positions.shape[1]
        W = self._child_mean_weights[:, :J]  # (24, J) — trim to actual joint count
        
        # Batched: end_pos[f] = W @ world_positions[f]
        # (F, 24, 3) = (24, J) @ (F, J, 3)
        end_pos = np.einsum('ij,fjk->fik', W, world_positions)  # (F, 24, 3)
        
        # For leaf joints (no children), W row is zero → end_pos = 0.
        # Replace with joint position so com = joint (safe fallback).
        leaf_mask = ~self._has_children_mask  # (24,)
        end_pos[:, leaf_mask, :] = world_positions[:, :24, :][:, leaf_mask, :]
        
        return end_pos


    def _compute_subtree_inertia(self, joint_idx, world_positions, world_orientations, limb_lengths, limb_masses):
        """
        Computes the effective moment of inertia of the subtree rooted at joint_idx,
        relative to the joint_idx position.
        
        Approximation:
        - Segments are thin rods.
        - I_local = 1/12 * m * L^2
        - Parallel axis theorem: I_effective = I_local + m * d^2
        
        LEGACY: Kept for backwards compatibility. New code should use
        _compute_all_subtree_inertias() for batch computation.
        """
        parents = self._get_hierarchy()
        children = [i for i, p in enumerate(parents) if p == joint_idx]
        
        # Traverse subtree
        subtree_indices = []
        stack = [joint_idx]
        while stack:
            curr = stack.pop()
            subtree_indices.append(curr)
            # Find children of curr
            curr_children = [i for i, p in enumerate(parents) if p == curr]
            stack.extend(curr_children)
            
        total_inertia = np.zeros(world_positions.shape[0])
        
        joint_pos = world_positions[..., joint_idx, :]
        
        for idx in subtree_indices:
            if idx >= 24: continue
            
            name = self.joint_names[idx]
            
            m = 0.0
            l = 0.0
            
            if 'pelvis' in name: m, l = limb_masses['pelvis'], 0.1
            elif 'hip' in name: m, l = limb_masses['upper_leg'], limb_lengths['upper_leg']
            elif 'knee' in name: m, l = limb_masses['lower_leg'], limb_lengths['lower_leg']
            elif 'ankle' in name: m, l = limb_masses['foot'], limb_lengths['foot']
            elif 'spine' in name: m, l = limb_masses['spine'], limb_lengths['spine_segment']
            elif 'neck' in name: m, l = limb_masses.get('head', 1.0)*0.2, limb_lengths['neck']
            elif 'head' in name: m, l = limb_masses['head']*0.8, limb_lengths['head']
            elif 'collar' in name: m, l = limb_masses['upper_arm']*0.2, 0.1
            elif 'shoulder' in name: m, l = limb_masses['upper_arm']*0.8, limb_lengths['upper_arm']
            elif 'elbow' in name: m, l = limb_masses['lower_arm'], limb_lengths['lower_arm']
            elif 'wrist' in name: m, l = limb_masses['hand'], limb_lengths['hand']
            elif 'hand' in name: continue
            elif 'foot' in name: m, l = limb_masses['foot'] * 0.4, 0.08
            
            if m <= 0: continue

            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            
            com_pos = world_positions[..., idx, :]
            
            if len(child_nodes) > 0:
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            else:
                p_idx = parents[idx]
                if p_idx != -1:
                    dir_vec = joint_pos - world_positions[..., p_idx, :]
                    norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
                    norm_safe = norm.copy()
                    norm_safe[norm_safe < 1e-6] = 1.0
                    dir_vec_normalized = dir_vec / norm_safe
                    
                    is_small = (norm < 1e-6).flatten()
                    if np.any(is_small):
                         dir_vec_normalized[is_small] = np.array([0.0, 0.0, 1.0])
                    
                    dir_vec = dir_vec_normalized
                    
                    com_pos = joint_pos + dir_vec * (l * 0.5)

            r_vec = com_pos - joint_pos
            r_sq = np.sum(r_vec**2, axis=-1)
            
            i_local = (1.0/12.0) * m * (l**2)
            
            total_inertia += (i_local + m * r_sq)
            
        return total_inertia
    
    def _compute_passive_torque(self, joint_name, active_torque_vec, pose_aa):
        """
        Calculates passive ligament torque if joint exceeds soft limits.
        Supports:
        - Symmetric 'Cone' limits (default): Limit on total angle magnitude.
        - Asymmetric 'Hinge' limits: Min/Max on specific axis.
        
        Args:
            joint_name (str): Name to look up limits.
            active_torque_vec (np.array): (F, 3) The active/net torque (used to determine opposition direction).
            pose_aa (np.array): (F, 3) The local pose axis-angle vector.
            
        Returns:
            passive_torque (np.array): (F, 3) Vector opposing the limit violation.
        """
        # Find limits
        limits = self.joint_limits.get('default')
        for key in self.joint_limits:
            if key in joint_name:
                limits = self.joint_limits[key]
                break
        
        limit_type = limits.get('type', 'cone')
        k = limits['k']
        
        if limit_type == 'hinge':
             # Hinge / Single Axis Limit
             axis_idx = limits.get('axis', 0)
             min_val = limits.get('min', -np.pi)
             max_val = limits.get('max', np.pi)
             
             # --- 1. Hinge Axis Logic ---
             angle = pose_aa[:, axis_idx] # (F,)
             torque_val = np.zeros_like(angle)
             
             
             # Check Min Violation (Angle < Min)
             mask_min = angle < min_val
             if np.any(mask_min):
                 deviation = min_val - angle[mask_min]
                 # Linear Stiffness for Hard Stops
                 torque_val[mask_min] = k * deviation
                 
             # Check Max Violation (Angle > Max)
             mask_max = angle > max_val
             if np.any(mask_max):
                 deviation = angle[mask_max] - max_val
                 # Linear Stiffness for Hard Stops
                 torque_val[mask_max] = -k * deviation
                 
             # --- 2. Locked Axes Logic (Structural Support) ---
             # If axes are explicitly locked, they absorb ALL Net Torque (Active = 0).
             # "Structure takes the load".
             locked_axes = limits.get('locked_axes', [])
             
             passive_torque = np.zeros_like(pose_aa)
             
             # Primary Axis (Spring Limit)
             passive_torque[:, axis_idx] = torque_val
             
             
             # Locked Axes (Force Match)
             if locked_axes:
                 for ax in locked_axes:
                     # Passive = Net.  => Active = Net - Passive = 0.
                     passive_torque[:, ax] = active_torque_vec[:, ax]
            
             
             # Fallback for non-locked, non-primary axes?
             # They assume 0 passive torque (Full Active).
             
             return passive_torque
             
        else:
            # Default: Symmetric Cone
            limit_angle = limits['limit']
            
            # Calculate deviation from neutral
            # Magnitude of rotation
            angle = np.linalg.norm(pose_aa, axis=1) # (F,)
            
            # Check excess
            excess = np.maximum(0, angle - limit_angle) # (F,)
            
            # If no excess, return 0
            if np.all(excess == 0):
                return np.zeros_like(pose_aa)
                
            # Passive Torque Magnitude (Exponential Spring)
            # T = k * excess^2 (Non-linear stiffening)
            t_mag = k * (excess ** 2)
            
            # Direction: Opposes the displacement (Restoring force)
            # Axis of rotation is pose_aa / angle
            # Restoring torque acts NEGATIVE to this axis.
            
            # Normalize axis safely
            axis = np.zeros_like(pose_aa)
            mask = angle > 1e-6
            axis[mask] = pose_aa[mask] / angle[mask][:, np.newaxis]
            
            # Passive Torque Vector
            # Opposes displacement -> -axis * magnitude
            passive_torque = -axis * t_mag[:, np.newaxis]
            
            return passive_torque

    def _precompute_passive_limit_tables(self):
        """Pre-build per-joint limit arrays for vectorized passive torque."""
        J = self.target_joint_count
        
        # Arrays for cone-type joints
        self._pl_is_cone = np.zeros(J, dtype=bool)
        self._pl_cone_limit = np.zeros(J)
        self._pl_cone_k = np.zeros(J)
        
        # Arrays for hinge-type joints
        self._pl_is_hinge = np.zeros(J, dtype=bool)
        self._pl_hinge_axis = np.zeros(J, dtype=int)
        self._pl_hinge_min = np.full(J, -np.pi)
        self._pl_hinge_max = np.full(J, np.pi)
        self._pl_hinge_k = np.zeros(J)
        self._pl_hinge_locked = [[] for _ in range(J)]  # list of locked axis lists
        
        for j in range(J):
            name = self.joint_names[j]
            limits = self.joint_limits.get('default')
            for key in self.joint_limits:
                if key in name:
                    limits = self.joint_limits[key]
                    break
            
            limit_type = limits.get('type', 'cone')
            if limit_type == 'hinge':
                self._pl_is_hinge[j] = True
                self._pl_hinge_axis[j] = limits.get('axis', 0)
                self._pl_hinge_min[j] = limits.get('min', -np.pi)
                self._pl_hinge_max[j] = limits.get('max', np.pi)
                self._pl_hinge_k[j] = limits['k']
                self._pl_hinge_locked[j] = limits.get('locked_axes', [])
            else:
                self._pl_is_cone[j] = True
                self._pl_cone_limit[j] = limits['limit']
                self._pl_cone_k[j] = limits['k']
        
        self._passive_limits_precomputed = True
    
    def _compute_passive_torques_batch(self, t_net_all, pose_aa_all):
        """Compute passive torques for all joints at once.
        
        Args:
            t_net_all: (F, J, 3) net torques per joint
            pose_aa_all: (F, J, 3) local pose axis-angle per joint
            
        Returns:
            t_passive: (F, J, 3) passive torques
        """
        if not hasattr(self, '_passive_limits_precomputed'):
            self._precompute_passive_limit_tables()
        
        F, J, _ = t_net_all.shape
        t_passive = np.zeros_like(t_net_all)
        
        # --- Cone joints (batch) ---
        cone_mask = self._pl_is_cone[:J]
        if np.any(cone_mask):
            cone_idx = np.where(cone_mask)[0]
            cone_pose = pose_aa_all[:, cone_idx, :]  # (F, n_cone, 3)
            
            # Rotation angle magnitude
            angle = np.linalg.norm(cone_pose, axis=2)  # (F, n_cone)
            
            # Excess beyond limit
            limits = self._pl_cone_limit[cone_idx]  # (n_cone,)
            k_vals = self._pl_cone_k[cone_idx]  # (n_cone,)
            excess = np.maximum(0, angle - limits[np.newaxis, :])  # (F, n_cone)
            
            # Torque magnitude: k * excess²
            t_mag = k_vals[np.newaxis, :] * (excess ** 2)  # (F, n_cone)
            
            # Direction: -axis (opposes displacement)
            safe_angle = np.where(angle > 1e-6, angle, 1.0)
            axis = cone_pose / safe_angle[:, :, np.newaxis]  # (F, n_cone, 3)
            axis[angle <= 1e-6] = 0.0
            
            cone_passive = -axis * t_mag[:, :, np.newaxis]  # (F, n_cone, 3)
            
            # Zero out where no excess
            cone_passive[excess <= 0] = 0.0
            
            t_passive[:, cone_idx, :] = cone_passive
        
        # --- Hinge joints (per-joint, few of them) ---
        hinge_mask = self._pl_is_hinge[:J]
        if np.any(hinge_mask):
            hinge_idx = np.where(hinge_mask)[0]
            for j in hinge_idx:
                ax = self._pl_hinge_axis[j]
                angle = pose_aa_all[:, j, ax]  # (F,)
                torque_val = np.zeros(F)
                
                k = self._pl_hinge_k[j]
                min_val = self._pl_hinge_min[j]
                max_val = self._pl_hinge_max[j]
                
                mask_min = angle < min_val
                torque_val[mask_min] = k * (min_val - angle[mask_min])
                mask_max = angle > max_val
                torque_val[mask_max] = -k * (angle[mask_max] - max_val)
                
                t_passive[:, j, ax] = torque_val
                
                # Locked axes: passive absorbs all net torque
                for locked_ax in self._pl_hinge_locked[j]:
                    t_passive[:, j, locked_ax] = t_net_all[:, j, locked_ax]
        
        # --- Optimal passive support clipping (all joints) ---
        mask_neg = t_passive < 0
        t_passive[mask_neg] = np.clip(t_net_all[mask_neg], t_passive[mask_neg], 0)
        mask_pos = t_passive >= 0
        t_passive[mask_pos] = np.clip(t_net_all[mask_pos], 0, t_passive[mask_pos])
        
        return t_passive
    def _compute_subtree_com(self, joint_idx, world_positions, limb_lengths, limb_masses):
        """
        Computes the Center of Mass (COM) and Total Mass of the subtree at joint_idx.
        Returns: (total_mass, com_vec_world)
        """
        parents = self._get_hierarchy()
        
        # Traverse subtree (BFS/DFS)
        subtree_indices = []
        stack = [joint_idx]
        while stack:
            curr = stack.pop()
            subtree_indices.append(curr)
            curr_children = [i for i, p in enumerate(parents) if p == curr]
            stack.extend(curr_children)
            
        total_mass = 0.0
        F = world_positions.shape[0]
        weighted_pos_sum = np.zeros((F, 3))
        
        for idx in subtree_indices:
            if idx >= 24: continue
            
            name = self.joint_names[idx]
            
            # Map joint name to limb data keys (simplified)
            m = 0.0
            l = 0.0
            # ... (Limb mapping omitted for brevity, should use same as above or factor out)
            if 'pelvis' in name: m, l = limb_masses['pelvis'], 0.1 
            elif 'hip' in name: m, l = limb_masses['upper_leg'], limb_lengths['upper_leg']
            elif 'knee' in name: m, l = limb_masses['lower_leg'], limb_lengths['lower_leg']
            elif 'ankle' in name: m, l = limb_masses['foot'], limb_lengths['foot']
            elif 'spine' in name: m, l = limb_masses['spine'], limb_lengths['spine_segment']
            elif 'neck' in name: m, l = limb_masses.get('head', 1.0)*0.2, limb_lengths['neck']
            elif 'head' in name: m, l = limb_masses['head']*0.8, limb_lengths['head']
            elif 'collar' in name: m, l = limb_masses['upper_arm']*0.2, 0.1
            elif 'shoulder' in name: m, l = limb_masses['upper_arm']*0.8, limb_lengths['upper_arm']
            elif 'elbow' in name: m, l = limb_masses['lower_arm'], limb_lengths['lower_arm']
            elif 'wrist' in name: m, l = limb_masses['hand'], limb_lengths['hand']
            elif 'hand' in name: continue 
            
            if m <= 0: continue
            
            # Segment COM estimation
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            com_pos = world_positions[..., idx, :]
            if len(child_nodes) > 0:
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            
            total_mass += m
            weighted_pos_sum += m * com_pos
            
        if total_mass > 0:
            final_com = weighted_pos_sum / total_mass
        else:
            final_com = world_positions[..., joint_idx, :]
            
        return total_mass, final_com

    def _compute_full_body_com(self, world_positions):
        """Computes Total Body Center of Mass (F, 3)."""
        if not hasattr(self, '_inertia_scatter_built'):
            self._build_inertia_scatter_tables()
        
        F = world_positions.shape[0]
        
        # Vectorized end positions for all joints
        end_pos_all = self._compute_nonleaf_end_positions(world_positions)  # (F, 24, 3)
        
        # Use cached segment masses from _precompute_inertia_tables
        joint_segment_masses = self._grav_seg_masses if hasattr(self, '_grav_seg_masses') else self._seg_mass
        
        total_mass = 0.0
        weighted_pos_sum = np.zeros((F, 3))
        
        for idx in range(24):
            m = joint_segment_masses[idx]
            if m <= 0:
                continue
            
            # COM = midpoint(joint, end_pos) for non-leaf, joint for leaf
            if self._has_children_mask[idx]:
                com_pos = (world_positions[:, idx, :] + end_pos_all[:, idx, :]) * 0.5
            else:
                com_pos = world_positions[:, idx, :]
            
            total_mass += m
            weighted_pos_sum += m * com_pos
            
        if total_mass > 0:
            return weighted_pos_sum / total_mass
        else:
            return world_positions[..., 0, :] # Fallback to Pelvis
    def _compute_gravity_torques(self, world_pos, options, global_rots=None, tips=None):
        """
        Compute gravity torque vectors for all joints.
        
        Args:
            world_pos (np.array): (F, 24, 3) World positions of joints.
            options (SMPLProcessingOptions): Usage: add_gravity, input_up_axis.
            global_rots (list[R] or F,24,4): Optional rotations for accurate leaf CoM.
            tips (dict): Optional virtual tip positions.
            
        Returns:
            t_grav_vecs (np.array): (F, 24, 3) Gravity torque vectors.
        """
        F = world_pos.shape[0]
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))
        
        if not options.add_gravity and not options.enable_apparent_gravity:
            return t_grav_vecs
            
        J_full = world_pos.shape[1]
        
        # DEBUG MASS ACCUMULATION
        # verify mass logic
        pass
            
        # Determine Gravity Vector
        g_mag = 9.81
        
        # Gravity should align with the INTERNAL simulation frame (usually Y-Up).
        # We respect 'internal_y_dim' if set, otherwise assume Y (1).
        y_dim = getattr(self, 'internal_y_dim', 1)
        
        g_vec = np.zeros(3)
        g_vec[y_dim] = -g_mag
            
        parents = self._get_hierarchy()
        limb_masses = self.limb_data['masses']
        limb_lengths = self.limb_data['lengths']
        
        # 1. Compute Individual Segment Properties (Mass & CoM)
        node_masses = np.zeros((F, J_full))
        node_weighted_com = np.zeros((F, J_full, 3))
        
        # Cache segment masses and children (constant across frames)
        if not hasattr(self, '_grav_seg_masses') or self._grav_seg_masses is None:
            joint_segment_masses = np.zeros(24)
            for idx in range(24):
                name = self.joint_names[idx]
                m = 0.0
                if 'pelvis' in name: m = limb_masses['pelvis']
                elif 'hip' in name: m = limb_masses['upper_leg']
                elif 'knee' in name: m = limb_masses['lower_leg']
                elif 'ankle' in name: m = limb_masses['foot']
                elif 'spine' in name: m = limb_masses['spine']
                elif 'neck' in name: m = limb_masses.get('head', 1.0)*0.2
                elif 'head' in name: m = limb_masses['head']*0.8
                elif 'collar' in name: m = limb_masses['upper_arm']*0.2
                elif 'shoulder' in name: m = limb_masses['upper_arm']*0.8
                elif 'elbow' in name: m = limb_masses['lower_arm']
                elif 'wrist' in name: m = limb_masses['hand']
                joint_segment_masses[idx] = m
            
            children_list = [[] for _ in range(24)]
            for c in range(len(parents)):
                p = parents[c]
                if 0 <= p < 24:
                    children_list[p].append(c)
            
            # Classify joints for vectorized CoM
            self._grav_seg_masses = joint_segment_masses
            self._grav_children = children_list
            self._grav_nonleaf = [idx for idx in range(24) if joint_segment_masses[idx] > 0 and len(children_list[idx]) > 0]
            self._grav_leaf = [idx for idx in range(24) if joint_segment_masses[idx] > 0 and len(children_list[idx]) == 0]
        
        joint_segment_masses = self._grav_seg_masses
        children_list = self._grav_children
        
        # Non-leaf joints: seg_com = midpoint(joint, mean(children)) — vectorized
        if not hasattr(self, '_inertia_scatter_built'):
            self._build_inertia_scatter_tables()
        end_pos_all = self._compute_nonleaf_end_positions(world_pos)  # (F, 24, 3)
        for idx in self._grav_nonleaf:
            m = joint_segment_masses[idx]
            seg_com = (world_pos[:, idx, :] + end_pos_all[:, idx, :]) * 0.5
            node_masses[:, idx] = m
            node_weighted_com[:, idx, :] = seg_com * m
        
        # Leaf joints: need special CoM handling (tips, rotation, parent vector)
        for idx in self._grav_leaf:
            m = joint_segment_masses[idx]
            joint_pos = world_pos[:, idx, :]
            seg_com = joint_pos.copy()
            
            # Priority 1: Virtual Tip (Best for Hands/Feet)
            if tips is not None and idx in tips:
                tip_pos = tips[idx]
                if tip_pos.shape[0] == F:
                     seg_com = (joint_pos + tip_pos) * 0.5
                
            # Priority 2: Joint Rotation (Best for Head / Leaves without tips)
            elif global_rots is not None:
                  name = self.joint_names[idx]
                  ext_len = 0.0
                  if 'head' in name: ext_len = limb_lengths.get('head', 0.20)
                  elif 'hand' in name: ext_len = limb_lengths.get('hand', 0.18)
                  elif 'foot' in name: ext_len = limb_lengths.get('foot', 0.20)
                  
                  if ext_len > 0:
                      local_axis = np.array([0.0, 1.0, 0.0])
                      if isinstance(global_rots, list):
                           r_obj = global_rots[idx]
                           axis_world = r_obj.apply(local_axis)
                           seg_com = joint_pos + axis_world * (ext_len * 0.5)
                      elif hasattr(global_rots, 'shape') and global_rots.ndim == 3 and global_rots.shape[-1] == 4:
                           r_obj = R.from_quat(global_rots[:, idx, :])
                           axis_world = r_obj.apply(local_axis)
                           seg_com = joint_pos + axis_world * (ext_len * 0.5)

            # Priority 3: Fallback to Parent Vector
            else: 
                 parent_idx = parents[idx]
                 if parent_idx >= 0:
                     parent_pos = world_pos[:, parent_idx, :]
                     bone_vec = joint_pos - parent_pos
                     norm = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
                     valid_norm = norm > 1e-6
                     direction = np.zeros_like(bone_vec)
                     np.divide(bone_vec, norm, out=direction, where=valid_norm)
                     
                     name = self.joint_names[idx]
                     ext_len = 0.0
                     if 'head' in name: ext_len = limb_lengths.get('head', 0.20)
                     elif 'hand' in name: ext_len = limb_lengths.get('hand', 0.18)
                     elif 'foot' in name: ext_len = limb_lengths.get('foot', 0.20)
                     
                     seg_com = joint_pos + direction * (ext_len * 0.5)
            
            node_masses[:, idx] = m
            node_weighted_com[:, idx, :] = seg_com * m
            
        # Store for debug/CoM calc
        self.node_masses = node_masses
        self.node_weighted_com = node_weighted_com
            
        # 2. Bottom-Up Accumulation (Subtree Properties)
        # Iterate reverse (Leaves -> Root)
        # Exclude indices >= 24 if they exist? target_joint_count is 22 usually, but mapped to 24 skeleton.
        
        subtree_masses = node_masses.copy()
        subtree_weighted_com = node_weighted_com.copy()
        
        for idx in range(23, 0, -1): # Skip 0 (Root has no parent in array logic for accumulation)
            parent = parents[idx]
            if parent >= 0:
                subtree_masses[:, parent] += subtree_masses[:, idx]
                subtree_weighted_com[:, parent, :] += subtree_weighted_com[:, idx, :]
        
        # 3. Compute Torque
        # T = r_com_rel x F_gravity
        # r_com_rel = (Weighted_Sum / Total_Mass) - Joint_Pos
        
        # Avoid division by zero
        mask = subtree_masses > 0
        
        # Initialize Com Vectors (F, 24, 3)
        subtree_com = np.zeros_like(world_pos)
        subtree_com[mask] = subtree_weighted_com[mask] / subtree_masses[mask][:, np.newaxis]
        
        # For joints with mass=0, torque is 0 (already init to 0)
        
        # Calculate R vector
        r_vecs = subtree_com - world_pos
        
        # Calculate Force (M * g)
        # Broadcaset g_vec (3,) to (F, 24, 3)
        f_grav = subtree_masses[:, :, np.newaxis] * g_vec[np.newaxis, np.newaxis, :]
        
        # Cross Product
        t_grav_vecs = np.cross(r_vecs, f_grav)
        
        # Old Debug Block Removed
        
        # --- Apparent Gravity (Airborne Physics) ---
        # If in freefall, body accelerates with gravity (a ~ -g). Effective weight is 0.
        # If landing, body decelerates (a > 0). Effective weight is > mg.
        # FIX: Use ROOT Position (System Frame) instead of CoM (which includes internal Dynamics).
        # Using CoM causes internal limb swings to trigger global gravity scaling.
        
        # --- Apparent Gravity (Airborne Physics & Locomotion Effort) ---
        # 1. Inertial Force Logic (if enabled)
        
        current_root = world_pos[:, 0, :] # (F, 3)
        g_acc_vec = np.zeros_like(current_root)
        
        if options.enable_apparent_gravity:
            # Calculate Root Acceleration (System Acceleration)
            dt = options.dt
            # current_root already defined
            
            if not hasattr(self, 'prev_root_gravity'):
                 self.prev_root_gravity = current_root.copy()
                 self.prev_vel_gravity = np.zeros_like(current_root)
                 root_acc = np.zeros_like(current_root)
            else:
                 # Handle Teleport (Resets)
                 dist = np.linalg.norm(current_root - self.prev_root_gravity, axis=-1)
                 if np.any(dist > 2.0): 
                      self.prev_root_gravity = current_root.copy()
                      self.prev_vel_gravity = np.zeros_like(current_root)
                      root_acc = np.zeros_like(current_root)
                 else:
                      current_vel = (current_root - self.prev_root_gravity) / dt
                      root_acc = (current_vel - self.prev_vel_gravity) / dt
                      
                      self.prev_root_gravity = current_root.copy()
                      self.prev_vel_gravity = current_vel
                      
            # Construct Effective Gravity Vector: g_eff = g_base - a_root
            # D'Alembert's Principle: Inertial Force = -ma. Equivalent to gravity field -a.
            
            # Clamp acceleration to prevent noise scaling (e.g. +/- 30 m/s^2)
            root_acc = np.clip(root_acc, -30.0, 30.0)
            
            # Rate-of-change limit: max change in acceleration per frame
            # This prevents sudden acceleration spikes from mocap noise
            if hasattr(self, 'prev_root_acc_smooth'):
                dt_acc = 1.0 / self.framerate
                max_acc_change = 15.0 * dt_acc * 120  # 15 m/s^2/s at 120fps baseline
                acc_delta = root_acc - self.prev_root_acc_smooth
                acc_delta_clamped = np.clip(acc_delta, -max_acc_change, max_acc_change)
                root_acc = self.prev_root_acc_smooth + acc_delta_clamped
            
            # Smooth it (Stronger EMA) to remove mocap jitter
            if not hasattr(self, 'prev_root_acc_smooth'):
                 self.prev_root_acc_smooth = np.zeros_like(root_acc)
                 
            alpha_acc = 0.03  # Stronger low-pass filter (3% blend, ~33 frame settling)
            root_acc = self.prev_root_acc_smooth * (1.0 - alpha_acc) + root_acc * alpha_acc
            self.prev_root_acc_smooth = root_acc
            
            g_acc_vec = -root_acc
        else:
             # Reset history if disabled (to avoid stale start if re-enabled?)
             # self.prev_root_gravity = None # Might cause error
             pass
        
        # Combine with Base Gravity (g_vec)
        # g_vec is (3,)
        # g_acc_vec is (F, 3)
        g_vec_eff = g_vec[np.newaxis, :] + g_acc_vec # (F, 3)
        
        # Store for debug (magnitude scaling relative to 9.81)
        g_mag_eff = np.linalg.norm(g_vec_eff, axis=-1)
        self.g_factor = (g_mag_eff[0] / 9.81) if len(g_mag_eff) > 0 else 1.0
        
        # --- Update Gravity Force Vectors ---
        # f_grav was computed using static g_vec at Line 1555.
        # We must RECOMPUTE it using g_vec_eff.
        
        # f_grav = m * g_eff
        # subtree_masses: (F, 24)
        # g_vec_eff: (F, 3) -> (F, 1, 3)
        f_grav = subtree_masses[:, :, np.newaxis] * g_vec_eff[:, np.newaxis, :]
        
        # Recompute Torque (r x f_new)
        # r_vecs: (F, 24, 3)
        t_grav_vecs = np.cross(r_vecs, f_grav)
        
        # We NO LONGER scale by scalar g_factor. The vector logic handles it.
        # Removed: t_grav_vecs *= g_factor_reshaped
        
        # Note: 'f_grav' used for RNE subtraction later is derived from 'g_vec'.
        # We rely on 'contact_pressure' being scaled by GRF factor (process_frame logic) 
        # to scale the GRF term. 
        # So both T_grav and T_grf scale together. 
        
        # Optimization: Cache Full Body CoM (Root Subtree) for Floor Contact logic
        # subtree_com[:, 0, :] is the CoM of the whole tree (rooted at 0)
        # Note: If F=1, simple assignment. If F>1, whole array.
        self.current_com = subtree_com[:, 0, :] # (F, 3)
        self.current_total_mass = subtree_masses[:, 0] # (F,)
        
        # --- 4. Standing Correction (Ground Reaction Gravity) ---
        # Standard gravity calc treats Hips as "Roots" of the Legs (calculating lift torque).
        # When Standing, Hips support the Upper Body (calculating load torque).
        # We blend between these models based on Foot Contact Probability.
        
        # --- 4. Generalized Ground Reaction Forces (RNE) ---
        # Instead of heuristic "Support Logic", we calculate the actual Moment of Ground Reaction Forces.
        # Torque_net = Torque_gravity - Torque_GRF
        
        if hasattr(self, 'contact_pressure') and self.contact_pressure is not None:
             # 1. Compute GRF Vectors (With Friction/Alignment)
             # Instead of Pure Vertical GRF (High Torque in wide stance), we align GRF with Support Leg Axis.
             
             contact_masses = self.contact_pressure # (F, J)
             if contact_masses.ndim == 1: contact_masses = contact_masses[np.newaxis, :]
             
             f_grf_vecs = np.zeros_like(world_pos)
             
             # Leg Mapping: Toe/Ankle -> Hip
             # Arm Mapping: Wrist/Hand -> Shoulder (Fix Cartwheel Effort)
             leg_map = {
                 10: 1, 7: 1, 11: 2, 8: 2,           # Legs -> Hips
                 20: 16, 22: 16, 21: 17, 23: 17,     # Arms -> Shoulders
                 24: 1, 25: 2,                       # Toes -> Hips (if present)
                 26: 16, 27: 17,                     # Fingers -> Shoulders (if present)
                 28: 1, 29: 2                        # Heels -> Hips (Virtual, Index 28/29)
             }
             
             # Base Vertical Force (Magnitude)
             # contact_masses is Kg. Force = kg * g_eff.
             # Use g_vec_eff magnitude for scaling
             # Base Vertical Force (Magnitude)
             # contact_masses is Kg. Force = kg * g_eff.
             # Use g_vec_eff magnitude for scaling
             g_mag = np.linalg.norm(g_vec_eff, axis=-1, keepdims=True) + 1e-9 # (F, 1)
             f_vert_mag = contact_masses * g_mag # (F, J)
             
             # Loop over joints (Vectorized over F)
             for j in range(world_pos.shape[1]):
                 if np.any(contact_masses[:, j] > 0):
                      # All contact GRF is vertical (upward).
                      # This ensures the total vertical force propagating up the
                      # leg is independent of the ankle/foot pressure split.
                      # Ankle torque arises from the foot's GRF lever arm in the
                      # backward pass; no special alignment needed.
                      dir_up = -g_vec_eff / g_mag
                      f_grf_vecs[:, j, :] = f_vert_mag[:, j, np.newaxis] * dir_up

             # 2. Backward Pass (Accumulate GRF Moments)
             f_cum_grf = f_grf_vecs.copy()
             m_cum_grf = np.zeros_like(f_grf_vecs)
             
             parents = np.array(self._get_hierarchy())
             J = world_pos.shape[1]
             
             # Pre-compute all bone vectors: r_bone[j] = pos[j] - pos[parent[j]]
             # This avoids repeated slicing in the loop.
             parent_indices = np.clip(parents[:J], 0, J-1)  # clip -1 to 0 (root, unused)
             r_bones = world_pos[:, :J, :] - world_pos[:, parent_indices, :]  # (F, J, 3)
             
             for j in range(J-1, 0, -1):
                 parent = parents[j]
                 if parent >= 0:
                      # Propagate Force
                      f_child = f_cum_grf[:, j, :]
                      f_cum_grf[:, parent, :] += f_child
                      
                      # Propagate Moment: M_parent += M_child + r_bone x f_child
                      # Inline cross product (avoids numpy dispatch overhead)
                      rb = r_bones[:, j, :]  # (F, 3)
                      t_lever_0 = rb[:, 1] * f_child[:, 2] - rb[:, 2] * f_child[:, 1]
                      t_lever_1 = rb[:, 2] * f_child[:, 0] - rb[:, 0] * f_child[:, 2]
                      t_lever_2 = rb[:, 0] * f_child[:, 1] - rb[:, 1] * f_child[:, 0]
                      
                      m_cum_grf[:, parent, 0] += m_cum_grf[:, j, 0] + t_lever_0
                      m_cum_grf[:, parent, 1] += m_cum_grf[:, j, 1] + t_lever_1
                      m_cum_grf[:, parent, 2] += m_cum_grf[:, j, 2] + t_lever_2
                      
             # Combine with Gravity (Opposing)
             # Gravity Moment (t_grav) and GRF Moment (m_cum) should naturally oppose.
             # e.g. Gravity pulls CoM down (Adduction), GRF pushes Foot up (Abduction).
             # Signs should be opposite, so Addition is correct.
             t_grav_vecs += m_cum_grf
             
             # --- Structural Support Damping Removed ---
             # We rely on Friction Alignment (above) to stabilize stance.
             
             # --- Global CoP Stability Model ---
             # REVERTED: We now rely on correct ZMP-based Force Distribution to handle balance.
             # Pure Physics: Net Torque = Gravity Torque + Support Torque.
             # If balanced, these cancel naturally.
        
        return t_grav_vecs

    def _compute_floor_contacts_RENAMED(self, world_pos, pose_data_aa, options):
        """
        Simplified Floor Contact Logic (State-based).
        
        Rules:
        1. Entry: Height < Tolerance AND Velocity_Down approx 0 (Stopped).
        2. Exit: Height > Exit_Tolerance OR Velocity_Horiz > Slip_Limit.
        3. Distribution: ZMP-based Gaussian weighting.
        4. Geometry: Heel/Toe priority band.
        """
        F = world_pos.shape[0]
        if not options.floor_enable:
             return np.zeros((F, 24), dtype=np.float32)
             
        # Use tracked internal dimension (defaults to 1, but 2 if unrotated Z-up)
        y_dim = getattr(self, 'internal_y_dim', 1)
        dt = options.dt
        floor_height = options.floor_height
        tol_enter = options.floor_tolerance
        tol_exit = tol_enter * 1.5 # Hysteresis
        
        # --- 1. Compute Velocities ---
        if self.prev_world_pos is None:
             self.prev_world_pos = world_pos[0].copy() if world_pos.ndim == 3 else world_pos.copy()
             
        lin_vel = np.zeros_like(world_pos)
        if F > 1:
             lin_vel[0] = (world_pos[0] - self.prev_world_pos) / dt
             lin_vel[1:] = (world_pos[1:] - world_pos[:-1]) / dt
             self.prev_world_pos = world_pos[-1].copy()
             pass
        else:
             lin_vel = (world_pos - self.prev_world_pos) / dt
             self.prev_world_pos = world_pos.copy()
             
        vel_y = lin_vel[..., y_dim] # (F, 24) or (24,)
        
        vel_h_vec = lin_vel.copy()
        vel_h_vec[..., y_dim] = 0.0
        vel_h = np.linalg.norm(vel_h_vec, axis=-1)
        
        # --- 2. Determine Contact Candidates (State Machine) ---
        # Initialize with history if available
        if self.prev_contact_mask is None:
            self.prev_contact_mask = np.zeros(24, dtype=bool)
        
        # We need a boolean mask for logic
        current_contact_state = np.zeros((F, 24), dtype=bool)
        
        # Iterate (Vectorized per frame usually, but here we loop f for clarity/state)
        # Actually, for batch F>1, state propagation is tricky. 
        # But usually F=1 in streaming.
        
        curr_state = self.prev_contact_mask.astype(bool) # (24,)
        
        # Tip Positions (Heels/Toes/Hands)
        # Need to use 'temp_tips' which are computed in forward kinematics
        # But we need them aligned with world_pos frame index.
        # process_frame stores self.temp_tips (dict of joint -> pos array)
        
        # Ensure temp_tips exists
        tips_available = hasattr(self, 'temp_tips') and self.temp_tips is not None
        
        for f in range(F):
            # Per-Joint Checking
            for j in range(24):
                # Use Tip Position if available
                pos = world_pos[f, j, :]
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                    
                h = pos[y_dim] - floor_height
                vy = vel_y[f, j]
                vh = vel_h[f, j]
                
                was_contact = curr_state[j]
                
                if was_contact:
                    # EXIT Conditions
                    # 1. Lifted
                    if h > tol_exit:
                        curr_state[j] = False
                    # 2. Sliding (Slip)
                    elif vh > 0.5: # 0.5 m/s slip limit
                        curr_state[j] = False
                    # 3. Active Lift (Jump/Rebound)
                    elif vy > 0.1: # Rising fast (> 10cm/s) -> Break Contact immediately
                        curr_state[j] = False
                    else:
                        # Sustain
                        curr_state[j] = True
                else:
                    # ENTRY Conditions
                    # 1. In Zone
                    if h < tol_enter:
                        # 2. Descent Stopped (Soft Landing)
                        # vy should be close to 0 or positive (rising CHECK)
                        # allow slight negative drift (-0.05) due to noise
                        # AND Prevent Re-Entry if Rising Fast (Oscillation Fix)
                        # Hysteresis: Entry < 0.05, Exit > 0.1.
                        if vy > -0.05 and vy < 0.05: 
                            curr_state[j] = True
                            
            # --- 3. Heel/Toe Geometric Resolution ---
            # If both Heel and Toe are candidates, apply geometric priority.
            # Left: Heel(7), Toe(10). Right: Heel(8), Toe(11).
            for (heel, toe) in [(7, 10), (8, 11)]:
                # Get Heights
                if tips_available and heel in self.temp_tips and toe in self.temp_tips:
                    h_heel = self.temp_tips[heel][f, y_dim]
                    h_toe = self.temp_tips[toe][f, y_dim]
                else:
                    # Fallback to joint positions
                    h_heel = world_pos[f, heel, y_dim]
                    h_toe = world_pos[f, toe, y_dim]

                # Diff: Positive = Heel Higher
                diff = h_heel - h_toe
                band = 0.05 # 5cm blending band
                
                r_heel = 0.5
                r_toe = 0.5
                
                if diff > band: # Heel is > 5cm above Toe -> Toe takes all
                    r_heel = 0.0
                    r_toe = 1.0
                elif diff < -band: # Toe is > 5cm above Heel -> Heel takes all
                    r_heel = 1.0
                    r_toe = 0.0
                else:
                    # Blend [-band, +band] -> [0, 2*band]
                    # t goes 0 (Heel Only) to 1 (Toe Only)
                    t = (diff + band) / (2.0 * band)
                    r_toe = t
                    r_heel = 1.0 - t
                    
                # Apply strictly? 
                # If a contact is OFF, it stays OFF.
                # If both are ON, we modulate them? 
                # Actually, the user wants "Heel gets no contact" if lifted.
                # So we act on the boolean state? 
                # Better: We keep boolean state as "Physical Candidate", 
                # and use these ratios for Mass Distribution Weighting.
                
                # Store ratios for this frame/pair to use in Weighting step
                # We can hack this by storing in a temporary dict
                pass 
                
            current_contact_state[f] = curr_state.copy()
            
        # Update History
        self.prev_contact_mask = curr_state.astype(float) # Store as float mask for compatibility
        
        # --- 4. Mass Distribution (ZMP) ---
        contact_masses = np.zeros((F, 24), dtype=np.float32)
        total_mass = self.total_mass_kg
        
        # Compute Ref Point (ZMP)
        if self.current_com is not None:
             com = self.current_com # (F, 3)
        else:
             com = self._compute_full_body_com(world_pos)
             
        # Compute COM Accel (Horizontal)
        # Use existing filter state
        com_vel = np.zeros_like(com)
        if F > 1:
            com_vel[1:] = (com[1:] - com[:-1]) / dt
            if self.prev_com_pos is not None: com_vel[0] = (com[0] - self.prev_com_pos) / dt
        elif self.prev_com_pos is not None:
            com_vel = (com - self.prev_com_pos) / dt
            
        accel_h = np.zeros_like(com) # Simplified: Ignore acceleration for stability first?
        # User requested: "Use center of mass and momentum"
        # ZMP = CoM - (h/g)*accel
        g = 9.81
        h_com = np.maximum(com[..., y_dim] - floor_height, 0.1)
        
        # Calculate Accel for ZMP
        if self.prev_com_vel is not None:
             accel = (com_vel - self.prev_com_vel) / dt
             accel_h = accel.copy()
             accel_h[..., y_dim] = 0.0
             
        zmp_offset = - (h_com[:, np.newaxis] / g) * accel_h
        target_point = com + zmp_offset
        
        # Explicitly set total_mass for distribution
        total_mass = self.total_mass_kg
        
        # DEBUG MASS
        # print(f"DEBUG: Inside _compute_floor_contacts. total_mass_kg: {self.total_mass_kg}")
        raise RuntimeError(f"CRASH TEST: Total Mass is {self.total_mass_kg}")
        
        # Update COM State
        if F > 0:
             self.prev_com_pos = com[-1].copy()
             self.prev_com_vel = com_vel[-1].copy()

        # Distribute
        for f in range(F):
            # Active Indices
            active_indices = np.where(current_contact_state[f])[0]
            if len(active_indices) == 0:
                continue
                
            weights = []
            
            # Recalculate Heel/Toe Ratios for this frame
            # (Repeating logic for clarity inside loop)
            geo_weights = np.ones(24)
            for (heel, toe) in [(7, 10), (8, 11)]:
                # Get Heights
                if tips_available and heel in self.temp_tips and toe in self.temp_tips:
                    h_heel = self.temp_tips[heel][f, y_dim]
                    h_toe = self.temp_tips[toe][f, y_dim]
                else:
                    h_heel, h_toe = world_pos[f, heel, y_dim], world_pos[f, toe, y_dim]
                
                diff = h_heel - h_toe
                band = 0.05
                if diff > band: 
                    geo_weights[heel] = 0.0
                    geo_weights[toe] = 1.0
                elif diff < -band:
                    geo_weights[heel] = 1.0
                    geo_weights[toe] = 0.0
                else:
                    t = (diff + band) / (2.0 * band)
                    geo_weights[toe] = t
                    geo_weights[heel] = 1.0 - t
            
            # Distance Weights
            sigma = 0.15 # Sharp (was 0.5)
            
            for idx in active_indices:
                # Position
                pos = world_pos[f, idx, :]
                if tips_available and idx in self.temp_tips:
                    pos = self.temp_tips[idx][f]
                    
                # Dist to ZMP
                dist_vec = pos - target_point[f]
                dist_vec[y_dim] = 0.0
                d_sq = np.sum(dist_vec**2)
                
                w_dist = np.exp(-d_sq / (2 * sigma**2))
                
                # Combine with Geometric Weight
                w_final = w_dist * geo_weights[idx]
                weights.append(w_final)
            
            total_w = sum(weights)
            if total_w > 1e-4:
                for i, idx in enumerate(active_indices):
                    contact_masses[f, idx] = total_mass * (weights[i] / total_w)
            
            # --- Refine Foot/Ankle Balance (Post-Pass: ZMP Projection) ---
            for (idx_a, idx_f) in [(7, 10), (8, 11)]:
                m_a = contact_masses[f, idx_a]
                m_f = contact_masses[f, idx_f]
                total_pair = m_a + m_f
                
                if total_pair < 0.1:
                    continue

                # 1. Geometric Validity (Height Logic)
                # Re-evaluate validity based on the 5cm band
                h_heel = 0.0
                h_toe = 0.0
                if tips_available and idx_f in self.temp_tips and idx_a in self.temp_tips:
                     h_heel = self.temp_tips[idx_a][f, y_dim]
                     h_toe = self.temp_tips[idx_f][f, y_dim]
                else:
                     h_heel = world_pos[f, idx_a, y_dim]
                     h_toe = world_pos[f, idx_f, y_dim]
                     
                diff = h_heel - h_toe
                band = 0.05
                
                # Default: Both valid
                valid_heel = 1.0
                valid_toe = 1.0
                
                if diff > band: # Heel High
                    valid_heel = 0.0
                    valid_toe = 1.0
                elif diff < -band: # Toe High
                    valid_heel = 1.0
                    valid_toe = 0.0
                else: # Mixing Band
                    t_geo = (diff + band) / (2.0 * band)
                    valid_toe = t_geo
                    valid_heel = 1.0 - t_geo
                    
                # 2. ZMP Demand (Project ZMP onto Foot Axis)
                # Use 'target_point' which includes CoM + Momentum (ZMP)
                zmp = target_point[f]
                
                plane_dims = [d for d in [0, 1, 2] if d != y_dim]
                p_heel = world_pos[f, idx_a, plane_dims]
                p_toe = world_pos[f, idx_f, plane_dims]
                p_zmp = zmp[plane_dims]
                
                if tips_available and idx_f in self.temp_tips and idx_a in self.temp_tips:
                     p_heel = self.temp_tips[idx_a][f][plane_dims]
                     p_toe = self.temp_tips[idx_f][f][plane_dims]

                vec_ht = p_toe - p_heel
                vec_hz = p_zmp - p_heel
                
                dist_sq = np.sum(vec_ht**2)
                demand_toe = 0.5
                demand_heel = 0.5
                
                if dist_sq > 1e-6:
                    # Projection
                    t_zmp = np.dot(vec_hz, vec_ht) / dist_sq
                    # Clamp [0, 1] means strictly between heel/toe.
                    t_zmp = np.clip(t_zmp, 0.0, 1.0)
                    
                    demand_toe = t_zmp
                    demand_heel = 1.0 - t_zmp
                
                # 3. Combine
                # Start with 'demand' (Physics preference), gated by 'validity' (Geometry)
                
                share_toe = demand_toe * valid_toe
                share_heel = demand_heel * valid_heel
                
                total_share = share_toe + share_heel
                
                if total_share > 1e-4:
                    normalized_toe = share_toe / total_share
                    normalized_heel = share_heel / total_share
                    
                    contact_masses[f, idx_f] = total_pair * normalized_toe
                    contact_masses[f, idx_a] = total_pair * normalized_heel
                    
        return contact_masses
    def set_axis_permutation(self, axis_permutation):
        """
        Parse and cache the axis permutation matrix.
        Args:
            axis_permutation (str): Format "x,y,z" or "y,z,-x" etc.
        """
        if not axis_permutation:
            self.perm_basis = None
            self.perm_basis_rot = None
            return

        try:
            parts = [p.strip().lower() for p in axis_permutation.split(',')]
            if len(parts) != 3:
                print(f"Error: Invalid axis_permutation '{axis_permutation}'. Expected 3 parts.")
                self.perm_basis = None
                self.perm_basis_rot = None
                return

            basis = np.zeros((3, 3))
            
            for i, p in enumerate(parts): # i is target index (0=X, 1=Y, 2=Z)
                sign = -1.0 if '-' in p else 1.0
                axis_char = p.replace('-', '').strip()
                src_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis_char, -1)
                
                if src_idx != -1:
                    basis[i, src_idx] = sign
            
            self.perm_basis = basis
            
            # Check Determinant for Rotation Matrix
            det = np.linalg.det(basis)
            basis_for_rot = basis
            
            if abs(det - 1.0) > 0.01:
                # msg = f"Warning: Axis Permutation '{axis_permutation}' matrix has det={det:.2f}. "
                if det < 0:
                    # Reflection (Left-Handed). Flip Z col for rotation.
                    basis_for_rot = basis.copy()
                    basis_for_rot[:, 2] *= -1.0
            
            self.perm_basis_rot = basis_for_rot
            
        except Exception as e:
             print(f"Error parsing axis permutation: {e}")
             self.perm_basis = None
             self.perm_basis_rot = None

    def _prepare_trans_and_pose(self, pose_data, trans_data, options):
        """
        Handles input reshaping, coordinate system conversion, and initial type conversion.
        Returns:
            trans_data (F, 3)
            pose_data_aa (F, 24, 3)
            quats (F, 24, 4)
        """
        # Unpack Options
        input_type = options.input_type
        input_up_axis = options.input_up_axis
        axis_permutation = options.axis_permutation
        quat_format = options.quat_format

        pose_data = np.array(pose_data)
        trans_data = np.array(trans_data)
        
        # Reshape trans_data to (F, 3)
        if trans_data.ndim == 1:
            if trans_data.size >= 3:
                trans_data = trans_data[:3].reshape(1, 3) # (1, 3)
            else:
                trans_data = np.zeros((1, 3)) # Fallback
        elif trans_data.ndim == 2:
            if trans_data.shape[1] > 3:
                trans_data = trans_data[:, :3] # Slice extra components
            elif trans_data.shape[1] < 3:
                 pass
        
        # --- Robust Input Reshap/Padding ---
        
        # 1. Handle Flattened Inputs
        if pose_data.ndim == 1:
            if pose_data.size == 66: # 22*3 Axis Angle
                pose_data = pose_data.reshape(1, 22, 3)
            elif pose_data.size == 88: # 22*4 Quats
                pose_data = pose_data.reshape(1, 22, 4)
            elif pose_data.size == 72: # 24*3
                pose_data = pose_data.reshape(1, 24, 3)
            elif pose_data.size == 96: # 24*4
                pose_data = pose_data.reshape(1, 24, 4)
            else:
                 pose_data = pose_data[np.newaxis, ...] # Fallback
                 
        # 2. Handle Single Frame 2D Inputs
        if pose_data.ndim == 2:
            if pose_data.shape[0] == 22: # (22, C)
                 pose_data = pose_data[np.newaxis, ...]
            elif pose_data.shape[0] == 24: # (24, C)
                 if pose_data.shape[1] in [3, 4]:
                     pose_data = pose_data[np.newaxis, ...]

        # 3. Handle 22-Joint Padding (F, 22, C) -> (F, 24, C)
        if pose_data.ndim == 3 and pose_data.shape[1] == 22:
            F = pose_data.shape[0]
            C = pose_data.shape[2]
            p24 = np.zeros((F, 24, C))
            p24[:, :22, :] = pose_data
            
            # Pad Identity if Quats
            if C == 4:
                if quat_format == 'xyzw':
                     p24[:, 22:, 3] = 1.0
                else: # wxyz
                     p24[:, 22:, 0] = 1.0
            
            pose_data = p24
            
        F = pose_data.shape[0]
        
        # 4. Handle AMASS / Extra Joints (Crop to 24)
        if pose_data.shape[-1] > 72 and input_type == 'axis_angle':
            temp = pose_data.reshape(F, -1, 3)
            pose_data = temp[:, :24, :].reshape(F, -1)
            
        elif pose_data.shape[-1] > 96 and input_type == 'quat':
             temp = pose_data.reshape(F, -1, 4)
             pose_data = temp[:, :24, :].reshape(F, -1)
        
        # Convert to Axis Angle
        if input_type == 'axis_angle':
            if pose_data.shape[-1] == 72:
                pose_data_aa = pose_data.reshape(F, 24, 3)
            else:
                pose_data_aa = pose_data
                
            flat_pose = pose_data_aa.reshape(-1, 3)
            r = R.from_rotvec(flat_pose)
            quats = r.as_quat().reshape(F, 24, 4)
            
        elif input_type == 'quat':
             quats = pose_data.reshape(F, 24, 4).copy()
             
             # Handle Format
             if quat_format == 'wxyz':
                 quats = np.roll(quats, -1, axis=-1)
             
             flat_q = quats.reshape(-1, 4)
             r = R.from_quat(flat_q)
             pose_data_aa = r.as_rotvec().reshape(F, 24, 3)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
            
        # --- Handle Coordinate System Conversion ---
        # Default Internal Up Axis is Y (1)
        self.internal_y_dim = 1
        
        if axis_permutation is not None:
             self.set_axis_permutation(axis_permutation)
             
        if hasattr(self, 'perm_basis') and self.perm_basis is not None:
             basis = self.perm_basis
             basis_for_rot = self.perm_basis_rot
             
             # Transform Translation
             trans_data = trans_data @ basis.T
             
             # Transform Root Orientation
             r_perm = R.from_matrix(basis_for_rot)
             r_root = R.from_rotvec(pose_data_aa[:, 0, :])
             r_root_new = r_perm * r_root
             pose_data_aa[:, 0, :] = r_root_new.as_rotvec()
             
             # Regenerate quats
             flat_pose = pose_data_aa.reshape(-1, 3)
             r_all = R.from_rotvec(flat_pose)
             quats = r_all.as_quat().reshape(F, 24, 4)
             
             r_all = R.from_rotvec(flat_pose)
             quats = r_all.as_quat().reshape(F, 24, 4)
             
        # Post-Permutation Alignment
        # If the resulting data is Z-up (as indicated by options), rotate it to Y-up.
        if input_up_axis == 'Z':
            # Convert Z-up to Y-up
            tx, ty, tz = trans_data[..., 0], trans_data[..., 1], trans_data[..., 2]
            trans_data = np.stack([tx, tz, -ty], axis=-1)
            
            r_conv = R.from_euler('x', -90, degrees=True)
            r_root = R.from_rotvec(pose_data_aa[:, 0, :])
            r_root_new = r_conv * r_root
            pose_data_aa[:, 0, :] = r_root_new.as_rotvec()
            
            flat_pose = pose_data_aa.reshape(-1, 3)
            r_all = R.from_rotvec(flat_pose)
            quats = r_all.as_quat().reshape(F, 24, 4)
            
            # Rotated to Y-up
            self.internal_y_dim = 1
            
        elif input_up_axis == 'Y':
            pass
            
        else:
             print(f"Warning: input_up_axis='{input_up_axis}' not fully supported. Assuming Y-up.")
             
        return trans_data, pose_data_aa, quats

    def _compute_forward_kinematics(self, trans_data, quats):
        """
        Computes World Positions and Tip Locations via Forward Kinematics.
        Args:
            trans_data (F, 3): Root translation.
            quats (F, 24, 4): Joint rotations.
        Returns:
            world_pos (F, 30, 3): Global positions.
            global_rots (List[R]): Global rotation objects per joint (relative to world).
            tips (dict): Index -> Position (F, 3) for end effectors (Feet, Hands).
        """
        F = trans_data.shape[0]
        # Extending to 30 output joints (24 original + 6 virtual: Toes, Fingers, Heels)
        world_pos = np.zeros((F, 30, 3))
        
        # Optimization: Create ONE Rotation object for all joints at once to speed up init
        # We need frames for a specific joint to be contiguous for easy slicing later.
        # quats input is (F, 24, 4), so we transpose to (24, F, 4)
        quats_perm = quats.transpose(1, 0, 2).reshape(-1, 4) # (24*F, 4)
        all_rots = R.from_quat(quats_perm)
        
        # We store slices for each joint to avoid re-slicing (though slicing R is cheapish)
        # Actually we just slice on the fly to keep code clean.
        
        global_rots = [None] * 30
        parents = self._get_hierarchy()

        for i in range(30):
            parent = parents[i]
            
            if i < 24:
                # Slice the global rotation object
                # Because we transposed to (24, F, 4), the frames for joint i are at [i*F : (i+1)*F]
                start = i * F
                end = (i + 1) * F
                rot_local = all_rots[start:end]
            else:
                # Virtual Joint (Fixed extension)
                # Local rotation is Identity.
                rot_local = None # Flag to skip multiplication
            
            if parent == -1:
                global_rots[i] = rot_local
                world_pos[:, i, :] = trans_data # (F, 3)
            else:
                if rot_local is not None:
                    global_rots[i] = global_rots[parent] * rot_local
                else:
                    # Identity local: Global = Parent Global
                    global_rots[i] = global_rots[parent]
                
                # Calculate Offset (using 30-element skeleton_offsets)
                local_pos = self.skeleton_offsets[i]

                # Apply Rotation (Parent's Global Rotation) to the Local Offset
                offset_vec = global_rots[parent].apply(local_pos)
                world_pos[:, i, :] = world_pos[:, parent, :] + offset_vec
                
        # --- Tip Projection for Contact Detection ---
        tips = {} # index -> pos (F, 3)
        
        # Use Virtual End-Effectors (Indices 24-29) for accurate contact
        # L_Foot (10) -> L_Toe (24)
        # R_Foot (11) -> R_Toe (25)
        # L_Hand (22) -> L_Tip (26)
        # R_Hand (23) -> R_Tip (27)
        
        # Verify we have enough joints (28)
        if world_pos.shape[1] >= 28:
            # User Contact Indices Convention:
            # 7/8: L/R Ankle (Native).
            # 10/11: L/R Foot/ToeBase (Native).
            # 22/23: L/R Hand (Native) - Do NOT overwrite.
            # 24/25: L/R Toe Tip (Virtual).
            # 26/27: L/R Finger Tip (Virtual).
            # 28/29: L/R Heel (Virtual).
            
            # Map Indices 24-27 to Virtual Tips
            tips[24] = world_pos[:, 24, :] # L_Toe (Virtual) -> Index 24
            tips[25] = world_pos[:, 25, :] # R_Toe (Virtual) -> Index 25
            tips[26] = world_pos[:, 26, :] # L_Finger (Virtual) -> Index 26
            tips[27] = world_pos[:, 27, :] # R_Finger (Virtual) -> Index 27
            
            # Heels (if computed in FK)
            if world_pos.shape[1] >= 30:
                tips[28] = world_pos[:, 28, :] # L_Heel (Virtual) -> Index 28
                tips[29] = world_pos[:, 29, :] # R_Heel (Virtual) -> Index 29
        else:
             # Fallback if virtual joints missing (should not happen with new logic)
             pass
                  
        # Store tips and world_pos for later contact logic/debug
        self.temp_tips = tips 
        self.last_world_pos = world_pos 
        
        return world_pos, global_rots, tips




    # Kinematic neighbourhoods for the coherence gate (SMPL 22-joint layout):
    # {proximal, self, distal}. Only limbs are listed — a joint absent here
    # gets gate=0 (never lifted, stays canonical). Arm chains are validated
    # (2026-07-06 Subject10 popping); leg chains follow the same pattern but
    # are not yet validated for footwork.
    _COH_NEIGH = {
        16: (13, 16, 18), 17: (14, 17, 19),   # shoulders: collar, self, elbow
        18: (16, 18, 20), 19: (17, 19, 21),   # elbows:    shoulder, self, wrist
        20: (18, 20),     21: (19, 21),         # wrists:    elbow, self
        1: (0, 1, 4),     2: (0, 2, 5),         # hips:      pelvis, self, knee
        4: (1, 4, 7),     5: (2, 5, 8),         # knees:     hip, self, ankle
        7: (4, 7, 10),    8: (5, 8, 11),        # ankles:    knee, self, foot
    }

    def _coherence_envelope_gate(self, pose_aa, options, n_j):
        """Per-joint soft gate in [0,1] flagging coordinated, ballistic
        (popping-like) motion, for lifting the adaptive-effort alpha_max cap.

        pose_aa: (>=22, 3) local axis-angle for the current frame.
        Three NON-LOCAL soft valves in [0,1], multiplied:
          coherence — min normalised angular speed across the joint's
                      kinematic neighbourhood. Rejects spatially-local noise
                      (soft-tissue ringing, per-IMU magnetometer jitter),
                      which does not co-activate the whole chain.
          envelope  — trailing windowed-mean / windowed-peak angular speed.
                      Rejects temporally-isolated 1-frame events (sensor
                      spikes, 2,2,1 cadence steps). Window scales with fps
                      (`_env_window_ms`) so a real accent fills it at any rate.
          abs floor — the joint's OWN absolute speed in deg/s (fps-independent).
                      Rejects the cumulative-mean normalisation firing on
                      absolutely-slow motion in low-energy files (a modest
                      move looks "fast" relative to a quiet baseline). Uses
                      OWN speed, not min-neighbour, so distal flicks (fast
                      wrist, slower elbow) still pass.
        The gate is then EMA-smoothed with an fps-aware time constant
        (`_smooth_ms`) so its cutoff ramps instead of snapping (avoids OEF
        transient spikes at high fps). Normalisation uses a cumulative
        per-joint mean speed (causal analogue of the prototype's whole-file
        mean). Returns (n_j,); zeros outside _COH_NEIGH and during warm-up.
        """
        from scipy.spatial.transform import Rotation as _R
        n = min(n_j, 24)
        aa = np.asarray(pose_aa[:n], dtype=np.float64)

        prev = getattr(self, '_cg_prev_aa', None)
        self._cg_prev_aa = aa.copy()
        if prev is None or prev.shape != aa.shape:
            return np.zeros(n_j)

        # Per-joint geodesic angular speed (deg/frame).
        rel = _R.from_rotvec(aa) * _R.from_rotvec(prev).inv()
        speed = np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1))  # (n,)

        # Cumulative per-joint mean → normalisation scale (floored).
        s_sum = getattr(self, '_cg_speed_sum', None)
        s_cnt = getattr(self, '_cg_speed_cnt', 0)
        if s_sum is None or s_sum.shape != speed.shape:
            s_sum = np.zeros_like(speed); s_cnt = 0
        s_sum = s_sum + speed; s_cnt += 1
        self._cg_speed_sum = s_sum; self._cg_speed_cnt = s_cnt
        scale = np.maximum(s_sum / s_cnt, 0.5)   # floor 0.5 deg/frame
        norm_sp = speed / scale

        # Trailing envelope ring of raw speed (per joint). Window scales with
        # fps from a duration (ms) so it captures a real accent at any rate.
        env_ms = float(getattr(options, 'coherence_gate_env_window_ms', 50.0))
        W = max(3, int(round(env_ms / 1000.0 * self.framerate)))
        ring = getattr(self, '_cg_env_ring', None)
        rptr = getattr(self, '_cg_env_ptr', 0)
        rcnt = getattr(self, '_cg_env_cnt', 0)
        if ring is None or ring.shape != (W, n):
            ring = np.zeros((W, n)); rptr = 0; rcnt = 0
        ring[rptr] = speed; rptr = (rptr + 1) % W; rcnt = min(rcnt + 1, W)
        self._cg_env_ring = ring; self._cg_env_ptr = rptr; self._cg_env_cnt = rcnt

        coh_lo = getattr(options, 'coherence_gate_coh_lo', 0.7)
        coh_hi = getattr(options, 'coherence_gate_coh_hi', 1.8)
        env_lo = getattr(options, 'coherence_gate_env_lo', 0.30)
        env_hi = getattr(options, 'coherence_gate_env_hi', 0.65)
        abs_lo = getattr(options, 'coherence_gate_abs_lo', 400.0)   # deg/s
        abs_hi = getattr(options, 'coherence_gate_abs_hi', 1000.0)  # deg/s
        strength = float(getattr(options, 'coherence_gate_strength', 1.0))
        speed_degps = speed * self.framerate   # (n,) fps-independent absolute speed

        gate = np.zeros(n_j)
        if rcnt >= 3:
            win = ring[:rcnt]
            for j, nb in self._COH_NEIGH.items():
                if j >= n:
                    continue
                nb = [k for k in nb if k < n]
                coh_raw = float(np.min(norm_sp[nb]))
                coh_v = np.clip((coh_raw - coh_lo) / max(coh_hi - coh_lo, 1e-6), 0.0, 1.0)
                w_j = win[:, j]
                env_raw = float(w_j.mean() / (w_j.max() + 1e-6))
                env_v = np.clip((env_raw - env_lo) / max(env_hi - env_lo, 1e-6), 0.0, 1.0)
                # abs floor uses the MAX speed over the joint's neighbourhood
                # (deg/s), so a proximal joint whose torque is driven by fast
                # distal motion still passes, while a limb that is uniformly
                # slow (low-energy false-open) is rejected.
                abs_nb = float(np.max(speed_degps[nb]))
                abs_v = np.clip((abs_nb - abs_lo) / max(abs_hi - abs_lo, 1e-6), 0.0, 1.0)
                gate[j] = coh_v * env_v * abs_v
        # Optional Hill sharpening of the valve product (see options doc):
        # decisive on real accents without saturating quiet-material flicker.
        hill_n = float(getattr(options, 'coherence_gate_hill_n', 0.0))
        if hill_n > 0.0:
            p50 = max(float(getattr(options, 'coherence_gate_hill_p50', 0.15)), 1e-6)
            pn = np.power(gate, hill_n)
            gate = pn / (pn + p50 ** hill_n)
        gate *= strength
        # strength is documented [0,1] but not enforced; gate > 1 would push
        # alpha_max past 1 (unstable EMA) and cutoffs past cutoff_hi — clamp.
        np.clip(gate, 0.0, 1.0, out=gate)

        # Temporal EMA on the gate (fps-aware time constant) so the cutoff
        # ramps up/down instead of snapping — avoids OEF transient spikes when
        # the gate opens for only 1-2 frames at high fps. Optionally
        # asymmetric: a fast attack lets the cutoff open during a 1-2 frame
        # accent (a symmetric 25 ms EMA opens only after the impulse energy
        # has passed) while a slow release keeps the ramp-down smooth.
        smooth_ms = float(getattr(options, 'coherence_gate_smooth_ms', 25.0))
        attack_ms = float(getattr(options, 'coherence_gate_attack_ms', -1.0))
        release_ms = float(getattr(options, 'coherence_gate_release_ms', -1.0))
        if attack_ms < 0.0:
            attack_ms = smooth_ms
        if release_ms < 0.0:
            release_ms = smooth_ms
        if attack_ms > 0.0 or release_ms > 0.0:
            dt = 1.0 / self.framerate
            a_att = 1.0 if attack_ms <= 0.0 else 1.0 - np.exp(-dt / (attack_ms / 1000.0))
            a_rel = 1.0 if release_ms <= 0.0 else 1.0 - np.exp(-dt / (release_ms / 1000.0))
            prev_g = getattr(self, '_cg_gate_smooth', None)
            if prev_g is None or prev_g.shape != gate.shape:
                prev_g = np.zeros_like(gate)
            a = np.where(gate > prev_g, a_att, a_rel)
            gate = a * gate + (1.0 - a) * prev_g
            self._cg_gate_smooth = gate
        return gate

    def _compute_angular_kinematics(self, F, pose_data_aa, quats, options, use_filter=True, state_suffix=''):
        """
        Computes angular velocity and acceleration.
        Dual Path Support: Can optionally bypass filter and use separate state history.
        
        Args:
            F (int): Frame count.
            pose_data_aa (F, 24, 3): Pose axis-angle.
            quats (F, 24, 4): Pose quaternions.
            options (SMPLProcessingOptions): Usage: dt, filter_min_cutoff, filter_beta.
            use_filter (bool): Whether to use OneEuroFilter (True for Contact Path, False for Effort Path).
            state_suffix (str): Suffix for state variables (e.g. '', '_effort').
            
        Returns:
            ang_vel (F, 24, 3): Angular velocity vectors.
            ang_acc (F, 24, 3): Angular acceleration vectors.
        """
        dt = options.dt
        
        # Resolve State Variable Names
        name_prev_pose_aa = 'prev_pose_aa' + state_suffix
        name_prev_pose_q = 'prev_pose_q' + state_suffix
        name_prev_vel_aa = 'prev_vel_aa' + state_suffix
        
        # Access State
        prev_pose_aa = getattr(self, name_prev_pose_aa, None)
        prev_pose_q = getattr(self, name_prev_pose_q, None)
        prev_vel_aa = getattr(self, name_prev_vel_aa, None)
        
        # KEY FIX: Always enter streaming block if F==1. 
        # Do not rely on hasattr() because the attribute might not exist yet for new suffixes.
        if F == 1:
            # Streaming mode
            curr_aa = pose_data_aa[0] # (24, 3)
            
            if prev_pose_aa is None:
                # First frame, no velocity/accel
                ang_vel = np.zeros_like(curr_aa)
                ang_acc = np.zeros_like(curr_aa)
                
                # Init State (using setattr for dynamic names)
                setattr(self, name_prev_pose_aa, curr_aa)
                setattr(self, name_prev_vel_aa, ang_vel)
                
                # Check for Shape Mismatch (Re-init filter if needed)
                if use_filter and options.enable_one_euro_filter and self.pose_filter is None:
                     self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
            else:
                dt = 1.0 / self.framerate if self.framerate > 0 else 0.016
                
                filtered_q = None
                
                # One Euro Filter for Pose
                if use_filter and options.enable_one_euro_filter:
                    if self.pose_filter is None:
                        # Initialize
                        self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                    
                    # Update params dynamicallly (if changed via UI)
                    self.pose_filter._mincutoff = options.filter_min_cutoff
                    self.pose_filter._beta = options.filter_beta
                    
                    # Filter the signal (Pose Quaternions to handle wrapping)
                    curr_q = quats[0].flatten() # (96,)
                    
                    # Check for Shape Mismatch (Re-init filter if needed)
                    if self.pose_filter._x.last_value is not None:
                        if self.pose_filter._x.last_value.shape != curr_q.shape:
                            # Reset filter state completely
                            self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                            setattr(self, name_prev_pose_q, None)
                            prev_pose_q = None
                            
                    # Align Quaternions to avoid flips (Double Cover)
                    if self.pose_filter._x.last_value is not None:
                        prev_filtered_q = self.pose_filter._x.last_value
                        c_rs = curr_q.reshape(-1, 4)
                        p_rs = prev_filtered_q.reshape(-1, 4)
                        dots = np.sum(c_rs * p_rs, axis=1) # (24,)
                        mask = dots < 0
                        if np.any(mask):
                            c_rs[mask] *= -1.0
                            curr_q = c_rs.flatten()
                    
                    # Filter
                    filtered_q_flat = self.pose_filter(curr_q)
                    filtered_q = filtered_q_flat.reshape(24, 4)
                else:
                    # Bypass Filter
                    curr_q = quats[0] # (24, 4)
                    
                    # Ensure continuity (Unwrap) just like Filter does
                    # This prevents numerical discontinuities if Quats flip sign
                    # Although Rotation object handles it, manual unwrap is safer for debugging/consistency
                    if prev_pose_q is not None:
                         # prev_pose_q is the LAST STORED STATE (Unfiltered)
                         # We need to access it properly.
                         # Logic below accesses it via `getattr`.
                         pass 
                         
                    # We can't easily access 'prev_pose_q' here because it's loaded later in the function?
                    # No, it's loaded at line 2506 *after* filtering.
                    # To unwrap, we need the previous value NOW.
                    
                    # Access Previous State
                    prev_state_q = getattr(self, name_prev_pose_q, None)
                    
                    if prev_state_q is not None:
                        # Align to previous
                         c_rs = curr_q.reshape(-1, 4)
                         p_rs = prev_state_q.reshape(-1, 4)
                         dots = np.sum(c_rs * p_rs, axis=1)
                         mask = dots < 0
                         if np.any(mask):
                             c_rs[mask] *= -1.0
                             curr_q = c_rs 
                             
                    filtered_q = curr_q
                
                # Normalize Filtered Quats
                norms = np.linalg.norm(filtered_q, axis=1, keepdims=True)
                filtered_q /= (norms + 1e-8)
                
                # Store filtered local quats for world-frame composition method
                self._current_filtered_local_quats = filtered_q.copy()  # (24, 4) scipy xyzw
                
                # Velocity computation using FILTERED Quaternions
                r_curr = R.from_quat(filtered_q)
                
                # Need previous FILTERED quat (from State)
                if prev_pose_q is None:
                    ang_vel = np.zeros((24, 3))
                    raw_ang_acc = np.zeros((24, 3))
                    # Initialize history
                    setattr(self, name_prev_pose_q, filtered_q)
                else:
                    r_prev = R.from_quat(prev_pose_q)
                    r_diff = r_curr * r_prev.inv()
                    diff_vec = r_diff.as_rotvec()
                    
                    ang_vel = diff_vec.reshape(24, 3) / dt
                    
                    # --- One Euro filter on angular velocity for acceleration ---
                    # Raw ang_vel has noise spikes from mocap jitter. When 
                    # differenced to get acceleration, each spike produces a 
                    # ±pair of impulses. A One Euro filter adapts: heavy
                    # smoothing when velocity changes slowly (noise), light
                    # smoothing when velocity changes rapidly (real bursts).
                    # Only the filtered velocity is used for acceleration;
                    # raw velocity is stored for next frame's vel computation.
                    #
                    # Parameters are per-joint: central/high-inertia joints need
                    # aggressive smoothing (low cutoff) while extremities need
                    # minimal smoothing to preserve rapid dynamics.
                    name_vel_oef = f'_vel_one_euro{state_suffix}'
                    vel_oef = getattr(self, name_vel_oef, None)
                    if vel_oef is None:
                        # Per-joint-group parameters: (min_cutoff_Hz, beta)
                        # Lower min_cutoff = more smoothing at rest
                        # Higher beta = faster tracking of rapid changes
                        vel_filter_params = {
                            'pelvis':   (0.8,  0.04),   # very heavy — high inertia, slow dynamics
                            'hip':      (1.0,  0.05),   # heavy — large range but slow
                            'spine':    (0.8,  0.04),   # very heavy — amplified noise
                            'knee':     (1.5,  0.08),   # moderate
                            'ankle':    (2.0,  0.10),   # moderate-light
                            'foot':     (3.0,  0.15),   # light
                            'neck':     (1.5,  0.08),   # moderate
                            'head':     (2.0,  0.10),   # moderate
                            'collar':   (1.5,  0.08),   # moderate
                            'shoulder': (2.5,  0.15),   # light — fast dynamics
                            'elbow':    (3.0,  0.20),   # very light — rapid movements
                            'wrist':    (4.0,  0.25),   # minimal — fastest dynamics
                            'hand':     (5.0,  0.30),   # minimal
                        }
                        default_params = (1.5, 0.10)
                        
                        # Build per-element arrays (24 joints × 3 axes = 72)
                        mc_arr = np.zeros(72)
                        beta_arr = np.zeros(72)
                        for j_idx in range(min(24, len(self.joint_names))):
                            jname = self.joint_names[j_idx] if j_idx < len(self.joint_names) else ''
                            mc, bt = default_params
                            for key, (mc_val, bt_val) in vel_filter_params.items():
                                if key in jname:
                                    mc, bt = mc_val, bt_val
                                    break
                            mc_arr[j_idx*3:(j_idx+1)*3] = mc
                            beta_arr[j_idx*3:(j_idx+1)*3] = bt
                        
                        vel_oef = OneEuroFilter(
                            min_cutoff=mc_arr,
                            beta=beta_arr,
                            d_cutoff=1.0,
                            framerate=1.0/dt
                        )
                        setattr(self, name_vel_oef, vel_oef)
                    
                    # Filter flattened velocity, then reshape back
                    smooth_vel = vel_oef(ang_vel.flatten()).reshape(24, 3)
                    
                    # Update History (raw velocity for next frame's velocity computation)
                    setattr(self, name_prev_pose_q, filtered_q)
                    
                    # Acceleration from ONE-EURO-FILTERED velocity
                    name_prev_smooth_vel = f'_prev_smooth_vel{state_suffix}'
                    prev_smooth = getattr(self, name_prev_smooth_vel, None)
                    if prev_smooth is None or prev_smooth.shape != smooth_vel.shape:
                        raw_ang_acc = np.zeros_like(ang_vel)
                    else:
                        raw_ang_acc = (smooth_vel - prev_smooth) / dt
                    setattr(self, name_prev_smooth_vel, smooth_vel.copy())
                        
                    # Update Velocity History (raw, for next frame's velocity computation)
                    setattr(self, name_prev_vel_aa, ang_vel)
                
                # Smoothing implicitly handled by OneEuroFilter on Pose (if enabled)
                ang_acc = raw_ang_acc
                
                # Update AA History
                # We track raw AA for spike detection logic usually?
                # Or filtered AA? Original code tracked curr_aa (Raw/Input).
                setattr(self, name_prev_pose_aa, curr_aa)
                
                # Wait, I might have just assigned filtered_q (Quat) to prev_pose_aa (AA) in my thought process?
                # Let's be careful.
                


        else:
            # Batch Mode (F > 1)
            # Calculate derivatives across the batch
            dt = 1.0 / self.framerate
            
            # Robust Angular Velocity using Quaternions (Handling wrapping/discontinuities)
            # Vectorized using linear R array
            flat_quats = quats.reshape(-1, 4)
            r_all = R.from_quat(flat_quats)
            
            # Backward Difference: vel[t] = (rot[t] * rot[t-1].inv) / dt
            # We skip the first frame (index 0) as it has no prev
            # Slices: 24 (1 frame) to End vs 0 to End-24
            
            r_curr = r_all[24:]
            r_prev = r_all[:-24]
            
            r_diff = r_curr * r_prev.inv()
            vel_flat = r_diff.as_rotvec() / dt
            
            ang_vel = np.zeros((F, 24, 3))
            ang_vel[1:] = vel_flat.reshape(F-1, 24, 3)
            ang_vel[0] = ang_vel[1] # Pad start
            
            # Smoothing (Batch)
            smoothing = 0.0 # Disabled for now as param was removed.
            if smoothing > 0.0:
                 # Gaussian filter or similar?
                 # For consistency with streaming (Exponential Moving Average), we could simulate EMA.
                 # Or just use scipy.ndimage.gaussian_filter1d
                 from scipy.ndimage import gaussian_filter1d
                 sigma = smoothing * 10.0 # Heuristic mapping
                 ang_vel = gaussian_filter1d(ang_vel, sigma=sigma, axis=0)

            # 2. Angular Acceleration
            ang_acc = np.gradient(ang_vel, dt, axis=0)
            
            if smoothing > 0.0:
                 ang_acc = gaussian_filter1d(ang_acc, sigma=sigma, axis=0)

            # Acceleration Clamping (Batch)
            # Acceleration Clamping REMOVED per task.
                    
        return ang_vel, ang_acc


    def _compute_world_angular_kinematics(self, F, global_rots, options, state_suffix=''):
        """
        Compute joint angular acceleration using world-frame composition.
        
        For each joint j:
          R_child_world(t) = R_parent_global_filtered(t) × R_local_filtered(t)
        
        - R_parent_global is the FK global rotation of the parent, strongly filtered
          (SLERP EMA) to provide a clean, slowly-varying joint base.
        - R_local is the per-joint One Euro filtered local rotation (from existing pipeline).
        - The product R_child_world is differentiated for ω and α.
        
        Properties:
        - Infinitely strong global filter → R_parent ≈ constant → recovers local-only α.
        - Weak global filter → R_parent tracks real motion → stationary arm gives α ≈ 0.
        
        Args:
            F (int): Frame count (expected 1 for streaming).
            global_rots (list[Rotation]): Unfiltered world-frame rotations from FK.
            options: Processing options.
            state_suffix (str): State namespace for dual-path support.
            
        Returns:
            ang_vel (F, 24, 3): World-frame angular velocity of composed rotation.
            ang_acc (F, 24, 3): World-frame angular acceleration.
        """
        dt = options.dt if hasattr(options, 'dt') and options.dt > 0 else 1.0 / self.framerate
        parents = self._get_hierarchy()
        n_joints = min(24, len(global_rots))
        
        # State names
        name_prev_composed = f'_prev_composed_rots{state_suffix}'
        name_prev_vel = f'_prev_composed_vel{state_suffix}'
        name_filt_global = f'_filtered_global_rots{state_suffix}'
        
        # Global filter strength: alpha for SLERP EMA (0 = infinitely strong, 1 = no filter)
        # Low alpha → very smooth base motion → approaches local-only result
        GLOBAL_FILTER_ALPHA = getattr(self, '_world_global_filter_alpha', 0.15)
        
        if F != 1:
            # Batch mode fallback: use raw world-frame differential (no composition)
            ang_vel_world = np.zeros((F, n_joints, 3))
            for j in range(n_joints):
                if global_rots[j] is None:
                    continue
                q_all = global_rots[j].as_quat()
                if q_all.ndim == 1:
                    q_all = q_all[np.newaxis, :]
                r_curr = R.from_quat(q_all[1:])
                r_prev = R.from_quat(q_all[:-1])
                r_diff = r_curr * r_prev.inv()
                v = r_diff.as_rotvec() / dt
                ang_vel_world[1:, j, :] = v
                ang_vel_world[0, j, :] = v[0] if len(v) > 0 else 0.0
            
            # Differential for batch
            ang_vel_diff = ang_vel_world.copy()
            for j in range(n_joints):
                p = parents[j] if j < len(parents) else -1
                if p >= 0 and p < n_joints:
                    ang_vel_diff[:, j, :] = ang_vel_world[:, j, :] - ang_vel_world[:, p, :]
            ang_acc = np.gradient(ang_vel_diff, dt, axis=0)
            
            # Pad
            if n_joints < 24:
                pad_v = np.zeros((F, 24, 3))
                pad_a = np.zeros((F, 24, 3))
                pad_v[:, :n_joints, :] = ang_vel_diff
                pad_a[:, :n_joints, :] = ang_acc
                return pad_v, pad_a
            return ang_vel_diff, ang_acc
        
        # --- Streaming Mode (F=1) ---
        
        # 1. Extract current unfiltered global rotations as quaternions
        curr_global_q = np.zeros((n_joints, 4))
        for j in range(n_joints):
            if global_rots[j] is not None:
                q = global_rots[j].as_quat()
                curr_global_q[j] = q[0] if q.ndim > 1 else q
        
        # 2. Get filtered local quaternions (from _compute_angular_kinematics)
        filtered_local_q = getattr(self, '_current_filtered_local_quats', None)
        if filtered_local_q is None:
            # Fallback: use raw from FK
            filtered_local_q = np.zeros((n_joints, 4))
            for j in range(n_joints):
                p = parents[j] if j < len(parents) else -1
                if p >= 0 and global_rots[p] is not None and global_rots[j] is not None:
                    r_p = R.from_quat(curr_global_q[p])
                    r_j = R.from_quat(curr_global_q[j])
                    r_local = r_p.inv() * r_j
                    filtered_local_q[j] = r_local.as_quat()
                else:
                    filtered_local_q[j] = curr_global_q[j]  # Root
        
        # 3. Compose: R_child_world = R_raw_global_parent × R_local_filtered
        # Vectorized quaternion multiply — no scipy Rotation objects needed.
        composed_q = np.zeros((n_joints, 4))
        # Root: composed = local (world rotation IS the local rotation)
        composed_q[0] = filtered_local_q[0]
        # Non-root: composed = quat_mul(parent_raw_global, local_filtered)
        for j in range(1, n_joints):
            p = parents[j] if j < len(parents) else -1
            if p >= 0:
                # Quaternion multiply: q1 * q2 (scipy xyzw format)
                a = curr_global_q[p]  # parent raw global
                b = filtered_local_q[j]  # child local filtered
                composed_q[j] = np.array([
                    a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
                    a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
                    a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
                    a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
                ])
            else:
                composed_q[j] = filtered_local_q[j]
        
        # 5. Differentiate composed rotation for angular velocity
        prev_composed = getattr(self, name_prev_composed, None)
        if prev_composed is None:
            ang_vel = np.zeros((1, 24, 3))
            ang_acc = np.zeros((1, 24, 3))
            setattr(self, name_prev_composed, composed_q.copy())
            setattr(self, name_prev_vel, np.zeros((n_joints, 3)))
            return ang_vel, ang_acc
        
        # Vectorized angular velocity from quaternion differences
        # Ensure shortest path (flip prev if dot < 0)
        dots = np.sum(composed_q * prev_composed, axis=1)  # (n_joints,)
        prev_q = prev_composed.copy()
        prev_q[dots < 0] *= -1
        
        # r_diff = composed * prev.inv()
        # quat inverse in xyzw format: [-x, -y, -z, w]
        prev_inv = prev_q.copy()
        prev_inv[:, :3] *= -1  # negate xyz
        
        # Batch quaternion multiply: composed_q * prev_inv
        a = composed_q   # (N, 4) xyzw
        b = prev_inv      # (N, 4) xyzw
        diff_q = np.empty_like(a)
        diff_q[:, 0] = a[:, 3]*b[:, 0] + a[:, 0]*b[:, 3] + a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
        diff_q[:, 1] = a[:, 3]*b[:, 1] - a[:, 0]*b[:, 2] + a[:, 1]*b[:, 3] + a[:, 2]*b[:, 0]
        diff_q[:, 2] = a[:, 3]*b[:, 2] + a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0] + a[:, 2]*b[:, 3]
        diff_q[:, 3] = a[:, 3]*b[:, 3] - a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1] - a[:, 2]*b[:, 2]
        
        # Convert diff quaternion to rotation vector (axis-angle)
        # rotvec = 2 * atan2(|xyz|, w) * xyz / |xyz|
        xyz = diff_q[:, :3]                                  # (N, 3)
        w = diff_q[:, 3]                                      # (N,)
        sin_half = np.linalg.norm(xyz, axis=1)                # (N,)
        angle = 2.0 * np.arctan2(sin_half, w)                 # (N,)
        # Normalize to [-pi, pi]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        safe_sin = np.where(sin_half > 1e-10, sin_half, 1.0)
        axis = xyz / safe_sin[:, np.newaxis]
        rotvec = angle[:, np.newaxis] * axis                   # (N, 3)
        rotvec[sin_half <= 1e-10] = 0.0  # Zero rotation → zero velocity
        
        vel = rotvec / dt
        
        # 5b. Apply per-joint One Euro filter on composed velocity (matching local pipeline)
        name_vel_oef = f'_vel_one_euro_composed{state_suffix}'
        vel_oef = getattr(self, name_vel_oef, None)
        if vel_oef is None and options.enable_one_euro_filter:
            vel_filter_params = {
                'pelvis':   (0.8,  0.04),
                'hip':      (1.0,  0.05),
                'spine':    (0.8,  0.04),
                'knee':     (1.5,  0.08),
                'ankle':    (2.0,  0.10),
                'foot':     (3.0,  0.15),
                'neck':     (1.5,  0.08),
                'head':     (2.0,  0.10),
                'collar':   (1.5,  0.08),
                'shoulder': (2.5,  0.15),
                'elbow':    (3.0,  0.20),
                'wrist':    (4.0,  0.25),
                'hand':     (5.0,  0.30),
            }
            default_params = (1.5, 0.10)
            mc_arr = np.zeros(n_joints * 3)
            beta_arr = np.zeros(n_joints * 3)
            for j_idx in range(min(n_joints, len(self.joint_names))):
                jname = self.joint_names[j_idx] if j_idx < len(self.joint_names) else ''
                mc, bt = default_params
                for key, (mc_val, bt_val) in vel_filter_params.items():
                    if key in jname:
                        mc, bt = mc_val, bt_val
                        break
                mc_arr[j_idx*3:(j_idx+1)*3] = mc
                beta_arr[j_idx*3:(j_idx+1)*3] = bt
            vel_oef = OneEuroFilter(
                min_cutoff=mc_arr, beta=beta_arr,
                d_cutoff=1.0, framerate=1.0/dt
            )
            setattr(self, name_vel_oef, vel_oef)
        
        if vel_oef is not None:
            smooth_vel = vel_oef(vel.flatten()).reshape(n_joints, 3)
        else:
            smooth_vel = vel
        
        # 6. Compute acceleration from velocity differences
        # If acc_smooth_window is set, use Savitzky-Golay derivative
        # over a ring buffer of recent velocities.
        sg_win = getattr(options, 'acc_smooth_window', 0)
        if sg_win >= 3:
            # Cached SG coefficients
            cache_name = f'_ang_sg_cache{state_suffix}'
            sg_cache = getattr(self, cache_name, None)
            if sg_cache is None or sg_cache[0] != sg_win or sg_cache[1] != dt:
                center = (sg_win - 1) / 2.0
                denom = sum((k - center) ** 2 for k in range(sg_win)) * dt
                coeffs = np.array([(k - center) / denom for k in range(sg_win)])
                setattr(self, cache_name, (sg_win, dt, coeffs))
            else:
                coeffs = sg_cache[2]
            
            # Numpy ring buffer
            arr_name = f'_ang_vel_ring_arr{state_suffix}'
            ptr_name = f'_ang_vel_ring_ptr{state_suffix}'
            cnt_name = f'_ang_vel_ring_cnt{state_suffix}'
            ring_arr = getattr(self, arr_name, None)
            ring_ptr = getattr(self, ptr_name, 0)
            ring_cnt = getattr(self, cnt_name, 0)
            
            if ring_arr is None or ring_arr.shape[0] != sg_win or ring_arr.shape[1:] != smooth_vel.shape:
                ring_arr = np.zeros((sg_win,) + smooth_vel.shape, dtype=smooth_vel.dtype)
                ring_ptr = 0
                ring_cnt = 0
            
            ring_arr[ring_ptr] = smooth_vel
            ring_ptr = (ring_ptr + 1) % sg_win
            ring_cnt = min(ring_cnt + 1, sg_win)
            
            setattr(self, arr_name, ring_arr)
            setattr(self, ptr_name, ring_ptr)
            setattr(self, cnt_name, ring_cnt)
            
            N = ring_cnt
            if N >= 3:
                if N == sg_win:
                    ordered = np.roll(ring_arr, -ring_ptr, axis=0)
                else:
                    ordered = ring_arr[:N]
                acc = np.tensordot(coeffs[:N], ordered, axes=([0], [0]))
            else:
                prev_vel = getattr(self, name_prev_vel, None)
                if prev_vel is not None and prev_vel.shape == smooth_vel.shape:
                    acc = (smooth_vel - prev_vel) / dt
                else:
                    acc = np.zeros((n_joints, 3))
        else:
            prev_vel = getattr(self, name_prev_vel, None)
            if prev_vel is None or prev_vel.shape != smooth_vel.shape:
                acc = np.zeros((n_joints, 3))
            else:
                acc = (smooth_vel - prev_vel) / dt

        # Update state
        setattr(self, name_prev_composed, composed_q.copy())
        setattr(self, name_prev_vel, smooth_vel.copy())
        
        # Pack to (1, 24, 3)
        ang_vel_out = np.zeros((1, 24, 3))
        ang_acc_out = np.zeros((1, 24, 3))
        ang_vel_out[0, :n_joints, :] = vel
        ang_acc_out[0, :n_joints, :] = acc
        
        return ang_vel_out, ang_acc_out


    def _compute_com_dynamic_torque(self, F, world_pos, global_rots, tips, options):
        """
        Compute dynamic torque from subtree center-of-mass linear acceleration.
        
        τ_j = r_j × (m_subtree_j × a_com_j)
        
        Segment CoMs are computed from SMPL mesh vertex centroids (LBS weights)
        when available, rotated into world frame by each joint's global rotation.
        Falls back to midpoint approximation if mesh data unavailable.
        
        Args:
            F: Frame count (1 for streaming)
            world_pos: (F, J, 3) world positions from FK
            global_rots: (F, J, 3, 3) global rotation matrices from FK
            tips: dict mapping tip names to positions, from FK
            options: Processing options
            
        Returns:
            t_dyn_com: (F, 24, 3) dynamic torque vectors from CoM acceleration
        """
        dt = options.dt if hasattr(options, 'dt') and options.dt > 0 else 1.0 / self.framerate
        parents = self._get_hierarchy()
        n_joints = min(22, world_pos.shape[1])
        
        # Build segment masses (cached)
        if not hasattr(self, '_com_seg_mass') or self._com_seg_mass is None:
            limb_masses = self.limb_data['masses']
            self._com_seg_mass = np.zeros(n_joints)
            for idx in range(n_joints):
                name = self.joint_names[idx].lower() if idx < len(self.joint_names) else ''
                m = 0.0
                if 'pelvis' in name: m = limb_masses['pelvis']
                elif 'hip' in name: m = limb_masses['upper_leg']
                elif 'knee' in name: m = limb_masses['lower_leg']
                elif 'ankle' in name: m = limb_masses['foot']
                elif 'spine' in name: m = limb_masses['spine']
                elif 'neck' in name: m = limb_masses.get('head', 1.0) * 0.2
                elif 'head' in name: m = limb_masses['head'] * 0.8
                elif 'collar' in name: m = limb_masses['upper_arm'] * 0.2
                elif 'shoulder' in name: m = limb_masses['upper_arm'] * 0.8
                elif 'elbow' in name: m = limb_masses['lower_arm']
                elif 'wrist' in name: m = limb_masses['hand']
                self._com_seg_mass[idx] = m
            
            # Build children list
            children_list = [[] for _ in range(n_joints)]
            for j in range(n_joints):
                p = parents[j] if j < len(parents) else -1
                if 0 <= p < n_joints:
                    children_list[p].append(j)
            self._com_children = children_list
            
            # Map leaf joints to their tip indices (virtual joints from FK)
            # Tips: 24=L_Toe, 25=R_Toe, 26=L_Finger, 27=R_Finger
            self._com_tip_map = {}
            for idx in range(n_joints):
                name = self.joint_names[idx].lower() if idx < len(self.joint_names) else ''
                if not children_list[idx]:  # leaf joint
                    if 'l_wrist' in name: self._com_tip_map[idx] = 26  # L_Finger
                    elif 'r_wrist' in name: self._com_tip_map[idx] = 27  # R_Finger
                    elif 'l_foot' in name: self._com_tip_map[idx] = 24  # L_Toe
                    elif 'r_foot' in name: self._com_tip_map[idx] = 25  # R_Toe
                    # Head has no tip — use an offset along local up
            
            def _get_subtree(j):
                result = [j]
                for c in children_list[j]:
                    result.extend(_get_subtree(c))
                return result
            
            self._com_subtrees = [_get_subtree(j) for j in range(n_joints)]
            self._com_subtree_mass = np.array([
                np.sum([self._com_seg_mass[k] for k in st]) for st in self._com_subtrees
            ])
            self._com_subtree_weights = []
            for j in range(n_joints):
                st = self._com_subtrees[j]
                masses = np.array([self._com_seg_mass[k] for k in st])
                total = np.sum(masses)
                self._com_subtree_weights.append(masses / max(total, 1e-8))
        
        seg_mass = self._com_seg_mass
        subtrees = self._com_subtrees
        subtree_mass = self._com_subtree_mass
        subtree_weights = self._com_subtree_weights
        children_list = self._com_children
        tip_map = self._com_tip_map
        
        # Get mesh-based segment CoM offsets (T-pose, relative to each joint)
        mesh_com_offsets = self.limb_data.get('segment_com_offsets', None)
        use_mesh_com = mesh_com_offsets is not None and len(mesh_com_offsets) >= n_joints
        
    # Compute per-SEGMENT CoM world positions (vectorized)
        com_pos = np.zeros((F, n_joints, 3))
        if use_mesh_com:
            # Batch matmul: rotate all T-pose offsets by global rotations
            # global_rots: (F, J, 3, 3), mesh_com_offsets: (J, 3)
            offsets = np.asarray(mesh_com_offsets)[:n_joints]  # (J, 3)
            # global_rots is a list of scipy Rotation objects — convert to matrices
            rot_mats = np.zeros((F, n_joints, 3, 3))
            for j in range(n_joints):
                if global_rots[j] is not None:
                    rot_mats[:, j] = global_rots[j].as_matrix()
                else:
                    rot_mats[:, j] = np.eye(3)
            # einsum: (F, J, 3, 3) @ (J, 3) → (F, J, 3)
            offset_world = np.einsum('fjik,jk->fji', rot_mats, offsets)
            seg_com = world_pos[:, :n_joints] + offset_world
        else:
            seg_com = world_pos[:, :n_joints].copy()
            for j in range(n_joints):
                if children_list[j]:
                    child_pos = world_pos[:, children_list[j][0]]
                    seg_com[:, j] = 0.5 * (world_pos[:, j] + child_pos)
                elif j in tip_map and tip_map[j] < world_pos.shape[1]:
                    seg_com[:, j] = 0.5 * (world_pos[:, j] + world_pos[:, tip_map[j]])
        
        # Subtree CoM = mass-weighted average (vectorized with pre-built arrays)
        if not hasattr(self, '_com_subtree_idx') or self._com_subtree_idx is None:
            # Build padded index array for vectorized gathering
            max_st = max(len(st) for st in subtrees)
            st_idx = np.zeros((n_joints, max_st), dtype=int)
            st_mask = np.zeros((n_joints, max_st))
            st_w = np.zeros((n_joints, max_st))
            for j in range(n_joints):
                st = subtrees[j]
                n = len(st)
                st_idx[j, :n] = st
                st_mask[j, :n] = 1.0
                st_w[j, :n] = subtree_weights[j]
            self._com_subtree_idx = st_idx
            self._com_subtree_w = st_w
        
        # Gather segment CoMs for all subtrees: (F, J, max_st, 3)
        gathered = seg_com[:, self._com_subtree_idx]  # (F, J, max_st, 3)
        # Weighted sum: (F, J, max_st, 3) * (J, max_st, 1) → sum → (F, J, 3)
        com_pos[:, :n_joints] = np.einsum('fjsk,js->fj k', gathered, self._com_subtree_w).reshape(F, n_joints, 3)
        
        t_dyn_com = np.zeros((F, max(24, n_joints), 3))
        
        # Read base filter params from options (user-tunable via widgets)
        base_pos_mc = options.com_pos_min_cutoff
        base_pos_beta = options.com_pos_beta
        base_vel_mc = options.com_vel_min_cutoff
        base_vel_beta = options.com_vel_beta
        base_acc_mc = options.com_acc_min_cutoff
        base_acc_beta = options.com_acc_beta
        current_params = (base_pos_mc, base_pos_beta, base_vel_mc, base_vel_beta, base_acc_mc, base_acc_beta)
        
        # Detect param changes → rebuild filter arrays and reset filters
        prev_params = getattr(self, '_com_filter_params', None)
        if prev_params != current_params:
            self._com_filter_arrays = None
            self._com_pos_filter = None
            self._com_vel_filter = None
            self._com_acc_filter = None
            self._com_prev_smooth_pos = None
            self._com_prev_smooth_vel = None
            self._com_filter_params = current_params
        
        # Build per-joint mass-scaled filter arrays (cached until params change)
        if not hasattr(self, '_com_filter_arrays') or self._com_filter_arrays is None:
            n_dims = n_joints * 3
            pos_mc = np.zeros(n_dims)
            pos_beta = np.zeros(n_dims)
            vel_mc = np.zeros(n_dims)
            vel_beta = np.zeros(n_dims)
            acc_mc = np.zeros(n_dims)
            acc_beta = np.zeros(n_dims)
            
            for j in range(n_joints):
                m = max(subtree_mass[j], 0.1)
                s = 1.0 / np.sqrt(m)
                pos_mc[j*3:(j+1)*3] = base_pos_mc * s
                pos_beta[j*3:(j+1)*3] = base_pos_beta * s
                vel_mc[j*3:(j+1)*3] = base_vel_mc * s
                vel_beta[j*3:(j+1)*3] = base_vel_beta * s
                acc_mc[j*3:(j+1)*3] = base_acc_mc * s
                acc_beta[j*3:(j+1)*3] = base_acc_beta * s
            
            self._com_filter_arrays = (pos_mc, pos_beta, vel_mc, vel_beta, acc_mc, acc_beta)
        
        pos_mc, pos_beta, vel_mc, vel_beta, acc_mc, acc_beta = self._com_filter_arrays
        
        if F == 1:
            # --- Streaming mode ---
            skip_pos = base_pos_mc >= 999
            skip_vel = base_vel_mc >= 999
            skip_acc = base_acc_mc >= 999
            
            if not skip_pos and (not hasattr(self, '_com_pos_filter') or self._com_pos_filter is None):
                n_dims = n_joints * 3
                self._com_pos_filter = OneEuroFilter(
                    min_cutoff=pos_mc, beta=pos_beta,
                    d_cutoff=1.0, framerate=1.0/dt
                )
            if not skip_vel and (not hasattr(self, '_com_vel_filter') or self._com_vel_filter is None):
                n_dims = n_joints * 3
                self._com_vel_filter = OneEuroFilter(
                    min_cutoff=vel_mc, beta=vel_beta,
                    d_cutoff=1.0, framerate=1.0/dt
                )
            if not skip_acc and (not hasattr(self, '_com_acc_filter') or self._com_acc_filter is None):
                self._com_acc_filter = OneEuroFilter(
                    min_cutoff=acc_mc, beta=acc_beta,
                    d_cutoff=1.0, framerate=1.0/dt
                )
            if not hasattr(self, '_com_prev_smooth_pos'):
                self._com_prev_smooth_pos = None
                self._com_prev_smooth_vel = None
            
            curr_pos = com_pos[0, :n_joints, :]
            
            # Position: filter or pass-through
            smooth_pos = curr_pos if skip_pos else self._com_pos_filter(
                curr_pos.flatten()
            ).reshape(n_joints, 3)
            
            # Velocity from positions
            if self._com_prev_smooth_pos is None:
                self._com_prev_smooth_pos = smooth_pos.copy()
                self._com_prev_smooth_vel = np.zeros((n_joints, 3))
                return t_dyn_com
            
            raw_vel = (smooth_pos - self._com_prev_smooth_pos) / dt
            
            # Velocity: filter or pass-through
            smooth_vel = raw_vel if skip_vel else self._com_vel_filter(
                raw_vel.flatten()
            ).reshape(n_joints, 3)
            
            # Acceleration: Savitzky-Golay derivative when acc_smooth_window > 0
            sg_win = getattr(options, 'acc_smooth_window', 0)
            if sg_win >= 3:
                # --- Cached SG coefficients (only recomputed when N or dt changes) ---
                sg_cache = getattr(self, '_com_sg_cache', None)
                if sg_cache is None or sg_cache[0] != sg_win or sg_cache[1] != dt:
                    center = (sg_win - 1) / 2.0
                    denom = sum((k - center) ** 2 for k in range(sg_win)) * dt
                    coeffs = np.array([(k - center) / denom for k in range(sg_win)])
                    self._com_sg_cache = (sg_win, dt, coeffs)
                else:
                    coeffs = sg_cache[2]
                
                # --- Numpy ring buffer (fixed-size array + write pointer) ---
                ring_arr = getattr(self, '_com_vel_ring_arr', None)
                ring_ptr = getattr(self, '_com_vel_ring_ptr', 0)
                ring_cnt = getattr(self, '_com_vel_ring_cnt', 0)
                
                if ring_arr is None or ring_arr.shape[0] != sg_win or ring_arr.shape[1:] != smooth_vel.shape:
                    ring_arr = np.zeros((sg_win,) + smooth_vel.shape, dtype=smooth_vel.dtype)
                    ring_ptr = 0
                    ring_cnt = 0
                
                ring_arr[ring_ptr] = smooth_vel
                ring_ptr = (ring_ptr + 1) % sg_win
                ring_cnt = min(ring_cnt + 1, sg_win)
                
                self._com_vel_ring_arr = ring_arr
                self._com_vel_ring_ptr = ring_ptr
                self._com_vel_ring_cnt = ring_cnt
                
                N = ring_cnt
                if N >= 3:
                    # Reorder ring buffer oldest→newest for coefficient alignment
                    if N == sg_win:
                        ordered = np.roll(ring_arr, -ring_ptr, axis=0)
                    else:
                        # Buffer not full yet: entries 0..ring_ptr-1 are in order
                        ordered = ring_arr[:N]
                    # Vectorized: coeffs[:N] · ordered → (J, 3)
                    raw_acc = np.tensordot(coeffs[:N], ordered, axes=([0], [0]))
                else:
                    raw_acc = (smooth_vel - self._com_prev_smooth_vel) / dt
            else:
                raw_acc = (smooth_vel - self._com_prev_smooth_vel) / dt
            
            # Acceleration: filter or pass-through
            acc = raw_acc if skip_acc else self._com_acc_filter(
                raw_acc.flatten()
            ).reshape(n_joints, 3)
            
            # Vectorized torque = r × (m × a)
            r = smooth_pos - world_pos[0, :n_joints]  # (J, 3)
            F_inertial = subtree_mass[:, np.newaxis] * acc  # (J, 3)
            t_dyn_com[0, :n_joints, :] = np.cross(r, F_inertial)
            
            self._com_prev_smooth_pos = smooth_pos.copy()
            self._com_prev_smooth_vel = smooth_vel.copy()
            
        else:
            # --- Batch mode ---
            skip_pos = base_pos_mc >= 999
            skip_vel = base_vel_mc >= 999
            skip_acc = base_acc_mc >= 999
            n_dims = n_joints * 3
            
            # Position: filter or pass-through
            if skip_pos:
                smooth_pos = com_pos.copy()
            else:
                pf = OneEuroFilter(min_cutoff=pos_mc, beta=pos_beta, d_cutoff=1.0, framerate=1.0/dt)
                smooth_pos = np.zeros_like(com_pos)
                for f in range(F):
                    smooth_pos[f, :n_joints] = pf(com_pos[f, :n_joints].flatten()).reshape(n_joints, 3)
            
            # Velocity
            vel = np.zeros_like(smooth_pos)
            vel[1:] = (smooth_pos[1:] - smooth_pos[:-1]) / dt
            vel[0] = vel[1]
            
            # Velocity: filter or pass-through
            if skip_vel:
                smooth_vel = vel.copy()
            else:
                vf = OneEuroFilter(min_cutoff=vel_mc, beta=vel_beta, d_cutoff=1.0, framerate=1.0/dt)
                smooth_vel = np.zeros_like(vel)
                for f in range(F):
                    smooth_vel[f, :n_joints] = vf(vel[f, :n_joints].flatten()).reshape(n_joints, 3)
            
            # Acceleration
            raw_acc = np.zeros_like(smooth_vel)
            raw_acc[1:] = (smooth_vel[1:] - smooth_vel[:-1]) / dt
            raw_acc[0] = raw_acc[1]
            
            # Acceleration: filter or pass-through
            if skip_acc:
                acc = raw_acc.copy()
            else:
                af = OneEuroFilter(min_cutoff=acc_mc, beta=acc_beta, d_cutoff=1.0, framerate=1.0/dt)
                acc = np.zeros_like(raw_acc)
                for f in range(F):
                    acc[f, :n_joints] = af(raw_acc[f, :n_joints].flatten()).reshape(n_joints, 3)
            
            # Torque
            for f in range(F):
                for j in range(n_joints):
                    r = smooth_pos[f, j] - world_pos[f, j]
                    F_i = subtree_mass[j] * acc[f, j]
                    t_dyn_com[f, j, :] = np.cross(r, F_i)
        
        return t_dyn_com

    def _compute_joint_torques(self, F, ang_acc, world_pos, parents, global_rots, pose_data_aa, tips, options, contact_forces=None, _frame_cache=None, skip_rate_limiting=False, dyn_override_world=None):
        """
        Computes Joint Torques, Efforts, and Inertias.
        Iterates through joints to calculate dynamic vs gravity torques.
        
        Args:
            F (int): Frame count.
            ang_acc (F, 24, 3): Angular acceleration.
            world_pos (F, 24, 3): World positions.
            parents (list): Parent indices.
            global_rots (list[R]): Global rotation objects per joint.
            pose_data_aa (F, 24, 3): Pose for passive limits.
            tips (dict): Cached tip positions (head, hands, feet).
            contact_forces (F, 24, 3): Contact forces (gravity distribution).
            
        Returns:
            torques, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs
        """
        # Output containers
        torques_vec = np.zeros((F, self.target_joint_count, 3))
        inertias = np.zeros((F, self.target_joint_count))
        efforts_net = np.zeros((F, self.target_joint_count, 3))
        efforts_dyn = np.zeros((F, self.target_joint_count, 3))
        efforts_grav = np.zeros((F, self.target_joint_count, 3))
        
        # Vectors debugging arrays
        t_net_vecs = np.zeros((F, self.target_joint_count, 3))
        t_dyn_vecs = np.zeros((F, self.target_joint_count, 3))
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))
        t_passive_vecs = np.zeros((F, self.target_joint_count, 3))
        
        
        
        # Inverse Rotations for Torque Projection (World -> Local)
        # 2. Torque / Effort Calculation (Vectorized)
        
        # Pre-compute Gravity Torques
        t_grav_vecs = self._compute_gravity_torques(world_pos, options, global_rots, tips)
        
        # Pre-compute Contact Torques (World Frame)
        t_contact_vecs = np.zeros((F, self.target_joint_count, 3))
        if contact_forces is not None:
             # Iterate all contact joints (including Tips/Virtual if present)
             num_contact_indices = contact_forces.shape[1]
             for c in range(num_contact_indices):
                  # Check if any force present in batch
                  f_c = contact_forces[:, c, :] # (F, 3)
                  if np.all(np.max(np.abs(f_c), axis=1) < 1e-6): continue
                  
                  p_c = world_pos[:, c, :] # (F, 3)
                  
                  # Traverse up
                  curr = c
                  while curr != -1:
                       if curr < self.target_joint_count:
                            p_curr = world_pos[:, curr, :] # (F, 3)
                            r = p_c - p_curr # (F, 3)
                            tau = np.cross(r, f_c) # (F, 3)
                            t_contact_vecs[:, curr, :] += tau
                       
                       # Move to parent
                       if curr < len(parents):
                            curr = parents[curr]
                       else:
                            break
        


        # Optimization: Cache expensive computations across repeated calls within the same frame.
        # global_rot_invs, inertias, and t_dyn_raw are identical across calls since
        # world_pos, global_rots, and ang_acc don't change — only contact_pressure does.
        if _frame_cache is None:
            _frame_cache = {}
        
        if 'global_rot_invs' in _frame_cache:
            global_rot_invs = _frame_cache['global_rot_invs']
        else:
            global_rot_invs = [r.inv() for r in global_rots]
            _frame_cache['global_rot_invs'] = global_rot_invs

        # --- PASS 1: Compute Raw Dynamic Torques and Inertias ---
        if 'inertias' in _frame_cache:
            inertias = _frame_cache['inertias']
            t_dyn_raw = _frame_cache['t_dyn_raw']
        else:
            inertias = self._compute_all_subtree_inertias(world_pos)  # (F, target_joint_count)
            t_dyn_raw = inertias[:, :, np.newaxis] * ang_acc[:, :self.target_joint_count, :]  # (F, J, 3)
            _frame_cache['inertias'] = inertias
            _frame_cache['t_dyn_raw'] = t_dyn_raw
        
        # --- Velocity Envelope (for adaptive dynamic torque smoothing) ---
        # Track per-joint angular velocity envelope. Used to set an adaptive
        # EMA alpha: low velocity → heavy smoothing (suppress noise on
        # stationary joints), high velocity → pass-through (preserve genuine
        # motion). No hard gate — continuous shading via tanh ramp.
        VEL_REF = 0.5           # rad/s — velocity at which smoothing is ~76% off
        ALPHA_MIN = 0.08        # EMA alpha floor (strongest smoothing when still)
        VEL_SMOOTH_ALPHA = 0.05 # EMA alpha for velocity tracking (slow, rejects spikes)
        VEL_CAP = 3.0           # rad/s — cap per-frame velocity to reject glitches
        
        # Use local-frame angular velocity (each joint's own rotation change)
        # rather than world-frame (which includes inherited parent rotation).
        # Local velocity gives true near-zero for stationary joints.
        ang_vel_for_gate = getattr(self, '_local_ang_vel', None)
        if ang_vel_for_gate is None:
            ang_vel_for_gate = getattr(self, '_current_ang_vel', None)
        vel_alpha = None  # (target_joint_count,) per-joint EMA alpha
        if ang_vel_for_gate is not None:
            if ang_vel_for_gate.ndim == 2:
                ang_vel_for_gate = ang_vel_for_gate[np.newaxis, ...]
            
            # Initialize or retrieve the smoothed velocity envelope
            if not hasattr(self, '_vel_envelope') or self._vel_envelope is None:
                self._vel_envelope = np.zeros(self.target_joint_count)
            if self._vel_envelope.shape[0] != self.target_joint_count:
                self._vel_envelope = np.zeros(self.target_joint_count)
            
            vel_alpha = np.ones(self.target_joint_count)
            for j in range(self.target_joint_count):
                vel_mag = np.linalg.norm(ang_vel_for_gate[:F, j, :], axis=-1)
                for fi in range(F):
                    # Cap extreme single-frame spikes before EMA to prevent
                    # rare glitches (e.g., 22 rad/s during standing) from
                    # inflating the envelope for many subsequent frames.
                    v_capped = min(vel_mag[fi], VEL_CAP)
                    # EMA-smoothed velocity: averages out noise spikes instead
                    # of peaking on them. Tracks sustained motion accurately.
                    self._vel_envelope[j] += VEL_SMOOTH_ALPHA * (v_capped - self._vel_envelope[j])
                
                # Smooth ramp: tanh gives gradual transition, no sharp edge
                vel_frac = np.tanh(self._vel_envelope[j] / VEL_REF)
                vel_alpha[j] = ALPHA_MIN + (1.0 - ALPHA_MIN) * vel_frac
        
        # --- Apply Rate Limiting to Raw Dynamic Torques ---
        if dyn_override_world is not None:
            # CoM-based dynamic torque provided — convert from world to local frame
            n_override = min(self.target_joint_count, dyn_override_world.shape[1])
            t_dyn_limited = np.zeros_like(t_dyn_raw)
            parents_list = self._get_hierarchy()
            for j in range(n_override):
                parent_idx = parents_list[j] if j < len(parents_list) else -1
                if parent_idx >= 0 and parent_idx < len(global_rot_invs):
                    t_dyn_limited[:, j, :] = global_rot_invs[parent_idx].apply(
                        dyn_override_world[:, j, :].reshape(-1, 3)
                    ).reshape(F, 3)
                else:
                    t_dyn_limited[:, j, :] = dyn_override_world[:, j, :]
        elif skip_rate_limiting:
            t_dyn_limited = t_dyn_raw.copy()
        else:
            t_dyn_limited = self._apply_torque_rate_limiting(t_dyn_raw, options)
        
        # --- Velocity-Adaptive Dynamic Torque Smoothing ---
        # Applied to ALL paths (CoM and I×α). Instead of gating (which cuts),
        # this smooths: at low velocity, dynamic torque changes slowly via
        # heavy EMA. At high velocity, it passes through nearly unmodified.
        if vel_alpha is not None and getattr(options, 'enable_velocity_gate', False):
            if not hasattr(self, '_vel_smooth_dyn') or self._vel_smooth_dyn is None:
                self._vel_smooth_dyn = t_dyn_limited.copy()
            elif self._vel_smooth_dyn.shape != t_dyn_limited.shape:
                self._vel_smooth_dyn = t_dyn_limited.copy()
            else:
                for j in range(min(self.target_joint_count, t_dyn_limited.shape[1])):
                    a = vel_alpha[j]
                    self._vel_smooth_dyn[:, j, :] += a * (t_dyn_limited[:, j, :] - self._vel_smooth_dyn[:, j, :])
                t_dyn_limited = self._vel_smooth_dyn.copy()
        
        # --- PASS 2: Compute Net Torques, Passive, and Efforts using Limited Dynamic ---
        # Vectorized world-to-local frame transform using cached numpy rotation matrices
        parents_arr = np.array(self._get_hierarchy())
        
        # Build parent inverse rotation matrices as numpy arrays (cached in _frame_cache)
        if 'parent_inv_mats' not in _frame_cache:
            # global_rots is (F, J, 3, 3) or list of scipy Rotation
            if hasattr(global_rots, 'shape') and global_rots.ndim == 4:
                # Direct numpy rotation matrices
                parent_inv_mats = np.zeros((F, self.target_joint_count, 3, 3))
                for j in range(self.target_joint_count):
                    p = parents_arr[j]
                    if p >= 0 and p < global_rots.shape[1]:
                        parent_inv_mats[:, j] = np.transpose(global_rots[:, p], (0, 2, 1))
                    else:
                        parent_inv_mats[:, j] = np.eye(3)
            else:
                # scipy Rotation objects
                parent_inv_mats = np.zeros((F, self.target_joint_count, 3, 3))
                for j in range(self.target_joint_count):
                    p = parents_arr[j]
                    if p >= 0:
                        parent_inv_mats[:, j] = global_rot_invs[p].as_matrix().reshape(F, 3, 3)
                    else:
                        parent_inv_mats[:, j] = np.eye(3)
            _frame_cache['parent_inv_mats'] = parent_inv_mats
        
        parent_inv_mats = _frame_cache['parent_inv_mats']
        
        # Stash world-frame gravity torques for next-frame contact detection
        # (the structural stream's effort-relief prior needs them in world
        # frame for lean-direction extraction).
        self._last_gravity_torques_world = t_grav_vecs[:, :self.target_joint_count].copy()

        # Batch transform gravity and contact to local frame: (F, J, 3, 3) @ (F, J, 3, 1) -> (F, J, 3)
        t_grav_local_all = np.einsum('fjik,fjk->fji', parent_inv_mats, t_grav_vecs[:, :self.target_joint_count])
        t_contact_local_all = np.einsum('fjik,fjk->fji', parent_inv_mats, t_contact_vecs[:, :self.target_joint_count])
        
        # Dynamic is already in local frame
        t_dyn_vecs[:, :self.target_joint_count] = t_dyn_limited[:, :self.target_joint_count]
        
        # Net local = dyn - grav_local - contact_local
        t_net_all = t_dyn_limited[:, :self.target_joint_count] - t_grav_local_all - t_contact_local_all
        
        # Store local-frame gravity for output consistency
        t_grav_vecs[:, :self.target_joint_count] = t_grav_local_all
        
        # Passive limits (vectorized batch computation)
        t_passive_all = np.zeros_like(t_net_all)
        if options.enable_passive_limits:
            t_passive_all = self._compute_passive_torques_batch(
                t_net_all, pose_data_aa[:, :self.target_joint_count, :]
            )
        
        t_passive_vecs[:, :self.target_joint_count] = t_passive_all
        
        # Active = net - passive
        t_active_all = t_net_all - t_passive_all
        torques_vec[:, :self.target_joint_count] = t_active_all
        
        # Efforts (vectorized over all joints)
        # Magnitude: capacity-relative (||τ / max_torque||) — high when the joint
        # is working close to its limit on any axis.
        # Direction: raw torque direction (τ / ||τ||) — un-skewed by per-axis
        # max_torque differences. Keeping direction aligned with the raw
        # torque vector avoids amplifying L/R asymmetry when one axis of
        # max_torque is much smaller than the others (e.g. shoulder twist).
        max_torque = self.max_torque_array[:self.target_joint_count]  # (J, 3)
        denom = max_torque + 1e-6  # (J, 3)
        efforts_dyn[:, :self.target_joint_count] = t_dyn_limited[:, :self.target_joint_count] / denom[np.newaxis, :, :]
        efforts_grav[:, :self.target_joint_count] = t_grav_local_all / denom[np.newaxis, :, :]
        efforts_net[:, :self.target_joint_count] = self._effort_with_raw_direction(
            t_active_all, denom)

        return torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs



    @staticmethod
    def _effort_with_raw_direction(torques, denom):
        """Effort vector with capacity-relative magnitude and raw torque direction.

        Magnitude: ||τ / max_torque|| (per-axis capacity-relative, same as before).
        Direction: τ / ||τ|| (raw torque direction, NOT skewed by per-axis
        max_torque differences).

        This avoids the prior behavior where the effort vector was tilted
        toward whichever axis had the smallest max_torque — a skew that
        amplified L/R asymmetry for muscles whose flex_axes differed in sign
        on a small-max axis (e.g. shoulder pecs).

        Args:
            torques: (F, J, 3) torque vectors.
            denom:   (J, 3) or (F, J, 3) max_torque + epsilon.

        Returns:
            (F, J, 3) effort vectors: magnitude × raw direction.
        """
        per_axis = torques / (denom if denom.ndim == torques.ndim else denom[np.newaxis])
        eff_mag = np.linalg.norm(per_axis, axis=-1, keepdims=True)  # (..., 1)
        tau_mag = np.linalg.norm(torques, axis=-1, keepdims=True)   # (..., 1)
        # Avoid div-by-zero when a joint has zero torque this frame.
        safe = np.where(tau_mag > 1e-9, tau_mag, 1.0)
        direction = np.where(tau_mag > 1e-9, torques / safe, 0.0)
        return eff_mag * direction

    def _apply_torque_rate_limiting(self, t_dyn_vecs, options):
        """
        Apply rate limiting to dynamic torque vectors to prevent glitches.
        
        Vectorized implementation: uses NumPy array operations instead of
        per-joint Python loops for passes 1-4.
        
        Args:
            t_dyn_vecs (F, J, 3): Raw dynamic torque vectors.
            options (SMPLProcessingOptions): Contains enable_rate_limiting and rate_limit_strength.
            
        Returns:
            t_dyn_limited (F, J, 3): Rate-limited dynamic torque vectors.
        """
        F = t_dyn_vecs.shape[0]
        t_dyn_limited = t_dyn_vecs.copy()
        
        # Initialize previous state if needed
        if self.prev_dynamic_torque is None:
            self.prev_dynamic_torque = np.zeros((self.target_joint_count, 3))
        
        # Ensure shape compatibility
        if self.prev_dynamic_torque.shape[0] != t_dyn_vecs.shape[1]:
            self.prev_dynamic_torque = np.zeros((t_dyn_vecs.shape[1], 3))
        
        num_joints = min(self.target_joint_count, t_dyn_vecs.shape[1])
        
        # Jitter detection parameters (needed for state init regardless)
        JITTER_WINDOW = 8
        JITTER_THRESHOLD = 0.3
        JITTER_DAMPING = 0.3
        
        # Initialize jitter detection state if needed
        if self.torque_sign_history is None:
            self.torque_sign_history = np.zeros((JITTER_WINDOW, num_joints, 3))
            self.jitter_history_idx = 0
        
        if self.torque_sign_history.shape[1] != num_joints:
            self.torque_sign_history = np.zeros((JITTER_WINDOW, num_joints, 3))
            self.jitter_history_idx = 0
        
        # Rate limiting setup (only needed if rate limiting is on)
        if options.enable_rate_limiting:
            if not hasattr(self, 'torque_rate_limits') or self.torque_rate_limits is None:
                self.torque_rate_limits = self._compute_torque_rate_limits()
            
            fps_scale = self.framerate / 60.0
            strength = max(0.01, options.rate_limit_strength)
            effective_limits = (self.torque_rate_limits / fps_scale) * strength
            
            SPIKE_MULTIPLIER = 1.5
            NEIGHBOR_MULTIPLIER = 1.0
            
            # Pre-build parent/children arrays for neighbor propagation
            parents_arr = np.array(self._get_hierarchy()[:num_joints])
            children_list = [[] for _ in range(num_joints)]
            for j in range(num_joints):
                p = parents_arr[j]
                if 0 <= p < num_joints:
                    children_list[p].append(j)
            
            eff_lim_col = effective_limits[:num_joints, np.newaxis]
            spike_thresholds = effective_limits[:num_joints] * SPIKE_MULTIPLIER
            neighbor_thresholds = effective_limits[:num_joints] * NEIGHBOR_MULTIPLIER
        
        for f in range(F):
            curr_torque = t_dyn_vecs[f, :num_joints]
            prev_torque = self.prev_dynamic_torque[:num_joints]
            
            n_spikes = 0
            frame_rate_limits = 0
            
            if options.enable_rate_limiting:
                # --- PASSES 1-3: Spike detection, neighbor propagation, rate clipping ---
                delta = curr_torque - prev_torque
                delta_mags = np.linalg.norm(delta, axis=1)
                curr_mags = np.linalg.norm(curr_torque, axis=1)
                prev_mags = np.linalg.norm(prev_torque, axis=1)
                
                # PASS 1: Detect primary spikes
                is_increasing = curr_mags > prev_mags
                spike_mask = (delta_mags > spike_thresholds) & is_increasing
                
                n_spikes = int(np.sum(spike_mask))
                if n_spikes > 0:
                    spike_indices = np.where(spike_mask)[0]
                    excess = delta_mags[spike_mask] - spike_thresholds[spike_mask]
                    self.noise_stats.spike_severity += float(np.sum(excess))
                    self.noise_stats.joint_spike_counts[spike_indices] += 1
                    self.noise_stats.joint_spike_severity[spike_indices] += excess
                    max_excess = float(np.max(excess))
                    if max_excess > self.noise_stats.max_spike_severity:
                        self.noise_stats.max_spike_severity = max_excess
                    self.noise_stats.spike_severity_list.extend(excess.tolist())
                
                # PASS 2: Propagate to neighbors
                neighbor_mask = np.zeros(num_joints, dtype=bool)
                if n_spikes > 0:
                    spike_indices = np.where(spike_mask)[0]
                    spike_parents = parents_arr[spike_indices]
                    valid_parents = spike_parents[(spike_parents >= 0) & (spike_parents < num_joints)]
                    if len(valid_parents) > 0:
                        neighbor_mask[valid_parents] = True
                    for j in spike_indices:
                        for child in children_list[j]:
                            neighbor_mask[child] = True
                    neighbor_mask &= ~spike_mask
                
                # PASS 3: Apply suppression
                clamped = prev_torque + np.clip(delta, -eff_lim_col, eff_lim_col)
                
                exceeded_pos = delta > eff_lim_col
                if np.any(exceeded_pos):
                    clamped_amounts = delta[exceeded_pos] - np.broadcast_to(eff_lim_col, delta.shape)[exceeded_pos]
                    self.noise_stats.rate_limit_severity += float(np.sum(clamped_amounts))
                    joints_clamped = np.any(exceeded_pos, axis=1)
                    self.noise_stats.joint_rate_limit_counts[np.where(joints_clamped)[0]] += 1
                
                frame_rate_limits = int(np.sum(exceeded_pos))
                
                result = clamped.copy()
                if n_spikes > 0:
                    result[spike_mask] = prev_torque[spike_mask]
                if np.any(neighbor_mask):
                    neighbor_exceeded = (delta_mags > neighbor_thresholds) & is_increasing & neighbor_mask
                    result[neighbor_exceeded] = prev_torque[neighbor_exceeded]
                
                t_dyn_limited[f, :num_joints] = result
            
            # --- PASS 4: Jitter Detection and Damping (vectorized) ---
            if options.enable_jitter_damping:
                curr_signs = np.sign(t_dyn_limited[f, :num_joints])  # (J, 3)
                self.torque_sign_history[self.jitter_history_idx] = curr_signs
                self.jitter_history_idx = (self.jitter_history_idx + 1) % JITTER_WINDOW
                
                # Sign-flip rate per joint (already vectorized in original)
                sign_changes = np.sum(np.abs(np.diff(self.torque_sign_history, axis=0)) > 0, axis=(0, 2))  # (J,)
                flip_rate = sign_changes / (JITTER_WINDOW - 1)
                
                # Vectorized jitter damping
                jittery = flip_rate > JITTER_THRESHOLD  # (J,) bool
                if np.any(jittery):
                    jitter_excess = np.clip(
                        (flip_rate[jittery] - JITTER_THRESHOLD) / (1.0 - JITTER_THRESHOLD + 1e-6),
                        0.0, 1.0
                    )  # (n_jittery,)
                    damping = 1.0 - jitter_excess * (1.0 - JITTER_DAMPING)  # (n_jittery,)
                    # Apply damping: blend between current and previous
                    t_dyn_limited[f, np.where(jittery)[0]] = (
                        damping[:, np.newaxis] * t_dyn_limited[f, np.where(jittery)[0]] +
                        (1.0 - damping[:, np.newaxis]) * prev_torque[jittery]
                    )
                    frame_jitter = int(np.sum(jittery))
                    self.noise_stats.joint_jitter_counts[np.where(jittery)[0]] += 1
                else:
                    frame_jitter = 0
            else:
                frame_jitter = 0
            
            # --- PASS 5: SmartClampKF Filtering (unchanged) ---
            frame_innovation_clamps = 0
            if options.enable_kf_smoothing:
                dt = 1.0 / self.framerate
                
                if self.dynamic_torque_kf is None:
                    self.dynamic_torque_kf = NumpySmartClampKF(dt, num_joints, 3)
                
                # Update params each frame so widget changes take effect
                self.dynamic_torque_kf.update_params(
                    options.kf_responsiveness, options.kf_smoothness,
                    options.kf_clamp_radius, dt
                )
                
                self.dynamic_torque_kf.predict()
                
                pre_update = t_dyn_limited[f].copy()
                t_dyn_limited[f] = self.dynamic_torque_kf.update(t_dyn_limited[f])
                
                innovation_mag = np.linalg.norm(pre_update - t_dyn_limited[f], axis=1)
                clamped_mask = innovation_mag > options.kf_clamp_radius * 0.9
                clamp_count = int(np.sum(clamped_mask))
                frame_innovation_clamps = clamp_count
                if clamp_count > 0:
                    excess_innovations = innovation_mag[clamped_mask] - options.kf_clamp_radius * 0.9
                    self.noise_stats.innovation_severity += float(np.sum(excess_innovations))
            
            # Update prev for next frame
            self.prev_dynamic_torque = t_dyn_limited[f, :num_joints].copy()
            
            # Update cumulative noise stats
            self.noise_stats.total_frames += 1
            self.noise_stats.spike_detections += n_spikes
            self.noise_stats.rate_limit_clamps += frame_rate_limits
            self.noise_stats.jitter_damping_events += frame_jitter
            self.noise_stats.innovation_clamps += frame_innovation_clamps
        
        return t_dyn_limited

    def _compute_balance_stability(self, world_pos, tips, options):
        """
        Compute continuous balance stability metrics.

        Combines CoM, ZMP, and the support polygon (convex hull of active
        contact points) into a 0–1 stability score, signed margins, an
        imbalance direction vector, and the polygon vertices.

        Returns:
            dict with keys:
                stability_score (float): 0 = falling/airborne, 1 = centered.
                com_margin (float): Signed distance of CoM to polygon edge (m).
                zmp_margin (float): Signed distance of ZMP to polygon edge (m).
                imbalance_vector (np.array (3,)): World-space direction from
                    polygon centroid toward ZMP. Magnitude = distance from
                    center.  Lies on the ground plane (y_dim = 0).
                support_polygon (np.array (N, 2)): Ordered convex hull vertices.
                support_radius (float): Inscribed radius of support polygon (m).
        """
        y_dim = getattr(self, 'internal_y_dim', 1)
        plane_dims = [0, 2] if y_dim == 1 else [0, 1]
        floor_h = getattr(self, '_inferred_floor_height', options.floor_height)

        # Defaults for the airborne / no-contact case
        zero3 = np.zeros(3)
        default = {
            'stability_score': 0.0,
            'com_margin': 0.0,
            'zmp_margin': 0.0,
            'imbalance_vector': zero3.copy(),
            'support_polygon': np.zeros((0, 2)),
            'support_radius': 0.0,
        }

        # --- 1. Collect active support points ---
        pressure = self.contact_pressure  # (F, J)
        if pressure is None or pressure.size == 0:
            return default

        f = 0  # streaming: single frame
        p_f = pressure[f] if pressure.ndim > 1 else pressure
        PRESSURE_THRESH = 0.5  # kg — minimum load to count as support

        pts_2d = []
        J = p_f.shape[0] if hasattr(p_f, 'shape') else len(p_f)
        for j in range(J):
            if p_f[j] < PRESSURE_THRESH:
                continue
            if tips is not None and j in tips:
                pos = tips[j][0] if tips[j].ndim > 1 else tips[j]
            elif j < world_pos.shape[1]:
                pos = world_pos[0, j]
            else:
                continue
            pts_2d.append(pos[plane_dims])

        if len(pts_2d) < 1:
            return default

        pts_2d = np.array(pts_2d)  # (N, 2)

        # --- 2. Build support polygon (convex hull) ---
        from scipy.spatial import ConvexHull

        if len(pts_2d) == 1:
            # Single contact: treat as small circle
            centroid = pts_2d[0]
            support_radius = 0.05  # ~5 cm effective radius for a foot point
            hull_verts = pts_2d
        elif len(pts_2d) == 2:
            # Line segment: use perpendicular distance
            centroid = np.mean(pts_2d, axis=0)
            support_radius = np.linalg.norm(pts_2d[1] - pts_2d[0]) * 0.5
            hull_verts = pts_2d
        else:
            try:
                hull = ConvexHull(pts_2d)
                hull_verts = pts_2d[hull.vertices]
                centroid = np.mean(hull_verts, axis=0)

                # Inscribed radius: min distance from centroid to each edge
                n_v = len(hull_verts)
                min_edge_dist = np.inf
                for i in range(n_v):
                    a = hull_verts[i]
                    b = hull_verts[(i + 1) % n_v]
                    edge = b - a
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-8:
                        continue
                    edge_n = np.array([-edge[1], edge[0]]) / edge_len  # outward normal
                    d = abs(np.dot(centroid - a, edge_n))
                    if d < min_edge_dist:
                        min_edge_dist = d
                support_radius = min_edge_dist if np.isfinite(min_edge_dist) else 0.05
            except Exception:
                # Degenerate hull (collinear points etc)
                centroid = np.mean(pts_2d, axis=0)
                support_radius = 0.05
                hull_verts = pts_2d

        # --- 3. Signed distance of CoM and ZMP to polygon boundary ---
        com = self.current_com
        if com is not None:
            com_hz = (com[0] if com.ndim > 1 else com)[plane_dims]
        else:
            com_hz = centroid.copy()

        zmp = self.current_zmp
        if zmp is not None:
            zmp_hz = (zmp[0] if zmp.ndim > 1 else zmp)[plane_dims]
        else:
            zmp_hz = com_hz.copy()

        def signed_distance_to_polygon(point, verts):
            """Positive = inside, negative = outside."""
            n_v = len(verts)
            if n_v < 2:
                return np.linalg.norm(point - verts[0]) * -1.0 if n_v == 1 else 0.0
            if n_v == 2:
                # Line segment: perpendicular distance (always "outside")
                a, b = verts[0], verts[1]
                seg = b - a
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-8:
                    return -np.linalg.norm(point - a)
                t = np.clip(np.dot(point - a, seg) / (seg_len**2), 0, 1)
                proj = a + t * seg
                return -np.linalg.norm(point - proj)

            # Full polygon: min distance to each edge, sign from winding
            min_dist = np.inf
            all_positive = True
            for i in range(n_v):
                a = verts[i]
                b = verts[(i + 1) % n_v]
                edge = b - a
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-8:
                    continue
                # Outward normal (CCW hull → left normal is outward)
                normal = np.array([-edge[1], edge[0]]) / edge_len
                d = np.dot(point - a, normal)
                if d > 0:
                    all_positive = False
                abs_d = abs(d)
                if abs_d < min_dist:
                    min_dist = abs_d

            # For CCW-ordered convex hull, all normals should point outward.
            # Point is inside if dot products with ALL outward normals are ≤ 0.
            # scipy ConvexHull returns CCW vertices, so outward normal = left normal.
            # d ≤ 0 for ALL edges → inside.
            if all_positive:
                # All cross products > 0 → outside (depends on winding)
                return -min_dist
            elif not all_positive and min_dist < np.inf:
                # Check: inside = all d ≤ 0 for outward normals
                # With left normal as outward, inside → d ≤ 0
                # Let's do proper inside check
                pass

            # Robust inside check: use winding-based approach
            inside = self._point_in_convex_polygon(point, verts)
            return min_dist if inside else -min_dist

        com_margin = signed_distance_to_polygon(com_hz, hull_verts)
        zmp_margin = signed_distance_to_polygon(zmp_hz, hull_verts)

        # --- 4. Stability score ---
        # Sigmoid scoring: gives continuous feedback even near/outside polygon edge.
        # margin=0 (on edge)  → 0.5
        # margin=+3cm (inside) → ~0.73
        # margin=+10cm         → ~0.96
        # margin=-3cm (outside) → ~0.27
        # margin=-10cm          → ~0.04
        # Falloff of 3cm matches typical contact polygon noise level.
        FALLOFF = 0.03  # meters — characteristic transition width
        # Clamp the exponent so math.exp can't overflow when ZMP lies far
        # outside the polygon (near-freefall toe-off/heel-strike, where GRF≈0
        # and the cart-table ZMP is ill-conditioned). Casting to a plain
        # Python float also avoids ufunc dispatch issues with exotic scalar
        # dtypes that have surfaced as scipy.special.expit _UFuncNoLoopError.
        ratio = max(-50.0, min(50.0, float(zmp_margin) / FALLOFF))
        stability_score = 1.0 / (1.0 + math.exp(-ratio))

        # --- 5. Imbalance direction vector (3D, on ground plane) ---
        offset_2d = zmp_hz - centroid
        imbalance = np.zeros(3)
        imbalance[plane_dims[0]] = offset_2d[0]
        imbalance[plane_dims[1]] = offset_2d[1]
        # y_dim stays 0 (lies on ground plane)

        return {
            'stability_score': stability_score,
            'com_margin': float(com_margin),
            'zmp_margin': float(zmp_margin),
            'imbalance_vector': imbalance,
            'support_polygon': hull_verts,
            'support_radius': float(support_radius),
        }

    @staticmethod
    def _point_in_convex_polygon(point, verts):
        """Check if 2D point is inside a convex polygon using cross-product test."""
        n = len(verts)
        if n < 3:
            return False
        sign = None
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
            if abs(cross) < 1e-10:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    def process_frame(self, pose_data, trans_data, options, effort_pose_data=None):
        """
        Process a single frame or batch of frames.
        Tracks state for torque calculation if frames are passed one by one.
        quat_format: 'xyzw' (default, Scipy) or 'wxyz' (Scalar first). Only used if input_type='quat'.
        effort_pose_data: Optional separate pose stream for Calculation of Effort (AngAcc).
        """
        # --- S-Curve Spine Toggle Detection ---
        # If use_s_curve_spine changed since last call, recompute geometry
        new_s_curve = getattr(options, 'use_s_curve_spine', True)
        if new_s_curve != self.use_s_curve_spine:
            self.use_s_curve_spine = new_s_curve
            self.limb_data = self._compute_limb_properties()
            self.skeleton_offsets = self._compute_skeleton_offsets()
            self.reset_physics_state()  # Prevent torque spikes from geometry change
        
        # --- Coherence gate (opt-in): compute ONCE per frame from the RAW
        # pose (before any smoothing), then reuse at the adaptive-effort EMA
        # alpha_max lift and (if enabled) the adaptive front-end filter.
        # gate[j] in [0,1]: 0 = full smoothing (noise floor protected — e.g.
        # quiet-period magnetometer jitter / 1-frame optical glitch), 1 =
        # lighter smoothing (ballistic accent recovered).
        # Geodesic speed is permutation-invariant, so raw pre-permutation
        # pose is fine here. axis_angle input only; else no gating (safe). ---
        self._cg_gate = None
        if (getattr(options, 'coherence_gate_enable', False)
                or getattr(options, 'adaptive_frontend', False)) \
                and getattr(options, 'input_type', 'axis_angle') == 'axis_angle':
            aa_raw = np.asarray(pose_data, dtype=np.float64).reshape(-1, 3)
            if aa_raw.shape[0] >= 1:
                self._cg_gate = self._coherence_envelope_gate(
                    aa_raw, options, min(24, aa_raw.shape[0]))

        # --- Adaptive Front-End Filter (opt-in) ---
        # Single per-joint One Euro Filter on the pose whose cutoff is driven
        # by the coherence gate (gate 0 → low cutoff = heavy smoothing =
        # glitch-safe; gate 1 → high cutoff = accent passes). Replaces the
        # fixed moving-average below; one recursive state, so no dual-pipeline
        # / state-continuity issue. Trans keeps the fixed MA (see below).
        used_adaptive_frontend = False
        _af_stream = (np.ndim(pose_data) <= 2) or (np.ndim(pose_data) == 3 and np.shape(pose_data)[0] == 1)
        if getattr(options, 'adaptive_frontend', False) and self._cg_gate is not None \
                and _af_stream \
                and getattr(options, 'input_type', 'axis_angle') == 'axis_angle':
            pose_data = np.array(pose_data, dtype=np.float64)
            p_flat = pose_data.reshape(-1)
            n_dof = p_flat.size
            g = self._cg_gate
            clo = float(getattr(options, 'adaptive_frontend_cutoff_lo', 1.0))
            chi = float(getattr(options, 'adaptive_frontend_cutoff_hi', 12.0))
            njg = min(g.shape[0], n_dof // 3)
            cutoff = np.full(n_dof, clo, dtype=np.float64)
            if njg > 0:
                cutoff[:njg * 3] = np.repeat(clo + g[:njg] * (chi - clo), 3)
            oef = getattr(self, '_frontend_oef', None)
            if oef is None or getattr(self, '_frontend_oef_n', 0) != n_dof:
                oef = OneEuroFilter(min_cutoff=cutoff,
                                    beta=float(getattr(options, 'adaptive_frontend_beta', 0.0)),
                                    framerate=self.framerate)
                self._frontend_oef = oef
                self._frontend_oef_n = n_dof
            oef._mincutoff = cutoff
            oef._beta = float(getattr(options, 'adaptive_frontend_beta', 0.0))
            pose_data = oef(p_flat).reshape(pose_data.shape)
            used_adaptive_frontend = True

        # --- Optional Input Smoothing ---
        # Causal moving average for general-purpose smoothing.
        # A 3-frame window naturally nulls the 33.3Hz magnetometer cadence
        # artifact present in Shadow IMU data.
        # In adaptive-front-end mode the pose is already filtered above; the MA
        # then only smooths trans (root translation). Force an effective trans
        # window in that mode — otherwise smooth_input_window=0 (the candidate
        # adaptive config) would silently leave trans completely unsmoothed.
        win = options.smooth_input_window
        if used_adaptive_frontend:
            win = max(win, int(getattr(options, 'adaptive_frontend_trans_window', 5)))

        # Independent trans window (ms → frames at this fps). 0 = legacy:
        # trans follows the pose/front-end window above. See the option doc —
        # the trans sensor flops during extreme movement, so its window is
        # fixed and never gate-driven.
        trans_ms = float(getattr(options, 'smooth_trans_window_ms', 0.0))
        if trans_ms > 0.0:
            trans_win = max(2, int(round(trans_ms / 1000.0 * self.framerate)))
        else:
            trans_win = win

        if win >= 2 or trans_win >= 2:
            pose_data = np.array(pose_data, dtype=np.float64)
            trans_data = np.array(trans_data, dtype=np.float64)

            # Determine if streaming (single-frame) or batch
            p_flat = pose_data.reshape(-1) if pose_data.ndim > 0 else pose_data
            t_flat = trans_data.reshape(-1) if trans_data.ndim > 0 else trans_data
            n_pose = p_flat.size
            n_trans = t_flat.size

            # Check if this is streaming mode (single frame input)
            is_streaming = (pose_data.ndim <= 2) or (pose_data.ndim == 3 and pose_data.shape[0] == 1)

            if is_streaming:
                # Ring buffers for streaming (pose and trans windows may differ)
                ring = getattr(self, '_input_smooth_ring', None)
                if ring is None \
                        or ring.get('n_pose') != n_pose \
                        or ring.get('win') != win \
                        or ring.get('trans_win') != trans_win:
                    ring = {
                        'win': win,
                        'trans_win': trans_win,
                        'n_pose': n_pose,
                        'n_trans': n_trans,
                        'pose_buf': np.tile(p_flat, (win, 1)) if win >= 2 else None,
                        'trans_buf': np.tile(t_flat, (trans_win, 1)) if trans_win >= 2 else None,
                        'idx': 0,
                        'trans_idx': 0,
                    }
                    self._input_smooth_ring = ring
                if ring['pose_buf'] is not None:
                    ring['pose_buf'][ring['idx']] = p_flat
                    ring['idx'] = (ring['idx'] + 1) % win
                    # Skip the pose MA when the adaptive front-end already
                    # filtered the pose (avoid double smoothing).
                    if not used_adaptive_frontend:
                        pose_data = np.mean(ring['pose_buf'], axis=0).reshape(pose_data.shape)
                if ring['trans_buf'] is not None:
                    ring['trans_buf'][ring['trans_idx']] = t_flat
                    ring['trans_idx'] = (ring['trans_idx'] + 1) % trans_win
                    trans_data = np.mean(ring['trans_buf'], axis=0).reshape(trans_data.shape)
            else:
                # Batch mode: causal moving average with edge padding
                F_in = pose_data.shape[0] if pose_data.ndim >= 2 else 1
                if F_in > 1:
                    from numpy.lib.stride_tricks import sliding_window_view
                    if win >= 2:
                        p2d = pose_data.reshape(F_in, -1)
                        p_pad = np.concatenate([np.tile(p2d[0:1], (win - 1, 1)), p2d], axis=0)
                        pose_data = np.mean(sliding_window_view(p_pad, win, axis=0), axis=-1).reshape(pose_data.shape)
                    if trans_win >= 2:
                        t2d = trans_data.reshape(F_in, -1)
                        t_pad = np.concatenate([np.tile(t2d[0:1], (trans_win - 1, 1)), t2d], axis=0)
                        trans_data = np.mean(sliding_window_view(t_pad, trans_win, axis=0), axis=-1).reshape(trans_data.shape)
        
        # Prepare Data (Reshape, Permute, Convert)
        trans_data, pose_data_aa, quats = self._prepare_trans_and_pose(
            pose_data, trans_data, options
        )
        
        # Reset Per-Frame Caches
        # Save previous CoM for stability contact method (which runs before new CoM is computed)
        self._prev_com_for_stability = getattr(self, 'current_com', None)
        self.current_com = None
        self.current_total_mass = None
        
        F = trans_data.shape[0]
        
        # 1. Forward Kinematics (Vectorized)
        world_pos, global_rots, tips = self._compute_forward_kinematics(trans_data, quats)
        parents = self._get_hierarchy()
        
        # Store global rotation matrices for frame evaluator surface distance
        n_rots = min(24, len(global_rots))
        _global_rot_mats = np.eye(3)[np.newaxis].repeat(n_rots, axis=0)  # (24, 3, 3)
        for j in range(n_rots):
            if global_rots[j] is not None:
                _global_rot_mats[j] = global_rots[j].as_matrix()
        self._prev_global_rots = _global_rot_mats





        # --- Angular Kinematics (Main Path: Contact/Gravity) ---
        # Always compute local angular kinematics (needed as fallback)
        ang_vel_local, ang_acc_local = self._compute_angular_kinematics(
            F, pose_data_aa, quats, options, use_filter=options.enable_one_euro_filter, state_suffix=''
        )
        
        # --- World-Frame vs Local-Frame Selection ---
        if options.world_frame_dynamics:
            # World-frame: compute angular velocity/acceleration from global rotations
            # Each segment measured independently — parent noise doesn't propagate
            ang_vel_world, ang_acc_world = self._compute_world_angular_kinematics(
                F, global_rots, options, state_suffix=''
            )
            ang_acc = ang_acc_world
            self._current_ang_vel = ang_vel_world  # World-frame velocity for gating
            self._local_ang_vel = ang_vel_local     # Local velocity for vel-adaptive smoothing
        else:
            # Legacy local-frame: parent-relative angular acceleration
            ang_acc = ang_acc_local
            self._current_ang_vel = ang_vel_local  # Local velocity for gating
            self._local_ang_vel = ang_vel_local     # Same as above when local mode
        
        # --- Angular Kinematics (Dual Path: Effort) ---
        ang_acc_for_effort = ang_acc # Default to main path
        
        if effort_pose_data is not None:
             # Process Effort Pose (Assume same Trans/Options)
             dummy_trans = trans_data
             _, effort_aa, effort_quats = self._prepare_trans_and_pose(
                 effort_pose_data, dummy_trans, options
             )
             
             if options.world_frame_dynamics:
                 # Effort path needs its own FK for global rotations
                 # Run FK on effort pose to get global_rots for effort
                 _, effort_global_rots, _ = self._compute_forward_kinematics(dummy_trans, effort_quats)
                 _, effort_ang_acc = self._compute_world_angular_kinematics(
                     F, effort_global_rots, options, state_suffix='_effort'
                 )
             else:
                 _, effort_ang_acc = self._compute_angular_kinematics(
                     F, effort_aa, effort_quats, options, use_filter=options.enable_one_euro_filter, state_suffix='_effort'
                 )
             ang_acc_for_effort = effort_ang_acc


        if ang_acc_for_effort.ndim == 2:
             ang_acc_for_effort = ang_acc_for_effort[np.newaxis, ...]
             
        if ang_acc.ndim == 2:
             ang_acc = ang_acc[np.newaxis, ...]
        
        # --- Probabilistic Contact (Moved Early for Dynamics) ---
        # --- Probabilistic Contact (Moved Early for Dynamics) ---
        self._update_physics_state(
             world_pos, pose_data_aa, options
        )
        
        
        # --- Contact Method Selection ---
        if options.contact_method == 'stability_v2':
             contact_probs_fusion = self._compute_probabilistic_contacts_stability_v2(
                  F, world_pos.shape[1], world_pos, options
             )
        elif options.contact_method == 'logodds':
             contact_probs_fusion = self._compute_probabilistic_contacts_logodds(
                  F, world_pos.shape[1], world_pos, options
             )
        elif options.contact_method == 'logodds_valved':
             contact_probs_fusion = self._compute_probabilistic_contacts_logodds(
                  F, world_pos.shape[1], world_pos, options, enable_valving=True
             )
        else:  # Default fallback: logodds_valved
             contact_probs_fusion = self._compute_probabilistic_contacts_logodds(
                  F, world_pos.shape[1], world_pos, options, enable_valving=True
             )
        
        # Standalone frame evaluator call removed — logodds structural stream
        # provides FE results internally; stability_v2 has its own pressure path.
        _frame_cache = {}
        dt = options.dt if hasattr(options, 'dt') and options.dt > 0 else 1.0/30.0
        working_probs = contact_probs_fusion.copy()
        
        # --- Adaptive floor height estimation ---
        # Runs BEFORE the unified/legacy branch so ALL methods benefit.
        # Previously this was inside the legacy pipeline and skipped by
        # unified/equilibrium, causing the floor to stay at 0.0.
        inferred_floor = self._update_adaptive_floor(
            world_pos, tips, working_probs, options)
        
        if options.contact_method in ('logodds', 'logodds_valved'):
            # Unified/equilibrium: use the method's own pressure directly.
            # Skip both frame_eval and legacy pipelines — they are redundant
            # and cause oscillation artifacts.
            J_cp = world_pos.shape[1]
            self.contact_pressure = np.zeros((F, J_cp))
            stab_press = getattr(self, '_stability_computed_pressure', None)
            if stab_press is not None and len(stab_press) == J_cp:
                for f_idx in range(F):
                    self.contact_pressure[f_idx] = stab_press
        else:
            # --- Legacy Pressure Pipeline ---
            working_probs = contact_probs_fusion.copy()
            # --- Weighted Mass Distribution (for RNE GRF) ---
            self.contact_pressure = np.zeros((F, world_pos.shape[1]))
            
            # Per-joint vertical velocity for liftoff suppression
            yd = getattr(self, 'internal_y_dim', 1)
            dt = options.dt if hasattr(options, 'dt') and options.dt > 0 else 1.0/30.0
            n_joints = world_pos.shape[1]
            
            # Compute current joint heights (using tips where available)
            curr_heights = np.zeros(n_joints)
            for j in range(n_joints):
                if j in tips:
                    curr_heights[j] = tips[j][0, yd] if tips[j].ndim > 1 else tips[j][yd]
                elif j < world_pos.shape[1]:
                    curr_heights[j] = world_pos[0, j, yd]
            
            # Compute per-joint VY
            if not hasattr(self, '_prev_joint_heights') or self._prev_joint_heights is None or self._prev_joint_heights.shape[0] != n_joints:
                self._prev_joint_heights = curr_heights.copy()
            
            joint_vy = (curr_heights - self._prev_joint_heights) / dt
            self._prev_joint_heights = curr_heights.copy()
            
            # --- Adaptive floor height estimation ---
            # Maintain a running estimate of the actual floor height from
            # the lowest confirmed-contact joints. This disambiguates
            # tip-toes (foot near inferred floor) from en pointe (foot
            # clearly above inferred floor).
            # STABILITY: Very slow EMA + per-frame clamp. Real floors
            # don't move, so this estimate should be rock-solid.
            # But for IMU data, body-shape offsets can put planted feet
            # at 0.10m+, so we need fast initial convergence.
            _floor_age = getattr(self, '_floor_adapt_age', 0)
            if _floor_age < 60:
                # First ~1 second: converge quickly to planted-foot height
                FLOOR_ALPHA = 0.15
                FLOOR_MAX_CHANGE = 0.01   # 10mm per frame
            else:
                FLOOR_ALPHA = 0.02        # Very slow EMA
                FLOOR_MAX_CHANGE = 0.002  # Max 2mm per frame
            self._floor_adapt_age = _floor_age + 1
            
            if not hasattr(self, '_inferred_floor_height') or self._inferred_floor_height is None:
                # Initialize from minimum foot height (not from options)
                foot_joints = [j for j in [10, 11] if j < len(curr_heights)]
                if foot_joints:
                    self._inferred_floor_height = min(
                        curr_heights[j] for j in foot_joints)
                else:
                    self._inferred_floor_height = options.floor_height
            
            # Update floor estimate from high-confidence contact joints
            all_contact_joints = [7, 8, 10, 11]
            for vi in [24, 25, 28, 29]:
                if vi < n_joints:
                    all_contact_joints.append(vi)
            
            confirmed_heights = []
            for j in all_contact_joints:
                if j < working_probs.shape[1] and working_probs[0, j] > 0.5:
                    confirmed_heights.append(curr_heights[j])
            
            # Fallback for equilibrium method: use computed pressure
            # (breaks chicken-and-egg: floor needs contacts, contacts need floor)
            if not confirmed_heights:
                stab_press = getattr(self, '_stability_computed_pressure', None)
                if stab_press is not None:
                    for j in all_contact_joints:
                        if j < len(stab_press) and stab_press[j] > 1.0:
                            confirmed_heights.append(curr_heights[j])
            
            # Secondary fallback: track minimum foot height (very slow)
            # This ensures floor adapts even during single-foot stance
            if not confirmed_heights:
                foot_joints = [j for j in [10, 11] if j < len(curr_heights)]
                if foot_joints:
                    min_foot_h = min(curr_heights[j] for j in foot_joints)
                    # Only adapt upward from current estimate and only if
                    # the foot is reasonably close to current floor
                    if min_foot_h < self._inferred_floor_height + options.floor_tolerance:
                        confirmed_heights.append(min_foot_h)
            
            if confirmed_heights:
                lowest_confirmed = min(confirmed_heights)
                raw_update = (
                    self._inferred_floor_height * (1 - FLOOR_ALPHA) +
                    lowest_confirmed * FLOOR_ALPHA
                )
                # Clamp change per frame for stability
                delta = raw_update - self._inferred_floor_height
                delta = np.clip(delta, -FLOOR_MAX_CHANGE, FLOOR_MAX_CHANGE)
                self._inferred_floor_height += delta
            
            inferred_floor = self._inferred_floor_height
            
            # --- Foot probability promotion ---
            # Toe joint rotations are poorly measured in mocap.
            # If toe or heel shows contact, the ball-of-foot (joint 10/11)
            # is almost certainly also in contact. Promote foot prob to
            # match, UNLESS the foot is clearly above the inferred floor
            # (en pointe: foot elevated >8cm above where contacts occur).
            EN_POINTE_ABOVE_FLOOR = 0.08  # 8cm above inferred floor
            foot_pairs = [
                (10, 24, 28),  # left:  foot, toe, heel
                (11, 25, 29),  # right: foot, toe, heel
            ]
            for f in range(F):
                probs_f = working_probs[f]
                for foot_j, toe_j, heel_j in foot_pairs:
                    if foot_j >= probs_f.shape[0]: continue
                    
                    # Get extremity probabilities
                    toe_p = probs_f[toe_j] if toe_j < probs_f.shape[0] else 0
                    heel_p = probs_f[heel_j] if heel_j < probs_f.shape[0] else 0
                    extremity_max = max(toe_p, heel_p)
                    
                    if extremity_max > probs_f[foot_j]:
                        # En pointe check: is the foot clearly above the
                        # inferred floor? If yes, toes take all pressure.
                        # If near the floor (tip-toes), promote foot.
                        foot_above_floor = curr_heights[foot_j] - inferred_floor
                        if foot_above_floor < EN_POINTE_ABOVE_FLOOR:
                            # Normal foot or tip-toes — promote foot prob
                            working_probs[f, foot_j] = extremity_max
            
            for f in range(F):
                 probs = working_probs[f]
                 
                 # Scale total mass by the maximum contact probability among
                 # ALL contact candidate joints. This ensures cartwheels,
                 # handstands, crawling, and kneeling produce proper pressure.
                 # When airborne (max p ≈ 0.02), near-zero pressure.
                 # When any joint is grounded (max p ≥ 0.5), full weight.
                 # Uses 2x scaling clamped to [0,1] so p=0.5 → full weight.
                 all_contact_indices = [0, 4, 5, 7, 8, 10, 11, 18, 19, 20, 21, 22, 23]
                 for vi in [24, 25, 28, 29]:     # toes, heels (virtual)
                     if vi < probs.shape[0]:
                         all_contact_indices.append(vi)
                 max_contact_prob = max(probs[j] for j in all_contact_indices if j < probs.shape[0])
                 grf_scale = min(1.0, 2.0 * max_contact_prob)  # Saturates at p=0.5
                 total_contact_mass = self.total_mass_kg * grf_scale
                 
                 # Use CoM (not ZMP) as the Gaussian center for pressure
                 # distribution. During weight transfer, the CoM shifts
                 # laterally toward the receiving foot faster than the ZMP,
                 # correctly unloading the departing foot.
                 if self.current_com is not None:
                     com_f = self.current_com[f] if self.current_com.ndim > 1 else self.current_com
                 else:
                     com_f = self.current_zmp[f] if self.current_zmp.ndim > 1 else self.current_zmp
                 pd = [0, 2] if yd == 1 else [0, 1]
                 center_hz = com_f[pd]
                 
                 w_dist = np.zeros_like(probs)
                 
                 for j in range(probs.shape[0]):
                      p = probs[j]
                      if p < 0.001: continue
                      
                      # Per-joint upward velocity suppression:
                      # If this joint is moving upward, it's lifting off —
                      # suppress its contact pressure contribution.
                      vy_j = joint_vy[j] if j < len(joint_vy) else 0.0
                      if vy_j > 0.05:  # Small deadzone to ignore noise
                          # Exponential suppression: vy=0.15 → 37%, vy=0.30 → 14%, vy=0.5 → 4%
                          vel_suppress = np.exp(-vy_j / 0.15)
                          p = p * vel_suppress
                          if p < 0.001: continue
                      
                      if j in tips: pos = tips[j][f]
                      else: pos = world_pos[f, j]
                      
                      pos_hz = pos[pd]
                      dist = np.sqrt(np.sum((pos_hz - center_hz)**2))
                      
                      SIGMA_PRESS = 0.60  # Wide Gaussian — reduces sensitivity to CoM jitter
                      gauss_weight = np.exp(-0.5 * (dist / SIGMA_PRESS)**2)
                      # Use linear p for stability-boosted contacts, p² for standard
                      # This prevents the squaring from crushing boosted contacts
                      p_weight = p * p
                      w_dist[j] = p_weight * gauss_weight
                      
                 w_sum = np.sum(w_dist)
                 
                 if w_sum > 1e-6:
                      weights = w_dist / w_sum
                      
                      for j in range(weights.shape[0]):
                           if weights[j] < 0.001:
                                continue
                           if j in tips:
                                h = tips[j][f][yd] - inferred_floor
                           else:
                                h = world_pos[f, j, yd] - inferred_floor
                           # Soft height penalty instead of hard cutoff
                           if h > 0.15:
                                height_fade = np.exp(-((h - 0.15) / 0.10)**2)
                                weights[j] *= height_fade
                      
                      w_sum_2 = np.sum(weights)
                      if w_sum_2 > 0:
                           weights = weights / w_sum_2
                           
                           # --- Toe pressure capping ---
                           # Toe positions are unreliable (rotation poorly
                           # measured). Cap each toe's weight to not exceed
                           # its foot (ball) weight, unless en pointe.
                           toe_foot_pairs = [(24, 10), (25, 11)]
                           for toe_j, foot_j in toe_foot_pairs:
                                if toe_j >= weights.shape[0] or foot_j >= weights.shape[0]:
                                     continue
                                if weights[toe_j] <= weights[foot_j]:
                                     continue
                                # En pointe check
                                foot_h = curr_heights[foot_j] - inferred_floor
                                if foot_h > EN_POINTE_ABOVE_FLOOR:
                                     continue
                                # Cap: redistribute excess to foot
                                excess = weights[toe_j] - weights[foot_j]
                                weights[toe_j] = weights[foot_j]
                                weights[foot_j] += excess
                           
                           self.contact_pressure[f] = weights * total_contact_mass
                      else:
                           self.contact_pressure[f] = 0.0
                 else:
                      self.contact_pressure[f] = 0.0
            
            # --- Stability Pressure Override ---
            # When using inverse statics, the stability method computes
            # physics-based pressure directly. Replace the probability-
            # derived pressure entirely.
            if options.contact_method in ('stability_v2',):
                stab_press = getattr(self, '_stability_computed_pressure', None)
                if stab_press is not None and len(stab_press) == self.contact_pressure.shape[1]:
                    for f in range(F):
                        self.contact_pressure[f] = stab_press
            # end legacy pressure pipeline
        
        # --- Contact Pressure Smoothing (Asymmetric + Rate Clamp) ---
        # Runs OUTSIDE the refinement loop so it always applies.
        # Time-constant based smoothing (framerate-adaptive).
        #
        # Skip for unified/equilibrium: these methods have internal state
        # machines with hysteresis — external smoothing is redundant and
        # creates oscillation artifacts (zero-snap → ramp-up → zero-snap).
        # LogOdds: the accumulator handles on/off transitions, but the
        # pressure distribution (force share via XCoM lever) is stateless
        # and needs smoothing to prevent gravity torque flicker.
        _skip_smoothing = False
        
        if not _skip_smoothing:
            if options.contact_method in ('logodds', 'logodds_valved'):
                # LogOdds structural stream provides frame-evaluator-grade results
                TAU_UP = 0.015
                TAU_DOWN = 0.025
            elif options.contact_method in ('stability_v2',):
                TAU_UP = 0.030
                TAU_DOWN = 0.030
            else:
                TAU_UP = 0.150
                TAU_DOWN = 0.050
            MAX_RATE = 5.0 * self.total_mass_kg

            alpha_up   = 1.0 - np.exp(-dt / TAU_UP)
            alpha_down = 1.0 - np.exp(-dt / TAU_DOWN)
            max_change_per_frame = MAX_RATE * dt

            if not hasattr(self, 'prev_contact_pressure_smooth') or self.prev_contact_pressure_smooth is None:
                self.prev_contact_pressure_smooth = self.contact_pressure.copy()

            if self.prev_contact_pressure_smooth.shape != self.contact_pressure.shape:
                self.prev_contact_pressure_smooth = self.contact_pressure.copy()

            for f in range(F):
                curr = self.contact_pressure[f]
                prev = self.prev_contact_pressure_smooth[0] if F == 1 else self.prev_contact_pressure_smooth[f]

                drop = np.maximum(0, prev - curr)
                drop_ratio = drop / (np.maximum(prev, 1.0))
                blend = np.clip(drop_ratio / 0.3, 0, 1)
                alpha = alpha_up * (1 - blend) + alpha_down * blend

                smoothed = prev * (1.0 - alpha) + curr * alpha
                delta = smoothed - prev
                delta = np.clip(delta, -max_change_per_frame, max_change_per_frame)
                smoothed = prev + delta

                if options.contact_method in ('logodds', 'logodds_valved', 'stability_v2'):
                    smoothed = np.where(curr <= 0, 0.0, smoothed)

                self.contact_pressure[f] = smoothed
                if F == 1:
                    self.prev_contact_pressure_smooth = smoothed[np.newaxis, :].copy()
                else:
                    self.prev_contact_pressure_smooth[f] = smoothed
        else:
            # For unified/equilibrium: use raw pressure directly, update smooth cache
            self.prev_contact_pressure_smooth = self.contact_pressure.copy()
        
        # --- Compute CoM dynamic torque (before final torque call) ---
        t_dyn_com_world = None
        if options.world_frame_dynamics:
            t_dyn_com_world = self._compute_com_dynamic_torque(F, world_pos, global_rots, tips, options)
        
        # Recompute torques with the smoothed contact pressure
        torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs = self._compute_joint_torques(
            F, ang_acc_for_effort, world_pos, parents, global_rots,
            pose_data_aa, tips, options, contact_forces=None,
            _frame_cache=_frame_cache, skip_rate_limiting=False,
            dyn_override_world=t_dyn_com_world
        )
        
        # --- Contact Torque KF Smoothing ---
        # Smooth only the contact torque component to preserve dynamic fidelity.
        # τ_contact = τ_total - τ_dyn - τ_grav - τ_passive (the residual from floor forces)
        # Apply KF to τ_contact, then recombine: τ_output = τ_dyn + τ_grav + τ_passive + τ_contact_smoothed
        if options.enable_kf_smoothing and F == 1:
            dt = 1.0 / self.framerate
            n_j = min(torques_vec.shape[1], t_dyn_vecs.shape[1],
                      t_grav_vecs.shape[1], t_passive_vecs.shape[1])
            
            # Extract contact torque
            t_contact = (torques_vec[0, :n_j] - t_dyn_vecs[0, :n_j]
                         - t_grav_vecs[0, :n_j] - t_passive_vecs[0, :n_j])
            
            # Lazy-init a dedicated KF for contact torque
            if not hasattr(self, 'contact_torque_kf') or self.contact_torque_kf is None:
                self.contact_torque_kf = NumpySmartClampKF(dt, n_j, 3)
            
            self.contact_torque_kf.update_params(
                options.kf_responsiveness, options.kf_smoothness,
                options.kf_clamp_radius, dt
            )
            
            self.contact_torque_kf.predict()
            t_contact_smoothed = self.contact_torque_kf.update(t_contact)
            
            # Recombine: smoothed contact + unmodified dynamic/gravity/passive
            torques_vec[0, :n_j] = (t_dyn_vecs[0, :n_j] + t_grav_vecs[0, :n_j]
                                     + t_passive_vecs[0, :n_j] + t_contact_smoothed)
        
        # Store active torques and contact probs for next frame's comparison
        if F == 1 and torques_vec.shape[1] >= 22:
            self._prev_active_torques = torques_vec[0, :self.target_joint_count, :].copy()
            self._prev_contact_probs = working_probs[0, :self.target_joint_count].copy()

        
        # Store for consensus contact feedback (uses previous frame's torques)
        if F == 1 and t_dyn_vecs.shape[1] >= 22:
            self._prev_torque_vecs = t_dyn_vecs[0, :self.target_joint_count, :]


        
        # Scalar torque magnitude for output
        # torques is now removed as per user request


            
        output_quats = quats[:, :self.target_joint_count, :]
        

        
        # Update tip history
        if hasattr(self, 'temp_tips') and self.temp_tips:
             self.last_tip_positions = {k: v[-1].copy() for k, v in self.temp_tips.items()}

        # --- Balance Stability ---
        balance = self._compute_balance_stability(world_pos, tips, options)

        # --- SG Output Smoothing ---
        # Apply Savitzky-Golay smoothing (zeroth derivative, quadratic fit)
        # to the final active torque. Smooths ALL flicker sources (gravity,
        # contact, dynamic) in one pass without affecting upstream physics.
        tsw = getattr(options, 'torque_smooth_window', 0)
        if tsw >= 3 and F == 1:
            # Cached SG smoothing coefficients (quadratic fit, right-edge evaluation)
            sg_s_cache = getattr(self, '_torque_sg_smooth_cache', None)
            if sg_s_cache is None or sg_s_cache[0] != tsw:
                x = np.arange(tsw) - (tsw - 1)  # oldest = -(N-1), newest = 0
                V = np.column_stack([np.ones(tsw), x])  # linear fit
                # Row 0 of pseudoinverse = weights for constant term at x=0
                sg_coeffs = np.linalg.pinv(V)[0]
                self._torque_sg_smooth_cache = (tsw, sg_coeffs)
            else:
                sg_coeffs = sg_s_cache[1]
            
            # Ring buffer for torque vectors
            t_ring = getattr(self, '_torque_ring_arr', None)
            t_ptr = getattr(self, '_torque_ring_ptr', 0)
            t_cnt = getattr(self, '_torque_ring_cnt', 0)
            
            tv_shape = torques_vec[0].shape  # (J, 3)
            if t_ring is None or t_ring.shape[0] != tsw or t_ring.shape[1:] != tv_shape:
                t_ring = np.zeros((tsw,) + tv_shape, dtype=torques_vec.dtype)
                t_ptr = 0
                t_cnt = 0
            
            t_ring[t_ptr] = torques_vec[0]
            t_ptr = (t_ptr + 1) % tsw
            t_cnt = min(t_cnt + 1, tsw)
            
            self._torque_ring_arr = t_ring
            self._torque_ring_ptr = t_ptr
            self._torque_ring_cnt = t_cnt
            
            if t_cnt >= 3:
                if t_cnt == tsw:
                    ordered = np.roll(t_ring, -t_ptr, axis=0)
                else:
                    ordered = t_ring[:t_cnt]
                    # Recompute coeffs for partial window
                    x = np.arange(t_cnt) - (t_cnt - 1)
                    V = np.column_stack([np.ones(t_cnt), x])  # linear fit
                    sg_coeffs = np.linalg.pinv(V)[0]
                torques_vec = np.tensordot(sg_coeffs[:t_cnt], ordered, axes=([0], [0]))[np.newaxis]
                
                # Recompute efforts_net from smoothed torques so the heatmap
                # sees the same smoothing. Without this, efforts_net reflects
                # pre-smoothed torques and the heatmap bypasses the filter.
                max_torque = self.max_torque_array[:self.target_joint_count]  # (J, 3)
                denom = max_torque + 1e-6
                n_j = min(torques_vec.shape[1], efforts_net.shape[1], denom.shape[0])
                efforts_net[:, :n_j] = self._effort_with_raw_direction(
                    torques_vec[:, :n_j], denom[:n_j])

        # --- Effort-Adaptive Smoothing ---
        # At low effort (||τ/τ_max|| << 1), SNR is poor and direction/magnitude
        # noise dominates. At high effort, sub-degree parent orientation
        # jitter creates 10-20 Hz torque noise via local-frame gravity
        # projection. Apply per-joint EMA with:
        #   - effort-adaptive alpha (heavier smoothing at low effort)
        #   - joint-specific alpha_max (heavier for high-inertia proximal
        #     joints, lighter for nimble distal joints)
        #
        # The threshold is on effort (torque normalized by max_torque), not
        # absolute torque magnitude — this matters for small-muscle joints
        # (wrist, hand) whose maximum physiological torque is small enough
        # that an absolute-Nm threshold would lock them permanently in
        # heavy-smoothing regime regardless of motion speed.
        #
        # Per-joint alpha_max rationale (at 100fps):
        #   Proximal (pelvis/hip/spine): α=0.3 → ~5.7 Hz cutoff
        #     High inertia, can't physically change >5 Hz. Most sensitive
        #     to parent orientation noise amplification.
        #   Mid (knee/shoulder/neck/collar): α=0.4 → ~8 Hz cutoff
        #   Distal (ankle/elbow): α=0.6 → ~15 Hz cutoff
        #   Extremity (wrist/hand/foot/head): α=0.8 → ~26 Hz cutoff
        #     Low inertia, fast movements (e.g. gestures, footwork).
        if getattr(options, 'adaptive_effort_smooth', False) and F == 1:
            lo = getattr(options, 'adaptive_effort_lo', 0.1)
            hi = getattr(options, 'adaptive_effort_hi', 0.5)
            alpha_min = getattr(options, 'adaptive_effort_alpha_min', 0.05)
            global_alpha_max = getattr(options, 'adaptive_effort_alpha_max', 0.5)
            
            # Per-joint alpha_max (SMPL 22-joint layout)
            # Scaled by global_alpha_max as a master control
            _PER_JOINT_ALPHA_MAX = getattr(self, '_per_joint_alpha_max', None)
            n_j = torques_vec.shape[1]
            if _PER_JOINT_ALPHA_MAX is None or len(_PER_JOINT_ALPHA_MAX) != n_j:
                _base = np.full(n_j, 0.5)  # default for unknown joints
                # SMPL joint indices:
                #  0=pelvis  1=L_hip  2=R_hip  3=spine1
                #  4=L_knee  5=R_knee  6=spine2
                #  7=L_ankle  8=R_ankle  9=spine3
                # 10=L_foot 11=R_foot 12=neck
                # 13=L_collar 14=R_collar 15=head
                # 16=L_shoulder 17=R_shoulder
                # 18=L_elbow 19=R_elbow
                # 20=L_wrist 21=R_wrist
                _joint_alpha = {
                    0: 0.3,   # pelvis — root, highest inertia
                    1: 0.3,   # L_hip — heavy leg
                    2: 0.3,   # R_hip — heavy leg
                    3: 0.3,   # spine1 — trunk mass
                    6: 0.3,   # spine2 — trunk mass
                    9: 0.35,  # spine3 — upper trunk
                    4: 0.4,   # L_knee
                    5: 0.4,   # R_knee
                    12: 0.4,  # neck
                    13: 0.4,  # L_collar
                    14: 0.4,  # R_collar
                    16: 0.5,  # L_shoulder
                    17: 0.5,  # R_shoulder
                    7: 0.6,   # L_ankle
                    8: 0.6,   # R_ankle
                    18: 0.6,  # L_elbow
                    19: 0.6,  # R_elbow
                    15: 0.8,  # head — light, fast
                    10: 0.8,  # L_foot/toe — fast footwork
                    11: 0.8,  # R_foot/toe
                    20: 0.8,  # L_wrist — very nimble
                    21: 0.8,  # R_wrist
                }
                for j_idx, a in _joint_alpha.items():
                    if j_idx < n_j:
                        _base[j_idx] = a
                # Scale by global alpha_max (user tuning knob)
                self._per_joint_alpha_max = np.clip(_base * (global_alpha_max / 0.5), 0.05, 1.0)
                _PER_JOINT_ALPHA_MAX = self._per_joint_alpha_max
            
            # Initialize EMA state if needed
            prev_tv = getattr(self, '_adapt_smooth_tv', None)
            prev_en = getattr(self, '_adapt_smooth_en', None)
            
            tv_cur = torques_vec[0]  # (J, 3)
            en_cur = efforts_net[0]  # (J, 3)
            
            if prev_tv is not None and prev_tv.shape == tv_cur.shape:
                # ─── Magnitude branch ───
                # Effort = torque / max_torque (per axis). Norm of the effort
                # vector is dimensionless and naturally joint-scaled.
                eff_mags = np.linalg.norm(en_cur, axis=-1)  # (J,)
                eff_blend = np.clip((eff_mags - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

                # ─── Change branch ───
                # Per-joint |Δeffort| this frame, then median over a K-frame
                # window. Median (not mean) survives glitch spikes: a single
                # spike contributes 2 large deltas (in+out) out of K=5 → median
                # picks the small 3rd value, gate stays closed. Sustained
                # oscillation overwhelms the median and opens the gate.
                deff_now = np.linalg.norm(en_cur - prev_en, axis=-1)  # (J,)
                K = max(3, int(getattr(options, 'adaptive_effort_change_window', 5)))
                ring = getattr(self, '_adapt_smooth_deff_ring', None)
                rptr = getattr(self, '_adapt_smooth_deff_ptr', 0)
                rcnt = getattr(self, '_adapt_smooth_deff_cnt', 0)
                if ring is None or ring.shape != (K, n_j):
                    ring = np.zeros((K, n_j), dtype=np.float64)
                    rptr = 0
                    rcnt = 0
                ring[rptr] = deff_now
                rptr = (rptr + 1) % K
                rcnt = min(rcnt + 1, K)
                self._adapt_smooth_deff_ring = ring
                self._adapt_smooth_deff_ptr = rptr
                self._adapt_smooth_deff_cnt = rcnt

                if rcnt >= 3:
                    deff_med = np.median(ring[:rcnt], axis=0)  # (J,)
                else:
                    deff_med = np.zeros(n_j)  # warmup: no change boost

                c_lo = getattr(options, 'adaptive_effort_change_lo', 0.02)
                c_hi = getattr(options, 'adaptive_effort_change_hi', 0.10)
                chg_blend = np.clip((deff_med - c_lo) / max(c_hi - c_lo, 1e-6), 0.0, 1.0)

                # ─── Combine: either condition opens the gate ───
                blend = np.maximum(eff_blend, chg_blend)

                # ─── Coherence gate (opt-in): lift alpha_max toward 1.0 for
                # coordinated ballistic (popping-like) limb motion, so genuine
                # sharp accents survive without opening the gate to local noise
                # or isolated spikes. gate=0 → canonical cap unchanged. ───
                amax_j = _PER_JOINT_ALPHA_MAX[:n_j]
                if getattr(options, 'coherence_gate_enable', False):
                    gate = getattr(self, '_cg_gate', None)
                    if gate is not None:
                        amax_j = amax_j + gate[:n_j] * (1.0 - amax_j)

                alpha = alpha_min + blend * (amax_j - alpha_min)  # (J,)
                alpha_3 = alpha[:, np.newaxis]  # (J, 1) for broadcasting

                tv_smooth = alpha_3 * tv_cur + (1.0 - alpha_3) * prev_tv
                en_smooth = alpha_3 * en_cur + (1.0 - alpha_3) * prev_en

                torques_vec = tv_smooth[np.newaxis]  # (1, J, 3)
                efforts_net = en_smooth[np.newaxis]  # (1, J, 3)
            
            # Store for next frame (use the smoothed values for continuity)
            self._adapt_smooth_tv = torques_vec[0].copy()
            self._adapt_smooth_en = efforts_net[0].copy()

        # --- Output Dictionary ---
        res = {
            'pose': output_quats if options.return_quats else pose_data_aa[:, :self.target_joint_count, :],
            'trans': trans_data,
            'contact_probs': contact_probs_fusion, # Original sensory prior
            'contact_probs_fusion': contact_probs_fusion, # Legacy alias
            'contact_probs_refined': working_probs, # After torque refinement
            'torques_vec': torques_vec,
            'torques_grav_vec': t_grav_vecs,
            'torques_passive_vec': t_passive_vecs,

            'torques_dyn_vec': t_dyn_vecs,
            'inertias': inertias,
            'efforts_dyn': efforts_dyn,
            'efforts_grav': efforts_grav,
            'efforts_net': efforts_net,
            'positions': world_pos,

            'contact_pressure': self.contact_pressure,
            'balance': balance,
        }
        
        # Add frame evaluator results if available
        frame_eval = getattr(self, '_frame_eval_result', None)
        if frame_eval is not None:
            res['frame_eval'] = frame_eval
        
        # --- Optional World-Frame Output Conversion ---
        # Internal computation is in local (parent) frame.
        # If user requests world frame, rotate all vectors to world.
        if getattr(options, 'torque_output_frame', 'local') == 'world':
            parents_list = self._get_hierarchy()
            global_rot_fwd = _frame_cache.get('global_rot_fwd', None)
            if global_rot_fwd is None:
                from scipy.spatial.transform import Rotation as R_scipy
                global_rot_fwd = [R_scipy.from_matrix(global_rots[0, j])
                                  for j in range(global_rots.shape[1])]
                _frame_cache['global_rot_fwd'] = global_rot_fwd
            
            for vec_key in ['torques_vec', 'torques_dyn_vec', 'torques_grav_vec', 'torques_passive_vec']:
                arr = res[vec_key]
                n_out = min(arr.shape[1], len(parents_list))
                for j in range(n_out):
                    parent_idx = parents_list[j] if j < len(parents_list) else -1
                    if parent_idx >= 0 and parent_idx < len(global_rot_fwd):
                        arr[:, j, :] = global_rot_fwd[parent_idx].apply(
                            arr[:, j, :].reshape(-1, 3)
                        ).reshape(arr.shape[0], 3)
                    # Root: already in world frame (parent = identity)
        
        return res


    def align_to_up_axis(self, trans, rotations, input_axis='Y', target_axis='Z'):
        """
        Helper to rotate root pose and translation to new up axis.
        """
        if input_axis == target_axis:
            return trans, rotations

        # Rotation matrix to go from Y-up to Z-up (rotate -90 deg around X-axis)
        # Y -> Z, Z -> -Y, X -> X
        if input_axis == 'Y' and target_axis == 'Z':
            # Rx(-90)
            rot_convert = R.from_euler('x', -90, degrees=True)
        elif input_axis == 'Z' and target_axis == 'Y':
            # Rx(90)
            rot_convert = R.from_euler('x', 90, degrees=True)
        else:
            logging.warning("Unsupported axis conversion")
            return trans, rotations
            
        # Apply to translation
        trans_out = rot_convert.apply(trans)
        
        # Apply to root rotation (index 0)
        # global_rot_new = rot_convert * global_rot_old
        r_root = R.from_quat(rotations[:, 0, :])
        r_root_new = rot_convert * r_root
        
        rotations_out = rotations.copy()
        rotations_out[:, 0, :] = r_root_new.as_quat()
        
        return trans_out, rotations_out

    def get_limb_physics_data(self):
        return self.limb_data

    def get_joint_limits(self):
        """
        Returns approximate biomechanical limits for the first 22 joints.
        Values are in RADIANS.
        Format: {joint_name: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]}
        
        Note: SMPL local coordinate systems usually have the primary rotation axis 
        aligned such that flexion/extension is the dominant component. 
        These are rough approximations useful for constraints.
        """
        d2r = np.deg2rad
        
        # Ranges: [min, max]
        # Common ranges (approximate)
        # 0: pelvis (root) - usually unlimited (free joint)
        # 1/2: L/R hips - ball joints, large range
        # 4/5: L/R knees - hinge, primarily flexion (0 to ~140/150)
        # 1/2/3/6/9... spines - limited ball joints
        
        limits = {}
        
        # Default loose limit for unspecified/ball joints
        full_range = [-np.pi, np.pi]
        limited_ball = [d2r(-45), d2r(45)]
        
        for name in self.joint_names[:self.target_joint_count]:
            # Default to full range for root, or reasonable limits for others
            if name == 'pelvis':
                 limits[name] = [full_range, full_range, full_range]
                 continue
            
            x_lim = limited_ball 
            y_lim = limited_ball 
            z_lim = limited_ball
            
            if 'knee' in name:
                # Knees are hinges. In SMPL, X-axis is usually flexion.
                # Flexion 0 to 150 deg approx.
                # Allow small range for twist/abduction (laxity)
                x_lim = [d2r(0), d2r(150)] 
                y_lim = [d2r(-10), d2r(10)]
                z_lim = [d2r(-10), d2r(10)]
                
            elif 'elbow' in name:
                # Elbows: Hinge. Flexion 0 to 150.
                # Warning: Check SMPL axis. Often Z or Y is main axis? 
                # Assuming X is primary for consistency or symmetric to knee in many rigs, 
                # but SMPL template is T-pose.
                # Let's assume standard SMPL: 
                # Be careful, we will provide broad 'primary Hinge' limits.
                # User asked for 'approximate', so we establish the pattern.
                 x_lim = [d2r(-150), d2r(0)] # or 0 to 150 depending on sign convention
                 # Actually, usually better to provide symmetric large range if unsure of sign
                 # or verify. For now, we'll use a specific range but warn.
                 # Let's use 0 to 160 magnitude.
                 x_lim = [d2r(-160), d2r(160)] 
                 y_lim = [d2r(-20), d2r(20)]
                 z_lim = [d2r(-20), d2r(20)]
                 
            elif 'hip' in name:
                # Ball joint, wide range
                x_lim = [d2r(-120), d2r(30)] # Flexion/Ext
                y_lim = [d2r(-50), d2r(50)]  # Abduction
                z_lim = [d2r(-50), d2r(50)]  # Rotation
                
            elif 'shoulder' in name:
                x_lim = [d2r(-180), d2r(180)] # Huge range
                y_lim = [d2r(-90), d2r(90)]
                z_lim = [d2r(-90), d2r(90)]

            elif 'spine' in name:
                # Stiffer
                stiff = [d2r(-30), d2r(30)]
                x_lim = stiff
                y_lim = stiff
                z_lim = stiff
                
            limits[name] = [x_lim, y_lim, z_lim]
            
        return limits
