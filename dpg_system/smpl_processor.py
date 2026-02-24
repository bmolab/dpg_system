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
    add_gravity: bool = False
    enable_passive_limits: bool = True # Enabled for Structural Support
    enable_apparent_gravity: bool = True
    torque_output_frame: str = 'local'  # 'local' (parent frame) or 'world'
    
    # --- Filtering / Signal Processing ---
    enable_one_euro_filter: bool = True
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.0
    
    # --- Floor / Environment ---
    floor_enable: bool = False
    floor_height: float = 0.0
    floor_tolerance: float = 0.15
    heel_toe_bias: float = 0.0
    contact_method: str = 'fusion'  # 'fusion', 'stability', 'com_driven', or 'consensus'
    
    # --- Torque Rate Limiting ---
    enable_rate_limiting: bool = True
    rate_limit_strength: float = 1.0  # Multiplier for per-joint rate limits
    enable_jitter_damping: bool = True  # Pass 4: sign-flip oscillation damping
    enable_velocity_gate: bool = True   # Suppress dynamic torque at low angular velocity
    enable_kf_smoothing: bool = True  # SmartClampKF filter for dynamic torque
    kf_responsiveness: float = 10.0   # How quickly KF tracks changes (higher = faster)
    kf_smoothness: float = 1.0        # Process noise (higher = trusts new data more)
    kf_clamp_radius: float = 15.0     # Max innovation per frame (Nm)
    
    # --- World-Frame Dynamics ---
    world_frame_dynamics: bool = False  # CoM-based dynamic torque (off by default for A/B comparison)
    com_pos_min_cutoff: float = 8.0    # Base One Euro min_cutoff for CoM position filter (scaled by 1/√mass)
    com_pos_beta: float = 0.05         # Base One Euro beta for CoM position filter
    com_vel_min_cutoff: float = 3.0    # Base One Euro min_cutoff for CoM velocity filter (scaled by 1/√mass)
    com_vel_beta: float = 0.05         # Base One Euro beta for CoM velocity filter
    com_acc_min_cutoff: float = 2.0    # Base One Euro min_cutoff for CoM acceleration filter (999 = disabled)
    com_acc_beta: float = 0.8          # Base One Euro beta — high for adaptive responsiveness during impacts
    smooth_input_window: int = 0       # Causal moving average window for pose+trans input (0 = off, 3 = recommended for 33Hz cadence removal)
    
    # --- Spine Geometry ---
    use_s_curve_spine: bool = True      # Use biomechanical S-curve spine instead of SMPL cantilevered spine

class SMPLProcessor:
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
            # Create Model
            # use_pca=False is safer for just shape exploration? No, defaults are fine.
            model = smplx.create(model_path=model_path, 
                                 model_type='smplh',
                                 gender=g_tag, 
                                 num_betas=10, 
                                 ext='pkl')
                                 
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
            # Ankle joint center is ~10-12cm above heel ground contact.
            heel_dir = np.array([0.0, -0.9, -0.4])  # Primarily downward, slightly backward
            heel_dir = heel_dir / np.linalg.norm(heel_dir) * 0.12  # ~12cm to floor
            offsets[28] = heel_dir  # L_heel from L_ankle
            offsets[29] = heel_dir  # R_heel from R_ankle
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
            elif 'heel' in node_name: length = 0.12

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
                # Ankle joint center is ~10-12cm above heel ground contact point
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
        if not hasattr(self, 'prob_prev_com') or self.prob_prev_com is None:
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = np.zeros_like(c)
            self.prob_prev_com_acc = np.zeros_like(c)
            self.com_acc_filter = OneEuroFilter(min_cutoff=0.3, beta=0.1, d_cutoff=1.0)  # Lower cutoff for more smoothing
            
            com_vel = np.zeros_like(c)
            com_acc = np.zeros_like(c)
        else:
            # Kinematics
            if F > 1:
                 com_vel = (c - self.prob_prev_com) / dt # Approx
            else:
                 com_vel = (c - self.prob_prev_com) / dt
                 
            # Acceleration
            raw_acc = (com_vel - self.prob_prev_com_vel) / dt
            
            if not hasattr(self, 'com_acc_filter'):
                self.com_acc_filter = OneEuroFilter(min_cutoff=0.3, beta=0.1, d_cutoff=1.0)  # Lower cutoff

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

    def _compute_probabilistic_contacts_fusion(self, F, J, world_pos, vel_y, vel_h, floor_height, options):
        """
        New Continuous Fusion Logic:
        P_final = P_prox * P_vel * P_acc
        
        Removes hard gates. Weights factors fluidly.
        """
        # Init Result Array (F, J)
        # Note: We compute per-frame, but usually F=1 in real-time mode.
        # DEBUG
        # f is local loop var, here we don't have loop yet.
        # Wait, f is used later.
        # We can just print once.
        
        
        if not options.floor_enable:
             return np.zeros((F, J))
             
        contact_probs = np.zeros((F, J))
        
        # Init State if needed
        if not hasattr(self, 'prob_contact_probs_fusion') or self.prob_contact_probs_fusion is None:
             self.prob_contact_probs_fusion = np.zeros(J)
             
        # Plane dims
        if options.input_up_axis == 'Y':
             y_dim = 1
             plane_dims = [0, 2]
        elif options.input_up_axis == 'Z':
             y_dim = 2
             plane_dims = [0, 1]
        
        # ZMP state (Already computed in main loop)
        p_zmp = self.current_zmp # (J_subset or 3?) - It's (3,) in streaming.
        if p_zmp.ndim > 1: p_zmp = p_zmp[0]
        p_zmp_hz = p_zmp[plane_dims]
        
        # Process Frame 0 only (streaming assumption)
        f = 0 
        
        # Get CoM State
        if hasattr(self, 'prob_prev_com_acc'):
             com_acc = self.prob_prev_com_acc
        else:
             com_acc = np.zeros(3)
             
        # Dynamic Sigma H (Stricter than Legacy)
        # tol/3.0 -> contact extends to ~tol.
        # tol/4.0 -> contact fades faster (at 4.5cm/15cm, P drops to <0.5).
        sigma_h = options.floor_tolerance / 4.0
        if sigma_h < 0.02: sigma_h = 0.02
        
        # 2. Joint Loop
        for j in range(J):
            # Pos
            if hasattr(self, 'temp_tips') and j in self.temp_tips:
                pos = self.temp_tips[j][f]
            else:
                pos = world_pos[f, j]
            
            h = pos[y_dim] - floor_height
            
            # Apply Virtual Heel Offset
            if j in [7, 8, 10, 11, 20, 21]:
                h -= options.heel_toe_bias
            
            vy = vel_y[f, j] if vel_y.ndim > 1 else vel_y[j]
            vh = vel_h[f, j] if vel_h.ndim > 1 else vel_h[j]
            v_total = np.sqrt(vy**2 + vh**2)
            
            # --- 1. Proximity P_prox ---
            if h <= 0:
                p_prox = 1.0
            else:
                p_prox = np.exp(-0.5 * (h / sigma_h)**2)
            if h > 0.20: p_prox = 0.0
            
            # --- 2. Load Factor ---
            pos_hz = pos[plane_dims]
            dist_zmp = np.linalg.norm(pos_hz - p_zmp_hz)
            load_factor = np.exp(- (dist_zmp / 0.45)**2)
            
            # --- 3. Velocity P_vel ---
            # Split into Horizontal (Sliding allowed if loaded) and Vertical (Strict)
            
            # 3a. Horizontal (Load modulates pivot)
            # v_pivot_h shifts from 0.35 to 1.35
            v_pivot_h = 0.35 + 1.0 * load_factor
            exp_h = 10.0 * (vh - v_pivot_h)
            if exp_h > 20: exp_h = 20
            p_vel_h = 1.0 / (1.0 + np.exp(exp_h))
            
            # 3b. Vertical Velocity Penalty Removed
            # Reason: Downward velocity (plunging) is good (loading). 
            # Upward velocity (lifting) is handled by P_LIFT below.
            # Legacy P_VEL_Y penalized both, causing early contact loss during heel lift (dip).
            p_vel = p_vel_h
            
            # --- 4. Acceleration P_acc ---
            # Freefall penalty.
            acc_y = com_acc[y_dim]
            # Soft penalty starting at -5.0, max at -10.0
            if acc_y < -5.0 and v_total < 0.5 and load_factor < 0.2:
                 penalty = ((-5.0 - acc_y) / 5.0)
                 if penalty > 1.0: penalty = 1.0
                 if penalty < 0.0: penalty = 0.0
                 p_acc = 1.0 - penalty
            else:
                 p_acc = 1.0
            
            # --- Lift-Off Check (Asymmetric Trend & Softened) ---
            # Smoothing: Fast Decay (Landing), Slow Rise (Lift).
            # Fixes latency on contact onset.
            
            # Init trend state if needed (per joint)
            if not hasattr(self, 'prob_vel_trend') or self.prob_vel_trend is None or self.prob_vel_trend.shape[0] != J:
                 self.prob_vel_trend = np.zeros(J)
            
            vy = vel_y[f, j]
            
            if F == 1:
                 # Asymmetric Alpha
                 # If decelerating (vy < trend, typically landing/dropping), update fast (0.5).
                 # If accelerating (vy > trend, lifting), update slow (0.1) to ignore noise.
                 alpha_v = 0.5 if vy < self.prob_vel_trend[j] else 0.1
                 self.prob_vel_trend[j] = self.prob_vel_trend[j] * (1.0 - alpha_v) + vy * alpha_v
                 v_trend = self.prob_vel_trend[j]
            else:
                 v_trend = vy 
            
            # Check Trend
            # Threshold 0.02 m/s.
            if v_trend > 0.02: 
                 # Penalty: Softened steepness 50 -> 30
                 p_lift = np.exp(-30.0 * (v_trend - 0.02))
                 if p_lift > 1.0: p_lift = 1.0
            else:
                 p_lift = 1.0
            
            # --- Horizontal Slide Check (Asymmetric Trend & Softened) ---
            # Init trend state if needed
            if not hasattr(self, 'prob_vel_trend_h') or self.prob_vel_trend_h is None or self.prob_vel_trend_h.shape[0] != J:
                 self.prob_vel_trend_h = np.zeros(J)
            
            vh_inst = np.linalg.norm(vel_h[f, j])
            
            if F == 1:
                 # Asymmetric Alpha
                 # If slowing down (vh < trend), update fast (0.5).
                 # If speeding up (vh > trend), update slow (0.1).
                 alpha_h = 0.5 if vh_inst < self.prob_vel_trend_h[j] else 0.1
                 self.prob_vel_trend_h[j] = self.prob_vel_trend_h[j] * (1.0 - alpha_h) + vh_inst * alpha_h
                 vh_trend = self.prob_vel_trend_h[j]
            else:
                 vh_trend = vh_inst
            
            # Check Slide Trend
            # Threshold 0.3 m/s.
            if vh_trend > 0.3:
                 # Penalty: Softened steepness 20 -> 10
                 p_slide = np.exp(-10.0 * (vh_trend - 0.3))
                 if p_slide > 1.0: p_slide = 1.0
            else:
                 p_slide = 1.0
            
            # --- Fusion ---
            p_raw = p_prox * p_vel * p_acc * p_lift * p_slide
            

            

            

            
            # --- 5. Geometric "Toe-Off" Check (Updated for SMPL 24 convention) ---
            # Check Right Ankle (8) vs Right Foot (11)
            # User Feedback: "Penalize Ankle (8) if it is higher than Foot (11)"
            if j == 8:
                 if 11 in self.temp_tips: f_pos = self.temp_tips[11][f]
                 else: f_pos = world_pos[f, 11]
                 h_foot = f_pos[y_dim] - floor_height
                 
                 # Ankle is naturally above Foot (~4-8cm).
                 # If Ankle is significantly higher (e.g. > 15cm above foot), it implies heel lift/plantar flexion.
                 # Or if Foot is near ground and Ankle is high?
                 
                 # Let's say if Ankle is > Foot + 0.12 (12cm)
                 if h > (h_foot + 0.12):
                      p_raw *= 0.0

            # Check Left Ankle (7) vs Left Foot (10)
            if j == 7:
                 if 10 in self.temp_tips: f_pos = self.temp_tips[10][f]
                 else: f_pos = world_pos[f, 10]
                 h_foot = f_pos[y_dim] - floor_height
                 
                 if h > (h_foot + 0.12):
                      p_raw *= 0.0
            
            # --- Smoothing ---
            # Standard EMA 0.5
            self.prob_contact_probs_fusion[j] = self.prob_contact_probs_fusion[j] * 0.5 + p_raw * 0.5
            
            contact_probs[f, j] = self.prob_contact_probs_fusion[j]
            
        return contact_probs

    def _compute_probabilistic_contacts_stability(self, F, J, world_pos, options):
        """Dynamics-aware inverse statics contact determination.
        
        Uses the consensus method as a base prior for contact *candidates*,
        then determines contact *pressure* via inverse statics:
        
        1. F_required = M × (a_com - g)  →  how much support is needed
        2. Candidate contacts from consensus (prob > threshold)
        3. Partition F_required among candidates using inverse-distance
           from CoM and height proximity to floor
        4. Consolidate toes into feet (no toe rotation measured)
        5. Normalize total pressure to body_mass × support_fraction
        6. Store computed pressure and return boosted probabilities
        
        Physics (CoM dynamics) determines whether contacts are needed.
        Probability is only used for candidate selection.
        Apparent gravity is handled separately in _compute_com_for_zmp.
        """
        # Step 1: Get consensus probabilities as candidate detector
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
        
        # Contact candidate joint indices
        # ANY body joint can be a contact point (floor work, crawling, etc.).
        # Height threshold (0.35m) and velocity gating naturally filter
        # non-contact joints regardless of posture.
        # Excluded from direct candidacy:
        #   - Toes (24, 25): consolidated into feet (10, 11) — no toe rotation measured
        #   - Fingers (26, 27): consolidated into hands (22, 23) — no finger deflection measured
        contact_joints = list(range(min(J, 24)))  # All 24 SMPL body joints
        for vi in [28, 29]:  # heels (virtual)
            if vi < J:
                contact_joints.append(vi)
        
        # Consolidation maps: unmeasured extremities → parent joints
        consolidate_to_parent = {}
        if J > 24: consolidate_to_parent[24] = 10  # L_toe → L_foot
        if J > 25: consolidate_to_parent[25] = 11  # R_toe → R_foot
        if J > 26: consolidate_to_parent[26] = 22  # L_finger → L_hand
        if J > 27: consolidate_to_parent[27] = 23  # R_finger → R_hand
        
        # Need previous frame data
        prev_com = getattr(self, '_prev_com_for_stability', None)
        com_acc = getattr(self, 'prob_prev_com_acc', None)
        
        if prev_com is None or com_acc is None:
            return consensus_probs
        
        if prev_com.ndim > 1:
            com = prev_com[0]
        else:
            com = prev_com
        
        # Step 2: Compute required total support force
        # F_required = M × (a_com - g)
        g_vec = np.zeros(3)
        g_vec[y_dim] = -g_mag
        
        F_required = total_mass * (com_acc - g_vec)  # (3,)
        F_support_up = F_required[y_dim]  # Positive = upward = contacts needed
        
        # Raw support fraction: 0 = freefall, 1 = static/grounded
        # Capped at 1.0 because landing dynamics (>1× body weight) are
        # already handled by apparent gravity in _compute_com_for_zmp
        support_fraction = np.clip(F_support_up / (total_mass * g_mag), 0.0, 1.0)
        
        # --- Airborne detection for adaptive alpha ---
        # Check minimum joint height across all contact candidates
        min_joint_height = float('inf')
        for j in contact_joints:
            if j >= J:
                continue
            if hasattr(self, 'temp_tips') and j in self.temp_tips:
                h = self.temp_tips[j][0, y_dim] if self.temp_tips[j].ndim > 1 else self.temp_tips[j][y_dim]
            else:
                h = world_pos[0, j, y_dim] if world_pos.ndim > 2 else world_pos[j, y_dim]
            min_joint_height = min(min_joint_height, h - floor_height)
        
        # Also check consolidated joint heights (toes, fingers)
        for child_j in consolidate_to_parent:
            if child_j < J:
                if hasattr(self, 'temp_tips') and child_j in self.temp_tips:
                    h = self.temp_tips[child_j][0, y_dim] if self.temp_tips[child_j].ndim > 1 else self.temp_tips[child_j][y_dim]
                else:
                    h = world_pos[0, child_j, y_dim]
                min_joint_height = min(min_joint_height, h - floor_height)
        
        AIRBORNE_HEIGHT_THRESH = 0.08  # 8cm — if ALL joints above this, likely airborne
        
        # Adaptive alpha: fast when airborne indicators present, slow when grounded
        airborne_indicators = 0.0
        if support_fraction < 0.3:
            airborne_indicators += 0.5  # Physics says low support
        if min_joint_height > AIRBORNE_HEIGHT_THRESH:
            airborne_indicators += 0.5  # All joints above floor
        # Hard override: if literally nothing is near the floor, force airborne
        if min_joint_height > 0.20:
            airborne_indicators = 1.0
        
        # Alpha: 0.15 (grounded, stable) → 0.8 (airborne, responsive)
        alpha_sf = 0.15 + 0.65 * airborne_indicators
        
        # Init smoothed support fraction
        if not hasattr(self, '_stability_support_frac') or self._stability_support_frac is None:
            self._stability_support_frac = support_fraction
        self._stability_support_frac = (
            self._stability_support_frac * (1 - alpha_sf) + support_fraction * alpha_sf
        )
        smoothed_support = self._stability_support_frac
        
        if smoothed_support < 0.05:
            # Airborne — decay existing pressure and return
            if hasattr(self, '_stability_pressure') and self._stability_pressure is not None:
                self._stability_pressure *= 0.7  # Fast decay
            self._stability_computed_pressure = getattr(self, '_stability_pressure', np.zeros(J))
            return consensus_probs
        
        # Init smoothed contact pressure state
        if not hasattr(self, '_stability_pressure') or self._stability_pressure is None or len(self._stability_pressure) != J:
            self._stability_pressure = np.zeros(J)
        
        # --- Horizontal velocity EMA (for candidate exclusion) ---
        # A joint with sustained horizontal velocity can't be in floor contact.
        # EMA filters out transient mocap drift spikes (planted foot drift can
        # peak at ~1 m/s but is random; swinging is sustained at 3+ m/s).
        HZ_THRESH = 0.2     # m/s — safe with EMA (planted median ≈ 0.04)
        HZ_EMA_ATTACK = 0.7  # Fast attack — track rising velocity quickly
        HZ_EMA_DECAY = 0.9   # Faster decay — release quickly on landing
        
        if not hasattr(self, '_stability_hz_vel_ema') or self._stability_hz_vel_ema is None or len(self._stability_hz_vel_ema) != J:
            self._stability_hz_vel_ema = np.zeros(J)
        
        vel = getattr(self, 'prob_smoothed_vel', None)
        if vel is not None:
            for j in contact_joints:
                if j < J and j < vel.shape[0]:
                    v_hz = np.sqrt(vel[j, plane_dims[0]]**2 + vel[j, plane_dims[1]]**2)
                    # Asymmetric EMA: fast attack, faster decay
                    if v_hz > self._stability_hz_vel_ema[j]:
                        alpha = HZ_EMA_ATTACK
                    else:
                        alpha = HZ_EMA_DECAY
                    self._stability_hz_vel_ema[j] = (
                        self._stability_hz_vel_ema[j] * (1 - alpha) +
                        v_hz * alpha
                    )
        
        # CoM horizontal position (from previous frame, same as used in inverse statics)
        com_hz = com[plane_dims]
        
        for f in range(F):
            # Step 3: Identify candidate contacts
            # Use height threshold only — do NOT gate on consensus_probs,
            # which uses the tight floor_tolerance and excludes joints that
            # the stability method's own weight distribution can handle.
            candidates = []
            for j in contact_joints:
                if j >= J:
                    continue
                # Get position
                if hasattr(self, 'temp_tips') and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                h = pos[y_dim] - floor_height
                if h > 0.35:
                    continue
                candidates.append({
                    'j': j,
                    'pos': pos,
                    'pos_hz': pos[plane_dims],
                    'h': h,
                })
            
            # Also check consolidated joints (toes→feet, fingers→hands)
            # If a child joint has consensus prob, add parent as candidate
            CANDIDATE_THRESH = 0.01
            parent_in_candidates = {c['j'] for c in candidates}
            for child_j, parent_j in consolidate_to_parent.items():
                if child_j >= J or parent_j >= J:
                    continue
                if parent_j in parent_in_candidates:
                    continue  # Already there
                if consensus_probs[f, child_j] < CANDIDATE_THRESH:
                    continue
                # Child detected contact → add parent as candidate
                if hasattr(self, 'temp_tips') and parent_j in self.temp_tips:
                    pos = self.temp_tips[parent_j][f]
                else:
                    pos = world_pos[f, parent_j]
                h = pos[y_dim] - floor_height
                if h > 0.35:
                    continue
                candidates.append({
                    'j': parent_j,
                    'pos': pos,
                    'pos_hz': pos[plane_dims],
                    'h': h,
                })
                parent_in_candidates.add(parent_j)
            
            if not candidates:
                continue
            
            # Step 4: Partition body mass among candidates
            # Weight distribution is purely geometric (inverse statics):
            # closer to CoM → more weight. Height is already screened by
            # the 0.35m candidacy threshold; once a joint is a candidate,
            # height should not bias its share (unreliable due to mocap error).
            com_hz = com[plane_dims]
            
            dists = np.array([np.linalg.norm(c['pos_hz'] - com_hz) for c in candidates])
            
            SIGMA_DIST = 0.15
            inv_dist_weights = 1.0 / (dists + SIGMA_DIST)
            
            # Pure geometry — no height bias
            combined = inv_dist_weights
            w_sum = np.sum(combined)
            
            if w_sum < 1e-8:
                continue
            
            mass_fractions = combined / w_sum
            
            # Step 5: Compute raw pressure with horizontal velocity attenuation
            # Velocity and CoM distance interact multiplicatively:
            # - High velocity + far from CoM → strong attenuation (swinging)
            # - Low velocity + near CoM → minimal attenuation (planted)
            # - High velocity + near CoM → moderate (mid-swing or drift)
            # Velocity is squared to suppress drift-level speeds (~1 m/s)
            # while strongly attenuating at walking/swinging speeds (3+ m/s).
            HZ_REF = 3.0     # m/s — reference velocity (~walking speed)
            COM_REF = 0.3     # m — reference CoM distance (~stance width)
            
            target_pressure = np.zeros(J)
            for idx, c in enumerate(candidates):
                target_pressure[c['j']] = mass_fractions[idx] * total_mass * smoothed_support
            
            # Normalize: total equals total_mass × support_fraction
            raw_total = np.sum(target_pressure)
            desired_total = total_mass * smoothed_support
            if raw_total > 1e-6:
                target_pressure *= (desired_total / raw_total)
            
            # Step 4b: Multi-contact torque-aware reweighting
            # With 3+ contacts, the inverse-distance weighting from CoM
            # under-loads joints that must support most of the body mass.
            # Fix: redistribute pressure proportional to each contact's
            # ability to counteract the body's gravitational moment.
            # Contacts closer to the CoM's ground projection carry more load.
            n_candidates = len(candidates)
            if n_candidates >= 3:
                # Proximity-weighted pressure: contacts near CoM ground
                # projection carry more gravitational load. σ=0.25m spreads
                # weight broadly enough to support distributed contacts
                # (e.g. both knees in kneeling) without starving distant ones.
                SIGMA_PROX = 0.25
                prox_weights = np.zeros(J)
                for c in candidates:
                    j = c['j']
                    d_hz = np.linalg.norm(c['pos_hz'] - com_hz)
                    prox_weights[j] = np.exp(-0.5 * (d_hz / SIGMA_PROX)**2)
                
                prox_total = np.sum(prox_weights)
                if prox_total > 1e-6:
                    prox_pressure = (prox_weights / prox_total) * desired_total
                    # Blend factor increases with contact count
                    # 3 contacts → 0.3, 6 → 0.5, 10+ → 0.6
                    blend = min(0.6, 0.15 + 0.05 * (n_candidates - 2))
                    target_pressure = (1.0 - blend) * target_pressure + blend * prox_pressure
            
            # Step 4c: Minimum pressure floor for grounded joints
            # Any joint clearly on the floor (height < -2cm below floor_height)
            # should receive meaningful support pressure regardless of CoM
            # proximity, preventing torque flickering on secondary contacts.
            MIN_PRESSURE_FRAC = 0.05  # 5% of desired_total per grounded joint
            GROUNDED_DEPTH = -0.02    # 2cm below floor = definitely on floor
            min_pressure = MIN_PRESSURE_FRAC * desired_total if desired_total > 0 else 0
            for c in candidates:
                j = c['j']
                if c['h'] < GROUNDED_DEPTH and target_pressure[j] < min_pressure:
                    target_pressure[j] = min_pressure
            
            # Apply horizontal velocity attenuation AFTER normalization.
            # This reduction represents load not on the floor (swinging limb
            # mass), so the total decreases — that load is airborne.
            for idx, c in enumerate(candidates):
                j = c['j']
                hz_v = self._stability_hz_vel_ema[j] if j < len(self._stability_hz_vel_ema) else 0.0
                if hz_v > 0.01:
                    com_d = dists[idx]
                    velocity_signal = (hz_v / HZ_REF) ** 2
                    distance_signal = com_d / COM_REF
                    interaction = velocity_signal * distance_signal
                    attenuation = 1.0 / (1.0 + interaction)
                    target_pressure[j] *= attenuation
            
            # Step 6: Smooth pressure with adaptive alpha
            # Three signals drive fast decay:
            #   1. Vertical velocity — joint rising = losing contact
            #   2. Non-candidate — joint dropped out of candidacy
            #   3. Horizontal velocity × CoM distance — same multiplicative
            #      interaction as Step 5, draining old accumulated pressure
            
            # Multi-contact smoothing: with 3+ contacts the pressure distribution
            # is over-determined. Small pose changes shouldn't cause large
            # redistributions, so increase smoothing (reduce alpha) proportionally.
            n_contacts = len(candidates)
            if n_contacts <= 2:
                ALPHA_SLOW = 0.15   # Normal walking — responsive
            else:
                # Scale down: 3 contacts → 0.12, 6 → 0.075, 10+ → 0.05
                ALPHA_SLOW = 0.15 / (1.0 + 0.25 * (n_contacts - 2))
            
            ALPHA_FAST = 1.0    # Full evidence — instant tracking, no residual
            VEL_THRESH = 0.05   # m/s — vertical velocity above which we speed up
            VEL_SCALE = 1.5     # m/s — vertical velocity at which alpha is fully fast
            
            vel = getattr(self, 'prob_smoothed_vel', None)
            
            # Build per-joint CoM distance map from candidates
            candidate_com_dist = {}
            for c in candidates:
                candidate_com_dist[c['j']] = np.linalg.norm(c['pos_hz'] - com_hz)
            
            for j in range(J):
                if target_pressure[j] > 0 or self._stability_pressure[j] > 0:
                    # --- Signal 1: Vertical velocity ---
                    if vel is not None and j < vel.shape[0]:
                        v_up = vel[j, y_dim]  # Positive = rising
                    else:
                        v_up = 0.0
                    
                    vel_factor = 0.0
                    if v_up > VEL_THRESH:
                        vel_factor = min(1.0, (v_up - VEL_THRESH) / VEL_SCALE)
                    
                    # --- Signal 2: Non-candidate with lingering pressure ---
                    orphan_factor = 0.0
                    if self._stability_pressure[j] > 0 and target_pressure[j] == 0:
                        orphan_factor = 1.0
                    
                    # --- Signal 3: Horizontal velocity × CoM distance ---
                    # Same multiplicative interaction as target_pressure
                    # attenuation: convert to a 0→1 release factor
                    hz_com_factor = 0.0
                    hz_v = self._stability_hz_vel_ema[j] if j < len(self._stability_hz_vel_ema) else 0.0
                    if hz_v > 0.01:
                        com_d = candidate_com_dist.get(j, 0.5)  # default ~stance width for non-candidates
                        interaction = ((hz_v / HZ_REF) ** 2) * (com_d / COM_REF)
                        hz_com_factor = interaction / (1.0 + interaction)
                    
                    # Combine: any signal can trigger fast decay
                    release_factor = max(vel_factor, orphan_factor, hz_com_factor)
                    alpha_j = ALPHA_SLOW + (ALPHA_FAST - ALPHA_SLOW) * release_factor
                    
                    self._stability_pressure[j] = (
                        self._stability_pressure[j] * (1 - alpha_j) +
                        target_pressure[j] * alpha_j
                    )
            
            # Step 7: Convert pressure to probabilities for consensus override
            for idx, c in enumerate(candidates):
                j = c['j']
                computed_mass = self._stability_pressure[j]
                if computed_mass > 0.5:
                    boost_prob = min(0.95, computed_mass / (total_mass * 0.3))
                    consensus_probs[f, j] = max(consensus_probs[f, j], boost_prob)
        
        # Store computed pressure for downstream override
        self._stability_computed_pressure = self._stability_pressure.copy()
        
        return consensus_probs

    def _compute_probabilistic_contacts_com_driven(self, F, J, world_pos, floor_height, options):
        """
        CoM-Driven Contact Detection for IMU-noisy data.
        
        Uses Center of Mass position/movement to infer which joints MUST be 
        bearing load, overriding noisy joint motion/position.
        
        Key differences from fusion:
        - CoM projection is primary (not joint height)
        - Velocity penalty is suppressed when CoM says joint is loaded
        - Soft proximity gating for feet, hard for hands
        """
        # --- State Initialization ---
        if not hasattr(self, 'prob_contact_probs_com') or self.prob_contact_probs_com is None or len(self.prob_contact_probs_com) != J:
             self.prob_contact_probs_com = np.zeros(J)
             
        # Smoothed load factors to reduce frame-to-frame oscillation
        if not hasattr(self, 'prob_load_factors_smooth') or self.prob_load_factors_smooth is None or len(self.prob_load_factors_smooth) != J:
             self.prob_load_factors_smooth = np.zeros(J)
        
        # One-euro filter for CoM to smooth out motion capture noise
        if not hasattr(self, 'com_one_euro_filter') or self.com_one_euro_filter is None:
             # Conservative parameters: min_cutoff=1.0 (gentle smoothing), beta=0.5 (moderate speed response)
             self.com_one_euro_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5, framerate=self.framerate)
             
        # Use internal Y dimension
        y_dim = getattr(self, 'internal_y_dim', 1)
        if y_dim == 1:
             plane_dims = [0, 2]  # X, Z (Y-up)
        else:
             plane_dims = [0, 1]  # X, Y (Z-up)
        
        # --- Parameters ---
        SOFT_CEILING = 0.25       # Below this, height has minimal effect (for feet)
        HARD_CEILING = 0.40       # Above this, contact = 0 
        HAND_SOFT_CEILING = 0.10  # Stricter soft ceiling for hands (10cm)
        HAND_HARD_CEILING = 0.20  # Hands must be very close for contact
        LOAD_VEL_SUPPRESS = 0.85  # How much load factor suppresses velocity penalty
        
        # Candidate contact joints (includes virtual joints)
        # 7/8: ankles, 10/11: feet, 24/25: toes, 28/29: heels
        FOOT_JOINTS = {7, 8, 10, 11, 24, 25, 28, 29}
        # 20/21: wrists, 22/23: hands, 26/27: fingers
        HAND_JOINTS = {20, 21, 22, 23, 26, 27}
        
        # Kinematic chains for relative height comparison
        # Within each chain, only joints near the lowest get contact pressure
        # Note: ankle is NOT a contact point - heel is the contact point at the back of foot
        # Leg chains: ordered by typical vertical position when standing
        # ankle(7) -> heel(28) -> foot(10) -> toe(24)
        # knee(4) is above ankle so handled separately in the chain as: knee -> ankle -> heel -> foot -> toe
        LEFT_LEG_CHAIN = [4, 7, 28, 10, 24]   # knee, ankle, heel, foot, toe
        RIGHT_LEG_CHAIN = [5, 8, 29, 11, 25]
        # Left arm chain: elbow(18) -> wrist(20) -> hand(22) -> finger(26)
        # When hand is flat, wrist/hand/fingers all in contact
        # Right arm chain: elbow(19) -> wrist(21) -> hand(23) -> finger(27)
        LEFT_ARM_CHAIN = [18, 20, 22, 26]   # elbow, wrist, hand, finger
        RIGHT_ARM_CHAIN = [19, 21, 23, 27]
        
        # Threshold for "significantly lower" in relative height (meters)
        CHAIN_HEIGHT_THRESHOLD = 0.05  # 5cm difference means lower joint takes priority
        
        contact_probs = np.zeros((F, J), dtype=np.float32)
        
        # Tips availability
        tips_available = hasattr(self, 'temp_tips') and self.temp_tips is not None
        
        for f in range(F):
            # --- CoM Access ---
            if self.current_com is not None:
                com = self.current_com
                if com.ndim == 2: c = com[f]
                else: c = com
            else:
                c = self._compute_full_body_com(world_pos[f:f+1])[0]
            
            # --- Apply One-Euro Filter to CoM ---
            # This smooths out motion capture noise to prevent acceleration spikes
            c = self.com_one_euro_filter(c.flatten())
            
            # --- CoM Acceleration (for push-off detection) ---
            com_acc = getattr(self, 'prob_prev_com_acc', np.zeros(3))
            
            # --- CoM Horizontal Projection ---
            com_hz = c[plane_dims]
            
            # --- Compute Load Factor for All Joints ---
            # Based on inverse distance from CoM projection
            load_factors = np.zeros(J)
            total_weight = 0.0
            
            # Use Gaussian weighting with large sigma for soft falloff
            # This prevents small CoM offsets from causing dramatic weight differences
            SIGMA_LOAD = 0.50  # 50cm - very soft falloff
            
            for j in range(J):
                # Only compute for potential contact joints
                if j not in FOOT_JOINTS and j not in HAND_JOINTS:
                    continue
                
                # Get joint position
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                
                pos_hz = pos[plane_dims]
                dist = np.sqrt(np.sum((pos_hz - com_hz)**2))
                
                # Gaussian weight: exp(-0.5 * (dist/sigma)^2)
                # At dist=sigma (50cm), weight = 0.61
                # At dist=2*sigma (1m), weight = 0.14
                # This is much softer than inverse-distance
                weight = np.exp(-0.5 * (dist / SIGMA_LOAD)**2)
                load_factors[j] = weight
                total_weight += weight
            
            # Normalize load factors
            if total_weight > 1e-6:
                load_factors = load_factors / total_weight
            
            # --- Height-Proximity Boost for Multi-Support Poses ---
            # If a joint is close to floor, boost its load factor regardless of CoM distance
            # This helps all-fours and crawling poses where hands bear significant weight
            for j in range(J):
                if j not in FOOT_JOINTS and j not in HAND_JOINTS:
                    continue
                
                # Get height
                if tips_available and j in self.temp_tips:
                    h = self.temp_tips[j][f][y_dim] - options.floor_height
                else:
                    h = world_pos[f, j, y_dim] - options.floor_height
                
                # Height-proximity boost: exponential increase as height decreases
                # At h=0, boost = 2.0 (double the load factor)
                # At h=SOFT_CEILING (0.25m), boost = 1.0 (no change)
                # At h>SOFT_CEILING, boost = 1.0
                if h <= 0:
                    height_boost = 2.0
                elif h < SOFT_CEILING:
                    # Linear interpolation: 2.0 at h=0, 1.0 at h=SOFT_CEILING
                    height_boost = 2.0 - 1.0 * (h / SOFT_CEILING)
                else:
                    height_boost = 1.0
                
                load_factors[j] *= height_boost
            
            # Re-normalize after height boost
            load_sum = np.sum(load_factors)
            if load_sum > 1e-6:
                load_factors = load_factors / load_sum
            # --- Multi-Support Detection (All-Fours, Crawling, etc.) ---
            # Detect when both hands and feet are close to floor
            # Use torque-based load distribution: joints opposite to CoM need more force
            
            MULTI_SUPPORT_THRESHOLD = 0.20  # Height threshold for "potentially in contact"
            
            # Check front (hands) and back (feet) contact candidates
            front_low = []  # Hand joints below threshold
            back_low = []   # Foot joints below threshold
            front_positions = {}  # joint -> horizontal position
            back_positions = {}
            
            for j in HAND_JOINTS:
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                h = pos[y_dim] - options.floor_height
                if h < MULTI_SUPPORT_THRESHOLD:
                    front_low.append(j)
                    front_positions[j] = pos[plane_dims]
            
            for j in FOOT_JOINTS:
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                h = pos[y_dim] - options.floor_height
                if h < MULTI_SUPPORT_THRESHOLD:
                    back_low.append(j)
                    back_positions[j] = pos[plane_dims]
            
            # If both front and back have low joints, we're in multi-support mode
            if len(front_low) > 0 and len(back_low) > 0:
                # Calculate current front/back load sums
                front_load = sum(load_factors[j] for j in HAND_JOINTS)
                back_load = sum(load_factors[j] for j in FOOT_JOINTS)
                total_load = front_load + back_load
                
                if total_load > 1e-6:
                    # Target: More balanced distribution for all-fours
                    # Use 40% front, 60% back (arms bear less than legs naturally)
                    TARGET_FRONT_RATIO = 0.40
                    TARGET_BACK_RATIO = 0.60
                    
                    current_front_ratio = front_load / total_load
                    
                    # Blend toward target: the more joints are low, the more balanced
                    blend_factor = min(1.0, (len(front_low) + len(back_low)) / 6.0)
                    
                    target_front = TARGET_FRONT_RATIO * blend_factor + current_front_ratio * (1.0 - blend_factor)
                    target_back = TARGET_BACK_RATIO * blend_factor + (1.0 - current_front_ratio) * (1.0 - blend_factor)
                    
                    # Scale factors to achieve target distribution
                    if front_load > 1e-6:
                        front_scale = (target_front * total_load) / front_load
                    else:
                        front_scale = 1.0
                    
                    if back_load > 1e-6:
                        back_scale = (target_back * total_load) / back_load
                    else:
                        back_scale = 1.0
                    
                    # Apply scaling
                    for j in HAND_JOINTS:
                        load_factors[j] *= front_scale
                    for j in FOOT_JOINTS:
                        load_factors[j] *= back_scale
                    
                    # Re-normalize
                    load_sum = np.sum(load_factors)
                    if load_sum > 1e-6:
                        load_factors = load_factors / load_sum
            
            # --- Smooth Load Factors (reduces oscillation from normalization changes) ---
            LOAD_SMOOTH_ALPHA = 0.15  # 15% new, 85% old - more stable
            for j in range(J):
                self.prob_load_factors_smooth[j] = (
                    self.prob_load_factors_smooth[j] * (1.0 - LOAD_SMOOTH_ALPHA) + 
                    load_factors[j] * LOAD_SMOOTH_ALPHA
                )
            load_factors = self.prob_load_factors_smooth.copy()
            
            # --- Acceleration Boost ---
            # If CoM is accelerating, boost joints in the opposite direction (push-off)
            # Using conservative threshold and boost to avoid spiky contact probs
            acc_mag = np.linalg.norm(com_acc[plane_dims])
            if acc_mag > 2.0:  # Increased threshold (was 0.5)
                acc_dir = -com_acc[plane_dims] / (acc_mag + 1e-6)  # Opposite direction
                
                for j in FOOT_JOINTS:
                    if tips_available and j in self.temp_tips:
                        pos = self.temp_tips[j][f]
                    else:
                        pos = world_pos[f, j]
                    
                    pos_hz = pos[plane_dims]
                    joint_dir = pos_hz - com_hz
                    joint_dir_norm = np.linalg.norm(joint_dir)
                    if joint_dir_norm > 0.01:
                        joint_dir = joint_dir / joint_dir_norm
                        alignment = np.dot(joint_dir, acc_dir)
                        if alignment > 0:
                            # This joint is in the push direction - boost load (reduced from 30% to 15%)
                            boost = alignment * 0.15  # Up to 15% boost
                            load_factors[j] = min(1.0, load_factors[j] + boost)
            
            # --- Chain-Relative Height Suppression ---
            # Within each limb chain, suppress load factors for joints that are 
            # significantly higher than the lowest joint in the chain.
            # This prevents knee contact when foot is clearly lower.
            
            def get_joint_pos(joint_idx):
                """Get 3D position of a joint, using virtual tip if available."""
                if tips_available and joint_idx in self.temp_tips:
                    return self.temp_tips[joint_idx][f]
                elif joint_idx < world_pos.shape[1]:
                    return world_pos[f, joint_idx]
                else:
                    return None
            
            def apply_angle_chain_suppression(chain):
                """Height-based suppression within a chain with smoothing.
                
                Suppress joints that are higher than the lowest joint.
                Smooths the suppression factors over time for stable output.
                """
                SUPPRESS_RATE = 0.03      # 3cm - exponential decay rate
                SMOOTH_ALPHA = 0.15       # EMA alpha for smoothing suppression factors
                
                # Initialize suppression smoothing state if needed
                if not hasattr(self, '_chain_suppression_smooth'):
                    self._chain_suppression_smooth = {}
                
                chain_key = tuple(chain)
                if chain_key not in self._chain_suppression_smooth:
                    self._chain_suppression_smooth[chain_key] = {j: 1.0 for j in chain}
                
                # Get heights of all joints in chain
                chain_data = {}  # joint_idx -> height
                for joint_idx in chain:
                    joint_pos = get_joint_pos(joint_idx)
                    if joint_pos is not None:
                        chain_data[joint_idx] = joint_pos[y_dim]
                
                if len(chain_data) < 2:
                    return
                
                # Find lowest joint
                lowest_height = min(chain_data.values())
                
                # Calculate and smooth suppression factors
                for j, h in chain_data.items():
                    h_above = h - lowest_height
                    if h_above > 0:
                        raw_factor = np.exp(-h_above / SUPPRESS_RATE)
                    else:
                        raw_factor = 1.0  # No suppression for lowest
                    
                    # Smooth the suppression factor
                    prev_factor = self._chain_suppression_smooth[chain_key].get(j, raw_factor)
                    smooth_factor = prev_factor * (1 - SMOOTH_ALPHA) + raw_factor * SMOOTH_ALPHA
                    self._chain_suppression_smooth[chain_key][j] = smooth_factor
                    
                    # Apply smoothed suppression
                    load_factors[j] *= smooth_factor
            
            # Apply to all limb chains
            apply_angle_chain_suppression(LEFT_LEG_CHAIN)
            apply_angle_chain_suppression(RIGHT_LEG_CHAIN)
            apply_angle_chain_suppression(LEFT_ARM_CHAIN)
            apply_angle_chain_suppression(RIGHT_ARM_CHAIN)
            
            # Re-normalize after chain suppression to prevent cross-limb interference
            load_sum = np.sum(load_factors)
            if load_sum > 1e-6:
                load_factors = load_factors / load_sum
            
            # --- Per-Joint Contact Calculation ---
            for j in range(J):
                # Get position
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                
                h = pos[y_dim] - floor_height
                
                # Apply heel bias for ankles
                if j in [7, 8]:
                    h -= options.heel_toe_bias
                
                # --- Load Factor (Primary for feet) ---
                p_load = load_factors[j]
                
                # --- Velocity (from smoothed state) ---
                if hasattr(self, 'prob_smoothed_vel') and self.prob_smoothed_vel is not None:
                    lin_vel = self.prob_smoothed_vel  # (J, 3)
                else:
                    lin_vel = np.zeros((J, 3))
                
                vy = lin_vel[j, y_dim]
                vel_h_vec = lin_vel[j].copy()
                vel_h_vec[y_dim] = 0.0
                vh = np.linalg.norm(vel_h_vec)
                v_total = np.sqrt(vy**2 + vh**2)
                
                # --- Height Gating ---
                if j in HAND_JOINTS:
                    # Stricter gating for hands - must be very close to ground
                    if h > HAND_HARD_CEILING:
                        p_height = 0.0
                    elif h > HAND_SOFT_CEILING:
                        # Linear falloff between soft and hard ceiling
                        p_height = 1.0 - (h - HAND_SOFT_CEILING) / (HAND_HARD_CEILING - HAND_SOFT_CEILING)
                    else:
                        p_height = 1.0
                elif j in FOOT_JOINTS:
                    # Soft gating for feet (height has minimal effect below ceiling)
                    if h > HARD_CEILING:
                        p_height = 0.0
                    elif h > SOFT_CEILING:
                        # Very gradual falloff
                        p_height = np.exp(-2.0 * (h - SOFT_CEILING) / SOFT_CEILING)
                    elif h <= 0:
                        p_height = 1.0
                    else:
                        # Below soft ceiling: minimal height penalty
                        p_height = 1.0 - 0.2 * (h / SOFT_CEILING)  # At most 20% penalty
                else:
                    # Non-contact joints: hard cutoff
                    p_height = 0.0 if h > 0.10 else 1.0
                
                # --- Velocity Penalty (Suppressed by Load) ---
                # Softer base velocity penalty for IMU noise tolerance
                v_threshold = 0.4  # Higher threshold for IMU noise
                if v_total < v_threshold:
                    p_vel_base = 1.0
                else:
                    # Gentler exponential: -3.0 instead of -5.0
                    p_vel_base = np.exp(-3.0 * (v_total - v_threshold))
                
                # Suppress velocity penalty when loaded (IMU velocity is noisy)
                # High load -> almost ignore velocity
                # Low load -> velocity matters
                vel_suppression = 0.9 * p_load  # Increased from 0.85
                p_vel = p_vel_base + (1.0 - p_vel_base) * vel_suppression
                p_vel = min(1.0, p_vel)
                
                # --- Lift-off Check with Hysteresis ---
                # Once lifted, stay lifted until consistent descent pattern
                # IMPORTANT: Only mark as lifted if joint is actually rising AND above minimum height
                
                # Initialize lift state tracking if needed
                if not hasattr(self, '_lift_state'):
                    self._lift_state = {}  # joint_idx -> {'lifted': bool, 'lift_frames': int, 'descent_frames': int, 'prev_h': float}
                
                if j not in self._lift_state:
                    self._lift_state[j] = {'lifted': False, 'lift_frames': 0, 'descent_frames': 0, 'prev_h': h}
                
                state = self._lift_state[j]
                
                # Constants for lift-off detection
                MIN_LIFT_HEIGHT = 0.12  # 12cm - must be above this to be considered "lifted"
                VEL_THRESHOLD = 0.05    # 5cm/s - significant velocity threshold
                HEIGHT_RISING_THRESHOLD = 0.002  # 2mm/frame - check if height is actually increasing
                
                # Check if height is actually rising
                height_rising = h > state['prev_h'] + HEIGHT_RISING_THRESHOLD
                height_falling = h < state['prev_h'] - HEIGHT_RISING_THRESHOLD
                state['prev_h'] = h
                
                # Check current motion pattern
                # Must have BOTH positive velocity AND height actually increasing
                is_ascending = vy > VEL_THRESHOLD and height_rising
                is_descending = vy < -VEL_THRESHOLD or height_falling
                
                # Track lift/descent frame counts
                if is_ascending:
                    state['lift_frames'] = min(state['lift_frames'] + 1, 10)
                    state['descent_frames'] = 0
                elif is_descending:
                    state['descent_frames'] = min(state['descent_frames'] + 1, 10)
                    state['lift_frames'] = max(0, state['lift_frames'] - 1)
                else:
                    # Near-zero velocity: decay both counters
                    state['lift_frames'] = max(0, state['lift_frames'] - 1)
                    state['descent_frames'] = max(0, state['descent_frames'] - 1)
                
                # State transitions
                LIFT_THRESHOLD = 4  # Frames needed to confirm lift-off
                DESCENT_THRESHOLD = 3  # Frames needed to confirm re-contact
                
                if not state['lifted']:
                    # Not currently lifted - check for lift-off
                    # MUST be above minimum height AND have consistent lift motion
                    if state['lift_frames'] >= LIFT_THRESHOLD and h > MIN_LIFT_HEIGHT:
                        state['lifted'] = True
                else:
                    # Currently lifted - check for re-contact
                    # Can return to contact if: descending OR height drops below threshold
                    if state['descent_frames'] >= DESCENT_THRESHOLD or h < MIN_LIFT_HEIGHT * 0.8:
                        state['lifted'] = False
                
                # Calculate p_lift based on state
                if state['lifted']:
                    # Joint is lifted - strong suppression
                    # But still allow some response if velocity becomes very negative
                    if vy < -0.15:
                        # Fast descent - allow re-contact
                        p_lift = min(1.0, 1.0 + vy * 4.0)  # At vy=-0.25, p_lift=0
                    else:
                        p_lift = 0.2  # Maintain low probability while lifted
                else:
                    # Not lifted - normal upward velocity suppression (softer)
                    if vy > 0:
                        # Weaker suppression - mainly to smooth transitions
                        vel_suppression_strength = 3.0 * (1.0 - min(1.0, p_load / 0.5))
                        p_lift = np.exp(-vel_suppression_strength * vy)
                    else:
                        p_lift = 1.0
                
                # --- Fusion ---
                # For feet: load is primary, height is soft gating
                # For hands: height is hard gating, load and lift-off secondary
                if j in FOOT_JOINTS:
                    # Load-driven: p_load * soft_height * lift_check
                    p_raw = p_load * p_height * p_lift
                elif j in HAND_JOINTS:
                    # Height-gated: Load and lift-off only matter if height allows
                    # Rising hands should have reduced probability
                    p_raw = p_height * p_load * p_lift
                else:
                    p_raw = 0.0
                
                # Clamp
                p_raw = max(0.0, min(1.0, p_raw))
                
                # --- No Output Smoothing ---
                # Raw probabilities pass through directly
                # (Noise is handled upstream by one-euro filter on CoM and load factor smoothing)
                contact_probs[f, j] = p_raw
        
        # --- Aggregate Virtual Joints into Parents ---
        # Merge virtual joint probs into parent joints for visualization compatibility
        # (motion capture systems often don't track knuckle/ball-of-foot angles)
        # toe (24, 25) -> foot (10, 11)
        # heel (28, 29) -> ankle (7, 8)
        # finger (26, 27) -> hand (22, 23)
        for f in range(F):
            # Left side
            contact_probs[f, 10] = min(1.0, contact_probs[f, 10] + contact_probs[f, 24])  # toe -> foot
            contact_probs[f, 7] = min(1.0, contact_probs[f, 7] + contact_probs[f, 28])   # heel -> ankle
            contact_probs[f, 22] = min(1.0, contact_probs[f, 22] + contact_probs[f, 26]) # finger -> hand
            
            # Right side
            contact_probs[f, 11] = min(1.0, contact_probs[f, 11] + contact_probs[f, 25])  # toe -> foot
            contact_probs[f, 8] = min(1.0, contact_probs[f, 8] + contact_probs[f, 29])   # heel -> ankle
            contact_probs[f, 23] = min(1.0, contact_probs[f, 23] + contact_probs[f, 27]) # finger -> hand
        
        return contact_probs

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
        
        # Build consensus options from processing options.
        # Use adaptive inferred floor if available (from previous frame).
        floor_for_consensus = options.floor_height
        if hasattr(self, '_inferred_floor_height') and self._inferred_floor_height is not None:
            floor_for_consensus = self._inferred_floor_height
        
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
            
            probs = self._consensus.compute_contacts(pos, com, options.dt, consensus_opts)
            
            # Ensure output size matches
            contact_probs[f, :min(J, len(probs))] = probs[:min(J, len(probs))]
        
        return contact_probs

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
            
            if 'pelvis' in name: val = [500.0, 200.0, 500.0] # Core is strong in all axes
            elif 'hip' in name: val = [300.0, 50.0, 150.0] # Flex, Twist, Abd
            elif 'knee' in name: val = [250.0, 20.0, 20.0] # Hinge (Primary X)
            elif 'ankle' in name: val = [150.0, 20.0, 40.0] # Dorsi/Plantar strong
            elif 'foot' in name: val = [40.0, 10.0, 10.0]
            
            elif 'spine' in name: val = [400.0, 100.0, 300.0] # Flex, Twist, Bend
            elif 'neck' in name: val = [50.0, 20.0, 40.0]
            elif 'head' in name: val = [30.0, 10.0, 20.0]
            
            elif 'collar' in name: val = [100.0, 100.0, 500.0] # Structural Z support
            elif 'shoulder' in name: val = [60.0, 120.0, 100.0] # Twist, Flex, Abd
            elif 'elbow' in name: val = [20.0, 100.0, 20.0] # Hinge (Primary Y)
            elif 'wrist' in name: val = [10.0, 30.0, 20.0]
            elif 'hand' in name: val = [10.0, 10.0, 10.0]
            
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
            'pelvis': [500.0, 300.0, 500.0],
            'spine': [400.0, 100.0, 300.0],
            'hip': [300.0, 150.0, 200.0], 
            'knee': [250.0, 100.0, 100.0], # Hinge
            'ankle': [200.0, 60.0, 80.0],
            'foot': [80.0, 20.0, 20.0],
            'neck': [50.0, 20.0, 40.0],
            'head': [30.0, 10.0, 20.0],
            'collar': [100.0, 100.0, 500.0], # Z support
            'shoulder': [60.0, 120.0, 100.0],
            'elbow': [20.0, 100.0, 20.0], # Hinge
            'wrist': [10.0, 30.0, 20.0],
            'hand': [10.0, 10.0, 10.0]
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
        
        Args:
            world_positions: (F, J, 3) world positions
            
        Returns:
            inertias: (F, target_joint_count) effective inertia per joint
        """
        F = world_positions.shape[0]
        parents = self._get_hierarchy()
        inertias = np.zeros((F, self.target_joint_count))
        
        for j in range(self.target_joint_count):
            joint_pos = world_positions[:, j, :]  # (F, 3)
            total_inertia = np.zeros(F)
            
            for idx in self._subtree_members[j]:
                if self._seg_is_leaf_skip[idx]:
                    continue
                
                m = self._seg_mass[idx]
                if m <= 0:
                    continue
                
                l = self._seg_length[idx]
                i_local = self._seg_local_inertia[idx]
                
                # Compute COM position for this segment
                if self._seg_has_children[idx]:
                    child_nodes = self._hierarchy_children[idx]
                    # Vectorized mean of children positions
                    child_positions = world_positions[:, child_nodes, :]  # (F, n_children, 3)
                    end_pos = np.mean(child_positions, axis=1)  # (F, 3)
                    com_pos = (world_positions[:, idx, :] + end_pos) * 0.5
                else:
                    # Leaf node
                    p_idx = self._seg_leaf_parent[idx]
                    if p_idx != -1:
                        dir_vec = joint_pos - world_positions[:, p_idx, :]
                        norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
                        norm_safe = np.maximum(norm, 1e-6)
                        dir_vec_normalized = dir_vec / norm_safe
                        
                        is_small = (norm < 1e-6).flatten()
                        if np.any(is_small):
                            dir_vec_normalized[is_small] = np.array([0.0, 0.0, 1.0])
                        
                        com_pos = joint_pos + dir_vec_normalized * (l * 0.5)
                    else:
                        com_pos = world_positions[:, idx, :]
                
                # Distance from pivot to segment COM
                r_vec = com_pos - joint_pos
                r_sq = np.sum(r_vec**2, axis=-1)  # (F,)
                
                # Parallel axis theorem
                total_inertia += (i_local + m * r_sq)
            
            inertias[:, j] = total_inertia
        
        return inertias

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
        parents = self._get_hierarchy()
        limb_masses = self.limb_data['masses']
        limb_lengths = self.limb_data['lengths']
        
        total_mass = 0.0
        F = world_positions.shape[0]
        weighted_pos_sum = np.zeros((F, 3))
        
        for idx in range(24):
            name = self.joint_names[idx]
            
            # Map joint name to limb data keys
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
            # Hands/Others ignored or lumped
            
            if m <= 0: continue
            
            # Segment COM estimation
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            com_pos = world_positions[..., idx, :]
            if len(child_nodes) > 0:
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0) # (F, 3)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            
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
        
        # Non-leaf joints: seg_com = midpoint(joint, mean(children))
        for idx in self._grav_nonleaf:
            m = joint_segment_masses[idx]
            joint_pos = world_pos[:, idx, :]
            child_nodes = children_list[idx]
            child_positions = world_pos[:, child_nodes, :]  # (F, n_children, 3)
            end_pos = np.mean(child_positions, axis=1)  # (F, 3)
            seg_com = (joint_pos + end_pos) * 0.5
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
                      target_hip = leg_map.get(j, -1)
                      
                      # Alignment Logic (Only for Ankles to stabilize main leg strut)
                      # For Toes/Heels, we want Vertical Force to preserve Lever Arm Torque (Balance)
                      use_alignment = (target_hip != -1) and (options.floor_enable) and (j in [7, 8])
                      
                      if use_alignment:
                           v_leg = world_pos[:, target_hip, :] - world_pos[:, j, :] # (F, 3)
                           dy = v_leg[:, 1]
                           
                           # Valid Alignment Check (Leg not horizontal)
                           valid = dy > 0.1
                           
                           # Aligned Force: Scale so Y-component matches f_vert_mag
                           scale = f_vert_mag[:, j] / (dy + 1e-9)
                           f_aligned = v_leg * scale[:, np.newaxis]
                           
                           # Vertical Fallback (Up)
                           # g_vec_eff is Down. -g is Up.
                           # Normalized Direction
                           dir_up = -g_vec_eff / g_mag # (F, 3)
                           f_vertical = f_vert_mag[:, j, np.newaxis] * dir_up
                           
                           # Blend
                           mask_valid = valid
                           f_grf_vecs[mask_valid, j, :] = f_aligned[mask_valid]
                           f_grf_vecs[~mask_valid, j, :] = f_vertical[~mask_valid]
                           
                      else:
                           # Default Vertical
                           dir_up = -g_vec_eff / g_mag
                           f_grf_vecs[:, j, :] = f_vert_mag[:, j, np.newaxis] * dir_up

             # 2. Backward Pass (Accumulate GRF Moments)
             f_cum_grf = f_grf_vecs.copy()
             m_cum_grf = np.zeros_like(f_grf_vecs)
             
             parents = np.array(self._get_hierarchy())
             J = world_pos.shape[1]
             
             for j in range(J-1, 0, -1):
                 parent = parents[j]
                 if parent >= 0:
                      # Propagate Force
                      f_child = f_cum_grf[:, j, :]
                      f_cum_grf[:, parent, :] += f_child
                      
                      # Propagate Moment
                      # M_parent += M_child + (P_child - P_parent) x F_child
                      m_child = m_cum_grf[:, j, :]
                      r_bone = world_pos[:, j, :] - world_pos[:, parent, :]
                      t_lever = np.cross(r_bone, f_child)
                      
                      m_cum_grf[:, parent, :] += m_child + t_lever
                      
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
        # No rotation-level filtering on global — raw FK rotations are smooth
        # (visually confirmed by rendered body). Only the per-joint velocity
        # filter (step 5b) smooths the composed angular velocity.
        composed_q = np.zeros((n_joints, 4))
        for j in range(n_joints):
            p = parents[j] if j < len(parents) else -1
            r_local = R.from_quat(filtered_local_q[j])
            if p >= 0:
                r_parent_raw = R.from_quat(curr_global_q[p])
                r_composed = r_parent_raw * r_local
            else:
                # Root: world rotation IS the local rotation
                r_composed = r_local
            composed_q[j] = r_composed.as_quat()
        
        # 5. Differentiate composed rotation for angular velocity
        prev_composed = getattr(self, name_prev_composed, None)
        if prev_composed is None:
            ang_vel = np.zeros((1, 24, 3))
            ang_acc = np.zeros((1, 24, 3))
            setattr(self, name_prev_composed, composed_q.copy())
            setattr(self, name_prev_vel, np.zeros((n_joints, 3)))
            return ang_vel, ang_acc
        
        # Compute angular velocity from composed rotation differences
        vel = np.zeros((n_joints, 3))
        for j in range(n_joints):
            r_curr_c = R.from_quat(composed_q[j])
            r_prev_c = R.from_quat(prev_composed[j])
            # Ensure shortest path
            if np.dot(composed_q[j], prev_composed[j]) < 0:
                r_prev_c = R.from_quat(-prev_composed[j])
            r_diff = r_curr_c * r_prev_c.inv()
            vel[j] = r_diff.as_rotvec() / dt
        
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
        
        # 6. Compute acceleration from FILTERED velocity differences
        prev_vel = getattr(self, name_prev_vel, None)
        if prev_vel is None or prev_vel.shape != smooth_vel.shape:
            acc = np.zeros((n_joints, 3))
        else:
            acc = (smooth_vel - prev_vel) / dt
        
        # Update state
        setattr(self, name_prev_composed, composed_q.copy())
        setattr(self, name_prev_vel, smooth_vel.copy())  # Store FILTERED velocity
        
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
            
            # Acceleration
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
        efforts_net = np.zeros((F, self.target_joint_count))
        efforts_dyn = np.zeros((F, self.target_joint_count))
        efforts_grav = np.zeros((F, self.target_joint_count))
        
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
        
        # --- Update Velocity Envelope (for gating after rate limiting) ---
        # Track the per-joint velocity envelope BEFORE rate limiting
        # so it accurately reflects actual motion, but apply the gate AFTER
        # rate limiting to avoid cascading suppression (gate zeroes quiet
        # frames → rate limiter can't ramp up from zero fast enough).
        VEL_GATE_SIGMA = 0.5    # rad/s — gate at 50% when envelope ≈ 29°/s
        VEL_ENVELOPE_TAU = 0.10  # seconds — envelope decay time constant
        
        ang_vel_for_gate = getattr(self, '_current_ang_vel', None)
        vel_gate = None  # (target_joint_count,) gate values, computed now, applied later
        if ang_vel_for_gate is not None:
            if ang_vel_for_gate.ndim == 2:
                ang_vel_for_gate = ang_vel_for_gate[np.newaxis, ...]
            
            # Initialize or retrieve the velocity envelope
            if not hasattr(self, '_vel_envelope') or self._vel_envelope is None:
                self._vel_envelope = np.zeros(self.target_joint_count)
            if self._vel_envelope.shape[0] != self.target_joint_count:
                self._vel_envelope = np.zeros(self.target_joint_count)
            
            gate_dt = options.dt if hasattr(options, 'dt') and options.dt > 0 else 1.0 / max(self.framerate, 1.0)
            decay = np.exp(-gate_dt / VEL_ENVELOPE_TAU)  # ~0.920 at 120fps, ~0.717 at 30fps
            
            vel_gate = np.ones(self.target_joint_count)
            for j in range(self.target_joint_count):
                vel_mag = np.linalg.norm(ang_vel_for_gate[:F, j, :], axis=-1)  # (F,)
                for fi in range(F):
                    # Leaky max: rise instantly, decay slowly
                    self._vel_envelope[j] = max(vel_mag[fi], self._vel_envelope[j] * decay)
                
                vel_gate[j] = 1.0 - np.exp(-(self._vel_envelope[j] / VEL_GATE_SIGMA) ** 2)
        
        # --- Apply Rate Limiting to Raw Dynamic Torques ---
        if dyn_override_world is not None:
            # CoM-based dynamic torque provided — convert from world to local frame
            # and skip I×α rate limiting and velocity gate entirely.
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
            
            # --- Apply Velocity Gate (post rate-limiting) ---
            # Suppress dynamic torque when angular velocity is near-zero (stationary
            # joints). Applied AFTER rate limiting so the rate limiter can ramp up
            # properly during motion onset from the raw signal.
            if vel_gate is not None and getattr(options, 'enable_velocity_gate', True):
                for j in range(min(self.target_joint_count, t_dyn_limited.shape[1])):
                    t_dyn_limited[:, j, :] *= vel_gate[j]
        
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
        
        # Batch transform gravity and contact to local frame: (F, J, 3, 3) @ (F, J, 3, 1) -> (F, J, 3)
        t_grav_local_all = np.einsum('fjik,fjk->fji', parent_inv_mats, t_grav_vecs[:, :self.target_joint_count])
        t_contact_local_all = np.einsum('fjik,fjk->fji', parent_inv_mats, t_contact_vecs[:, :self.target_joint_count])
        
        # Dynamic is already in local frame
        t_dyn_vecs[:, :self.target_joint_count] = t_dyn_limited[:, :self.target_joint_count]
        
        # Net local = dyn - grav_local - contact_local
        t_net_all = t_dyn_limited[:, :self.target_joint_count] - t_grav_local_all - t_contact_local_all
        
        # Store local-frame gravity for output consistency
        t_grav_vecs[:, :self.target_joint_count] = t_grav_local_all
        
        # Passive limits (per-joint, can't easily vectorize due to name-based lookup)
        t_passive_all = np.zeros_like(t_net_all)
        if options.enable_passive_limits:
            for j in range(self.target_joint_count):
                name = self.joint_names[j]
                curr_pose_aa = pose_data_aa[:, j, :]
                t_p = self._compute_passive_torque(name, t_net_all[:, j], curr_pose_aa)
                
                # Optimal passive support clipping
                mask_neg = t_p < 0
                t_p[mask_neg] = np.clip(t_net_all[:, j][mask_neg], t_p[mask_neg], 0)
                mask_pos = t_p >= 0
                t_p[mask_pos] = np.clip(t_net_all[:, j][mask_pos], 0, t_p[mask_pos])
                t_passive_all[:, j] = t_p
        
        t_passive_vecs[:, :self.target_joint_count] = t_passive_all
        
        # Active = net - passive
        t_active_all = t_net_all - t_passive_all
        torques_vec[:, :self.target_joint_count] = t_active_all
        
        # Efforts (vectorized over all joints)
        max_torque = self.max_torque_array[:self.target_joint_count]  # (J, 3)
        denom = max_torque + 1e-6  # (J, 3)
        efforts_net[:, :self.target_joint_count] = np.linalg.norm(
            np.abs(t_active_all) / denom[np.newaxis, :, :], axis=-1)
        efforts_dyn[:, :self.target_joint_count] = np.linalg.norm(
            np.abs(t_dyn_limited[:, :self.target_joint_count]) / denom[np.newaxis, :, :], axis=-1)
        efforts_grav[:, :self.target_joint_count] = np.linalg.norm(
            np.abs(t_grav_local_all) / denom[np.newaxis, :, :], axis=-1)

        return torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs



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
        stability_score = float(1.0 / (1.0 + np.exp(-zmp_margin / FALLOFF)))

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
        
        # --- Optional Input Smoothing ---
        # Causal moving average on both pose and trans to remove sensor cadence artifacts.
        # Applied BEFORE _prepare_trans_and_pose so FK and all downstream use consistently smoothed data.
        win = options.smooth_input_window
        if win >= 2:
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
                # Ring buffer for streaming
                if not hasattr(self, '_input_smooth_ring') or self._input_smooth_ring is None \
                        or self._input_smooth_ring.get('n_pose') != n_pose \
                        or self._input_smooth_ring.get('win') != win:
                    self._input_smooth_ring = {
                        'win': win,
                        'n_pose': n_pose,
                        'n_trans': n_trans,
                        'pose_buf': np.tile(p_flat, (win, 1)),
                        'trans_buf': np.tile(t_flat, (win, 1)),
                        'idx': 0,
                    }
                ring = self._input_smooth_ring
                ring['pose_buf'][ring['idx']] = p_flat
                ring['trans_buf'][ring['idx']] = t_flat
                ring['idx'] = (ring['idx'] + 1) % win
                
                pose_data = np.mean(ring['pose_buf'], axis=0).reshape(pose_data.shape)
                trans_data = np.mean(ring['trans_buf'], axis=0).reshape(trans_data.shape)
            else:
                # Batch mode: causal moving average with edge padding
                F_in = pose_data.shape[0] if pose_data.ndim >= 2 else 1
                if F_in > 1:
                    p2d = pose_data.reshape(F_in, -1)
                    t2d = trans_data.reshape(F_in, -1)
                    p_pad = np.concatenate([np.tile(p2d[0:1], (win - 1, 1)), p2d], axis=0)
                    t_pad = np.concatenate([np.tile(t2d[0:1], (win - 1, 1)), t2d], axis=0)
                    from numpy.lib.stride_tricks import sliding_window_view
                    pose_data = np.mean(sliding_window_view(p_pad, win, axis=0), axis=-1).reshape(pose_data.shape)
                    trans_data = np.mean(sliding_window_view(t_pad, win, axis=0), axis=-1).reshape(trans_data.shape)
        
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
        else:
            # Legacy local-frame: parent-relative angular acceleration
            ang_acc = ang_acc_local
            self._current_ang_vel = ang_vel_local  # Local velocity for gating
        
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
        
        # --- Velocity Prep for Fusion ---
        smoothed_vel_vec = self.prob_smoothed_vel # (J, 3)
        if F == 1:
             vel_y_in = smoothed_vel_vec[np.newaxis, :, getattr(self, 'internal_y_dim', 1)]
             # Horizontal
             yd = getattr(self, 'internal_y_dim', 1)
             pd = [0, 2] if yd == 1 else [0, 1]
             vel_h_in = np.linalg.norm(smoothed_vel_vec[np.newaxis, :, pd], axis=-1)
        else:
             # Batch - simplified
             vel_y_in = np.zeros((F, world_pos.shape[1]))
             vel_h_in = np.zeros((F, world_pos.shape[1]))
        
        # --- Contact Method Selection ---
        if options.contact_method == 'com_driven':
             contact_probs_fusion = self._compute_probabilistic_contacts_com_driven(
                  F, world_pos.shape[1], world_pos, options.floor_height, options
             )
        elif options.contact_method == 'consensus':
             contact_probs_fusion = self._compute_probabilistic_contacts_consensus(
                  F, world_pos.shape[1], world_pos, options
             )
        elif options.contact_method == 'stability':
             contact_probs_fusion = self._compute_probabilistic_contacts_stability(
                  F, world_pos.shape[1], world_pos, options
             )
        else:  # Default: 'fusion'
             contact_probs_fusion = self._compute_probabilistic_contacts_fusion(
                  F, world_pos.shape[1], world_pos, vel_y_in, vel_h_in, options.floor_height, options
             )
        
        # --- Torque-Aware Contact Refinement Loop ---
        # Iteratively: compute floor pressure → torques → check for
        # torque discontinuities → if a contact change explains the burst,
        # refine the contacts → re-compute. Real torque bursts from genuine
        # motion are left alone for downstream rate limiting.
        
        # Per-frame cache: shared across repeated _compute_joint_torques calls.
        # Avoids recomputing global_rot_invs, inertias, and t_dyn_raw which are
        # identical between calls (only contact_pressure changes).
        _frame_cache = {}
        
        working_probs = contact_probs_fusion.copy()
        
        if True:  # Single pass (refinement loop removed)
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
            FLOOR_ALPHA = 0.02        # Very slow EMA
            FLOOR_MAX_CHANGE = 0.002  # Max 2mm per frame
            if not hasattr(self, '_inferred_floor_height') or self._inferred_floor_height is None:
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
                      p_weight = p if (options.contact_method == 'stability' and hasattr(self, '_stability_boost') and self._stability_boost[j] > 0.05) else (p * p)
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
            if options.contact_method == 'stability':
                stab_press = getattr(self, '_stability_computed_pressure', None)
                if stab_press is not None and len(stab_press) == self.contact_pressure.shape[1]:
                    for f in range(F):
                        self.contact_pressure[f] = stab_press
        
        # --- Contact Pressure Smoothing (Asymmetric + Rate Clamp) ---
        # Runs OUTSIDE the refinement loop so it always applies.
        # Time-constant based smoothing (framerate-adaptive).
        TAU_UP   = 0.150  # seconds — build-up time constant
        TAU_DOWN = 0.050  # seconds — release time constant
        MAX_RATE = 5.0 * self.total_mass_kg  # kg/s — max pressure change rate per joint
        
        # Convert time constants to per-frame alphas
        alpha_up   = 1.0 - np.exp(-dt / TAU_UP)    # ~0.054 at 120fps, ~0.189 at 30fps
        alpha_down = 1.0 - np.exp(-dt / TAU_DOWN)   # ~0.154 at 120fps, ~0.487 at 30fps
        max_change_per_frame = MAX_RATE * dt          # kg/frame
        
        if not hasattr(self, 'prev_contact_pressure_smooth') or self.prev_contact_pressure_smooth is None:
            self.prev_contact_pressure_smooth = self.contact_pressure.copy()
        
        if self.prev_contact_pressure_smooth.shape != self.contact_pressure.shape:
            self.prev_contact_pressure_smooth = self.contact_pressure.copy()
        
        for f in range(F):
            curr = self.contact_pressure[f]
            prev = self.prev_contact_pressure_smooth[0] if F == 1 else self.prev_contact_pressure_smooth[f]
            
            # Soft alpha blending based on drop magnitude.
            # Instead of a binary switch (which is fragile near zero change),
            # blend smoothly: small changes → slow alpha (stable),
            # large drops → fast alpha (responsive release).
            drop = np.maximum(0, prev - curr)  # How much each joint is dropping
            drop_ratio = drop / (np.maximum(prev, 1.0))  # Relative drop (safe div)
            blend = np.clip(drop_ratio / 0.3, 0, 1)  # 0→30% drop maps to 0→1
            alpha = alpha_up * (1 - blend) + alpha_down * blend
            
            smoothed = prev * (1.0 - alpha) + curr * alpha
            # Rate clamp: prevent single-frame radical shifts
            delta = smoothed - prev
            delta = np.clip(delta, -max_change_per_frame, max_change_per_frame)
            smoothed = prev + delta
            self.contact_pressure[f] = smoothed
            if F == 1:
                self.prev_contact_pressure_smooth = smoothed[np.newaxis, :].copy()
            else:
                self.prev_contact_pressure_smooth[f] = smoothed
        
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

        # --- Output Dictionary ---
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
